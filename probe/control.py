"""Damped-least-squares differential IK on the grip site, tracked by the UR5e's position servos.

The IK target is the GRIP, never the tip: the tip goes wherever the physics sends it.

IK runs on a private kinematic copy of the model state (q_des), so servo tracking error under
contact load does not feed back into the reference: the arm behaves like a stiff-but-finite
hand following the recorded grip trajectory. The *achieved* grip pose is what gets logged.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .geometry import interp_poses
from .scene import HOME_Q, arm_actuator_ids, arm_joint_dof_adr, arm_joint_ids, arm_joint_qpos_adr
from .schema import GripTrajectory


class GripIK:
    def __init__(self, m: mujoco.MjModel, site_id: int, damping: float = 0.03, null_gain: float = 0.05,
                 rot_weight: float = 0.5, max_step: float = 0.05, iters: int = 3, home=HOME_Q):
        self.m, self.site = m, site_id
        self.d = mujoco.MjData(m)
        self.jid = arm_joint_ids(m)
        self.qadr = arm_joint_qpos_adr(m)
        self.vadr = arm_joint_dof_adr(m)
        lim = m.jnt_range[self.jid].copy()
        lim[~m.jnt_limited[self.jid].astype(bool)] = (-np.inf, np.inf)
        self.lo, self.hi = lim[:, 0], lim[:, 1]
        self.lam, self.kn, self.wr, self.max_step, self.iters = damping, null_gain, rot_weight, max_step, iters
        self.home = np.asarray(home, float)
        self.q = self.home.copy()
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))

    def reset(self, q):
        self.q = np.asarray(q, float).copy()

    def _fk(self):
        self.d.qpos[self.qadr] = self.q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def pose_error(self, pos, quat):
        self._fk()
        ep = pos - self.d.site_xpos[self.site]
        cur = np.zeros(4)
        mujoco.mju_mat2Quat(cur, self.d.site_xmat[self.site])
        er = np.zeros(3)
        mujoco.mju_subQuat(er, np.asarray(quat, float), cur)  # rotation cur -> target, local frame
        er = self.d.site_xmat[self.site].reshape(3, 3) @ er  # to world
        return ep, er

    def solve(self, pos, quat) -> np.ndarray:
        """A few DLS iterations toward (pos, quat). Returns the joint reference q_des."""
        for _ in range(self.iters):
            ep, er = self.pose_error(pos, quat)
            e = np.concatenate([ep, self.wr * er])
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
            J = np.vstack([self._jp[:, self.vadr], self.wr * self._jr[:, self.vadr]])
            JJt = J @ J.T + self.lam**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            # null-space bias toward a sane posture (6x6 arm: active only through the damping)
            N = np.eye(len(self.q)) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (self.kn * (self.home - self.q))
            n = np.abs(dq).max()
            if n > self.max_step:
                dq *= self.max_step / n
            self.q = np.clip(self.q + dq, self.lo, self.hi)
        return self.q

    def converge(self, pos, quat, tol=1e-5, max_iter=2000) -> float:
        """Iterate to convergence (for initial placement). Returns final position error (m)."""
        kn = self.kn
        for i in range(max_iter):
            self.kn = kn if i < max_iter // 2 else 0.0  # posture first, then exact pose
            self.solve(pos, quat)
            ep, er = self.pose_error(pos, quat)
            if i >= max_iter // 2 and np.linalg.norm(ep) < tol and np.linalg.norm(er) < 10 * tol:
                break
        self.kn = kn
        return float(np.linalg.norm(ep))


class TrajectoryFollower:
    """Interpolates a grip trajectory to the sim clock and produces actuator commands."""

    def __init__(self, m, d, traj: GripTrajectory, site_id: int, **ik_kw):
        self.traj = traj.validate()
        ik_kw.setdefault("home", d.qpos[arm_joint_qpos_adr(m)].copy())  # the scene's start posture (load_scene)
        self.ik = GripIK(m, site_id, **ik_kw)
        self.m, self.d = m, d
        self.qadr = arm_joint_qpos_adr(m)
        self.t0 = traj.t[0]
        # position servos: force = kp*(ctrl - q) - kv*qdot. Velocity feedforward:
        # ctrl = q_des + (kv/kp)*qdot_des turns that into kp*(q_des - q) + kv*(qdot_des - qdot).
        self.act = arm_actuator_ids(m)
        self.kv_over_kp = -m.actuator_biasprm[self.act, 2] / m.actuator_gainprm[self.act, 0]
        self._q_prev, self._t_prev = None, None

    @property
    def duration(self) -> float:
        return float(self.traj.t[-1] - self.traj.t[0])

    def target(self, t):
        p, q = interp_poses(self.traj.t, self.traj.pos, self.traj.quat, np.array([self.t0 + t]))
        return p[0], q[0]

    def place_at_start(self) -> float:
        """Teleport the arm to the first trajectory pose (no dynamics). Returns IK residual (m)."""
        p, q = self.target(0.0)
        err = self.ik.converge(p, q)
        self.d.qpos[self.qadr] = self.ik.q
        self.d.qvel[:] = 0
        self.d.ctrl[self.act] = self.ik.q
        self._q_prev, self._t_prev = self.ik.q.copy(), 0.0
        mujoco.mj_forward(self.m, self.d)
        return err

    def command(self, t) -> np.ndarray:
        p, q = self.target(t)
        q_des = self.ik.solve(p, q).copy()
        qd = np.zeros_like(q_des)
        if self._t_prev is not None and t > self._t_prev:
            qd = (q_des - self._q_prev) / (t - self._t_prev)
        self._q_prev, self._t_prev = q_des, t
        self.d.ctrl[self.act] = q_des + self.kv_over_kp * qd
        return self.d.ctrl[self.act]
