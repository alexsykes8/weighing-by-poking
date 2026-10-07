"""Drive the arm along a grip trajectory and log a ProbeObservation (the sim side of the contract)."""
from __future__ import annotations

import mujoco
import numpy as np

from .control import TrajectoryFollower
from .geometry import curvature_inplane, mat_to_quat_wxyz, resample_by_arclen, stick_local_frame
from .io import git_sha
from .scene import OBJECTS, SceneConfig, load_scene
from .schema import GripTrajectory, ProbeObservation
from .stick import StickParams, stick_geoms

K_DEFAULT = 41
CURV_WINDOW = 0.15  # m of arc length for the curvature filter (see tests/test_extraction.py)


def extract_frame(m, d, h, arclen, window=CURV_WINDOW):
    """Stick shape at the current state: (centreline_local (K,3), curvature (K,), grip pose (7,), frame (3,3))."""
    nodes = h.nodes_world(d, m)
    gp = d.site_xpos[h.grip_site].copy()
    gR = d.site_xmat[h.grip_site].reshape(3, 3)
    F = stick_local_frame(gR)
    local = (nodes - gp) @ F
    cl = resample_by_arclen(h.node_s, local, arclen)
    kappa = curvature_inplane(cl, arclen, window)
    return cl, kappa, np.concatenate([gp, mat_to_quat_wxyz(gR)]), F


def stick_contact_force(m, d, stick_geom_set, other_geoms) -> np.ndarray:
    """Total force (world frame) exerted ON the stick by `other_geoms`."""
    f = np.zeros(3)
    c6 = np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = con.geom1, con.geom2
        if g1 in stick_geom_set and g2 in other_geoms:
            sign = -1.0  # contact force acts on geom2 along +normal; on the stick (geom1) it's -
        elif g2 in stick_geom_set and g1 in other_geoms:
            sign = 1.0
        else:
            continue
        mujoco.mj_contactForce(m, d, i, c6)
        frame = con.frame.reshape(3, 3)  # rows: normal, tangent1, tangent2
        f += sign * frame.T @ c6[:3]
    return f


def run_rollout(traj: GripTrajectory, label: str, stick: StickParams | None = None,
                scene: SceneConfig | None = None, mass=None, friction=None, K: int = K_DEFAULT,
                record_hz: float = 100.0, extra_meta: dict | None = None, frame_callback=None,
                presettle: float = 1.0) -> ProbeObservation:
    stick = stick or StickParams()
    scene = scene or SceneConfig()
    m, d, h = load_scene(stick, scene, active=label, mass=mass, friction=friction)
    fol = TrajectoryFollower(m, d, traj, h.grip_site)
    ik_err = fol.place_at_start()
    if ik_err > 1e-4:
        raise RuntimeError(f"trajectory start pose unreachable (IK residual {ik_err*1e3:.2f} mm)")
    # let the stick sag to rest at the start pose before the clock starts (real footage starts at rest)
    while d.time < presettle:
        fol.command(0.0)
        mujoco.mj_step(m, d)
    d.time = 0.0
    fol._t_prev = 0.0

    arclen = np.linspace(0, h.length, K)
    sgeoms = set(stick_geoms(m, h))
    body = "heavy" if label == "free" else label  # 'free' (given mass) uses the free-joint box slot
    obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj_{body}") if body in OBJECTS else -1
    obj_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == obj_body} if obj_body >= 0 else set()
    env_geoms = {g for g in range(m.ngeom) if g not in sgeoms and m.geom_bodyid[g] == 0}
    s_force = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "mount_force")
    s_torque = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "mount_torque")

    every = max(1, int(round(1.0 / (record_hz * m.opt.timestep))))
    n_steps = int(np.floor(fol.duration / m.opt.timestep))
    log = {k: [] for k in ("t", "cl", "kappa", "grip", "cmd", "f_obj", "f_env", "mount_f", "mount_t", "obj_pos")}
    for step in range(n_steps + 1):
        t = d.time
        fol.command(t)
        if step % every == 0:
            mujoco.mj_forward(m, d)
            cl, kappa, grip, F = extract_frame(m, d, h, arclen)
            log["t"].append(t)
            log["cl"].append(cl)
            log["kappa"].append(kappa)
            log["grip"].append(grip)
            p, q = fol.target(t)
            log["cmd"].append(np.concatenate([p, q]))
            # forces in the stick-local frame, so they line up with the curvature signal
            log["f_obj"].append(stick_contact_force(m, d, sgeoms, obj_geoms) @ F)
            log["f_env"].append(stick_contact_force(m, d, sgeoms, env_geoms) @ F)
            Rs = d.site_xmat[h.grip_site].reshape(3, 3)  # sensor reads in the site frame
            log["mount_f"].append((Rs @ d.sensordata[m.sensor_adr[s_force]:m.sensor_adr[s_force] + 3]) @ F)
            log["mount_t"].append((Rs @ d.sensordata[m.sensor_adr[s_torque]:m.sensor_adr[s_torque] + 3]) @ F)
            log["obj_pos"].append(d.xpos[obj_body].copy() if obj_body >= 0 else np.full(3, np.nan))
            if frame_callback is not None:
                frame_callback(m, d, t)
        mujoco.mj_step(m, d)
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            raise FloatingPointError(f"simulation diverged at t={d.time:.3f}")

    cl = np.array(log["cl"])
    obj = np.array(log["obj_pos"])
    extras = {"grip_cmd": np.array(log["cmd"]), "f_contact_obj": np.array(log["f_obj"]),
              "f_contact_env": np.array(log["f_env"]), "mount_force": np.array(log["mount_f"]),
              "mount_torque": np.array(log["mount_t"])}
    if obj_body >= 0:
        extras["obj_pos"] = obj
    meta = {"young": stick.young, "radius": stick.radius, "length": stick.length, "timestep": m.opt.timestep,
            "source": "sim", "git_sha": git_sha(), "mujoco": mujoco.__version__, "stick": stick.to_dict(),
            "object_mass": mass, "object_friction": friction, "record_hz": record_hz,
            "frames": "centreline/tip/forces: stick-local; grip_pose/grip_cmd/obj_pos: world"}
    meta.update(extra_meta or {})
    return ProbeObservation(t=np.array(log["t"]), centreline=cl, tip=cl[:, -1].copy(),
                            curvature=np.array(log["kappa"]), arclen=arclen, grip_pose=np.array(log["grip"]),
                            label=label, meta=meta, extras=extras).validate()
