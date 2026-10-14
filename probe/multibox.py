"""Training-sim for "push the heaviest/lightest box": three identical-looking boxes in a row, a scripted expert.

The expert probes every box with the standard push (my measured push style: angle, contact height,
speed), runs the sim-trained mass estimator on each probe, keeps a running per-box summary, then pushes the
chosen box 10 cm. Every frame is logged for a VLA: camera image, state, language instruction and the next grip
target as the action.

State (22): grip pose (x y z qw qx qy qz) | needle bending (curvature at 0/25/50/75 % of the length, tip x, tip z;
stick-local, cubic fit as in the estimator) | per box: probed flag, estimated log-mass (0 if not probed),
its 1-sigma (0 if not probed). The VLA is given the needle and the estimator summary; an images-only ablation
drops them (see scripts/gen_vla_episodes.py).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .camera import CameraConfig
from .control import TrajectoryFollower
from .estimator import cubic_shape
from .rollout import extract_frame
from .scene import (CONTACT_SOLIMP, CONTACT_SOLREF, SceneConfig, _box_I, arm_actuator_ids, arm_joint_qpos_adr,
                    build_scene_xml, close_gripper, home_q)
from .schema import GripTrajectory
from .stick import StickParams, colour_bands, find_stick
from .trajectory import grip_rotation

ROW_Y = (-0.15, 0.0, 0.15)
MIN_MASS_RATIO = 1.3
BOX_RGBA = "0.93 0.93 0.95 1"  # identical, like the real white-wrapped box
INSTRUCTIONS = {
    "heaviest": ("push the heaviest box", "push the heaviest one", "find the heaviest box and push it",
                 "which box is heaviest? push it"),
    "lightest": ("push the lightest box", "push the lightest one", "find the lightest box and push it",
                 "which box is lightest? push it"),
}
STATE_NAMES = (["grip_x", "grip_y", "grip_z", "grip_qw", "grip_qx", "grip_qy", "grip_qz",
                "kappa_0", "kappa_25", "kappa_50", "kappa_75", "tip_x", "tip_z"]
               + [f"box{i}_{k}" for i in range(3) for k in ("probed", "logm", "logm_sd")])
ACTION_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]
VLA_CAMERA = CameraConfig(pos=(0.22, -0.95, 1.40), lookat=(0.38, 0.0, 0.84), resolution=(256, 256),
                          fx=215.0, fy=215.0, cx=128.0, cy=128.0)


@dataclass
class PushStyle:
    """One probe/push, parameterised like my recorded pushes."""
    pitch_deg: float = 45.0
    contact_height: float = 0.025
    push_speed: float = 0.10
    approach_speed: float = 0.12
    standoff: float = 0.06
    hold: float = 0.3
    retract: float = 0.08
    settle: float = 0.4
    lift: float = 0.07       # clearance when moving between boxes
    hit_offset: float = 0.0  # sideways miss of the face centre


@dataclass
class EpisodeConfig:
    masses: tuple = (0.16, 0.345, 0.52)
    friction: float = 0.55
    goal: str = "heaviest"
    instruction: str = "push the heaviest box"
    stick: StickParams = field(default_factory=StickParams)
    style: PushStyle = field(default_factory=PushStyle)
    probe_depth: float = 0.06
    final_depth: float = 0.10
    record_hz: float = 15.0
    est_hz: float = 30.0
    ys: tuple = ROW_Y
    scene: SceneConfig = field(default_factory=lambda: SceneConfig(camera=VLA_CAMERA))
    probe_order: tuple = (0, 1, 2)  # the arm always starts at box 0's approach pose, then probes in this order
    # recovery demo: after probing, drift (unrecorded) to this grip offset from the last retract pose, then record
    # only the recovery from there (go to the chosen box and push it); None = a normal demo
    recovery_offset: tuple | None = None
    # real-poke probes: per box, (repo-relative grip CSV, time scale) to replay my recorded straight poke
    # there instead of the synthetic push; None (or a None entry) = synthetic
    probe_replays: tuple | None = None


# ---- scene --------------------------------------------------------------------------------------

def build_row_scene(cfg: EpisodeConfig):
    """The probe scene with its three class boxes replaced by three identical free boxes in a row."""
    sc = cfg.scene
    root = ET.fromstring(build_scene_xml(cfg.stick, sc, active="freespace"))
    wb = root.find("worldbody")
    for b in [b for b in wb.findall("body") if (b.get("name") or "").startswith("obj_")]:
        wb.remove(b)
    hx, hy, hz = sc.object_half
    for i, (y, mass) in enumerate(zip(cfg.ys, cfg.masses)):
        wb.append(ET.fromstring(f"""
    <body name="box{i}" pos="{sc.object_pos[0]} {y} {sc.table_height + hz + 1e-4}">
      <freejoint/><inertial pos="0 0 0" mass="{mass}" diaginertia="{_box_I(mass, sc.object_half)}"/>
      <geom name="box{i}_geom" type="box" size="{hx} {hy} {hz}" rgba="{BOX_RGBA}"
            friction="{cfg.friction} 0.005 0.0001" condim="4" priority="1" solref="{CONTACT_SOLREF}"
            solimp="{CONTACT_SOLIMP}"/>
    </body>"""))
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    h = find_stick(m)
    colour_bands(m, h, cfg.stick)
    d = mujoco.MjData(m)
    q0 = home_q(sc)
    d.qpos[arm_joint_qpos_adr(m)] = q0
    d.ctrl[arm_actuator_ids(m)] = q0
    close_gripper(m, d)
    mujoco.mj_forward(m, d)
    boxes = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"box{i}") for i in range(3)]
    return m, d, h, boxes


# ---- trajectories -------------------------------------------------------------------------------

def _smooth(segs, rate=60.0):
    """[(a, b, duration)] -> (t, pos), cosine-eased per segment (zero-velocity ends, like a hand)."""
    ts, ps, t = [], [], 0.0
    for a, b, dur in segs:
        n = max(2, int(round(dur * rate)))
        u = np.linspace(0, 1, n, endpoint=False)
        s = 0.5 - 0.5 * np.cos(np.pi * u)
        ts.append(t + u * dur)
        ps.append(np.asarray(a) + s[:, None] * (np.asarray(b) - np.asarray(a)))
        t += dur
    ts.append([t])
    ps.append([segs[-1][1]])
    return np.concatenate(ts), np.vstack(ps)


def _traj(segs, R) -> GripTrajectory:
    t, p = _smooth(segs)
    from .geometry import mat_to_quat_wxyz
    return GripTrajectory(t, p, np.tile(mat_to_quat_wxyz(R), (len(t), 1))).validate()


def _traj_q(segs, q0, q1) -> GripTrajectory:
    """Like _traj, but the orientation turns from q0 to q1 (wxyz, normalised lerp) over the whole path."""
    t, p = _smooth(segs)
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if q0 @ q1 < 0:
        q1 = -q1
    u = (t / t[-1])[:, None]
    q = (1 - u) * q0 + u * q1
    return GripTrajectory(t, p, q / np.linalg.norm(q, axis=1, keepdims=True)).validate()


def replay_probe(path, time_scale, box_pos, stick: StickParams, scene: SceneConfig) -> GripTrajectory:
    """The person's recorded straight poke (grip CSV in the single-box scene's frame), moved onto a row box and
    lifted so the unbent tip clears the table (as in the estimator's training replays, probe/simdata.py)."""
    from pathlib import Path
    from .io import load_trajectory
    from .simdata import lift_to_clear_table
    from .trajectory import replay_variant
    tr = replay_variant(load_trajectory(Path(__file__).resolve().parent.parent / path), time_scale)
    pos = tr.pos.copy()
    pos[:, 0] += box_pos[0] - scene.object_pos[0]
    pos[:, 1] += box_pos[1] - scene.object_pos[1]
    return lift_to_clear_table(GripTrajectory(tr.t, pos, tr.quat).validate(), stick, scene, 0.003)


def push_start_pose(face_x, y, stick: StickParams, style: PushStyle, scene: SceneConfig):
    """Grip position at the start of the approach (tip `standoff` short of the face, at contact height)."""
    R = grip_rotation(style.pitch_deg)
    tip = np.array([face_x - stick.radius - style.standoff, y + style.hit_offset,
                    scene.table_height + style.contact_height])
    return tip - stick.length * R[:, 0], R


def transit_segs(p_from, p_to, lift, speed=0.20):
    up = max(p_from[2], p_to[2]) + lift
    a, b = np.array([*p_from[:2], up]), np.array([*p_to[:2], up])
    d = lambda x, y: max(0.25, np.linalg.norm(np.subtract(y, x)) / speed)  # noqa: E731
    return [(p_from, a, d(p_from, a)), (a, b, d(a, b)), (b, p_to, d(b, p_to))]


def push_segs(p0, depth, style: PushStyle):
    fwd = np.array([1.0, 0, 0])
    p1 = p0 + style.standoff * fwd
    p2 = p1 + depth * fwd
    p3 = p2 - style.retract * fwd
    return [(p0, p0, style.settle), (p0, p1, style.standoff / style.approach_speed),
            (p1, p2, depth / style.push_speed), (p2, p2, style.hold), (p2, p3, style.retract / style.approach_speed),
            (p3, p3, 0.2)]


# ---- episode ------------------------------------------------------------------------------------

class Estimator:
    """The sim-trained GRU mass estimator, applied to one probe (main-venv checkpoint, runs on CPU)."""

    def __init__(self, ckpt_path):
        import torch
        from .estimator import Estimator as Net
        ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        self.st, cfg = ck["stats"], ck["cfg"]
        self.net = Net(hidden=cfg["model"]["hidden"], layers=cfg["model"]["layers"], dropout=0.0)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.torch = torch

    def __call__(self, t, grip_pose, centreline, arclen, stick: StickParams, friction):
        from .estimator import cond_vector, sequence_features
        x = (sequence_features(t, grip_pose, centreline, arclen) - self.st["x_mean"]) / self.st["x_std"]
        c = (cond_vector(stick.young, stick.length, stick.radius, stick.damping, friction)
             - self.st["c_mean"]) / self.st["c_std"]
        with self.torch.no_grad():
            mu, lv, lg = self.net(self.torch.from_numpy(x.astype(np.float32))[None], self.torch.tensor([len(x)]),
                                  self.torch.from_numpy(c.astype(np.float32))[None])
        return float(mu), float(np.exp(0.5 * float(lv))), float(1 / (1 + np.exp(-float(lg))))


def needle_features(cl, arclen):
    kappa, tip = cubic_shape(cl[None], arclen)
    return np.concatenate([kappa[0], tip[0]])


def run_episode(cfg: EpisodeConfig, estimator: Estimator, render=True, rng=None):
    """Probe all three boxes (in cfg.probe_order), decide, push the chosen one. Returns the logged episode.
    With cfg.recovery_offset only the recovery from a hover pose is logged (see EpisodeConfig)."""
    rng = rng or np.random.default_rng()
    m, d, h, boxes = build_row_scene(cfg)
    sc, st, style = cfg.scene, cfg.stick, cfg.style
    hx = sc.object_half[0]
    site = h.grip_site
    renderer = None
    if render:
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, sc.camera.resolution[0])
        m.vis.global_.offheight = max(m.vis.global_.offheight, sc.camera.resolution[1])
        renderer = mujoco.Renderer(m, sc.camera.resolution[1], sc.camera.resolution[0])
    arclen_est = np.linspace(0, st.length, 21)
    summary = np.zeros((3, 3))  # probed, logm, sd
    log = {"image": [], "state": [], "action": [], "t": []}
    probes = []
    clock = {"next_rec": 0.0, "next_est": 0.0}

    def place(R):  # settle the arm at the first approach pose before the clock starts
        p, _ = push_start_pose(d.xpos[boxes[0]][0] - hx, cfg.ys[0], st, style, sc)
        fol = TrajectoryFollower(m, d, _traj([(p, p, 1.0)], R), site)
        if fol.place_at_start() > 1e-4:
            raise RuntimeError("start pose unreachable")
        while d.time < 1.0:
            fol.command(d.time - 0.0)
            mujoco.mj_step(m, d)
        d.time = 0.0
        return p

    def run(traj: GripTrajectory, probe_buf=None, record=True):
        fol = TrajectoryFollower(m, d, traj, site)
        t0 = d.time
        while d.time - t0 < fol.duration:
            tl = d.time - t0
            fol.command(tl)
            if record and d.time + 1e-9 >= clock["next_rec"]:
                mujoco.mj_forward(m, d)
                cl, _, gp, _ = extract_frame(m, d, h, arclen_est)
                log["state"].append(np.concatenate([gp, needle_features(cl, arclen_est), summary.ravel()]))
                p, q = fol.target(min(tl + 1.0 / cfg.record_hz, fol.duration))
                log["action"].append(np.concatenate([p, q]))
                log["t"].append(d.time)
                if renderer is not None:
                    renderer.update_scene(d, camera=sc.camera.name)
                    log["image"].append(renderer.render().copy())
                clock["next_rec"] += 1.0 / cfg.record_hz
            if probe_buf is not None and d.time + 1e-9 >= clock["next_est"]:
                mujoco.mj_forward(m, d)
                cl, _, gp, _ = extract_frame(m, d, h, arclen_est)
                probe_buf["t"].append(d.time)
                probe_buf["grip"].append(gp)
                probe_buf["cl"].append(cl)
                clock["next_est"] = d.time + 1.0 / cfg.est_hz
            mujoco.mj_step(m, d)
            if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
                raise FloatingPointError(f"simulation diverged at t={d.time:.3f}")
        return traj.pos[-1]

    R = grip_rotation(style.pitch_deg)
    from .geometry import mat_to_quat_wxyz, quat_wxyz_to_mat
    qR = mat_to_quat_wxyz(R)
    cur = place(R)
    cur_q = qR  # orientation at `cur` (a real-poke replay ends at its own, slightly different pitch)
    rec = cfg.recovery_offset is None
    for i in cfg.probe_order:
        rp = cfg.probe_replays[i] if cfg.probe_replays else None
        buf = {"t": [], "grip": [], "cl": []}
        if rp is not None:
            tr = replay_probe(rp[0], rp[1], (d.xpos[boxes[i]][0], cfg.ys[i]), st, sc)
            cur = run(_traj_q(transit_segs(cur, tr.pos[0], style.lift), cur_q, tr.quat[0]), record=rec)
            clock["next_est"] = d.time
            cur = run(tr, probe_buf=buf, record=rec)
            cur_q = tr.quat[-1]
            # back off until the tip is `standoff` short of the face (real clips can end still pushed in)
            ax = quat_wxyz_to_mat(cur_q)[:, 0]
            back = (cur[0] + st.length * ax[0]) - (d.xpos[boxes[i]][0] - hx - st.radius - style.standoff)
            if back > 1e-3:
                p_back = cur - np.array([back, 0.0, 0.0])
                cur = run(_traj_q([(cur, p_back, back / style.approach_speed), (p_back, p_back, 0.2)], cur_q, cur_q),
                          record=rec)
        else:
            p0, _ = push_start_pose(d.xpos[boxes[i]][0] - hx, cfg.ys[i], st, style, sc)
            if np.linalg.norm(p0 - cur) > 1e-6:
                segs = transit_segs(cur, p0, style.lift)
                cur = run(_traj(segs, R) if cur_q is qR else _traj_q(segs, cur_q, qR), record=rec)
            clock["next_est"] = d.time
            cur = run(_traj(push_segs(p0, cfg.probe_depth, style), R), probe_buf=buf, record=rec)
            cur_q = qR
        logm, sd, p_anch = estimator(np.array(buf["t"]), np.array(buf["grip"]), np.array(buf["cl"]), arclen_est,
                                     st, cfg.friction)
        summary[i] = (1.0, logm, sd)
        probes.append({"box": int(i), "logm": logm, "sd": sd, "p_anchored": p_anch})

    # decide from the posteriors: probability each box is the heaviest (lightest), by sampling
    draws = rng.normal(summary[:, 1], np.maximum(summary[:, 2], 1e-3), size=(4000, 3))
    pick = draws.argmax(1) if cfg.goal == "heaviest" else draws.argmin(1)
    p_best = np.bincount(pick, minlength=3) / len(pick)
    chosen = int(np.argmax(p_best))
    if not rec:  # drift to the hover pose like a lost policy, settle, and start recording there
        hover = cur + np.asarray(cfg.recovery_offset)
        segs = [(cur, hover, max(0.5, np.linalg.norm(hover - cur) / 0.15)), (hover, hover, 0.5)]
        cur = run(_traj(segs, R) if cur_q is qR else _traj_q(segs, cur_q, cur_q), record=False)
        clock["next_rec"] = d.time
    p0, _ = push_start_pose(d.xpos[boxes[chosen]][0] - hx, cfg.ys[chosen], st, style, sc)
    segs = transit_segs(cur, p0, style.lift)
    cur = run(_traj(segs, R) if cur_q is qR else _traj_q(segs, cur_q, qR))
    run(_traj(push_segs(p0, cfg.final_depth, style), R))
    if renderer is not None:
        renderer.close()

    truth = int(np.argmax(cfg.masses) if cfg.goal == "heaviest" else np.argmin(cfg.masses))
    return {k: np.asarray(v, np.float32 if k != "image" else np.uint8) for k, v in log.items()} | {
        "task": cfg.instruction, "goal": cfg.goal, "masses": list(cfg.masses), "friction": cfg.friction,
        "chosen": chosen, "truth": truth, "correct": chosen == truth, "p_best": p_best.tolist(), "probes": probes,
        "probe_order": [int(i) for i in cfg.probe_order],
        "probe_replays": None if cfg.probe_replays is None else [None if r is None else [r[0], float(r[1])]
                                                                  for r in cfg.probe_replays],
        "recovery_offset": None if rec else [float(x) for x in cfg.recovery_offset]}


def sample_episode(seed: int, style: str = "angled") -> EpisodeConfig:
    """Randomised training-sim episode around the measured setup and my push style."""
    if style == "straight":
        return sample_straight_episode(seed)
    rng = np.random.default_rng(seed)
    while True:  # like the real box set (100-520 g, neighbours 1.4-1.6x apart): near-ties carry no signal to imitate
        masses = tuple(float(np.exp(rng.uniform(np.log(0.07), np.log(0.6)))) for _ in range(3))
        o = np.sort(masses)
        if o[1] / o[0] >= MIN_MASS_RATIO and o[2] / o[1] >= MIN_MASS_RATIO:
            break
    goal = "heaviest" if rng.uniform() < 0.5 else "lightest"
    stick = StickParams(young=float(np.exp(rng.uniform(np.log(3.5e10), np.log(5e10)))),
                        length=float(rng.uniform(0.325, 0.343)), radius=float(rng.uniform(0.0017, 0.0018)))
    style = PushStyle(pitch_deg=float(rng.uniform(35, 50)), contact_height=float(rng.uniform(0.015, 0.045)),
                      push_speed=float(rng.uniform(0.05, 0.15)), approach_speed=float(rng.uniform(0.08, 0.15)),
                      hit_offset=float(rng.uniform(-0.005, 0.005)))
    return EpisodeConfig(masses=masses, friction=float(rng.uniform(0.45, 0.65)), goal=goal,
                         instruction=str(rng.choice(INSTRUCTIONS[goal])), stick=stick, style=style)


def sample_straight_episode(seed: int) -> EpisodeConfig:
    """Straight pokes (the needle ~vertical, along the gripper's approach axis, tip low on the box face), with
    the ranges of the straight estimator's training-sim (probe/simdata.py RANGES_STRAIGHT). Boxes 0.07-0.45 kg:
    the straight estimator's range (heavier boxes need >110 mm of tip deflection and the tip skids). The tip
    runs only 2-28 mm above the table, so it retracts further and lifts higher between boxes than an angled
    push, and the final push is longer so a heavy box visibly moves."""
    rng = np.random.default_rng(seed)
    while True:
        masses = tuple(float(np.exp(rng.uniform(np.log(0.07), np.log(0.45)))) for _ in range(3))
        o = np.sort(masses)
        if o[1] / o[0] >= MIN_MASS_RATIO and o[2] / o[1] >= MIN_MASS_RATIO:
            break
    goal = "heaviest" if rng.uniform() < 0.5 else "lightest"
    stick = StickParams(young=float(np.exp(rng.uniform(np.log(3.5e10), np.log(5e10)))),
                        length=float(rng.uniform(0.325, 0.343)), radius=float(rng.uniform(0.0017, 0.0018)),
                        tip_friction=float(rng.uniform(0.10, 0.60)))
    style = PushStyle(pitch_deg=float(rng.uniform(80, 90)), contact_height=float(rng.uniform(0.002, 0.028)),
                      push_speed=float(rng.uniform(0.03, 0.10)), approach_speed=float(rng.uniform(0.05, 0.15)),
                      standoff=0.04, retract=0.14, lift=0.11, hit_offset=float(rng.uniform(-0.005, 0.005)))
    return EpisodeConfig(masses=masses, friction=float(rng.uniform(0.45, 0.65)), goal=goal,
                         instruction=str(rng.choice(INSTRUCTIONS[goal])), stick=stick, style=style,
                         probe_depth=0.09, final_depth=0.14,
                         scene=SceneConfig(camera=VLA_CAMERA, exit_angle_deg=0.0))
