"""Open the interactive MuJoCo viewer and play a push live, in real time.

    python scripts/view.py                                         # a real straight poke (160 g), replayed
    python scripts/view.py --replay data/runs/real/100gS_b1798fd13e.npz --slow 2
    python scripts/view.py --trajectory data/trajectories/synthetic_push.csv --object heavy   # old angled push
    python scripts/view.py --loop                                  # replay until the window is closed
    python scripts/view.py --static                                # just the scene at the start pose, no motion

--replay sets the sim up as scripts/replay_real.py does (the sim-IRL): the tracked grip path drives the arm,
the needle is mounted straight along the gripper if the clip is steeper than 70 deg, the box has the clip's
scale mass, friction is the measured 0.55, and the path is lifted so the unbent tip clears the table.

Viewer tips: double-click a body to select it, ctrl+right-drag to push it, [ / ] cycles cameras
(the 'ipad' camera is the fixed video viewpoint), F1 for help. Contact points/forces are shown. Each box's
mass is written on the side facing you (viewer only: not part of the sim, never recorded). In a replay only
the box in the push path is used; the other two are parked props.
"""
import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import mujoco
import mujoco.viewer
import numpy as np

from probe.control import TrajectoryFollower
from probe.io import load_trajectory
from probe.rollout import run_rollout
from probe.camera import CameraConfig
from probe.estimator import grip_features
from probe.io import load_observation
from probe.scene import SceneConfig, load_scene
from probe.schema import LABELS
from probe.simdata import lift_to_clear_table
from probe.stick import StickParams

DEFAULT_REPLAY = _bootstrap.REPO / "data" / "runs" / "real" / "160gS_55b53993ad.npz"  # held out; box slides 11 cm

# Stroke font for the viewer-only mass labels: segments in a unit cell (x 0-1, y 0-1, baseline y=0)
_SEG7 = {"a": ((0, 1), (1, 1)), "b": ((1, 1), (1, .5)), "c": ((1, .5), (1, 0)), "d": ((0, 0), (1, 0)),
         "e": ((0, .5), (0, 0)), "f": ((0, 1), (0, .5)), "g": ((0, .5), (1, .5))}
GLYPHS = {ch: [_SEG7[k] for k in segs] for ch, segs in zip(
    "0123456789", ("abcdef", "bc", "abged", "abgcd", "fgbc", "afgcd", "afgedc", "abc", "abcdefg", "abcdfg"))}
GLYPHS.update({
    "g": [((0, .6), (1, .6)), ((1, .6), (1, -.3)), ((1, -.3), (0, -.3)), ((0, .6), (0, .15)), ((0, .15), (1, .15))],
    "F": [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((0, .5), (.7, .5))],
    "I": [((.5, 0), (.5, 1)), ((.2, 1), (.8, 1)), ((.2, 0), (.8, 0))],
    "X": [((0, 0), (1, 1)), ((0, 1), (1, 0))],
    "E": [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((0, .5), (.7, .5)), ((0, 0), (1, 0))],
    "D": [((0, 0), (0, 1)), ((0, 1), (.6, 1)), ((.6, 1), (1, .7)), ((1, .7), (1, .3)), ((1, .3), (.6, 0)),
          ((.6, 0), (0, 0))],
})


def box_labels(m):
    """(body id, half-size, text) for every box: its mass in grams, or FIXED for the static one."""
    out = []
    for b in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith("obj_"):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name + "_geom")
            text = "FIXED" if m.body_dofnum[b] == 0 else f"{1e3 * m.body_mass[b]:.0f}g"
            out.append((b, m.geom_size[g].copy(), text))
    return out


def _viewer_cam_pos(v, d):
    if v.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
        return d.cam_xpos[v.cam.fixedcamid].copy()
    az, el = np.radians(v.cam.azimuth), np.radians(v.cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.asarray(v.cam.lookat) - v.cam.distance * fwd


def draw_labels(v, d, labels, rgba=(0.05, 0.05, 0.05, 1)):
    """Viewer-only overlay (v.user_scn: never simulated or recorded): each box's label in big stroke text on
    whichever long side faces the viewer camera, redrawn every frame so it follows the box."""
    scn, cam = v.user_scn, _viewer_cam_pos(v, d)
    scn.ngeom = 0
    aspect, adv = 0.55, 1.35                                           # glyph width/height, advance/width
    for b, half, text in labels:
        c, R = d.xpos[b], d.xmat[b].reshape(3, 3)
        n = R[:, 1] if R[:, 1] @ (cam - c) > 0 else -R[:, 1]           # long side (129 x 80 mm) facing the camera
        right, up = np.cross(R[:, 2], n), R[:, 2]
        span = adv * len(text) - (adv - 1)                             # text width in glyph widths
        cw = min(0.9 * 2 * half[0] / span, 0.75 * 2 * half[2] * aspect)
        h = cw / aspect
        origin = c + (half[1] + 1e-3) * n - 0.5 * span * cw * right - 0.45 * h * up
        for k, ch in enumerate(text):
            for (ax, ay), (bx, by) in GLYPHS.get(ch, []):
                if scn.ngeom >= scn.maxgeom:
                    return
                p0 = origin + (k * adv + ax) * cw * right + ay * h * up
                p1 = origin + (k * adv + bx) * cw * right + by * h * up
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                    np.asarray(rgba, np.float32))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.07 * h, p0, p1)
                scn.ngeom += 1


class LiveView:
    """frame_callback for run_rollout: opens the viewer on first call, then syncs at real time."""

    def __init__(self, slow=1.0):
        self.v, self.slow, self.wall0, self.labels = None, slow, None, []

    def __call__(self, m, d, t):
        if self.v is None:
            self.v = mujoco.viewer.launch_passive(m, d)
            self.labels = box_labels(m)
            self._style()
        if not self.v.is_running():
            raise KeyboardInterrupt
        if self.wall0 is None:
            self.wall0 = time.perf_counter() - t * self.slow
        lag = self.wall0 + t * self.slow - time.perf_counter()
        if lag > 0:
            time.sleep(lag)
        with self.v.lock():
            draw_labels(self.v, d, self.labels)
        self.v.sync()

    def _style(self):
        with self.v.lock():
            self.v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self.v.cam.lookat[:] = (0.15, 0.0, 0.9)
            self.v.cam.distance, self.v.cam.azimuth, self.v.cam.elevation = 1.6, 90, -12


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--replay", help=f"tracked real clip .npz (default {DEFAULT_REPLAY.name})")
    src.add_argument("--trajectory", help="grip-trajectory CSV instead, e.g. the synthetic angled push")
    ap.add_argument("--object", default="fixed", choices=LABELS, help="with --trajectory only")
    ap.add_argument("--friction", type=float, default=0.55, help="box friction with --replay")
    ap.add_argument("--young", type=float, default=StickParams.young)
    ap.add_argument("--slow", type=float, default=1.0, help="slow-motion factor")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--static", action="store_true", help="show the scene at the start pose without moving")
    ap.add_argument("--camera", help="camera JSON from scripts/camera_from_stray.py (select it with [ / ])")
    a = ap.parse_args()
    stick = StickParams(young=a.young)
    scene_kw = {"camera": CameraConfig.from_json(a.camera)} if a.camera else {}
    if a.trajectory:
        traj, obj, mass, friction = load_trajectory(a.trajectory), a.object, None, None
        scene = SceneConfig(**scene_kw)
    else:
        clip = load_observation(a.replay or DEFAULT_REPLAY)
        traj = load_trajectory((a.replay or str(DEFAULT_REPLAY)).replace(".npz", ".grip.csv"))
        pitch = float(np.median(np.degrees(grip_features(clip.t, clip.grip_pose)[:, 4])))
        straight = pitch > 70
        scene = SceneConfig(**scene_kw, **({"exit_angle_deg": 0.0} if straight else {}))
        if straight:
            traj = lift_to_clear_table(traj, stick, scene, 0.003)
        obj, mass, friction = "free", clip.meta["object_mass"], a.friction
        print(f"replaying {Path(a.replay or DEFAULT_REPLAY).stem}: {1e3 * mass:.0f} g box, needle pitch "
              f"{pitch:.0f} deg ({'straight' if straight else 'angled'} mount)")

    if a.static:
        m, d, h = load_scene(stick, scene, active=obj, mass=mass, friction=friction)
        fol = TrajectoryFollower(m, d, traj, h.grip_site)
        fol.place_at_start()
        labels = box_labels(m)
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running():  # physics keeps running (stick settles, you can poke things)
                fol.command(0.0)
                mujoco.mj_step(m, d)
                with v.lock():
                    draw_labels(v, d, labels)
                v.sync()
                time.sleep(m.opt.timestep)
        return

    view = LiveView(a.slow)
    try:
        while True:
            view.wall0 = None
            run_rollout(traj, obj, stick, scene, mass=mass, friction=friction, frame_callback=view, record_hz=60)
            if not a.loop:
                print("done - close the viewer window to exit")
                while view.v.is_running():
                    time.sleep(0.05)
                break
            if view.v is not None:  # a fresh model is compiled each loop: reopen the window on it
                view.v.close()
                view.v = None
    except KeyboardInterrupt:
        pass
    finally:
        if view.v is not None:
            view.v.close()


if __name__ == "__main__":
    main()
