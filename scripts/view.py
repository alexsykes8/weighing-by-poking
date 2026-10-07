"""Open the interactive MuJoCo viewer and play a push live, in real time.

    python scripts/view.py                         # fixed object, default synthetic push
    python scripts/view.py --object heavy --loop   # replay until the window is closed
    python scripts/view.py --static                # just the scene at the start pose, no motion
    python scripts/view.py --slow 4                # 4x slow motion

Viewer tips: double-click a body to select it, ctrl+right-drag to push it, [ / ] cycles cameras
(the 'ipad' camera is the fixed video viewpoint), F1 for help. Contact points/forces are shown.
"""
import argparse
import time

import _bootstrap  # noqa: F401
import mujoco
import mujoco.viewer

from probe.control import TrajectoryFollower
from probe.io import load_trajectory
from probe.rollout import run_rollout
from probe.camera import CameraConfig
from probe.scene import SceneConfig, load_scene
from probe.schema import LABELS
from probe.stick import StickParams

DEFAULT_TRAJ = _bootstrap.REPO / "data" / "trajectories" / "synthetic_push.csv"


class LiveView:
    """frame_callback for run_rollout: opens the viewer on first call, then syncs at real time."""

    def __init__(self, slow=1.0):
        self.v, self.slow, self.wall0 = None, slow, None

    def __call__(self, m, d, t):
        if self.v is None:
            self.v = mujoco.viewer.launch_passive(m, d)
            self._style()
        if not self.v.is_running():
            raise KeyboardInterrupt
        if self.wall0 is None:
            self.wall0 = time.perf_counter() - t * self.slow
        lag = self.wall0 + t * self.slow - time.perf_counter()
        if lag > 0:
            time.sleep(lag)
        self.v.sync()

    def _style(self):
        with self.v.lock():
            self.v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self.v.cam.lookat[:] = (0.15, 0.0, 0.9)
            self.v.cam.distance, self.v.cam.azimuth, self.v.cam.elevation = 1.6, 90, -12


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", default="fixed", choices=LABELS)
    ap.add_argument("--trajectory", default=str(DEFAULT_TRAJ))
    ap.add_argument("--young", type=float, default=StickParams.young)
    ap.add_argument("--slow", type=float, default=1.0, help="slow-motion factor")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--static", action="store_true", help="show the scene at the start pose without moving")
    ap.add_argument("--camera", help="camera JSON from scripts/camera_from_stray.py (select it with [ / ])")
    a = ap.parse_args()
    stick = StickParams(young=a.young)
    traj = load_trajectory(a.trajectory)
    scene = SceneConfig(camera=CameraConfig.from_json(a.camera)) if a.camera else SceneConfig()

    if a.static:
        m, d, h = load_scene(stick, scene, active=a.object)
        fol = TrajectoryFollower(m, d, traj, h.grip_site)
        fol.place_at_start()
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running():  # physics keeps running (stick settles, you can poke things)
                fol.command(0.0)
                mujoco.mj_step(m, d)
                v.sync()
                time.sleep(m.opt.timestep)
        return

    view = LiveView(a.slow)
    try:
        while True:
            view.wall0 = None
            run_rollout(traj, a.object, stick, scene, frame_callback=view, record_hz=60)
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
