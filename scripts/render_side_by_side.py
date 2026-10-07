"""Side-by-side demo: the real clip | the sim-IRL UR5e replaying its grip path (same camera) | a wide sim view.

    MUJOCO_GL=egl python scripts/render_side_by_side.py data/usable/straight_345g/7c7e3afa1d/track.npz \
        --out out/demo/345g_7c7e3afa1d.mp4

The tracked clip's grip path (<clip>.grip.csv) drives the arm exactly as in scripts/replay_real.py (measured
needle, the clip's box mass, friction 0.55; straight pokes mount the needle along the gripper axis and are lifted
to clear the table by 3 mm). The middle panel renders through the clip's fitted iPad camera (camera.json), so it
should line up with the video; the right panel is a fixed wide view showing the whole arm. Frames are rotated
upright (the iPad recorded in portrait) and synchronised on the tracker's clock.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import _bootstrap  # noqa: E402,F401
import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from probe.camera import CameraConfig  # noqa: E402
from probe.estimator import grip_features  # noqa: E402
from probe.io import load_observation, load_trajectory  # noqa: E402
from probe.rollout import run_rollout  # noqa: E402
from probe.scene import SceneConfig  # noqa: E402
from probe.simdata import lift_to_clear_table  # noqa: E402
from probe.stick import StickParams  # noqa: E402

W, H = 640, 480  # per panel before rotating upright (the recording's 4:3)


def upright(img):
    return np.rot90(img, k=-1)  # stored frames have the table on the right: rotate clockwise so gravity is down


def label(img, text):
    """Burn a small dark title bar into the top of the panel (no font dependency)."""
    out = img.copy()
    out[:28] = (out[:28] * 0.35).astype(np.uint8)
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(out)
        ImageDraw.Draw(im).text((8, 7), text, fill=(255, 255, 255))
        out = np.asarray(im)
    except ImportError:
        pass
    return out


class SimFrames:
    """Renders the matched iPad view and a wide free-camera view at the requested sim times."""

    def __init__(self, times):
        self.times, self.k, self.frames = np.asarray(times), 0, []
        self.r = None

    def __call__(self, m, d, t):
        while self.k < len(self.times) and t + 1e-9 >= self.times[self.k]:
            if self.r is None:
                m.vis.global_.offwidth = max(m.vis.global_.offwidth, W, H)
                m.vis.global_.offheight = max(m.vis.global_.offheight, W, H)
                self.r = mujoco.Renderer(m, H, W)
                self.r_wide = mujoco.Renderer(m, W, H)  # already portrait: a free camera, nothing to rotate
                self.wide = mujoco.MjvCamera()
                self.wide.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.wide.lookat[:] = (0.40, 0.0, 1.0)
                self.wide.distance, self.wide.azimuth, self.wide.elevation = 1.5, -120.0, -15.0
            self.r.update_scene(d, camera="ipad")
            ipad = self.r.render().copy()
            self.r_wide.update_scene(d, camera=self.wide)
            self.frames.append((ipad, self.r_wide.render().copy()))
            self.k += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obs", help="tracked clip (.npz); grip path = .grip.csv beside it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--friction", type=float, default=0.55)
    a = ap.parse_args()
    path = Path(a.obs).resolve()
    video = load_observation(path)
    meta = video.meta
    export = _bootstrap.REPO / meta["stray_export"]
    traj = load_trajectory(path.with_suffix(".grip.csv"))
    pitch = float(np.median(np.degrees(grip_features(video.t, video.grip_pose)[:, 4])))
    straight = pitch > 70
    cam = CameraConfig.from_json(export / "camera.json")
    stick = StickParams()
    scene = SceneConfig(camera=cam, **({"exit_angle_deg": 0.0} if straight else {}))
    if straight:
        traj = lift_to_clear_table(traj, stick, scene, 0.003)

    # tracker clock -> video frames: the observation starts at the first kept frame, rest_trim_s after the clip's
    # first frame (odometry timestamps give each video frame's time)
    ts = np.genfromtxt(export / "odometry.csv", delimiter=",", skip_header=1, usecols=0)
    ts = ts - ts[0]
    t_out = np.arange(0.0, float(traj.t[-1]), 1.0 / a.fps)
    vid_idx = np.clip(np.searchsorted(ts, t_out + meta.get("rest_trim_s", 0.0)), 0, len(ts) - 1)

    sim = SimFrames(t_out)
    run_rollout(traj, "free", stick, scene, mass=meta["object_mass"], friction=a.friction, record_hz=2 * a.fps,
                frame_callback=sim)
    n = min(len(sim.frames), len(t_out))
    reader = imageio.get_reader(export / "rgb.mp4")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=1)
    want, cur, frame = {}, -1, None
    for i in range(n):
        j = int(vid_idx[i])
        while cur < j:
            frame = reader.get_next_data()
            cur += 1
        real = np.asarray(frame)[::3, ::3][:H, :W]
        ipad, wide = sim.frames[i]
        g = f"{1e3 * meta['object_mass']:.0f} g"
        panels = [label(upright(real), f"real clip ({g})"), label(upright(ipad), "sim-IRL, same camera"),
                  label(wide, f"sim UR5e replaying the grip path  t={t_out[i]:.1f}s")]
        writer.append_data(np.concatenate(panels, axis=1))
    writer.close()
    if sim.r is not None:
        sim.r.close()
        sim.r_wide.close()
    print(f"{a.out}: {n} frames @ {a.fps:g} fps ({json.dumps({'straight': straight, 'mass_g': 1e3 * meta['object_mass']})})")


if __name__ == "__main__":
    main()
