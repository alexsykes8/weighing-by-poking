"""Offscreen render of a rollout from the MJCF camera ('ipad') to mp4. Headless (EGL, falling back to OSMesa).

    python scripts/render.py --trajectory data/trajectories/synthetic_push.csv --object fixed \
        --out out/fixed.mp4 [--width 960 --fps 30] [--camera data/raw/clip01/camera.json] [--save-obs x.npz]
"""
import argparse
import os
import subprocess
import sys

BACKENDS = ("egl", "osmesa")
PROBE = "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); mujoco.Renderer(m, 8, 8).render()"


def pick_backend() -> str:
    """Find a working headless GL backend. MUJOCO_GL is read at import time, so probe in subprocesses."""
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    errors = {}
    for b in BACKENDS:
        r = subprocess.run([sys.executable, "-c", PROBE], env={**os.environ, "MUJOCO_GL": b},
                           capture_output=True, text=True)
        if r.returncode == 0:
            return b
        errors[b] = (r.stderr.strip().splitlines() or ["?"])[-1]
    msg = "\n".join(f"  {b}: {e}" for b, e in errors.items())
    sys.exit(f"No headless OpenGL backend available.\n{msg}\n"
             "Install one of: libegl1 (+ GPU driver) for EGL, or libosmesa6 for OSMesa.")


os.environ["MUJOCO_GL"] = pick_backend()

import _bootstrap  # noqa: E402,F401
import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from probe.io import load_trajectory, save_observation  # noqa: E402
from probe.rollout import run_rollout  # noqa: E402
from probe.schema import LABELS  # noqa: E402
from probe.camera import CameraConfig  # noqa: E402
from probe.scene import SceneConfig  # noqa: E402
from probe.stick import StickParams  # noqa: E402


class FrameWriter:
    def __init__(self, path, width, height, fps, camera="ipad"):
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        self.width, self.height, self.fps, self.camera = width, height, fps, camera
        self.renderer = None
        self.next_t = 0.0
        self.n = 0

    def __call__(self, m, d, t):
        if t + 1e-9 < self.next_t:
            return
        if self.renderer is None:
            m.vis.global_.offwidth = max(m.vis.global_.offwidth, self.width)
            m.vis.global_.offheight = max(m.vis.global_.offheight, self.height)
            self.renderer = mujoco.Renderer(m, self.height, self.width)
        self.renderer.update_scene(d, camera=self.camera)
        self.writer.append_data(self.renderer.render())
        self.next_t += 1.0 / self.fps
        self.n += 1

    def close(self):
        self.writer.close()
        if self.renderer is not None:
            self.renderer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--object", required=True, choices=LABELS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--young", type=float, default=StickParams.young)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, help="default: keep the camera's aspect ratio")
    ap.add_argument("--camera", help="camera JSON (e.g. from scripts/camera_from_stray.py); default: built-in")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--save-obs", help="also write the observation .npz")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    scene = SceneConfig(camera=CameraConfig.from_json(a.camera)) if a.camera else SceneConfig()
    res = scene.camera.resolution
    a.height = a.height or int(round(a.width * res[1] / res[0] / 2)) * 2
    if abs(a.width / a.height - res[0] / res[1]) > 0.01:
        print(f"warning: {a.width}x{a.height} does not match the camera aspect {res[0]}x{res[1]}; image will be stretched")
    fw = FrameWriter(a.out, a.width, a.height, a.fps)
    try:
        obs = run_rollout(load_trajectory(a.trajectory), a.object, StickParams(young=a.young), scene, frame_callback=fw,
                          record_hz=max(100.0, a.fps))
    finally:
        fw.close()
    if a.save_obs:
        save_observation(a.save_obs, obs)
    print(f"{a.out}: {fw.n} frames @ {a.fps} fps, {a.width}x{a.height}, backend={os.environ['MUJOCO_GL']}")


if __name__ == "__main__":
    main()
