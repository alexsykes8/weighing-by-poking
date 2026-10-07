"""Sim camera (pixel intrinsics + pose) from a tripod Stray Scanner export.

    python scripts/camera_from_stray.py data/raw/clip01 --anchor-px 1210 905 --axis-px 420 930
    python scripts/render.py ... --camera data/raw/clip01/camera.json

--anchor-px  RGB pixel (1920x1440 frame) at the centre of the active box's camera-facing face. Its sim
             position is known (probe.scene.object_front_face_centre), which anchors the translation.
             Pick a spot with solid LiDAR return: flat, matte, square to the camera.
--axis-px    a second pixel on the push line, >= 0.3 m from the anchor (e.g. a tape mark on the table at
             the same depth as the push plane). Fixes yaw; without it the camera is assumed square to the
             push plane. At 0.1 m separation depth noise gives ~8 px reprojection error; at >= 0.3 m, < 1 px.
--frames     frame range used for the tripod pose (default: all), e.g. --frames 0 120
--mirrored   the needle enters from the RIGHT of the frame and pushes leftward. The sim keeps pushing along +x
             and puts its camera (and mirrors the robot) on the other side of the push plane.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from probe.arkit import camera_from_stray, check_report, load_stray
from probe.scene import SceneConfig, object_front_face_centre


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="Stray Scanner export directory")
    ap.add_argument("--anchor-px", nargs=2, type=float, required=True, metavar=("U", "V"))
    ap.add_argument("--anchor-sim", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="sim position of the anchor (default: centre of the active box's front face)")
    ap.add_argument("--axis-px", nargs=2, type=float, metavar=("U", "V"))
    ap.add_argument("--frame", type=int, help="frame whose depth map is used for the clicked points (default: middle)")
    ap.add_argument("--frames", nargs=2, type=int, metavar=("START", "STOP"))
    ap.add_argument("--min-conf", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--mirrored", action="store_true", help="needle enters from the right of the frame")
    ap.add_argument("--out", help="default: <export>/camera.json")
    a = ap.parse_args()

    exp = load_stray(a.export)
    anchor_S = np.array(a.anchor_sim) if a.anchor_sim else object_front_face_centre(SceneConfig(mirrored=a.mirrored))
    frames = np.arange(*a.frames) if a.frames else None
    cam, rep = camera_from_stray(exp, tuple(a.anchor_px), anchor_S, axis_px=tuple(a.axis_px) if a.axis_px else None,
                                 frame=a.frame, frames=frames, min_conf=a.min_conf, mirrored=a.mirrored)
    out = Path(a.out or Path(a.export) / "camera.json")
    cam.to_json(out)

    print(f"export: {exp.root}  ({exp.n} frames, RGB {exp.rgb_size[0]}x{exp.rgb_size[1]}, "
          f"{len(exp.depth_files)} depth maps)")
    print(f"intrinsics: fx={cam.fx:.1f} fy={cam.fy:.1f} cx={cam.cx:.1f} cy={cam.cy:.1f} px")
    print(json.dumps(rep, indent=2))
    for w in check_report(rep):
        print(f"WARNING: {w}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
