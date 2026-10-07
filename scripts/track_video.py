"""Track the stick in a Stray Scanner clip -> data-contract observation (.npz) + grip trajectory (.csv).

    # once per tripod setup: camera from the clip (see camera_from_stray.py)
    python scripts/camera_from_stray.py data/raw/clip01 --anchor-px U V --axis-px U2 V2
    # then per clip
    python scripts/track_video.py data/raw/clip01 --label free --mass 0.520 --out data/runs/real/clip01.npz
    python scripts/track_video.py data/raw/clip01 --debug 15 ...   # overlays every 15th frame, to tune colours

Stick markings are measured from the GRIP (where the needle leaves the clamp block): --band-end is where the
yellow band meets the blue body (default 0.069), --tip-start where the orange tip begins (0.294), --length the
physical tip (0.334). Defaults are the measured needle. Hue ranges (0..1) are config: --base-h/--body-h/--tip-h. The push plane is
sim y = --plane-y (default 0 = through the box centre, matching the camera alignment's box-face anchor).
Frame times come from Stray's odometry.csv. Frames where the stick isn't found are dropped and counted.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probe.arkit import load_stray
from probe.camera import CameraConfig
from probe.io import git_sha, save_observation, save_trajectory
from probe.schema import LABELS
from probe.stick import StickParams
from probe.tracker import ColourThresholds, Plane, StickMarkings, StickTracker, TrackerConfig, fixes_to_observation
from probe.trajectory import first_still_index


REST_TOLS = (4e-3, 8e-3, 12e-3)  # grip stillness over 0.4 s, tightest first (hand-held needles jitter ~5 mm)


def rest_before_push(t, grip, max_travel_frac=0.25):
    """(index, tol) of the first still window that comes before the push, i.e. before the grip has covered
    `max_travel_frac` of its travel along the push (sim +x; lowering the needle into place doesn't count);
    the tolerance is relaxed step by step. (None, None) if none does."""
    d = np.abs(grip[:, 0] - grip[0, 0])
    for tol in REST_TOLS:
        i = first_still_index(t, grip, tol=tol)
        if i is not None and d[i] <= max_travel_frac * d.max():
            return i, tol
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="Stray Scanner export directory (rgb.mp4 + odometry.csv)")
    ap.add_argument("--camera", help="camera JSON (default: <export>/camera.json)")
    ap.add_argument("--label", required=True, choices=LABELS)
    ap.add_argument("--mass", type=float, help="scale reading of the box, kg (meta; never used by the tracker)")
    mk0, st = StickMarkings(), StickParams()
    ap.add_argument("--length", type=float, default=mk0.length, help="grip -> tip, m")
    ap.add_argument("--band-end", type=float, default=mk0.band_end, help="grip -> yellow/blue boundary, m")
    ap.add_argument("--tip-start", type=float, default=mk0.tip_start, help="grip -> blue/orange boundary, m")
    ap.add_argument("--radius", type=float, default=st.radius)
    ap.add_argument("--young", type=float, default=st.young, help="E from the bench calibration, Pa (meta)")
    ap.add_argument("--plane-y", type=float, default=0.0)
    ap.add_argument("--frames", nargs=2, type=int, metavar=("START", "STOP"))
    ap.add_argument("--temporal", type=int, default=0, help="odd temporal smoothing window in frames (0 = off)")
    ap.add_argument("--k", type=int, default=41)
    ap.add_argument("--keep-plane", action="store_true",
                    help="don't move the push plane to the clip's median needle depth (default: move it; a constant "
                         "length offset means the needle is nearer/farther than the box-centre plane)")
    ap.add_argument("--no-rest-trim", action="store_true",
                    help="keep frames before the needle first holds still (default: drop them; the needle is usually "
                         "brought into shot moving, and the features' zero-load baseline is the first 0.4 s)")
    ap.add_argument("--base-h", nargs=2, type=float, help="HSV hue range of the yellow base band (0..1)")
    ap.add_argument("--body-h", nargs=2, type=float, help="HSV hue range of the blue body (0..1)")
    ap.add_argument("--tip-h", nargs=2, type=float, help="HSV hue range of the orange tip (0..1; lo > hi wraps)")
    ap.add_argument("--debug", type=int, default=0, help="save an overlay every N frames to <out>_debug/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj-out", help="grip trajectory CSV (default: alongside --out)")
    a = ap.parse_args()

    exp = load_stray(a.export)
    cam = CameraConfig.from_json(a.camera or Path(a.export) / "camera.json")
    colours = ColourThresholds()
    for part in ("base", "body", "tip"):
        if getattr(a, f"{part}_h"):
            getattr(colours, part).h = tuple(getattr(a, f"{part}_h"))
    mk = StickMarkings(length=a.length, band_end=a.band_end, tip_start=a.tip_start)
    plane = Plane(point=(0.0, a.plane_y, 0.0), normal=(0.0, 1.0, 0.0))
    # out-of-plane gate applied after the pass, relative to the clip's median length: a constant offset (needle
    # pushed off the box centre, or a few mm of paint-length error) is not frame-to-frame out-of-plane motion
    gate = TrackerConfig().max_scale_dev
    tracker = StickTracker(cam, TrackerConfig(markings=mk, colours=colours, plane=plane, max_scale_dev=np.inf))

    reader = imageio.get_reader(Path(a.export) / "rgb.mp4")
    lo, hi = a.frames or (0, exp.n)
    debug_dir = Path(a.out).with_suffix("").parent / (Path(a.out).stem + "_debug")
    fixes, times = [], []
    for i, frame in enumerate(reader):
        if i < lo:
            continue
        if i >= min(hi, exp.n):
            break
        if frame.shape[1] != cam.resolution[0] or frame.shape[0] != cam.resolution[1]:
            raise SystemExit(f"frame {frame.shape[1]}x{frame.shape[0]} != camera {cam.resolution}; wrong camera.json?")
        if exp.K_frames is not None:  # autofocus moves the focal length by up to ~1.3% within a clip
            fx, fy, cx, cy = exp.K_frames[i]
            tracker.cam = replace(cam, fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
        f = tracker.track(frame)
        fixes.append(f)
        times.append(exp.t[i])
        if a.debug and (i - lo) % a.debug == 0:
            debug_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(9, 6.75), constrained_layout=True)
            ax.imshow(frame)
            if f.ok:
                ax.plot(f.px[:, 0], f.px[:, 1], "-", color="#eb6834", lw=1.2)
                ax.plot(*f.px[0], "o", color="#eb6834", ms=4)
                ax.set_title(f"frame {i}  length scale {f.scale:.3f}  width {f.width_px:.1f} px")
            else:
                ax.set_title(f"frame {i}  LOST: {f.reason}")
            ax.axis("off")
            fig.savefig(debug_dir / f"{i:06d}.png", dpi=90)
            plt.close(fig)
        if i % 60 == 0:
            print(f"  frame {i}: {'ok' if f.ok else 'LOST (' + f.reason + ')'}", flush=True)

    meta = {"young": a.young, "radius": a.radius, "length": a.length,
            "timestep": float(np.median(np.diff(times))) if len(times) > 1 else float("nan"),
            "git_sha": git_sha(), "object_mass": a.mass, "stray_export": str(a.export),
            "camera": cam.source, "frames": [int(lo), int(lo + len(fixes))]}
    for f in fixes:  # grossly wrong lengths first, so they cannot drag the clip median
        if f.ok and abs(f.scale - 1) > 0.15:
            f.ok, f.reason = False, "length implausible (>15% off): occluded, out of frame or far out of plane"
    scales = np.array([f.scale for f in fixes if f.ok])
    med = float(np.median(scales)) if len(scales) else 1.0
    for f in fixes:
        if f.ok and abs(f.scale / med - 1) > gate:
            f.ok, f.reason = False, f"stick out of the push plane (length off the clip median by >{gate:.0%})"
    if not a.no_rest_trim:
        ok_idx = [k for k, f in enumerate(fixes) if f.ok]
        i_rest, tol = rest_before_push(np.asarray(times)[ok_idx], np.array([fixes[k].grip for k in ok_idx])) \
            if ok_idx else (None, None)
        if i_rest is None:
            print("WARNING: the needle never holds still for 0.4 s before the push (grip within 12 mm): "
                  "baseline will be unreliable")
        else:
            if tol > REST_TOLS[0]:
                print(f"note: no rest within {REST_TOLS[0] * 1e3:.0f} mm before the push; "
                      f"used the first 0.4 s window within {tol * 1e3:.0f} mm")
            for k in ok_idx[:i_rest]:
                fixes[k].ok, fixes[k].reason = False, "before the needle came to rest"
            meta["rest_trim_s"] = float(times[ok_idx[i_rest]] - times[0])
            meta["rest_tol_m"] = tol
    cam_depth = abs(np.asarray(cam.pos)[1] - a.plane_y)
    meta["length_scale_clip_median"] = med
    meta["implied_plane_offset_m"] = float(cam_depth / med - cam_depth)  # + = needle farther from the camera
    if not a.keep_plane:
        # moving a plane along its normal scales every back-projected point about the camera centre by the same
        # factor, so this is exact: the median blue length becomes nominal, shapes are unchanged
        C, k = np.asarray(cam.pos, float), 1.0 / med
        for f in fixes:
            if f.ok:
                f.pts, f.grip, f.scale = C + (f.pts - C) * k, C + (f.grip - C) * k, f.scale / med
        py = C[1] + (a.plane_y - C[1]) * k
        plane = Plane(point=(0.0, float(py), 0.0), normal=plane.normal)
        meta["plane_y_used"] = float(py)
    obs, traj = fixes_to_observation(times, fixes, mk, plane, a.label, meta, K=a.k, temporal_window=a.temporal)
    save_observation(a.out, obs)
    tpath = Path(a.traj_out or Path(a.out).with_suffix(".grip.csv"))
    save_trajectory(tpath, traj)
    sc = obs.extras["length_scale"]
    print(json.dumps({"observation": a.out, "trajectory": str(tpath), "frames": len(fixes),
                      "dropped": obs.meta["frames_dropped"], "drop_reasons": obs.meta["drop_reasons"],
                      "length_scale_median": float(np.median(sc)), "length_scale_p5_p95": np.percentile(sc, [5, 95]).tolist(),
                      "max_tip_drop_mm": float(-obs.tip[:, 2].min() * 1e3),
                      "implied_plane_offset_mm": meta["implied_plane_offset_m"] * 1e3}, indent=2))
    if abs(med - 1) > 0.03:
        print("WARNING: tracked stick length is off by >3%: check --band-end/--tip-start, the camera alignment "
              "(plane depth), or whether the stick leaves the push plane.")


if __name__ == "__main__":
    main()
