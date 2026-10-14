"""Find the separate pokes in one long Stray clip (several pokes of the same box, the stick taken out of shot
between them) and print/write their frame ranges for track_video.py --frames.

    python scripts/split_pokes.py data/raw/bench/230g --body-v 0.55 --out data/raw/bench/230g/pokes.json

Cheap pass: every --step-th frame at quarter resolution, count the needle's bright-blue pixels. A poke is a run of
frames with the needle in shot lasting >= --min-s, split where it is out of shot for >= --gap-s. Each window is
padded by --pad-s so track_video's rest trim can find the still moment before the push. A window longer than
--long-s probably holds several pokes (box reset with the needle still in shot): it is flagged, split it by eye.
"""
import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from matplotlib.colors import rgb_to_hsv

from probe.tracker import ColourThresholds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    ap.add_argument("--body-v", type=float, help="as track_video.py --body-v")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--min-px", type=int, default=150, help="needle pixels (quarter res) to count as in shot")
    ap.add_argument("--min-s", type=float, default=3.0)
    ap.add_argument("--gap-s", type=float, default=1.0)
    ap.add_argument("--pad-s", type=float, default=0.3)
    ap.add_argument("--long-s", type=float, default=20.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    root = Path(a.export)
    body = ColourThresholds().body
    if a.body_v is not None:
        body.v_min = a.body_v
    t = np.genfromtxt(root / "odometry.csv", delimiter=",", skip_header=1)[:, 0]
    n, W, H = len(t), 480, 360
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(root / "rgb.mp4"), "-vf",
                             f"select=not(mod(n\\,{a.step})),scale={W}:{H}", "-vsync", "0", "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    idx, count = [], []
    k = 0
    while True:
        buf = proc.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        hsv = rgb_to_hsv(np.frombuffer(buf, np.uint8).reshape(H, W, 3) / 255.0)
        idx.append(k * a.step)
        count.append(int(body.mask(hsv[..., 0], hsv[..., 1], hsv[..., 2]).sum()))
        k += 1
    idx, on = np.array(idx), np.array(count) >= a.min_px
    fps = (n - 1) / (t[-1] - t[0])
    gap, pad = int(a.gap_s * fps / a.step), int(a.pad_s * fps)
    runs, start, off = [], None, 0
    for i, v in enumerate(on):
        if v:
            start = i if start is None else start
            off = 0
        elif start is not None:
            off += 1
            if off >= gap:
                runs.append((start, i - off))
                start, off = None, 0
    if start is not None:
        runs.append((start, len(on) - 1))
    pokes = []
    for s, e in runs:
        f0, f1 = max(0, idx[s] - pad), min(n, idx[e] + a.step + pad)
        if (f1 - f0) / fps >= a.min_s:
            pokes.append([int(f0), int(f1)])
    for i, (f0, f1) in enumerate(pokes):
        print(f"poke {i + 1:2d}: frames {f0}-{f1} ({(f1 - f0) / fps:.1f} s)"
              + ("  LONG: several pokes? split by eye" if (f1 - f0) / fps > a.long_s else ""))
    print(f"{len(pokes)} pokes in {n} frames ({n / fps:.0f} s)")
    if a.out:
        Path(a.out).write_text(json.dumps({"export": str(root), "fps": fps, "pokes": pokes}, indent=2))


if __name__ == "__main__":
    main()
