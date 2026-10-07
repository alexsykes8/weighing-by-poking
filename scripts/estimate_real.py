"""Run the sim-trained mass estimator on tracked real clips (data-contract .npz from track_video.py).

    python scripts/estimate_real.py data/runs/real/520g2_*.npz
    python scripts/estimate_real.py data/runs/real/*.npz --ckpt checkpoints/estimator/<run>/best.pt --friction 0.55

The whole tracked observation is one probe (track_video.py already trims it to start once the needle is still).
Stick parameters are the measured needle (StickParams defaults); friction is the measured static mu.
Prints the estimate with its 1-sigma range and P(anchored) next to the scale reading stored in the clip's meta.
"""
import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from probe.io import load_observation
from probe.multibox import Estimator
from probe.stick import StickParams

CKPT = _bootstrap.REPO / "checkpoints" / "estimator" / "gru-n5000-1006-1818" / "best.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obs", nargs="+")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--friction", type=float, default=0.55)
    ap.add_argument("--json", help="also write the rows here")
    a = ap.parse_args()
    est, st = Estimator(a.ckpt), StickParams()
    rows = []
    for path in a.obs:
        o = load_observation(path)
        logm, sd, p_anch = est(o.t, o.grip_pose, o.centreline, o.arclen, st, a.friction)
        true = o.meta.get("object_mass")
        r = {"clip": path, "true_kg": true, "est_kg": float(np.exp(logm)), "lo_kg": float(np.exp(logm - sd)),
             "hi_kg": float(np.exp(logm + sd)), "p_anchored": p_anch, "frames": len(o.t), "dur_s": float(o.t[-1])}
        rows.append(r)
        err = f"{100 * (r['est_kg'] / true - 1):+.0f}%" if true else ""
        print(f"{path}: {1e3 * r['est_kg']:.0f} g (1σ {1e3 * r['lo_kg']:.0f}-{1e3 * r['hi_kg']:.0f}) "
              f"true {1e3 * true if true else float('nan'):.0f} g {err}  P(anchored) {p_anch:.2f}  "
              f"[{r['frames']} frames, {r['dur_s']:.1f} s]")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
