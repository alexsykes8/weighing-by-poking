"""Run the sim-trained mass estimator on tracked real clips (data-contract .npz from track_video.py).

    python scripts/estimate_real.py data/runs/real/520g2_*.npz
    python scripts/estimate_real.py data/runs/real/*.npz --ckpt checkpoints/estimator/<run>/best.pt --friction 0.55

The whole tracked observation is one probe (track_video.py already trims it to start once the needle is still).
Stick parameters are the measured needle (StickParams defaults); friction is the measured static mu.
Prints the estimate with its 1-sigma range and P(anchored) next to the scale reading stored in the clip's meta.
The needle's rest bend (its shape over that first still 0.4 s) is subtracted first: see
probe.estimator.zero_rest_shape. --no-rest-zero turns this off.
"""
import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from probe.estimator import zero_rest_shape
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
    ap.add_argument("--no-rest-zero", action="store_true", help="feed the tracked centreline as is")
    a = ap.parse_args()
    est, st = Estimator(a.ckpt), StickParams()
    rows = []
    for path in a.obs:
        o = load_observation(path)
        if "rest_trim_s" not in o.meta and not a.no_rest_zero:
            print(f"note: {path} has no still window before the push; its rest bend is read off the first "
                  f"0.4 s anyway")
        cl = o.centreline if a.no_rest_zero else zero_rest_shape(o.t, o.centreline)
        logm, sd, p_anch = est(o.t, o.grip_pose, cl, o.arclen, st, a.friction)
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
