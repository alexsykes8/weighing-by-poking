"""Mass estimates for the bench pokes (one box, filled to 100/160/230/345 g, ~10 hand pokes per fill; tracked by
scripts/track_pokes.sh into data/runs/real/bench/<fill>_p<NN>.npz), read exactly like estimate_real.py.

    python scripts/bench_report.py data/runs/real/bench/*.npz --out out/bench

Each poke is flagged by two rules (applied after the pokes were read; both are reported, nothing is dropped):
  loaded start   the needle is already bent when tracking starts (median bow over the first 0.4 s more than
                 --loaded-mm above the bow over the last 0.4 s): the poke began in contact with the box, so there
                 is no unloaded moment to zero the rest bend from (tier A in data/usable/README.md needs one)
  no bend seen   95th-percentile bow under --min-peak-mm: the tracker never saw the needle bend
Scenes: every combination of three pokes with different fills, asked both "heaviest" and "lightest"; right when
the highest (lowest) estimate is the heaviest (lightest) fill. Also: for every pair of pokes with different fills,
whether the heavier fill reads heavier.
Writes <out>/pokes.json and <out>/summary.json, and prints per-fill results for all pokes and the clean ones.
"""
import itertools
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from probe.estimator import zero_rest_shape
from probe.io import load_observation
from probe.multibox import Estimator
from probe.stick import StickParams

CKPT = _bootstrap.REPO / "checkpoints" / "estimator" / "gru-n5000-1007-1935" / "best.pt"


def bow_mm(cl):
    """Per frame: the centreline's largest distance from the grip-tip chord, mm."""
    base, tip = cl[:, 0], cl[:, -1]
    c = (tip - base) / np.linalg.norm(tip - base, axis=-1, keepdims=True)
    rel = cl - base[:, None]
    perp = rel - (rel * c[:, None]).sum(-1, keepdims=True) * c[:, None]
    return 1e3 * np.linalg.norm(perp, axis=-1).max(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obs", nargs="+")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--friction", type=float, default=0.55)
    ap.add_argument("--loaded-mm", type=float, default=1.5)
    ap.add_argument("--min-peak-mm", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    est, st = Estimator(a.ckpt), StickParams()
    rows = []
    for path in sorted(a.obs):
        o = load_observation(path)
        b, t = bow_mm(o.centreline), o.t
        start, end = float(np.median(b[t < t[0] + 0.4])), float(np.median(b[t > t[-1] - 0.4]))
        peak = float(np.percentile(b, 95))
        logm, sd, _ = est(o.t, o.grip_pose, zero_rest_shape(o.t, o.centreline), o.arclen, st, a.friction)
        flags = (["loaded start"] if start - end > a.loaded_mm else []) + (["no bend seen"] if peak < a.min_peak_mm else [])
        true_g = 1e3 * o.meta["object_mass"]
        rows.append({"poke": Path(path).stem, "true_g": true_g, "est_g": 1e3 * float(np.exp(logm)), "sd": float(sd),
                     "err": float(np.exp(logm) * 1e3 / true_g - 1), "bow_start_mm": start, "bow_end_mm": end,
                     "bow_peak_mm": peak, "frames": len(t), "flags": flags})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pokes.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"{r['poke']:10s} {r['est_g']:4.0f} g ({100 * r['err']:+4.0f}%)  bow start/end/peak "
              f"{r['bow_start_mm']:4.1f}/{r['bow_end_mm']:4.1f}/{r['bow_peak_mm']:4.1f} mm  {', '.join(r['flags'])}")
    summary = {}
    for name, sel in (("all", rows), ("clean", [r for r in rows if not r["flags"]])):
        print(f"\n{name} pokes")
        summary[name] = {"per_fill": {}, "scenes": scenes(sel)}
        for g in sorted({r["true_g"] for r in rows}):
            e = np.array([r["est_g"] for r in sel if r["true_g"] == g])
            if not len(e):
                continue
            err = np.abs(e / g - 1)
            summary[name]["per_fill"][f"{g:.0f}"] = {"n": len(e), "median_g": float(np.median(e)), "iqr_g": np.percentile(e, [25, 75]).tolist(),
                                         "mean_abs_err": float(err.mean()), "within_20pct": float((err <= 0.2).mean())}
            print(f"  {g:4.0f} g: n {len(e):2d}  median {np.median(e):4.0f} g (IQR {np.percentile(e, 25):.0f}-"
                  f"{np.percentile(e, 75):.0f})  mean |err| {err.mean():.0%}  within 20%: {(err <= 0.2).mean():.0%}")
        sc = summary[name]["scenes"]
        print(f"  3-box scenes right: {sc['correct']}/{sc['n']} = {sc['rate']:.0%}; heavier of a pair: "
              f"{sc['pairwise']:.0%}")
        for k, v in sc["by_trio"].items():
            print(f"    {k}: {v:.0%}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def scenes(rows):
    by = {}
    for r in rows:
        by.setdefault(r["true_g"], []).append(r["est_g"])
    fills, k, n, by_trio = sorted(by), 0, 0, {}
    for trio in itertools.combinations(fills, 3):
        for goal, f in (("heaviest", np.argmax), ("lightest", np.argmin)):
            ok = [f(e) == f(trio) for e in itertools.product(*[by[w] for w in trio])]
            by_trio[f"{'/'.join(f'{w:.0f}' for w in trio)} {goal}"] = float(np.mean(ok))
            k, n = k + sum(ok), n + len(ok)
    pairs = [y > x for a, b in itertools.combinations(fills, 2) for x in by[a] for y in by[b]]
    return {"correct": int(k), "n": n, "rate": k / n, "pairwise": float(np.mean(pairs)), "by_trio": by_trio}


if __name__ == "__main__":
    main()
