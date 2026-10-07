"""Generate a labelled synthetic dataset and fit a small learned classifier on curvature features.

    python scripts/classify.py --per-class 25 --workers 16       # simulate + fit (in-domain sim evaluation)
    python scripts/classify.py --reuse                           # refit on data/runs/dataset/*.npz
    python scripts/classify.py --reuse --real data/runs/real     # train on ALL sim, test on real video observations

Each rollout randomises push speed, contact point (height), approach angle (pitch), push depth and
the object's mass/friction within its class. Classifier: standardised logistic regression.

--real DIR: every *.npz under DIR with meta.source == "video" is a test example (label from the file).
Reports real accuracy with an exact 95% CI, the sim->real gap (sim 5-fold CV minus real accuracy),
per-class accuracy, a confusion matrix, and a per-feature shift showing which part of the signature
doesn't transfer. --train-noise-mm trains on tracker-noised sim (a cheap domain-randomisation knob).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probe import features
from probe.classifier import evaluate_in_domain, evaluate_transfer, feature_table, real_paths
from probe.io import save_observation
from probe.schema import CLASSES as LABELS

REPO = _bootstrap.REPO
DATA = REPO / "data" / "runs" / "dataset"
OUT = REPO / "out"
INK, INK2, GRID, SURF, BLUE, ORANGE = "#1a1a19", "#5f5e58", "#e4e3dc", "#fcfcfb", "#2a78d6", "#eb6834"


def simulate_one(job):
    label, seed = job
    from probe.rollout import run_rollout
    from probe.scene import SceneConfig
    from probe.trajectory import PushParams, make_push, push_meta
    rng = np.random.default_rng(seed)
    pp = PushParams.sample(rng)
    cls = SceneConfig().classes.get(label)
    mass = float(rng.uniform(*cls.mass)) if cls and label != "fixed" else None
    fric = float(rng.uniform(*cls.friction)) if cls else None
    obs = run_rollout(make_push(pp), label, mass=mass, friction=fric, extra_meta={"push": push_meta(pp), "seed": seed})
    path = DATA / f"{label}_{seed:05d}.npz"
    save_observation(path, obs)
    return str(path)


def _confusion(ax, cm, title):
    cm = np.asarray(cm)
    vmax = max(cm.sum(1).max(), 1)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11, color="white" if cm[i, j] > vmax / 2 else INK)
    ax.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="predicted", ylabel="true", title=title)


def plot_in_domain(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(4.4 * len(results), 4), constrained_layout=True)
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        _confusion(ax, r["confusion"], f"{name}\nheld-out acc {r['heldout_acc']:.0%}  "
                                       f"(5-fold CV {r['cv_mean']:.0%} ± {r['cv_std']:.0%})")
    fig.savefig(path, dpi=150)


def plot_transfer(r, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1, 1.2]})
    lo, hi = r["real_ci95"]
    _confusion(a1, r["confusion"], f"real (n={r['n_real']}): acc {r['real_acc']:.0%} [95% CI {lo:.0%}–{hi:.0%}]\n"
                                   f"sim CV {r['sim_cv_acc']:.0%}  →  gap {r['gap']:+.0%}")
    names = list(features.FEATURES)
    z = np.array([r["feature_shift"][n]["z_shift"] for n in names])
    out = np.array([r["feature_shift"][n]["frac_outside_sim_range"] for n in names])
    y = np.arange(len(names))
    a2.barh(y, z, color=np.where(np.abs(z) > 2, ORANGE, BLUE), height=0.6)
    a2.axvline(0, color=INK2, lw=0.8)
    for x in (-2, 2):
        a2.axvline(x, color=INK2, lw=0.8, ls="--")
    a2.set_xscale("symlog", linthresh=2)  # linear inside +-2 sigma, log beyond: one huge shift can't hide the rest
    lim = max(4.0, np.abs(z).max() * 1.5)
    a2.set_xlim(-lim, lim)
    a2.set(yticks=y, yticklabels=[f"{n}  ({o:.0%} out of sim range)" for n, o in zip(names, out)],
           xlabel="real mean − sim mean  (sim std units, symlog)", title="Feature shift sim → real (orange: |z| > 2)")
    a2.invert_yaxis()
    a2.grid(axis="x", color=GRID, lw=0.6)
    fig.savefig(path, dpi=150)


def run_real(paths_sim, a):
    real, skipped = real_paths(a.real)
    for s in skipped:
        print(f"  skipped {s}")
    if not real:
        print(f"no real (meta.source == 'video') observations under {a.real}: nothing to evaluate yet.")
        return
    Xs, ys, _ = feature_table(paths_sim, a.train_noise_mm * 1e-3, a.seed)
    Xr, yr, _ = feature_table(real, 0.0)
    r = evaluate_transfer(Xs, ys, Xr, yr, names_real=[p.name for p in real], seed=a.seed)
    r.update(train_noise_mm=a.train_noise_mm, real_dir=str(a.real), skipped=skipped)
    lo, hi = r["real_ci95"]
    print(f"sim 5-fold CV {r['sim_cv_acc']:.1%} ± {r['sim_cv_std']:.1%} (n={r['n_sim']}, train noise "
          f"{a.train_noise_mm:g} mm)")
    print(f"real accuracy {r['real_acc']:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  (n={r['n_real']}, chance {r['chance']:.0%})")
    print(f"sim->real gap {r['gap']:+.1%}")
    for c, pc in r["per_class"].items():
        print(f"  {c:9s} {pc['acc']:.0%} of {pc['n']}  CI [{pc['ci95'][0]:.0%}, {pc['ci95'][1]:.0%}]")
    if r["classes_untested"]:
        print(f"  untested classes (no real examples): {r['classes_untested']}")
    big = [(n, s["z_shift"]) for n, s in r["feature_shift"].items() if abs(s["z_shift"]) > 2]
    if big:
        print("  features that do not transfer (|z| > 2): " + ", ".join(f"{n} ({z:+.1f})" for n, z in big))
    OUT.mkdir(exist_ok=True)
    plot_transfer(r, OUT / "sim2real.png")
    (OUT / "sim2real_report.json").write_text(json.dumps(r, indent=2))
    print(f"wrote {OUT/'sim2real.png'} and {OUT/'sim2real_report.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reuse", action="store_true", help="skip simulation, use existing dataset files")
    ap.add_argument("--noise-mm", type=float, default=1.0, help="tracker noise for the in-domain robustness eval")
    ap.add_argument("--real", help="directory of real (video) observations to test sim-trained model on")
    ap.add_argument("--train-noise-mm", type=float, default=0.0, help="with --real: tracker noise added to sim training")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    if not a.reuse:
        jobs = [(lab, 1000 * i + j) for i, lab in enumerate(LABELS) for j in range(a.per_class)]
        print(f"simulating {len(jobs)} rollouts on {a.workers} workers ...", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            for n, _ in enumerate(ex.map(simulate_one, jobs), 1):
                if n % 10 == 0:
                    print(f"  {n}/{len(jobs)}", flush=True)
    paths = sorted(DATA.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no sim dataset in {DATA}; run without --reuse first")

    if a.real:
        run_real(paths, a)
        return

    results = {}
    for name, noise in (("clean centreline", 0.0), (f"{a.noise_mm:g} mm tracker noise", a.noise_mm * 1e-3)):
        X, y, _ = feature_table(paths, noise, a.seed)
        results[name] = evaluate_in_domain(X, y, a.seed)
        r = results[name]
        print(f"[{name}] held-out acc {r['heldout_acc']:.1%} on {r['n_test']} | 5-fold CV {r['cv_mean']:.1%} ± {r['cv_std']:.1%}")
        print("  confusion (rows true, cols pred, order %s):\n  %s" % (LABELS, np.array(r["confusion"])))
    OUT.mkdir(exist_ok=True)
    plot_in_domain(results, OUT / "confusion.png")
    (OUT / "classifier_report.json").write_text(json.dumps({"n_rollouts": len(paths), "features": features.FEATURES,
                                                            "labels": list(LABELS), "results": results}, indent=2))
    print(f"wrote {OUT/'confusion.png'} and {OUT/'classifier_report.json'}")


if __name__ == "__main__":
    main()
