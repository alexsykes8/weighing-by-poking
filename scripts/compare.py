"""Compare two probe observations: tip error over time + side-by-side centreline frame grid.

    python scripts/compare.py data/runs/a.npz data/runs/b.npz --out out/compare [--names sim video]

Both files go through the same contract (stick-local frame), so either can come from sim or video.
They are aligned on time (B is interpolated onto A's clock, overlapping window only).
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probe.io import load_observation

INK, INK2, GRID, SURF = "#1a1a19", "#5f5e58", "#e4e3dc", "#fcfcfb"
C_A, C_B = "#2a78d6", "#eb6834"


def align(a, b):
    """Interpolate B's tip and centreline onto A's timestamps within the overlapping window."""
    if not np.allclose(a.arclen, b.arclen, atol=1e-6):
        raise ValueError("observations use different arclen sampling; resample one first")
    t0, t1 = max(a.t[0], b.t[0]), min(a.t[-1], b.t[-1])
    sel = (a.t >= t0) & (a.t <= t1)
    t = a.t[sel]
    tip_b = np.stack([np.interp(t, b.t, b.tip[:, i]) for i in range(3)], axis=1)
    flat = b.centreline.reshape(len(b.t), -1)
    cl_b = np.stack([np.interp(t, b.t, flat[:, j]) for j in range(flat.shape[1])], axis=1).reshape(len(t), -1, 3)
    return t, a.tip[sel], tip_b, a.centreline[sel], cl_b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default=str(_bootstrap.REPO / "out" / "compare"))
    ap.add_argument("--names", nargs=2, help="legend names (default: from meta)")
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()

    A, B = load_observation(a.a), load_observation(a.b)
    names = a.names or [f"{o.meta['source']} {Path(p).stem} (E={o.meta['young']:.2g})" for o, p in ((A, a.a), (B, a.b))]
    t, tipA, tipB, clA, clB = align(A, B)
    err = np.linalg.norm(tipA - tipB, axis=1) * 1e3
    err_xz = np.linalg.norm((tipA - tipB)[:, [0, 2]], axis=1) * 1e3
    shape_err = np.linalg.norm(clA - clB, axis=2).mean(axis=1) * 1e3
    summary = {"a": a.a, "b": a.b, "n": int(t.size), "tip_err_mm": {"mean": float(err.mean()), "rms": float(np.sqrt((err**2).mean())),
               "p95": float(np.percentile(err, 95)), "max": float(err.max())},
               "tip_err_inplane_mm_rms": float(np.sqrt((err_xz**2).mean())),
               "centreline_err_mm_mean": float(shape_err.mean())}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.facecolor": SURF, "figure.facecolor": SURF})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, constrained_layout=True)
    ax1.plot(t, tipA[:, 2] * 1e3, color=C_A, lw=2, label=names[0])
    ax1.plot(t, tipB[:, 2] * 1e3, color=C_B, lw=2, label=names[1])
    ax1.set(ylabel="tip z, stick-local (mm)", title="Tip deflection (held-out prediction)")
    ax1.legend(frameon=False, fontsize=8)
    ax2.plot(t, err, color=INK, lw=2, label="|tip_A − tip_B|  (3D)")
    ax2.plot(t, shape_err, color=INK2, lw=1.2, ls="--", label="mean centreline error")
    ax2.set(xlabel="t (s)", ylabel="error (mm)",
            title=f"tip error: RMS {summary['tip_err_mm']['rms']:.1f} mm, p95 {summary['tip_err_mm']['p95']:.1f} mm, "
                  f"max {summary['tip_err_mm']['max']:.1f} mm")
    ax2.legend(frameon=False, fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=0.6)
    fig.savefig(out / "tip_error.png", dpi=150)

    idx = np.linspace(0, len(t) - 1, a.frames).astype(int)
    fig, axes = plt.subplots(1, a.frames, figsize=(2.3 * a.frames, 2.6), sharey=True, constrained_layout=True)
    zmin = min(clA[..., 2].min(), clB[..., 2].min()) * 1e3
    for ax, i in zip(axes, idx):
        ax.plot(clA[i, :, 0] * 1e3, clA[i, :, 2] * 1e3, color=C_A, lw=2)
        ax.plot(clB[i, :, 0] * 1e3, clB[i, :, 2] * 1e3, color=C_B, lw=2)
        ax.plot(*tipA[i, [0, 2]] * 1e3, "o", color=C_A, ms=5)
        ax.plot(*tipB[i, [0, 2]] * 1e3, "o", color=C_B, ms=5)
        ax.set_title(f"t={t[i]:.2f}s  err {err[i]:.1f} mm", fontsize=8)
        ax.set_ylim(zmin - 10, 15)
        ax.set_xlabel("x (mm)")
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("z, stick-local (mm)")
    fig.legend(names, loc="outside upper center", ncol=2, frameon=False, fontsize=8)
    fig.savefig(out / "frames.png", dpi=150)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
