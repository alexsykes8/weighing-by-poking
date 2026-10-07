"""Cantilever calibration rig: validate the stick against beam theory and write the E inverse table.

    python scripts/calibrate.py                       # sweep, figure, probe/calibration.json
    python scripts/calibrate.py --measure 0.05 0.031  # E from a bench measurement (mass kg, delta m)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probe.calibration import (analytic_delta, build_master_curve, elastica_curve, load_calibration,
                               save_calibration, sim_delta, young_from_deflection)
from probe.io import git_sha
from probe.stick import StickParams, build_stick_xml

OUT = Path(__file__).resolve().parent.parent / "out"
LINEAR_LIMIT = 0.05  # delta/L

# reference palette (dataviz skill): blue sequential ramp for ordinal E, chrome/ink tokens
E_STEPS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK2, GRID, SURF, ACCENT = "#1a1a19", "#5f5e58", "#e4e3dc", "#fcfcfb", "#eb6834"


def self_weight_sag(p: StickParams) -> float:
    """Euler-Bernoulli tip sag of the unloaded stick under its own weight, wL^4/8EI."""
    w = p.density * np.pi * p.radius**2 * 9.81
    return w * p.length**4 / (8 * p.EI)


def sweep(p: StickParams, youngs, masses):
    rows = []
    for E in youngs:
        for m in masses:
            q = p.with_(young=E)
            rows.append((E, m, sim_delta(m, q), analytic_delta(m, q.length, q.radius, E), self_weight_sag(q)))
            print(f"  E={E:.1e} Pa  m={m*1e3:6.1f} g  sim={rows[-1][2]*1e3:8.2f} mm  "
                  f"EB={rows[-1][3]*1e3:8.2f} mm  ratio={rows[-1][2]/rows[-1][3]:.4f}", flush=True)
    return np.array(rows)


def linear_mask(rows, p):
    """Small-deflection regime judged on *total* sag (self-weight + hung mass), not the increment."""
    return (rows[:, 3] + rows[:, 4]) / p.length < LINEAR_LIMIT


def plot(rows, master, p, path):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)

    # left: dimensional sweep, sim vs Euler-Bernoulli
    lim = [rows[:, 2:4].min() * 0.6, rows[:, 2:4].max() * 1.6]
    a1.axvspan(lim[0], LINEAR_LIMIT * p.length, color=GRID, alpha=0.6, lw=0)
    a1.text(LINEAR_LIMIT * p.length * 0.85, lim[0] * 1.3, "small load\nδ/L < 0.05", ha="right",
            va="bottom", color=INK2, fontsize=8)
    a1.plot(lim, lim, color=INK2, lw=1, ls="--", label="sim = Euler–Bernoulli")
    for c, E in zip(E_STEPS, np.unique(rows[:, 0])):
        r = rows[rows[:, 0] == E]
        a1.plot(r[:, 3], r[:, 2], "-", color=c, lw=2)
        lin = linear_mask(r, p)
        a1.plot(r[lin, 3], r[lin, 2], "o", color=c, ms=6, mec=SURF, mew=1.5,
                label=f"E = {E:.0e} Pa  (self-weight sag {r[0, 4]/p.length:.1%} of L)")
        a1.plot(r[~lin, 3], r[~lin, 2], "o", color=SURF, ms=6, mec=c, mew=1.5)
    a1.plot([], [], "o", color=SURF, mec=INK2, mew=1.5, label="hollow: total sag/L > 0.05 (nonlinear)")
    a1.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Euler–Bernoulli δ = FL³/3EI  (m)",
           ylabel="simulated tip drop δ  (m)",
           title=f"Sweep: E × hanging mass, with self-weight\nL={p.length} m, r={p.radius*1e3:.0f} mm, "
                 f"ρ={p.density:.0f} kg/m³, N={p.n_elem}")
    a1.grid(color=GRID, lw=0.6)
    a1.legend(frameon=False, fontsize=8, loc="upper left")

    # right: dimensionless master curve vs linear theory and exact elastica
    al = np.asarray(master["alpha"])
    dl = np.asarray(master["delta_over_L"])
    af = np.logspace(np.log10(al.min()), np.log10(al.max()), 60)
    a2.axhspan(dl.min() * 0.5, LINEAR_LIMIT, color=GRID, alpha=0.6, lw=0)
    a2.plot(af, af / 3, color=INK2, lw=1, ls="--", label="linear theory  δ/L = α/3")
    a2.plot(af, elastica_curve(af), color=ACCENT, lw=2, label="exact elastica (BVP)")
    a2.plot(al, dl, "o", color=E_STEPS[2], ms=6, mec=SURF, mew=1.5, label="MuJoCo cable (sim)")
    err = np.abs(dl / elastica_curve(al) - 1).max()
    a2.set(xscale="log", yscale="log", ylim=(dl.min() * 0.5, 1.5), xlabel="load parameter  α = FL²/EI",
           ylabel="δ / L", title=f"Dimensionless master curve (weightless rod)\n"
                                 f"max |sim/elastica − 1| = {err:.2%}")
    a2.grid(color=GRID, lw=0.6)
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(path, dpi=160)
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=float, default=StickParams.length)
    ap.add_argument("--radius", type=float, default=StickParams.radius)
    ap.add_argument("--density", type=float, default=StickParams.density)
    ap.add_argument("--n-elem", type=int, default=StickParams.n_elem)
    ap.add_argument("--measure", nargs=2, type=float, metavar=("MASS_KG", "DELTA_M"),
                    help="skip the sweep; print E for a bench measurement using the saved table")
    ap.add_argument("--self-weight", action="store_true",
                    help="with --measure: include stick self-weight (uses --density, re-runs sim)")
    a = ap.parse_args()

    if a.measure:
        m, d = a.measure
        E = young_from_deflection(m, d, a.length, a.radius, density=a.density if a.self_weight else None)
        print(f"E = {E:.4g} Pa   (m={m} kg, delta={d} m, L={a.length} m, r={a.radius} m)")
        return

    p = StickParams(length=a.length, radius=a.radius, density=a.density, n_elem=a.n_elem)
    print("Sweep (E x mass):")
    rows = sweep(p, youngs=[1e9, 5e9, 2e10, 1e11], masses=[0.002, 0.01, 0.05, 0.2, 0.5])
    print("Master curve:")
    master = build_master_curve(np.logspace(-2, 1.3, 16), p)
    table = {"master": master, "sweep": {"cols": ["young", "mass", "sim_delta", "eb_delta", "eb_self_weight_sag"],
                                         "rows": rows.tolist(), "stick": p.to_dict()},
             "git_sha": git_sha()}
    save_calibration(table)
    models = Path(__file__).resolve().parent.parent / "models"
    (models / "cantilever.xml").write_text(build_stick_xml(p.length, p.radius, p.young, p.n_elem, p.damping,
                                                           density=p.density))

    OUT.mkdir(exist_ok=True)
    err = plot(rows, master, p, OUT / "calibration.png")
    lin = rows[linear_mask(rows, p)]
    print(f"\nlinear regime: sim/EB in [{(lin[:,2]/lin[:,3]).min():.4f}, {(lin[:,2]/lin[:,3]).max():.4f}] "
          f"over {len(lin)} points (total sag/L<{LINEAR_LIMIT}); master curve vs elastica max error {err:.3%}")
    # round-trip check of the inverse
    E_true, m = p.young, 0.05
    d = sim_delta(m, p, gravity=False)  # weightless rod, as the master curve
    print(f"inverse round-trip: E_true={E_true:.3e}  E_recovered={young_from_deflection(m, d, p.length, p.radius):.3e}")
    print(f"wrote {OUT/'calibration.png'} and probe/calibration.json")


if __name__ == "__main__":
    main()
