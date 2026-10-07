"""Cantilever tip-deflection error vs segment count, uncorrected (uniform) vs midpoint cable layout.

    python scripts/discretisation_study.py   # -> out/discretisation.png, out/discretisation.json

Weightless rod, small tip load (alpha = FL^2/EI = 0.06, delta/L ~ 0.02). The reference is the exact
elastica at the same alpha: even here it sits 4.1e-4 below Euler-Bernoulli (geometric nonlinearity),
which would otherwise show up as a fake error floor. Each layout is also checked against its closed-form
discrete-chain prediction (joint stiffness EI/l lumped at each joint):
  uniform : (N-1)N(2N-1)/(2N^3)  ~ 1 - 3/(2N)      first-order
  midpoint: 1 - 1/(4N^2)                            second-order
"""
import json

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probe.calibration import elastica_delta_over_L, sim_tip_drop
from probe.stick import StickParams

OUT = _bootstrap.REPO / "out"
NS = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
ALPHA = 0.06
THEORY = {"uniform": lambda N: (N - 1) * N * (2 * N - 1) / (2 * N**3),
          "midpoint": lambda N: 1 - 1 / (4 * N**2)}
INK2, GRID, SURF = "#5f5e58", "#e4e3dc", "#fcfcfb"
COL = {"uniform": "#eb6834", "midpoint": "#2a78d6"}


def main():
    exact = elastica_delta_over_L(ALPHA)  # delta/L, continuum reference
    rows = []
    for layout in ("uniform", "midpoint"):
        for N in NS:
            p = StickParams(n_elem=N, layout=layout)
            F = ALPHA * p.EI / p.length**2
            dl = sim_tip_drop(p, F, gravity=False) / p.length
            ratio = dl / exact
            rows.append({"layout": layout, "N": N, "sim_over_elastica": ratio, "sim_over_eb": dl / (ALPHA / 3),
                         "chain_theory": THEORY[layout](N)})
            print(f"{layout:8s} N={N:3d}  sim/elastica={ratio:.6f}  chain theory={THEORY[layout](N):.6f}  "
                  f"|err|={abs(ratio-1):.2e}  (sim/EB={dl/(ALPHA/3):.6f})", flush=True)

    # empirical convergence order from a log-log fit of |error| vs N
    orders = {}
    for layout in ("uniform", "midpoint"):
        r = [x for x in rows if x["layout"] == layout]
        N = np.array([x["N"] for x in r])
        e = np.abs(np.array([x["sim_over_elastica"] for x in r]) - 1)
        ok = e > 1e-5  # below this the settle tolerance dominates
        orders[layout] = float(-np.polyfit(np.log(N[ok]), np.log(e[ok]), 1)[0])
        print(f"{layout}: empirical order {orders[layout]:.2f} (fit over N={N[ok].tolist()})")

    plt.rcParams.update({"font.size": 9, "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": INK2})
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    Nf = np.geomspace(NS[0], NS[-1], 100)
    for layout in ("uniform", "midpoint"):
        r = [x for x in rows if x["layout"] == layout]
        ax.plot(Nf, np.abs(THEORY[layout](Nf) - 1), color=COL[layout], lw=1, ls="--")
        ax.plot([x["N"] for x in r], [abs(x["sim_over_elastica"] - 1) for x in r], "o", color=COL[layout], ms=6, mec=SURF,
                mew=1.5, label=f"{layout} (order {orders[layout]:.2f})")
    ax.axhline(0.05, color=INK2, lw=0.8)
    ax.text(NS[-1], 0.055, "5% acceptance", ha="right", va="bottom", color=INK2, fontsize=8)
    ax.plot([], [], color=INK2, ls="--", lw=1, label="discrete-chain closed form")
    ax.set(xscale="log", yscale="log", xlabel="segments N", ylabel="|sim / exact elastica − 1|",
           title="Cantilever tip-deflection error vs segment count")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "discretisation.png", dpi=150)
    (OUT / "discretisation.json").write_text(json.dumps({"alpha": ALPHA, "elastica_delta_over_L": exact, "rows": rows, "orders": orders}, indent=2))
    print(f"wrote {OUT/'discretisation.png'}")


if __name__ == "__main__":
    main()
