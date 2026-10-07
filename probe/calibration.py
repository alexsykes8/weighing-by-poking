"""Cantilever calibration: sim tip deflection vs Euler-Bernoulli, and the E-from-measurement inverse.

Protocol (identical in sim and on the bench):
  1. Clamp the stick horizontally at the grip point.
  2. Record the unloaded tip height (the stick sags under its own weight; that's fine).
  3. Hang mass m from the tip, wait to settle, record the tip height again.
  4. delta = drop of the tip between 2 and 3 (vertical, metres).

In sim the hanging mass is a vertical dead load m*g applied exactly at the tip point (a mass on a
string is a vertical dead load at steady state).
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

from .stick import StickParams, build_stick_xml, find_stick

G = 9.81
CALIBRATION_JSON = Path(__file__).resolve().parent / "calibration.json"
SETTLE_DAMPING = 0.05  # s; damping does not change statics, only how fast we get there


def analytic_delta(mass, length, radius, young):
    I = np.pi * radius**4 / 4
    return mass * G * length**3 / (3 * young * I)


def sim_tip_drop(p: StickParams, tip_force: float, gravity: bool = True, timestep: float = 1e-3,
                 t_max: float = 20.0, vtol: float = 1e-7) -> float:
    """Vertical drop of the tip caused by a vertical tip load `tip_force` (N), relative to unloaded."""
    def settle(F):
        m = mujoco.MjModel.from_xml_string(build_stick_xml(
            p.length, p.radius, p.young, p.n_elem, SETTLE_DAMPING, density=p.density,
            poisson=p.poisson, timestep=timestep, gravity=gravity, layout=p.layout))
        d = mujoco.MjData(m)
        h = find_stick(m)
        ramp = 0.3
        while d.time < t_max:
            mujoco.mj_forward(m, d)
            d.qfrc_applied[:] = 0
            if F:
                Fz = -F * min(1.0, d.time / ramp)
                mujoco.mj_applyFT(m, d, np.array([0, 0, Fz]), np.zeros(3), h.tip_world(d, m),
                                  int(h.chain[-1]), d.qfrc_applied)
            mujoco.mj_step(m, d)
            if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
                raise FloatingPointError(f"cantilever unstable at E={p.young:.3g}, dt={timestep}")
            if d.time > ramp + 0.2 and np.abs(d.qvel).max() < vtol:
                break
        mujoco.mj_forward(m, d)
        return h.tip_world(d, m)[2]

    # without gravity the unloaded stick is straight, at the clamp height (1 m)
    z0 = settle(0.0) if gravity else 1.0
    return float(z0 - settle(tip_force))


def sim_delta(mass: float, p: StickParams, **kw) -> float:
    return sim_tip_drop(p, mass * G, **kw)


# ---- exact large-deflection reference (Kirchhoff elastica, weightless rod, dead tip load) ----

def elastica_curve(alphas) -> np.ndarray:
    """Tip drop / L of a weightless inextensible cantilever under a vertical dead tip load.

    alpha = F L^2 / EI. With theta(s) the slope angle: theta'' = -alpha cos(theta),
    theta(0) = 0 (clamp), theta'(1) = 0 (free end), z' = sin(theta). Solved by continuation in
    alpha (each solve warm-started from the previous one) so the BVP stays on the physical branch.
    """
    s = np.linspace(0, 1, 200)
    y = np.zeros((3, s.size))
    out = []
    for a in np.sort(np.asarray(alphas, float)):
        sol = solve_bvp(lambda s, y: np.vstack([y[1], -a * np.cos(y[0]), np.sin(y[0])]),
                        lambda ya, yb: np.array([ya[0], yb[1], ya[2]]), s, y, tol=1e-8, max_nodes=100000)
        if not sol.success:
            raise RuntimeError(f"elastica BVP failed at alpha={a}: {sol.message}")
        y = sol.sol(s)
        out.append(float(y[2, -1]))
    order = np.argsort(np.argsort(np.asarray(alphas, float)))
    return np.asarray(out)[order]


def elastica_delta_over_L(alpha: float) -> float:
    a = np.geomspace(min(alpha, 1e-2), alpha, 40)
    return float(elastica_curve(a)[-1])


# ---- calibration table and inverse ----------------------------------------------------------

def build_master_curve(alphas, p: StickParams | None = None) -> dict:
    """Dimensionless sim curve delta/L = f(alpha), weightless rod (no gravity, tip load only).

    For a weightless rod this curve is universal: it does not depend on E, L or r separately.
    """
    p = p or StickParams()
    rows = []
    for a in alphas:
        F = a * p.EI / p.length**2
        rows.append(sim_tip_drop(p, F, gravity=False) / p.length)
    return {"alpha": list(map(float, alphas)), "delta_over_L": rows, "n_elem": p.n_elem}


def save_calibration(table: dict, path=CALIBRATION_JSON) -> Path:
    Path(path).write_text(json.dumps(table, indent=2))
    return Path(path)


def load_calibration(path=CALIBRATION_JSON) -> dict:
    return json.loads(Path(path).read_text())


def young_from_deflection(mass: float, delta: float, length: float, radius: float,
                          density: float | None = None, table: dict | None = None, **stick_kw) -> float:
    """Young's modulus (Pa) from a measured cantilever deflection, inverting the *sim* curve.

    mass: hung at the tip (kg). delta: tip drop caused by hanging it (m). length: grip->tip (m).
    density=None: use the stored dimensionless sim master curve (weightless rod; fine when the
    stick weighs much less than `mass`). density given (kg/m^3): root-find E by re-running the
    sim with self-weight included, i.e. the exact bench protocol.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    table = table or load_calibration()
    a = np.asarray(table["master"]["alpha"])
    r = np.asarray(table["master"]["delta_over_L"])
    y = delta / length
    if not (r[0] <= y <= r[-1]):
        raise ValueError(f"delta/L={y:.4f} outside calibrated range [{r[0]:.4f}, {r[-1]:.4f}]")
    alpha = float(np.exp(np.interp(np.log(y), np.log(r), np.log(a))))  # monotone in log-log
    I = np.pi * radius**4 / 4
    E0 = mass * G * length**2 / (alpha * I)
    if density is None:
        return E0

    def resid(logE):
        p = StickParams(length=length, radius=radius, young=float(np.exp(logE)), density=density, **stick_kw)
        return np.log(sim_delta(mass, p)) - np.log(delta)

    lo, hi = np.log(E0) - 1.0, np.log(E0) + 1.0
    while resid(lo) < 0:
        lo -= 1.0
    while resid(hi) > 0:
        hi += 1.0
    return float(np.exp(brentq(resid, lo, hi, xtol=1e-4)))
