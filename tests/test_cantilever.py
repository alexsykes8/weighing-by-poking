"""Acceptance: the stick model reproduces Euler-Bernoulli in the small-deflection limit.

If these fail the stick model is wrong and nothing downstream is worth running.
"""
import numpy as np
import pytest

from probe.calibration import analytic_delta, elastica_delta_over_L, sim_delta, sim_tip_drop, young_from_deflection
from probe.stick import StickParams

# model checks run on a fixed reference beam (the original 600 mm x 8 mm rod), so they do not drift when the
# default stick is re-measured; the real needle gets its own linear-regime case below
REF = StickParams(length=0.6, radius=0.004, density=1900.0)

LINEAR = [  # (E Pa, mass kg) on REF: total sag (self-weight + load) / L < 0.05
    (2e10, 0.01),
    (2e10, 0.05),
    (1e11, 0.2),
    (5e9, 0.01),
]


@pytest.mark.parametrize("E,mass", LINEAR)
def test_tip_deflection_matches_euler_bernoulli_within_5pct(E, mass):
    p = REF.with_(young=E)
    sim = sim_delta(mass, p)
    eb = analytic_delta(mass, p.length, p.radius, E)
    assert eb / p.length < 0.05, "test case is not in the linear regime"
    assert sim == pytest.approx(eb, rel=0.05), f"sim {sim*1e3:.3f} mm vs EB {eb*1e3:.3f} mm"


def test_real_needle_matches_euler_bernoulli():
    p = StickParams()  # measured needle: EI = 0.31 N m^2, 334 mm
    sim = sim_delta(0.01, p)
    eb = analytic_delta(0.01, p.length, p.radius, p.young)
    assert eb / p.length < 0.05, "test case is not in the linear regime"
    assert sim == pytest.approx(eb, rel=0.05), f"sim {sim*1e3:.3f} mm vs EB {eb*1e3:.3f} mm"


def test_linear_agreement_is_tight_not_just_within_tolerance():
    # midpoint discretisation predicts 1 - 1/(4N^2); anything >1% off means a regression
    p = StickParams(young=2e10)
    ratio = sim_delta(0.02, p) / analytic_delta(0.02, p.length, p.radius, p.young)
    assert abs(ratio - 1) < 0.01, ratio


def test_large_deflection_matches_exact_elastica():
    p = StickParams()
    alpha = 3.0  # delta/L ~ 0.6, far outside linear theory
    sim = sim_tip_drop(p, alpha * p.EI / p.length**2, gravity=False) / p.length
    assert sim == pytest.approx(elastica_delta_over_L(alpha), rel=0.01)
    assert sim < 0.7 * alpha / 3  # and genuinely nonlinear


def test_young_from_deflection_round_trip():
    p = REF.with_(young=8e9)
    for m in (0.01, 0.2):  # linear and nonlinear
        d = sim_delta(m, p, gravity=False)
        assert young_from_deflection(m, d, p.length, p.radius) == pytest.approx(p.young, rel=0.01)


def test_young_from_deflection_with_self_weight():
    p = StickParams(young=3e9)  # soft: self-weight sag ~4% of L, so it matters
    d = sim_delta(0.05, p)
    assert young_from_deflection(0.05, d, p.length, p.radius, density=p.density) == pytest.approx(3e9, rel=0.01)


def test_stiffness_is_a_single_exposed_parameter():
    soft, stiff = StickParams(young=1e10), StickParams(young=4e10)
    ratio = sim_delta(0.02, soft) / sim_delta(0.02, stiff)
    assert ratio == pytest.approx(4.0, rel=0.02)
