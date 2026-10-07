"""Shape extraction: stick-local frame, resampling, and curvature against beam theory."""
import mujoco
import numpy as np
import pytest

from probe.geometry import curvature_inplane, stick_local_frame
from probe.rollout import extract_frame
from probe.stick import StickParams, build_stick_xml, find_stick


def loaded_cantilever(p, F):
    m = mujoco.MjModel.from_xml_string(build_stick_xml(young=p.young, damping=0.05, gravity=False))
    d = mujoco.MjData(m)
    h = find_stick(m)
    for _ in range(4000):
        mujoco.mj_forward(m, d)
        d.qfrc_applied[:] = 0
        mujoco.mj_applyFT(m, d, np.array([0, 0, -F * min(1, d.time / 0.3)]), np.zeros(3), h.tip_world(d, m),
                          int(h.chain[-1]), d.qfrc_applied)
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    return m, d, h


def test_curvature_matches_beam_theory():
    p = StickParams()
    F = 0.09 * p.EI / p.length**2  # delta/L ~ 0.03
    m, d, h = loaded_cantilever(p, F)
    s = np.linspace(0, h.length, 41)
    cl, k, grip, _ = extract_frame(m, d, h, s)
    analytic = -F * (h.length - s) / p.EI  # M(s) = EI kappa(s); bending toward -z is negative
    assert np.sqrt(np.mean((k - analytic) ** 2)) < 0.03 * np.abs(analytic).max()
    assert k[0] == pytest.approx(analytic[0], rel=0.06)  # the clamp end is not forced to zero
    assert np.allclose(cl[0], 0, atol=1e-9)
    assert cl[-1, 2] < 0  # drooping toward -z in the stick-local frame


def test_curvature_of_circle():
    R = 0.8
    s = np.linspace(0, 0.6, 41)
    pts = np.column_stack([R * np.sin(s / R), np.zeros_like(s), R * (1 - np.cos(s / R))])
    assert np.allclose(curvature_inplane(pts, s), 1 / R, rtol=1e-3)


def test_stick_local_frame_is_gravity_aligned():
    a = np.radians(35)
    R = np.column_stack([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    roll = np.array([[1, 0, 0], [0, np.cos(0.4), -np.sin(0.4)], [0, np.sin(0.4), np.cos(0.4)]])
    F = stick_local_frame(R @ roll)  # rolling the grip about the stick axis must not change the frame
    assert np.allclose(F[:, 0], R[:, 0])
    assert F[:, 2] @ np.array([0, 0, 1]) > 0
    assert np.allclose(F, stick_local_frame(R))
