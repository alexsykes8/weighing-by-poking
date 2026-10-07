"""End-to-end smoke test: synthetic trajectory -> UR5e rollout -> valid observation on disk."""
import numpy as np

from probe import features
from probe.io import load_observation, save_observation
from probe.rollout import run_rollout
from probe.trajectory import PushParams, make_push


def test_fixed_push_bows_the_stick_and_writes_valid_observation(tmp_path):
    obs = run_rollout(make_push(PushParams()), "fixed", record_hz=50)
    obs2 = load_observation(save_observation(tmp_path / "fixed.npz", obs))
    assert obs2.meta["source"] == "sim"
    # the stick visibly bows on contact: > 2 cm tip deflection, real contact force
    assert -obs2.tip[:, 2].min() > 0.02
    assert np.linalg.norm(obs2.extras["f_contact_obj"], axis=1).max() > 1.0
    # the grip follows the commanded trajectory closely at hand-like loads (the tip is not commanded at all).
    # This push drives the tip into the box-table corner, where the pinned stick can carry ~200+ N
    # (clamped-pinned buckling, 2.05 pi^2 EI / L^2); under that the position servos yield, as a hand would.
    err = np.linalg.norm(obs2.grip_pose[:, :3] - obs2.extras["grip_cmd"][:, :3], axis=1)
    load = np.linalg.norm(obs2.extras["f_contact_obj"], axis=1) + np.linalg.norm(obs2.extras["f_contact_env"], axis=1)
    # the load chatters (stick-slip in the corner) and the arm's deflection lags it: mask on the recent peak load
    from scipy.ndimage import maximum_filter1d
    recent = maximum_filter1d(load, size=int(0.5 * 50), origin=int(0.5 * 50) // 2 - 1)
    assert err[recent < 20].max() < 5e-3
    assert err.max() < 0.03
    assert np.all(np.isfinite(features.extract(obs2)))
