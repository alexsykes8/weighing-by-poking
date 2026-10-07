"""Sim->real evaluation path: transfer metrics, honest intervals, and real-file discovery."""
import dataclasses
import warnings

import numpy as np
import pytest

from probe import features
from probe.classifier import evaluate_transfer, exact_ci, real_paths
from probe.io import save_observation
from probe.schema import CLASSES as LABELS
from test_contract import toy_obs

F = len(features.FEATURES)


def toy_features(n_per, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, c in enumerate(LABELS):
        mu = np.zeros(F)
        mu[0] = 3.0 * i  # classes separable along peak_bend
        X.append(rng.normal(mu, 0.5, (n_per, F)))
        y += [c] * n_per
    X = np.vstack(X)
    X[:, 0] += shift
    return X, np.array(y)


def test_transfer_without_shift_has_small_gap():
    Xs, ys = toy_features(30)
    Xr, yr = toy_features(5, seed=1)
    r = evaluate_transfer(Xs, ys, Xr, yr)
    assert r["real_acc"] > 0.9 and abs(r["gap"]) < 0.1
    lo, hi = r["real_ci95"]
    assert lo <= r["real_acc"] <= hi
    assert len(r["predictions"]) == 20 and set(r["per_class"]) == set(LABELS)


def test_transfer_with_shift_shows_gap_and_names_the_feature():
    Xs, ys = toy_features(30)
    Xr, yr = toy_features(5, shift=3.0, seed=1)  # e.g. real stick stiffer than sim: every bend reads one class up
    r = evaluate_transfer(Xs, ys, Xr, yr)
    assert r["gap"] > 0.4
    z = {k: v["z_shift"] for k, v in r["feature_shift"].items()}
    assert max(z, key=lambda k: abs(z[k])) == features.FEATURES[0]


def test_missing_classes_are_reported_not_hidden():
    Xs, ys = toy_features(30)
    Xr, yr = toy_features(5, seed=1)
    keep = yr != "light"
    with pytest.warns(UserWarning, match="light"):
        r = evaluate_transfer(Xs, ys, Xr[keep], yr[keep])
    assert r["classes_untested"] == ["light"]


def test_exact_ci():
    assert exact_ci(10, 10)[1] == 1.0 and exact_ci(10, 10)[0] == pytest.approx(0.6915, abs=1e-3)
    assert exact_ci(0, 4)[0] == 0.0
    lo, hi = exact_ci(3, 4)
    assert lo < 0.75 < hi and hi - lo > 0.5  # 4 clips can't pin accuracy down; the report must say so


def test_real_paths_keeps_only_video(tmp_path):
    sim = toy_obs()
    vid = dataclasses.replace(toy_obs(), meta={**sim.meta, "source": "video"})
    save_observation(tmp_path / "a_sim.npz", sim)
    save_observation(tmp_path / "sub" / "b_video.npz", vid)
    (tmp_path / "c_broken.npz").write_bytes(b"not an npz")
    keep, skipped = real_paths(tmp_path)
    assert [p.name for p in keep] == ["b_video.npz"]
    assert any("a_sim" in s for s in skipped) and any("c_broken" in s for s in skipped)
    assert real_paths(tmp_path / "nope") == ([], [f"{tmp_path / 'nope'} does not exist"])
