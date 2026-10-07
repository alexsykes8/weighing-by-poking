"""Features from the curvature signature of one push. Uses only what video also gives:
centreline (stick-local) and the grip trajectory. No sim-only force channels.

The core signal is the *grip bending* b(t): mean curvature over the first 30% of the stick,
minus its value at rest. Near the grip the bending moment is ~ F * lever arm, so b(t) is a
proxy for the contact load (M = EI kappa).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .geometry import curvature_inplane, quat_wxyz_to_mat
from .schema import ProbeObservation

FEATURES = ("peak_bend", "load_slope", "plateau_ratio", "decay_rate", "residual_bend", "bend_per_advance",
            "bend_impulse", "peak_tip_drop", "hold_tip_drift")
BASE_FRAC = 0.3
REST_TIME = 0.4  # s of rest at the start used as the zero-load baseline


def add_tracking_noise(obs: ProbeObservation, sigma: float, rng: np.random.Generator, window: float = 0.15):
    """Simulate a video tracker: iid Gaussian noise (m) on every centreline point, curvature recomputed."""
    cl = obs.centreline + rng.normal(0, sigma, obs.centreline.shape)
    cl[:, 0] = 0.0  # the grip is the frame origin by definition
    kappa = np.array([curvature_inplane(c, obs.arclen, window) for c in cl])
    return cl, kappa


def grip_bend(kappa: np.ndarray, arclen: np.ndarray, t: np.ndarray) -> np.ndarray:
    sel = arclen <= BASE_FRAC * arclen[-1]
    b = kappa[:, sel].mean(axis=1)
    b = uniform_filter1d(b, size=max(1, int(round(0.05 / np.median(np.diff(t))))))  # 50 ms smoothing
    return -(b - np.median(b[t < t[0] + REST_TIME]))  # pushing down-forward bends toward -z: make load positive


def extract(obs: ProbeObservation, centreline=None, curvature=None) -> np.ndarray:
    cl = obs.centreline if centreline is None else centreline
    kappa = obs.curvature if curvature is None else curvature
    t = obs.t
    dt = np.median(np.diff(t))
    b = grip_bend(kappa, obs.arclen, t)

    # grip advance along its own (initial) stick direction projected on the horizontal push axis
    g = obs.grip_pose[:, :3]
    fwd = _push_axis(obs)
    adv = (g - g[0]) @ fwd
    vel = np.gradient(adv, t)
    i_pk = int(np.argmax(b))
    pk = max(b[i_pk], 1e-9)

    # phases from the grip motion: deepest point -> hold until the grip starts to retract
    i_deep = int(np.argmax(adv))
    after = np.where(vel[i_deep:] < -0.25 * np.abs(vel).max())[0]
    i_ret = i_deep + (after[0] if after.size else len(t) - 1 - i_deep)
    hold = slice(i_deep, max(i_ret, i_deep + 2))

    db = np.gradient(b, t)
    tip_drop = -(cl[:, -1, 2] - np.median(cl[t < t[0] + REST_TIME, -1, 2]))
    hb = b[hold]
    decay = -np.polyfit(t[hold] - t[hold][0], hb / pk, 1)[0] if hb.size > 3 else 0.0
    return np.array([
        pk,                                             # peak_bend (1/m)
        db[(t > t[0] + REST_TIME) & (np.arange(len(t)) <= i_pk)].max(initial=0.0),  # load_slope (1/m/s)
        b[hold].mean() / pk,                            # plateau_ratio: ~1 fixed, <1 free (relaxes)
        decay,                                          # decay_rate during the hold (1/s, normalised)
        b[-int(0.3 / dt):].mean(),                      # residual_bend after retract
        pk / max(adv.max(), 1e-3),                      # bend_per_advance: stiffness seen by the probe
        np.trapezoid(np.clip(b, 0, None), t),           # bend_impulse
        tip_drop.max(),                                 # peak_tip_drop (m)
        (tip_drop[hold][-1] - tip_drop[hold][0]),       # hold_tip_drift (m)
    ])


def _push_axis(obs: ProbeObservation) -> np.ndarray:
    """Horizontal direction the stick points at the start (from the grip orientation)."""
    x = quat_wxyz_to_mat(obs.grip_pose[0, 3:])[:, 0]
    x[2] = 0
    return x / np.linalg.norm(x)
