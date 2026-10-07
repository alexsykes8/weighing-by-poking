"""Seam for a learned residual on top of the physics prediction. NOT IMPLEMENTED (by design, for now).

    centreline_pred = sim(grip trajectory)                      # analytic/physics world model
    centreline_hat  = centreline_pred + R(grip, centreline_pred) # learned correction
    trained on (video centreline - sim centreline) pairs, all in the stick-local frame.

Inputs and outputs are exactly the data-contract arrays, so a model can be dropped in without
touching the rollout or comparison code.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class ResidualModel(Protocol):
    def fit(self, grip_pose: list[np.ndarray], centreline_pred: list[np.ndarray],
            centreline_obs: list[np.ndarray]) -> "ResidualModel":
        """grip_pose[i]: (T,7) world; centreline_*[i]: (T,K,3) stick-local. One entry per clip."""
        ...

    def predict(self, grip_pose: np.ndarray, centreline_pred: np.ndarray) -> np.ndarray:
        """Return the correction (T,K,3), stick-local, to ADD to centreline_pred."""
        ...


class ZeroResidual:
    """Identity baseline: predicts no correction. Any learned model must beat this on tip error."""

    def fit(self, grip_pose, centreline_pred, centreline_obs):
        return self

    def predict(self, grip_pose, centreline_pred):
        return np.zeros_like(centreline_pred)
