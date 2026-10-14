"""The data contract shared by the sim pipeline and the video pipeline.

See docs/data_contract.md. Everything downstream of a rollout or a tracker consumes
`GripTrajectory` (input) and `ProbeObservation` (output) and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# "free": a movable box whose mass is meta["object_mass"] (the mass-estimation experiment).
# "heavy"/"light": the earlier 3-class protocol (CLASSES).
LABELS = ("fixed", "free", "heavy", "light", "freespace")
CLASSES = ("fixed", "heavy", "light", "freespace")
SOURCES = ("sim", "video")
TRAJ_COLUMNS = ("t", "x", "y", "z", "qw", "qx", "qy", "qz")
META_REQUIRED = ("young", "radius", "length", "timestep", "source", "git_sha")


class ContractError(ValueError):
    """Raised when data violates the contract. Never caught inside this package."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def _check_time(t: np.ndarray, name: str) -> None:
    _check(t.ndim == 1 and t.size >= 2, f"{name}: t must be 1-D with >=2 samples, got {t.shape}")
    _check(np.all(np.isfinite(t)), f"{name}: t has non-finite values")
    _check(np.all(np.diff(t) > 0), f"{name}: t must be strictly increasing")


def _check_quats(q: np.ndarray, name: str, tol: float = 1e-3) -> None:
    n = np.linalg.norm(q, axis=-1)
    _check(np.all(np.abs(n - 1) < tol), f"{name}: quaternions not unit (max |1-|q||={np.abs(n-1).max():.2e})")


@dataclass
class GripTrajectory:
    """Grip pose over time, world frame. Columns t,x,y,z,qw,qx,qy,qz (quaternion w-first)."""
    t: np.ndarray        # (T,)
    pos: np.ndarray      # (T,3) m
    quat: np.ndarray     # (T,4) wxyz, unit

    def validate(self) -> "GripTrajectory":
        self.t = np.asarray(self.t, float)
        self.pos = np.asarray(self.pos, float)
        self.quat = np.asarray(self.quat, float)
        _check_time(self.t, "GripTrajectory")
        T = self.t.size
        _check(self.pos.shape == (T, 3), f"GripTrajectory: pos shape {self.pos.shape} != ({T},3)")
        _check(self.quat.shape == (T, 4), f"GripTrajectory: quat shape {self.quat.shape} != ({T},4)")
        _check(np.all(np.isfinite(self.pos)), "GripTrajectory: non-finite positions")
        _check_quats(self.quat, "GripTrajectory")
        return self

    @property
    def pose7(self) -> np.ndarray:
        return np.hstack([self.pos, self.quat])


@dataclass
class ProbeObservation:
    """Stick shape over time in the stick-local frame.

    Stick-local frame: origin at the grip, +x along the undeformed stick axis at the grip,
    +z opposite gravity (the component of world-up orthogonal to +x), +y = z cross x.
    """
    t: np.ndarray            # (T,)
    centreline: np.ndarray   # (T,K,3) m, base -> tip, stick-local
    tip: np.ndarray          # (T,3) m, stick-local
    curvature: np.ndarray    # (T,K) 1/m, signed in-plane (x-z) curvature
    arclen: np.ndarray       # (K,) m, along the undeformed stick
    grip_pose: np.ndarray    # (T,7) world frame, x y z qw qx qy qz
    label: str
    meta: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)  # optional extra (T,...) arrays, e.g. sim wrench

    def validate(self, frame_tol: float = 2e-3) -> "ProbeObservation":
        for k in ("t", "centreline", "tip", "curvature", "arclen", "grip_pose"):
            setattr(self, k, np.asarray(getattr(self, k), float))
        _check_time(self.t, "ProbeObservation")
        T = self.t.size
        _check(self.arclen.ndim == 1 and self.arclen.size >= 3, f"arclen must be (K>=3,), got {self.arclen.shape}")
        K = self.arclen.size
        _check(np.isclose(self.arclen[0], 0.0, atol=1e-9), "arclen[0] must be 0 (grip)")
        _check(np.all(np.diff(self.arclen) > 0), "arclen must be strictly increasing")
        _check(self.centreline.shape == (T, K, 3), f"centreline shape {self.centreline.shape} != ({T},{K},3)")
        _check(self.tip.shape == (T, 3), f"tip shape {self.tip.shape} != ({T},3)")
        _check(self.curvature.shape == (T, K), f"curvature shape {self.curvature.shape} != ({T},{K})")
        _check(self.grip_pose.shape == (T, 7), f"grip_pose shape {self.grip_pose.shape} != ({T},7)")
        for k in ("centreline", "tip", "curvature", "grip_pose"):
            _check(np.all(np.isfinite(getattr(self, k))), f"{k} has non-finite values")
        _check_quats(self.grip_pose[:, 3:], "grip_pose")
        _check(self.label in LABELS, f"label {self.label!r} not in {LABELS}")
        for k in META_REQUIRED:
            _check(k in self.meta, f"meta missing required key {k!r}")
        _check(self.meta["source"] in SOURCES, f"meta.source {self.meta['source']!r} not in {SOURCES}")
        for k, v in self.extras.items():
            _check(np.shape(v)[0] == T, f"extras[{k!r}] first dim {np.shape(v)[0]} != T={T}")

        # frame checks: these are what make sim and video comparable
        _check(np.allclose(self.tip, self.centreline[:, -1], atol=1e-9), "tip must equal centreline[:, -1]")
        _check(np.all(np.linalg.norm(self.centreline[:, 0], axis=1) < frame_tol),
               "centreline[:,0] must be at the origin (stick-local frame is grip-centred)")
        # the stick leaves the grip along +x. Use the chord to the point at ~10% of the length (>= 2 samples),
        # not the first segment: with ~1-2 mm tracker noise on 15 mm segments the first segment's direction
        # wobbles by up to ~20 deg, while a wrong frame gives cos ~ 0 or < 0 regardless.
        j = max(2, int(np.searchsorted(self.arclen, 0.1 * self.arclen[-1])))
        d1 = self.centreline[:, j] - self.centreline[:, 0]
        cos = d1[:, 0] / np.maximum(np.linalg.norm(d1, axis=1), 1e-12)
        # threshold 0.7 (45 deg): a wrong frame gives cos ~ 0 or < 0, while a soft stick jammed against a box can
        # really bend ~30 deg within the first 6 cm (seen in randomised sim with E = 5 GPa, r = 3 mm)
        _check(np.all(cos > 0.7), f"stick does not leave the grip along +x (min cos={cos.min():.3f} over "
                                   f"the first {self.arclen[j]*100:.0f} cm)")
        # inextensibility sanity: polyline length within 5% of arclen span. Measured on ~10 coarse segments:
        # iid tracker noise sigma inflates a polyline by ~2 sigma^2/ds^2, i.e. 3.6% at 2 mm on 15 mm segments
        # but ~0.2% on 6 cm ones, so this still catches unit/scale errors without rejecting noisy video.
        idx = np.unique(np.linspace(0, K - 1, min(K, 11)).round().astype(int))
        seg = np.linalg.norm(np.diff(self.centreline[:, idx], axis=1), axis=2).sum(axis=1)
        rel = np.abs(seg / self.arclen[-1] - 1)
        _check(np.all(rel < 0.05), f"centreline length deviates from arclen by up to {rel.max():.1%}")
        return self
