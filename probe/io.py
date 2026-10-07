"""Load/save for the data contract. Everything here validates on the way in and out."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .schema import TRAJ_COLUMNS, ContractError, GripTrajectory, ProbeObservation

REPO = Path(__file__).resolve().parent.parent


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL)
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def validate(obj):
    """Hard-fail on any contract violation. Returns the object for chaining."""
    if not isinstance(obj, (GripTrajectory, ProbeObservation)):
        raise ContractError(f"cannot validate {type(obj).__name__}")
    return obj.validate()


# ---- grip trajectory CSV --------------------------------------------------------------------

def save_trajectory(path, traj: GripTrajectory) -> Path:
    traj.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack([traj.t, traj.pos, traj.quat])
    np.savetxt(path, arr, delimiter=",", header=",".join(TRAJ_COLUMNS), comments="", fmt="%.6f")
    return path


def load_trajectory(path) -> GripTrajectory:
    path = Path(path)
    with open(path) as f:
        header = f.readline().strip().split(",")
    if tuple(h.strip() for h in header) != TRAJ_COLUMNS:
        raise ContractError(f"{path}: header {header} != {list(TRAJ_COLUMNS)}")
    arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return GripTrajectory(arr[:, 0], arr[:, 1:4], arr[:, 4:8]).validate()


# ---- probe observation NPZ ------------------------------------------------------------------

def save_observation(path, obs: ProbeObservation) -> Path:
    obs.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, t=obs.t, centreline=obs.centreline, tip=obs.tip, curvature=obs.curvature,
        arclen=obs.arclen, grip_pose=obs.grip_pose, label=np.array(obs.label),
        meta=np.array(json.dumps(obs.meta, default=float)),
        **{f"extra_{k}": v for k, v in obs.extras.items()})
    return path


def load_observation(path) -> ProbeObservation:
    with np.load(path, allow_pickle=False) as z:
        obs = ProbeObservation(
            t=z["t"], centreline=z["centreline"], tip=z["tip"], curvature=z["curvature"],
            arclen=z["arclen"], grip_pose=z["grip_pose"], label=str(z["label"]),
            meta=json.loads(str(z["meta"])),
            extras={k[6:]: z[k] for k in z.files if k.startswith("extra_")})
    return obs.validate()
