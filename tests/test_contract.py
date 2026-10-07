"""The data contract hard-fails on shape and frame violations, and round-trips through disk."""
import dataclasses

import numpy as np
import pytest

from probe.io import load_observation, load_trajectory, save_observation, save_trajectory
from probe.schema import ContractError, GripTrajectory, ProbeObservation
from probe.trajectory import PushParams, make_push


def toy_obs(T=5, K=11, L=0.6):
    s = np.linspace(0, L, K)
    cl = np.zeros((T, K, 3))
    cl[:, :, 0] = s
    cl[:, :, 2] = -0.01 * (s / L) ** 2
    pose = np.tile([0, 0, 1, 1, 0, 0, 0], (T, 1)).astype(float)
    meta = dict(young=2e10, radius=0.004, length=L, timestep=1e-3, source="sim", git_sha="x")
    return ProbeObservation(np.arange(T) * 0.01, cl, cl[:, -1].copy(), np.zeros((T, K)), s, pose, "fixed", meta)


def test_observation_round_trip(tmp_path):
    o = toy_obs()
    o.extras["mount_force"] = np.ones((5, 3))
    o2 = load_observation(save_observation(tmp_path / "o.npz", o))
    assert np.allclose(o2.centreline, o.centreline) and o2.label == "fixed" and o2.meta == o.meta
    assert np.allclose(o2.extras["mount_force"], 1)


def test_trajectory_round_trip(tmp_path):
    tr = make_push(PushParams())
    tr2 = load_trajectory(save_trajectory(tmp_path / "t.csv", tr))
    assert np.allclose(tr2.pos, tr.pos, atol=1e-6) and np.allclose(tr2.t, tr.t, atol=1e-6)
    assert open(tmp_path / "t.csv").readline().strip() == "t,x,y,z,qw,qx,qy,qz"


@pytest.mark.parametrize("mutate", [
    lambda o: dataclasses.replace(o, centreline=o.centreline[:, :-1]),             # K mismatch
    lambda o: dataclasses.replace(o, tip=o.tip + 0.01),                             # tip != centreline[-1]
    lambda o: dataclasses.replace(o, centreline=o.centreline + [0.1, 0, 0]),        # not grip-centred
    lambda o: dataclasses.replace(o, centreline=o.centreline[..., [2, 1, 0]],       # stick not along +x
                                  tip=o.centreline[:, -1, [2, 1, 0]]),
    lambda o: dataclasses.replace(o, centreline=o.centreline * 1.2, tip=o.tip * 1.2),  # stretched / wrong units
    lambda o: dataclasses.replace(o, label="medium"),
    lambda o: dataclasses.replace(o, meta={k: v for k, v in o.meta.items() if k != "young"}),
    lambda o: dataclasses.replace(o, t=o.t[::-1].copy()),
    lambda o: dataclasses.replace(o, grip_pose=o.grip_pose * 2),                    # non-unit quaternion
])
def test_observation_violations_fail_loudly(mutate):
    with pytest.raises(ContractError):
        mutate(toy_obs()).validate()


def test_trajectory_violations_fail_loudly(tmp_path):
    with pytest.raises(ContractError):
        GripTrajectory(np.array([0, 1.0]), np.zeros((2, 3)), np.zeros((2, 4))).validate()
    (tmp_path / "bad.csv").write_text("t,x,y,z,qx,qy,qz,qw\n0,0,0,0,0,0,0,1\n1,0,0,0,0,0,0,1\n")
    with pytest.raises(ContractError):
        load_trajectory(tmp_path / "bad.csv")


def test_noisy_tracked_centreline_is_accepted():
    """Real video has ~1-2 mm of tracking noise; the frame checks must tolerate it (regression)."""
    o = toy_obs(T=50, K=41)
    rng = np.random.default_rng(0)
    cl = o.centreline + rng.normal(0, 2e-3, o.centreline.shape)
    cl[:, 0] = 0
    dataclasses.replace(o, centreline=cl, tip=cl[:, -1].copy()).validate()
