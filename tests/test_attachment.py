"""The stick is rigidly attached to the flange (directly, or glued in the gripper), and the IK drives the grip."""
import mujoco
import numpy as np
import pytest

from probe.control import TrajectoryFollower
from probe.scene import FLANGE_SITE, SceneConfig, load_scene
from probe.trajectory import PushParams, make_push


@pytest.mark.parametrize("exit_deg", [57.0, 80.0])  # at the home pose, steeper exits reach the table
def test_stick_glued_in_gripper_at_exit_angle_gravity_off_arm_static(exit_deg):
    m, d, h = load_scene(cfg=SceneConfig(exit_angle_deg=exit_deg), active="freespace")
    m.opt.gravity[:] = 0
    flange = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, FLANGE_SITE)
    base_body = h.chain[0]
    assert m.body_jntnum[base_body] == 0, "first stick body must have no joint (rigidly parented)"
    assert m.body_weldid[base_body] == m.body_weldid[m.site_bodyid[flange]], "stick base not welded to flange"
    q_arm = d.qpos[:6].copy()
    for _ in range(500):
        mujoco.mj_step(m, d)
    Rf = d.site_xmat[flange].reshape(3, 3)
    Rg = d.site_xmat[h.grip_site].reshape(3, 3)
    # stick axis at the measured exit angle from the gripper's approach (flange tool) axis, in a plane
    # perpendicular to the finger closing axis
    assert np.degrees(np.arccos(Rg[:, 0] @ Rf[:, 2])) == pytest.approx(exit_deg, abs=1e-6)
    assert abs(Rg[:, 1] @ Rf[:, 2]) < 1e-9
    # the grip sits ahead of the flange, between the fingertips (~16 cm out along the tool axis)
    off = d.site_xpos[h.grip_site] - d.site_xpos[flange]
    assert 0.14 < off @ Rf[:, 2] < 0.18
    # with gravity off and the arm holding still, the straight stick stays straight
    nodes = h.nodes_world(d, m) - d.site_xpos[h.grip_site]
    assert np.allclose(np.cross(nodes, Rg[:, 0]), 0, atol=1e-6)
    assert np.abs(d.qpos[:6] - q_arm).max() < 1e-3


def test_stick_base_pose_equals_flange_pose_without_gripper():
    m, d, h = load_scene(cfg=SceneConfig(gripper=False), active="freespace")
    m.opt.gravity[:] = 0
    flange = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, FLANGE_SITE)
    base_body = h.chain[0]
    assert m.body_jntnum[base_body] == 0, "first stick body must have no joint (rigidly parented)"
    assert m.body_weldid[base_body] == m.body_weldid[m.site_bodyid[flange]], "stick base not welded to flange"
    q0 = d.qpos.copy()
    for _ in range(500):
        mujoco.mj_step(m, d)
    Rf = d.site_xmat[flange].reshape(3, 3)
    Rg = d.site_xmat[h.grip_site].reshape(3, 3)
    assert np.allclose(d.site_xpos[h.grip_site], d.site_xpos[flange], atol=1e-9)
    assert np.allclose(Rg[:, 0], Rf[:, 2], atol=1e-9)  # stick axis = flange tool axis
    # stick base segment: origin l/2 behind the grip along the stick axis, same orientation
    rel = d.xpos[base_body] - d.site_xpos[flange]
    assert np.allclose(np.cross(rel, Rf[:, 2]), 0, atol=1e-9) and rel @ Rf[:, 2] < 0
    assert np.allclose(d.xmat[base_body].reshape(3, 3), Rg, atol=1e-9)
    # with gravity off and the arm holding still, the straight stick stays straight
    nodes = h.nodes_world(d, m) - d.site_xpos[h.grip_site]
    assert np.allclose(np.cross(nodes, Rg[:, 0]), 0, atol=1e-6)
    assert np.abs(d.qpos - q0).max() < 1e-3


def test_ik_targets_grip_not_tip():
    m, d, h = load_scene(active="freespace")
    traj = make_push(PushParams())
    fol = TrajectoryFollower(m, d, traj, h.grip_site)
    assert fol.ik.site == h.grip_site
    assert fol.place_at_start() < 1e-6
    assert np.allclose(d.site_xpos[h.grip_site], traj.pos[0], atol=1e-5)
    # the tip is NOT at grip + L*axis: it sags under gravity, i.e. it is free
    for _ in range(500):
        fol.command(d.time)
        mujoco.mj_step(m, d)
    straight = d.site_xpos[h.grip_site] + h.length * d.site_xmat[h.grip_site].reshape(3, 3)[:, 0]
    assert np.linalg.norm(h.tip_world(d, m) - straight) > 3e-4  # the short real needle sags ~0.8 mm at 35 deg
