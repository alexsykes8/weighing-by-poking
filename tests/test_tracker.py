"""Tracker accuracy on rendered sim frames vs ground truth (full 1920x1440, as Stray records)."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

from probe.camera import CameraConfig
from probe.control import TrajectoryFollower
from probe.geometry import curvature_inplane, stick_local_frame
from probe.rollout import extract_frame
from probe.scene import SceneConfig, load_scene
from probe.stick import StickParams, colour_bands
from probe.tracker import (Plane, StickMarkings, StickTracker, TrackerConfig, backproject_to_plane, fix_to_samples,
                           fixes_to_observation)
from probe.trajectory import PushParams, make_push


def render_state(label, t_end, cam):
    stick, scene = StickParams(), SceneConfig(camera=cam)
    m, d, h = load_scene(stick, scene, active=label, mass=0.55, friction=0.6)
    b = colour_bands(m, h, stick)
    fol = TrajectoryFollower(m, d, make_push(PushParams()), h.grip_site)
    fol.place_at_start()
    for _ in range(800):
        fol.command(0.0)
        mujoco.mj_step(m, d)
    d.time = 0.0
    while d.time < t_end:
        fol.command(d.time)
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    try:
        r = mujoco.Renderer(m, cam.resolution[1], cam.resolution[0])
    except Exception as e:
        pytest.skip(f"no offscreen GL: {e}")
    r.update_scene(d, camera=cam.name)
    img = r.render()
    r.close()
    mk = StickMarkings(length=h.length, band_end=float(b["band_end"]), tip_start=float(b["tip_start"]))
    return img, m, d, h, mk


@pytest.mark.parametrize("label,t_end", [("freespace", 0.0), ("heavy", 2.6)])
def test_tracker_matches_ground_truth(label, t_end):
    cam = CameraConfig()
    img, m, d, h, mk = render_state(label, t_end, cam)
    fix = StickTracker(cam, TrackerConfig(markings=mk)).track(img)
    assert fix.ok, fix.reason
    A = np.linspace(0, mk.length, 41)
    nodes = h.nodes_world(d, m)
    gt = np.column_stack([np.interp(A, h.node_s, nodes[:, i]) for i in range(3)])
    tr = fix_to_samples(fix, A)
    assert np.linalg.norm(tr - gt, axis=1).max() < 4e-3                       # world, mm-level
    Rg = d.site_xmat[h.grip_site].reshape(3, 3)
    assert np.degrees(np.arccos(np.clip(fix.tangent @ Rg[:, 0], -1, 1))) < 0.6  # grip axis
    F = stick_local_frame(Rg)
    _, k_gt, _, _ = extract_frame(m, d, h, A)
    k = curvature_inplane((tr - tr[0]) @ F, A)
    assert np.sqrt(np.mean((k - k_gt) ** 2)) < 0.1
    assert abs(fix.scale - 1) < 0.01


def test_fixes_assemble_into_valid_observation_and_trajectory():
    cam = CameraConfig()
    img, m, d, h, mk = render_state("freespace", 0.0, cam)
    tr = StickTracker(cam, TrackerConfig(markings=mk))
    fixes = [tr.track(img) for _ in range(4)]
    meta = dict(young=2e10, radius=0.004, length=mk.length, timestep=1 / 30, git_sha="t")
    obs, traj = fixes_to_observation(np.arange(4) / 30, fixes, mk, Plane(), "freespace", meta)
    assert obs.meta["source"] == "video" and obs.meta["frames_dropped"] == 0
    assert np.allclose(traj.pos, obs.grip_pose[:, :3])
    # the grip frame follows the sim convention: x along the stick, y = push-plane normal
    from probe.geometry import quat_wxyz_to_mat
    R = quat_wxyz_to_mat(traj.quat[0])
    assert R[:, 1] @ np.array([0, 1.0, 0]) > 0.999 and R[:, 0] @ fixes[0].tangent > 0.9999


def test_backprojection_inverts_projection():
    cam = CameraConfig(cx=940.0, cy=735.0)
    pts = np.array([[0.0, 0.0, 0.9], [0.4, 0.0, 0.8], [-0.2, 0.0, 1.1]])
    assert np.allclose(backproject_to_plane(cam.project(pts), cam, Plane()), pts, atol=1e-9)


def test_lost_stick_is_reported_not_guessed():
    cam = CameraConfig()
    fix = StickTracker(cam).track(np.full((1440, 1920, 3), 128, np.uint8))
    assert not fix.ok and "not found" in fix.reason


def test_yellow_block_touching_the_band_does_not_move_the_grip():
    # the real clamp block is painted the same yellow as the base band: paint a block-sized yellow slab over the
    # gripper, touching the band where the needle leaves it, and the fit must not change
    cam = CameraConfig()
    img, m, d, h, mk = render_state("freespace", 0.0, cam)
    clean = StickTracker(cam, TrackerConfig(markings=mk)).track(img)
    g = cam.project(d.site_xpos[h.grip_site])
    px_per_m = cam.fx / np.linalg.norm(d.site_xpos[h.grip_site] - np.asarray(cam.pos))
    w, ht = int(0.15 * px_per_m), int(0.165 * px_per_m)          # 150 x 165 mm block
    blocked = img.copy()
    u0, v0 = int(g[0]) - w // 2, int(g[1]) - ht + 2               # block above the exit point, touching it
    blocked[v0:int(g[1]) + 2, u0:u0 + w] = (255, 217, 13)
    fix = StickTracker(cam, TrackerConfig(markings=mk)).track(blocked)
    assert fix.ok, fix.reason
    Rg = d.site_xmat[h.grip_site].reshape(3, 3)
    assert np.degrees(np.arccos(np.clip(fix.tangent @ Rg[:, 0], -1, 1))) < 0.6
    assert np.linalg.norm(fix.grip - clean.grip) < 2e-3
    assert abs(fix.scale - 1) < 0.01


def test_stick_off_the_push_plane_is_dropped_not_guessed():
    cam = CameraConfig()
    img, m, d, h, mk = render_state("freespace", 0.0, cam)
    # back-projecting onto a plane 6 cm behind the stick stretches it ~5%: same signature as a sideways buckle
    fix = StickTracker(cam, TrackerConfig(markings=mk, plane=Plane(point=(0, 0.06, 0)))).track(img)
    assert not fix.ok and "push plane" in fix.reason and fix.scale > 1.03


def test_needle_running_out_of_frame_is_dropped():
    # crop the frame just past the yellow -> blue boundary (camera shifted to match): the blue body now ends at
    # the image edge, which must not be mistaken for the painted boundary
    from dataclasses import replace
    cam = CameraConfig()
    img, m, d, h, mk = render_state("freespace", 0.0, cam)
    fix = StickTracker(cam, TrackerConfig(markings=mk)).track(img)
    yb = cam.project(fix.pts[np.searchsorted(fix.s, mk.band_end)])
    shift = int(yb[0]) + 30
    cam_c = replace(cam, cx=cam.cx - shift, resolution=(cam.resolution[0] - shift, cam.resolution[1]))
    lost = StickTracker(cam_c, TrackerConfig(markings=mk)).track(img[:, shift:].copy())
    assert not lost.ok and ("out of frame" in lost.reason or "frame edge" in lost.reason), lost.reason
