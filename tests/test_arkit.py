"""Stray Scanner loader + ARKit->sim alignment, round-tripped through a synthetic export."""
import imageio.v3 as iio
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from probe.arkit import camera_from_stray, check_report, gravity_check, load_stray
from probe.camera import CameraConfig
from probe.scene import object_front_face_centre

DEPTH_WH = (256, 192)


def R_arkit_from_sim(yaw):
    """ARKit world is y-up: sim z -> ARKit y, sim x -> horizontal at `yaw`."""
    ex = np.array([np.cos(yaw), 0, -np.sin(yaw)])
    ez = np.array([0, 1.0, 0])
    return np.column_stack([ex, np.cross(ez, ex), ez])


def write_fake_export(root, cam: CameraConfig, wall_y, yaw=0.7, offset=(2.0, -0.3, 5.0), n=30, jitter=4e-4,
                      world_up_is_y=True, seed=0):
    """A Stray-format export of a scene seen by `cam` (sim frame). Depth: a wall at sim y = wall_y."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    np.savetxt(root / "camera_matrix.csv", cam.K, delimiter=",")
    R_AS = R_arkit_from_sim(yaw) if world_up_is_y else np.eye(3)
    t_AS = np.asarray(offset)
    R_cv_S = cam.rotation_cv()
    rows = []
    (root / "depth").mkdir(exist_ok=True)
    (root / "confidence").mkdir(exist_ok=True)
    # depth: per depth pixel, ray from the camera centre to the plane y_S = wall_y, z-depth along the optical axis
    W, H = DEPTH_WH
    sx, sy = W / cam.resolution[0], H / cam.resolution[1]
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays_c = np.stack([(uu / sx - cam.cx) / cam.fx, (vv / sy - cam.cy) / cam.fy, np.ones_like(uu)], -1)
    rays_S = rays_c @ R_cv_S.T
    s = (wall_y - cam.pos[1]) / rays_S[..., 1]
    depth_mm = np.round(s * 1000).astype(np.uint16)  # z-depth: s scales a ray whose camera z is 1
    for i in range(n):
        dp = rng.normal(0, jitter, 3)
        dR = Rotation.from_rotvec(rng.normal(0, np.radians(0.02), 3)).as_matrix()
        p_A = R_AS @ (np.asarray(cam.pos) + dp) + t_AS
        R_A = R_AS @ dR @ R_cv_S
        q = Rotation.from_matrix(R_A).as_quat()
        rows.append([i / 60.0, i, *p_A, *q])
        iio.imwrite(root / "depth" / f"{i:06d}.png", depth_mm)
        iio.imwrite(root / "confidence" / f"{i:06d}.png", np.full((H, W), 2, np.uint8))
    np.savetxt(root / "odometry.csv", np.array(rows), delimiter=",", header="timestamp, frame, x, y, z, qx, qy, qz, qw",
               comments="")
    return root


@pytest.fixture
def truth():
    # a realistic-but-imperfect tripod: off-centre principal point, pitched down, slightly off-square
    return CameraConfig(pos=(0.15, -1.1, 1.02), lookat=(0.5, 0.0, 0.8), resolution=(1920, 1440),
                        fx=1452.3, fy=1449.8, cx=951.2, cy=727.6)


def test_round_trip_recovers_camera(tmp_path, truth):
    anchor_S = object_front_face_centre()
    exp = load_stray(write_fake_export(tmp_path / "clip", truth, wall_y=anchor_S[1]))
    assert exp.n == 30 and exp.rgb_size == (1920, 1440) and len(exp.depth_files) == 30
    u, v = truth.project(anchor_S)
    axis_S = anchor_S + np.array([-0.3, 0, 0])  # a point further back along the push line
    cam, rep = camera_from_stray(exp, (u, v), anchor_S, axis_px=tuple(truth.project(axis_S)))
    assert np.allclose(cam.K, truth.K)
    assert np.linalg.norm(np.subtract(cam.pos, truth.pos)) < 3e-3, cam.pos
    ang = np.degrees(Rotation.from_matrix(cam.rotation_cv().T @ truth.rotation_cv()).magnitude())
    # yaw comes from two LiDAR points 0.3 m apart; 1 mm depth quantisation alone moves it by up to ~0.2 deg
    assert ang < 0.3, ang
    pts = np.array([[0.0, 0.0, 0.8], [0.55, -0.05, 0.8], [0.3, 0.0, 1.2]])
    assert np.abs(cam.project(pts) - truth.project(pts)).max() < 3.0  # px at 1920x1440
    assert rep["tripod"]["pos_dev_max_mm"] < 3 and rep["gravity"]["image_up_dot_world_up"] > 0.9
    assert rep["yaw_off_plane_normal_deg"] == pytest.approx(
        np.degrees(np.arctan2(*truth.rotation_cv()[:2, 2])), abs=0.2)


def test_without_axis_assumes_square_camera(tmp_path, truth):
    square = CameraConfig(pos=(0.15, -1.1, 1.02), lookat=(0.15, 0.0, 0.85), fx=1450, fy=1450, cx=960, cy=720)
    anchor_S = object_front_face_centre()
    exp = load_stray(write_fake_export(tmp_path / "clip", square, wall_y=anchor_S[1]))
    cam, rep = camera_from_stray(exp, tuple(square.project(anchor_S)), anchor_S)
    assert np.linalg.norm(np.subtract(cam.pos, square.pos)) < 3e-3
    assert any("square" in w for w in check_report(rep))


def test_wrong_world_convention_is_flagged(tmp_path, truth):
    anchor_S = object_front_face_centre()
    exp = load_stray(write_fake_export(tmp_path / "clip", truth, wall_y=anchor_S[1], world_up_is_y=False))
    _, rep = camera_from_stray(exp, tuple(truth.project(anchor_S)), anchor_S)
    assert any("world frame" in w for w in check_report(rep))


def test_gravity_check_reports_pitch():
    cam = CameraConfig(pos=(0, -1, 1), lookat=(0, 0, 0))  # looking 45 deg down, in sim (z-up) coordinates
    g = gravity_check(cam.rotation_cv(), up=np.array([0, 0, 1.0]))
    assert g["camera_pitch_deg"] == pytest.approx(-45, abs=1e-6) and abs(g["camera_roll_deg"]) < 1e-6


def test_missing_files_and_bad_header_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_stray(tmp_path)
    np.savetxt(tmp_path / "camera_matrix.csv", np.eye(3), delimiter=",")
    (tmp_path / "odometry.csv").write_text("t,x,y,z,qw,qx,qy,qz\n0,0,0,0,1,0,0,0\n")
    with pytest.raises(ValueError, match="header"):
        load_stray(tmp_path)


def test_mirrored_setup_puts_sim_camera_on_the_other_side(tmp_path):
    # needle enters from the right of the frame: the real camera is on the sim's +y side, looking -y
    truth = CameraConfig(pos=(0.6, 1.1, 1.02), lookat=(0.3, 0.0, 0.8), resolution=(1920, 1440),
                         fx=1452.3, fy=1449.8, cx=951.2, cy=727.6)
    from probe.scene import SceneConfig
    anchor_S = object_front_face_centre(SceneConfig(mirrored=True))
    assert anchor_S[1] > 0  # the camera-facing face is now the +y face
    exp = load_stray(write_fake_export(tmp_path / "clip", truth, wall_y=anchor_S[1]))
    axis_S = anchor_S + np.array([-0.3, 0, 0])
    cam, rep = camera_from_stray(exp, tuple(truth.project(anchor_S)), anchor_S,
                                 axis_px=tuple(truth.project(axis_S)), mirrored=True)
    assert np.linalg.norm(np.subtract(cam.pos, truth.pos)) < 3e-3, cam.pos
    pts = np.array([[0.0, 0.0, 0.8], [0.55, 0.05, 0.8], [0.3, 0.0, 1.2]])
    assert np.abs(cam.project(pts) - truth.project(pts)).max() < 3.0
    assert rep["mirrored"] and SceneConfig(camera=cam).side() == -1  # scenes built with it mirror themselves
    # without the flag the same clip would come out on the wrong side of the push plane
    cam_wrong, _ = camera_from_stray(exp, tuple(truth.project(anchor_S)), anchor_S, axis_px=tuple(truth.project(axis_S)))
    assert np.linalg.norm(np.subtract(cam_wrong.pos, truth.pos)) > 0.5


def test_mirrored_scene_tracks_a_push_with_the_robot_behind_the_stick():
    from probe.rollout import run_rollout
    from probe.scene import SceneConfig
    from probe.trajectory import PushParams, make_push
    cfg = SceneConfig(camera=CameraConfig(pos=(0.12, 1.2, 1.0), lookat=(0.12, 0.0, 0.93)))
    assert cfg.side() == -1
    obs = run_rollout(make_push(PushParams(pitch_deg=33), scene=cfg), "free", scene=cfg, mass=0.3, friction=0.55,
                      record_hz=50)
    err = np.linalg.norm(obs.grip_pose[:, :3] - obs.extras["grip_cmd"][:, :3], axis=1)
    assert err.max() < 5e-3
    assert np.linalg.norm(obs.extras["obj_pos"][-1, :2] - obs.extras["obj_pos"][0, :2]) > 0.02  # box was pushed


def test_portrait_camera_without_axis_point(tmp_path):
    # iPad in portrait: the recorded frame is the sensor's landscape frame rolled 90 deg about the optical axis
    level = CameraConfig(pos=(0.15, -1.1, 1.02), lookat=(0.15, 0.0, 0.85), fx=1450, fy=1450, cx=960, cy=720)
    roll = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])  # camera x/y swapped: image-up is now sideways
    portrait = CameraConfig(pos=level.pos, R_cv=(level.rotation_cv() @ roll).tolist(), fx=1450, fy=1450,
                            cx=960, cy=720)
    anchor_S = object_front_face_centre()
    exp = load_stray(write_fake_export(tmp_path / "clip", portrait, wall_y=anchor_S[1]))
    cam, rep = camera_from_stray(exp, tuple(portrait.project(anchor_S)), anchor_S)
    assert np.linalg.norm(np.subtract(cam.pos, portrait.pos)) < 3e-3, cam.pos
    pts = np.array([[0.0, 0.0, 0.8], [0.55, -0.05, 0.8], [0.3, 0.0, 1.2]])
    assert np.abs(cam.project(pts) - portrait.project(pts)).max() < 3.0
    assert abs(abs(rep["gravity"]["camera_roll_deg"]) - 90) < 1
    w = check_report(rep)
    assert not any("world frame" in x for x in w) and any("portrait" in x for x in w)
