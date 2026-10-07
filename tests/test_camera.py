"""Pixel intrinsics: a rendered marker lands where the OpenCV pinhole model says it should."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest

from probe.camera import CameraConfig


def _render_marker(cam: CameraConfig, p):
    xml = f"""<mujoco><visual><global offwidth="{cam.resolution[0]}" offheight="{cam.resolution[1]}"/>
      <headlight ambient="1 1 1" diffuse="0 0 0"/></visual>
      <worldbody>{cam.to_mjcf()}
        <geom type="sphere" size="0.004" pos="{p[0]} {p[1]} {p[2]}" rgba="1 0 0 1" contype="0" conaffinity="0"/>
      </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    try:
        r = mujoco.Renderer(m, cam.resolution[1], cam.resolution[0])
    except Exception as e:  # no headless GL on this machine
        pytest.skip(f"no offscreen GL: {e}")
    r.update_scene(d, camera=cam.name)
    img = r.render().astype(float)
    r.close()
    mask = (img[..., 0] > 150) & (img[..., 1] < 80) & (img[..., 2] < 80)
    assert mask.sum() > 3, "marker not visible"
    v, u = np.nonzero(mask)
    return np.array([u.mean() + 0.5, v.mean() + 0.5])  # pixel centres, OpenCV convention


@pytest.mark.parametrize("cx,cy", [(320, 240), (360, 200), (280, 275)])
def test_rendered_marker_matches_opencv_projection(cx, cy):
    cam = CameraConfig(pos=(0.1, -1.0, 1.0), lookat=(0.2, 0.0, 0.9), resolution=(640, 480),
                       fx=520.0, fy=500.0, cx=cx, cy=cy)
    for p in ([0.2, 0.0, 0.9], [0.45, 0.1, 1.1], [-0.1, -0.1, 0.75]):
        px = cam.project(np.array(p))
        assert 20 < px[0] < 620 and 20 < px[1] < 460
        assert np.linalg.norm(_render_marker(cam, p) - px) < 1.0, (p, px)


def test_explicit_rotation_equals_lookat():
    a = CameraConfig(pos=(0.1, -1.0, 1.0), lookat=(0.2, 0.0, 0.9))
    b = CameraConfig(pos=a.pos, R_cv=a.rotation_cv().tolist())
    p = np.array([[0.3, 0.2, 0.8], [0.0, 0.0, 1.0]])
    assert np.allclose(a.project(p), b.project(p))
    assert a.to_mjcf() == b.to_mjcf()


def test_scaled_and_json_round_trip(tmp_path):
    cam = CameraConfig(cx=1000.0, cy=700.0, R_cv=np.eye(3)[:, [0, 2, 1]].tolist(), source={"export": "x"})
    half = cam.scaled(960, 720)
    p = np.array([0.2, 0.5, 1.0])
    assert np.allclose(half.project(p), cam.project(p) / 2)
    back = CameraConfig.from_json(cam.to_json(tmp_path / "cam.json"))
    assert np.allclose(back.project(p), cam.project(p)) and back.source == {"export": "x"}
