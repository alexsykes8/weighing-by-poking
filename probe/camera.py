"""Pinhole camera config with pixel intrinsics (OpenCV conventions), MJCF export, and projection.

Intrinsics are what ARKit / Stray Scanner report: fx, fy, cx, cy in pixels, with (cx, cy) measured
from the top-left corner of a width x height image. Pose is either a look-at (pos + lookat, roll-free,
world +z up) or an explicit camera rotation `R_cv` (columns = OpenCV camera axes x right, y down,
z forward, expressed in the sim world), which is what the ARKit loader produces.

MuJoCo cameras look along their -z with +y up, and take the principal point as an offset from the
image centre; `to_mjcf()` does both conversions. `tests/test_camera.py` renders a marker and checks it
lands on the pixel `project()` predicts, which pins the principal-point sign convention.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CV_TO_MJ = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes -> MuJoCo camera axes (same x, flip y and z)


@dataclass
class CameraConfig:
    """Defaults: iPad-like 4:3 wide camera (Stray Scanner records 1920x1440), ~67 deg horizontal FOV."""
    pos: tuple = (0.12, -1.2, 1.0)
    lookat: tuple = (0.12, 0.0, 0.93)
    R_cv: list | None = None            # 3x3, overrides lookat when set
    resolution: tuple = (1920, 1440)
    fx: float = 1440.0
    fy: float = 1440.0
    cx: float = 960.0
    cy: float = 720.0
    name: str = "ipad"
    source: dict = field(default_factory=dict)  # provenance, e.g. which export/frames produced it

    @classmethod
    def from_K(cls, K, resolution, **kw) -> "CameraConfig":
        K = np.asarray(K, float)
        return cls(resolution=tuple(int(v) for v in resolution), fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], **kw)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def rotation_cv(self) -> np.ndarray:
        """World-frame rotation whose columns are the OpenCV camera axes (x right, y down, z forward)."""
        if self.R_cv is not None:
            return np.asarray(self.R_cv, float)
        fwd = np.subtract(self.lookat, self.pos).astype(float)
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return np.column_stack([right, down, fwd])

    def scaled(self, width: int, height: int) -> "CameraConfig":
        """Same camera at another image size (e.g. render at half resolution)."""
        sx, sy = width / self.resolution[0], height / self.resolution[1]
        d = asdict(self)
        d.update(resolution=(width, height), fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy)
        return CameraConfig(**d)

    def project(self, p_world) -> np.ndarray:
        """OpenCV pinhole projection of world points (..., 3) to pixels (..., 2), origin top-left."""
        R = self.rotation_cv()
        pc = (np.asarray(p_world, float) - np.asarray(self.pos)) @ R
        return np.stack([self.fx * pc[..., 0] / pc[..., 2] + self.cx, self.fy * pc[..., 1] / pc[..., 2] + self.cy], -1)

    def to_mjcf(self) -> str:
        R_mj = self.rotation_cv() @ CV_TO_MJ
        w, h = self.resolution
        # sensorsize only fixes the pixel pitch here; pick 1 um pixels (the image geometry is pitch-independent)
        pitch = 1e-6
        # principal point as an offset from the image centre. Sign convention established empirically on
        # MuJoCo 3.14 (tests/test_camera.py): both axes are (centre - principal), i.e. x is NOT (cx - w/2).
        px, py = w / 2.0 - self.cx, h / 2.0 - self.cy
        return (f'<camera name="{self.name}" pos="{" ".join(f"{v:.6f}" for v in self.pos)}" '
                f'xyaxes="{" ".join(f"{v:.9f}" for v in (*R_mj[:, 0], *R_mj[:, 1]))}" '
                f'sensorsize="{w * pitch:.9g} {h * pitch:.9g}" resolution="{w} {h}" '
                f'focalpixel="{self.fx:.6f} {self.fy:.6f}" principalpixel="{px:.6f} {py:.6f}"/>')

    def to_json(self, path) -> Path:
        d = asdict(self)
        d["R_cv"] = None if self.R_cv is None else np.asarray(self.R_cv).tolist()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=2, default=float))
        return path

    @classmethod
    def from_json(cls, path) -> "CameraConfig":
        d = json.loads(Path(path).read_text())
        for k in ("pos", "lookat", "resolution"):
            d[k] = tuple(d[k])
        return cls(**d)
