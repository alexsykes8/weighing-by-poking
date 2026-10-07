"""Stray Scanner (ARKit + LiDAR) export loader, and alignment of the recorded camera to the sim world.

Stray Scanner export layout (verified against the app's reference loader, github.com/kekeblom/StrayVisualizer):
    rgb.mp4               RGB video (1920x1440 on current iPads)
    camera_matrix.csv     3x3 intrinsic matrix K for the RGB resolution, comma separated
    odometry.csv          header + rows: timestamp, frame, x, y, z, qx, qy, qz, qw
                          = camera-to-world pose T_WC per RGB frame; camera axes in OpenCV convention
                          (x right, y down, z forward); world = ARKit's gravity-aligned frame (+y up)
    depth/000000.png      LiDAR depth, uint16 millimetres, 256x192 (or .npy)
    confidence/000000.png 0/1/2, higher = more confident
    imu.csv               not used here

The world +y-up assumption is ARKit's documented convention but is not stated by Stray; `gravity_check`
verifies it on every export (the camera's image-up direction must point roughly along world +y).

Alignment ARKit -> sim (z up, camera looking along +y, push direction +x):
    z_sim = ARKit up
    x_sim = horizontal push direction: from a second clicked point along the push line if given,
            otherwise horizontal and square to the optical axis (assumes the camera is square to the plane)
    mirrored setup (needle enters from the right of the frame): the sim camera is put at +y looking -y instead
    origin: one clicked pixel with good LiDAR depth (default: centre of the active object's front face),
            whose known sim position anchors the translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .camera import CameraConfig

ARKIT_UP = np.array([0.0, 1.0, 0.0])
DEFAULT_RGB_SIZE = (1920, 1440)


@dataclass
class StrayExport:
    root: Path
    K: np.ndarray          # (3,3) for the RGB image
    rgb_size: tuple        # (w, h)
    t: np.ndarray          # (N,) s
    frame: np.ndarray      # (N,) frame index
    T_WC: np.ndarray       # (N,4,4) camera -> ARKit world, OpenCV camera axes
    depth_files: list
    conf_dir: Path
    K_source: str = "camera_matrix.csv"  # or "odometry (per-frame median)" on newer Stray versions
    K_frames: np.ndarray | None = None   # (N,4) per-frame fx, fy, cx, cy when the export has them

    @property
    def n(self) -> int:
        return len(self.t)

    def depth(self, i: int) -> np.ndarray:
        """Depth in metres (float32), 0 where missing."""
        f = self.depth_files[i]
        mm = np.load(f) if f.suffix == ".npy" else _read_png(f)
        return mm.astype(np.float32) / 1000.0

    def confidence(self, i: int) -> np.ndarray | None:
        f = self.conf_dir / f"{Path(self.depth_files[i]).stem}.png"
        return _read_png(f) if f.exists() else None

    def K_depth(self, depth_shape) -> np.ndarray:
        sy, sx = depth_shape[0] / self.rgb_size[1], depth_shape[1] / self.rgb_size[0]
        return np.diag([sx, sy, 1.0]) @ self.K

    def depth_at(self, i: int, u: float, v: float, win: int = 1, min_conf: int = 2) -> float:
        """Median LiDAR depth (m) around RGB pixel (u, v), using only confident returns."""
        d = self.depth(i)
        c = self.confidence(i)
        Kd = self.K_depth(d.shape)
        ud = (u - self.K[0, 2]) * Kd[0, 0] / self.K[0, 0] + Kd[0, 2]
        vd = (v - self.K[1, 2]) * Kd[1, 1] / self.K[1, 1] + Kd[1, 2]
        r, q = int(round(vd - 0.5)), int(round(ud - 0.5))
        sl = (slice(max(r - win, 0), r + win + 1), slice(max(q - win, 0), q + win + 1))
        vals = d[sl]
        ok = vals > 0
        if c is not None:
            ok &= c[sl] >= min_conf
        if not ok.any():
            raise ValueError(f"no depth with confidence >= {min_conf} near pixel ({u:.0f}, {v:.0f}) in frame {i}; "
                             "pick a flat, matte, well-lit spot facing the camera")
        return float(np.median(vals[ok]))

    def backproject(self, i: int, u: float, v: float, T_WC=None, **kw) -> np.ndarray:
        """ARKit-world point seen at RGB pixel (u, v) in frame i."""
        z = self.depth_at(i, u, v, **kw)
        pc = np.array([(u - self.K[0, 2]) / self.K[0, 0] * z, (v - self.K[1, 2]) / self.K[1, 1] * z, z])
        T = self.T_WC[i] if T_WC is None else T_WC
        return T[:3, :3] @ pc + T[:3, 3]


def _read_png(path) -> np.ndarray:
    import imageio.v3 as iio
    a = iio.imread(path)
    return a[..., 0] if a.ndim == 3 else a


def load_stray(root) -> StrayExport:
    root = Path(root)
    for f in ("camera_matrix.csv", "odometry.csv"):
        if not (root / f).exists():
            raise FileNotFoundError(f"{root} is not a Stray Scanner export (missing {f})")
    K = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
    if K.shape != (3, 3):
        raise ValueError(f"camera_matrix.csv must be 3x3, got {K.shape}")
    header = [h.strip() for h in (root / "odometry.csv").read_text().splitlines()[0].split(",")]
    expected = ["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw"]
    if header[:9] != expected:
        raise ValueError(f"odometry.csv header {header} does not start with {expected}")
    # newer Stray versions append per-frame intrinsics (fx, fy, cx, cy) and an often-empty distortion centre
    has_k = header[9:13] == ["fx", "fy", "cx", "cy"]
    od = np.loadtxt(root / "odometry.csv", delimiter=",", skiprows=1, ndmin=2, usecols=range(13 if has_k else 9))
    K_frames, K_source = None, "camera_matrix.csv"
    if has_k:
        K_frames = od[:, 9:13]
        fx, fy, cx, cy = np.median(K_frames, axis=0)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])  # ARKit's own per-frame values: preferred
        K_source = "odometry (per-frame median)"
    T = np.tile(np.eye(4), (len(od), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(od[:, 5:9]).as_matrix()  # scipy: x, y, z, w
    T[:, :3, 3] = od[:, 2:5]
    depth_dir = root / "depth"
    depth_files = sorted([p for p in depth_dir.iterdir() if p.suffix in (".png", ".npy")]) if depth_dir.exists() else []
    return StrayExport(root, K, _rgb_size(root), od[:, 0], od[:, 1].astype(int), T, depth_files, root / "confidence",
                       K_source, K_frames)


def _rgb_size(root: Path) -> tuple:
    """RGB frame size from rgb.mp4; K in camera_matrix.csv is for this resolution."""
    video = root / "rgb.mp4"
    if video.exists():
        import imageio.v2 as iio2
        with iio2.get_reader(video) as r:
            return tuple(int(v) for v in r.get_meta_data()["size"])
    return DEFAULT_RGB_SIZE


# ---- tripod pose, sanity checks ---------------------------------------------------------------

def static_pose(exp: StrayExport, frames=None):
    """Robust single pose for a tripod clip: median position, mean rotation. Returns (T_WC, stats)."""
    idx = np.arange(exp.n) if frames is None else np.asarray(frames)
    P = exp.T_WC[idx, :3, 3]
    rots = Rotation.from_matrix(exp.T_WC[idx, :3, :3])
    R_mean = rots.mean()
    p = np.median(P, axis=0)
    T = np.eye(4)
    T[:3, :3] = R_mean.as_matrix()
    T[:3, 3] = p
    stats = {"frames": int(idx.size), "pos_dev_max_mm": float(np.linalg.norm(P - p, axis=1).max() * 1e3),
             "rot_dev_max_deg": float(np.degrees((R_mean.inv() * rots).magnitude().max()))}
    return T, stats


def gravity_check(R_WC: np.ndarray, up=ARKIT_UP) -> dict:
    """For any sane tripod shot world up lies close to the image plane (the camera isn't looking steeply up or
    down). Roll is free: an iPad in portrait records the sensor's landscape frame rotated by +/-90 deg."""
    img_up = -R_WC[:, 1]
    fwd = R_WC[:, 2]
    return {"image_up_dot_world_up": float(img_up @ up),
            "up_in_image_plane": float(np.hypot(R_WC[:, 0] @ up, R_WC[:, 1] @ up)),
            "camera_pitch_deg": float(np.degrees(np.arcsin(np.clip(fwd @ up, -1, 1)))),  # + = looking up
            "camera_roll_deg": float(np.degrees(np.arctan2(R_WC[:, 0] @ up, img_up @ up)))}


# ---- ARKit -> sim alignment ----------------------------------------------------------------------

@dataclass
class Alignment:
    R_SA: np.ndarray   # rotates ARKit-world vectors into sim-world vectors
    a_A: np.ndarray    # anchor in ARKit world
    a_S: np.ndarray    # same anchor in sim world

    def point(self, p_A):
        return (np.asarray(p_A) - self.a_A) @ self.R_SA.T + self.a_S

    def rotation(self, R_A):
        return self.R_SA @ R_A


def align(anchor_A, anchor_S, R_WC, axis_A=None, up=ARKIT_UP, mirrored=False) -> Alignment:
    """Sim axes in ARKit coordinates: z = up, x = horizontal push direction, y = z cross x.

    The push direction's sign comes from which side the camera is on: normally the needle enters from the left
    of the frame and the sim camera looks along +y; `mirrored` (needle enters from the right) puts the sim
    camera on the other side, looking along -y, so the push is still +x in sim.
    """
    z = np.asarray(up, float) / np.linalg.norm(up)
    # without a second point: the horizontal direction square to the optical axis (any camera roll, e.g. portrait)
    x = (np.asarray(axis_A) - np.asarray(anchor_A)) if axis_A is not None else np.cross(R_WC[:, 2], z)
    x = x - z * (x @ z)
    if np.linalg.norm(x) < 1e-6:
        raise ValueError("push axis is vertical; pick the second point along the push line")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    if (y @ R_WC[:, 2] < 0) != mirrored:  # fixes the sign, so any point on the push line works
        x, y = -x, -y
    return Alignment(np.vstack([x, y, z]), np.asarray(anchor_A, float), np.asarray(anchor_S, float))


def camera_from_stray(exp: StrayExport, anchor_px, anchor_S, axis_px=None, frame: int | None = None,
                      frames=None, min_conf: int = 2, mirrored: bool = False) -> tuple[CameraConfig, dict]:
    """Sim CameraConfig (pixel intrinsics + pose in sim world) from a tripod Stray clip.

    anchor_px: RGB pixel (u, v) of a point whose sim position anchor_S is known.
    axis_px:   optional second pixel anywhere else on the push line, e.g. a tape mark (fixes yaw without
               assuming the camera is square to the push plane; direction sign is resolved automatically).
    mirrored:  the needle enters from the right of the frame (see `align`); anchor_S must then be on the
               +y face, e.g. object_front_face_centre(SceneConfig(mirrored=True)).
    """
    T, tripod = static_pose(exp, frames)
    i = exp.n // 2 if frame is None else frame
    a_A = exp.backproject(i, *anchor_px, T_WC=T, min_conf=min_conf)
    ax_A = exp.backproject(i, *axis_px, T_WC=T, min_conf=min_conf) if axis_px is not None else None
    al = align(a_A, anchor_S, T[:3, :3], ax_A, mirrored=mirrored)
    pos = al.point(T[:3, 3])
    R_cv = al.rotation(T[:3, :3])
    cam = CameraConfig.from_K(exp.K, exp.rgb_size, pos=tuple(float(v) for v in pos), R_cv=R_cv.tolist(),
                              source={"stray_export": str(exp.root), "frame": int(i), "anchor_px": list(anchor_px),
                                      "axis_px": None if axis_px is None else list(axis_px)})
    fwd_S = R_cv[:, 2]
    report = {"tripod": tripod, "gravity": gravity_check(T[:3, :3]),
              "anchor_depth_m": float(np.linalg.norm(a_A - T[:3, 3])),
              "camera_pos_sim": pos.tolist(),
              # how square the camera is to the push plane: angle between optical axis and sim +y, horizontally
              "yaw_off_plane_normal_deg": float(np.degrees(np.arctan2(fwd_S[0], abs(fwd_S[1])))),
              "mirrored": bool(mirrored),
              "pitch_deg": float(np.degrees(np.arcsin(np.clip(fwd_S[2], -1, 1)))),
              "axis_given": axis_px is not None, "rgb_size": list(exp.rgb_size)}
    return cam, report


def check_report(report: dict, tripod_mm=5.0, tripod_deg=0.5) -> list[str]:
    """Human-readable warnings for a camera_from_stray report."""
    w = []
    g = report["gravity"]["up_in_image_plane"]
    if g < 0.5:
        w.append(f"world up is only {g:.2f} in the image plane: either the camera is pitched >60 deg, or the "
                 "export's world frame is not ARKit +y-up. Check before trusting the pose.")
    roll = report["gravity"]["camera_roll_deg"]
    if abs(abs(roll) - 90) < 20:
        w.append(f"note: camera roll {roll:.0f} deg (iPad in portrait): fine, frames are used as recorded")
    t = report["tripod"]
    if t["pos_dev_max_mm"] > tripod_mm or t["rot_dev_max_deg"] > tripod_deg:
        w.append(f"camera moved during the clip ({t['pos_dev_max_mm']:.1f} mm, {t['rot_dev_max_deg']:.2f} deg max): "
                 "tripod bumped, or ARKit tracking drift. Consider --frames to use a still segment.")
    if report["axis_given"] and abs(report["yaw_off_plane_normal_deg"]) > 10:
        w.append(f"camera is {report['yaw_off_plane_normal_deg']:.1f} deg off square to the push plane: in-plane "
                 "tracking will foreshorten the stick. Re-square the tripod if you can.")
    if not report["axis_given"]:
        w.append("no --axis-px given: assuming the camera is square to the push plane (yaw unverified).")
    return w
