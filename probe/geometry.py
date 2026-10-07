"""Frame transforms and centreline processing shared by sim extraction and video tracking."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat()[..., [3, 0, 1, 2]]


def stick_local_frame(grip_R: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation whose columns are the stick-local axes expressed in world.

    +x: the stick axis at the grip (column 0 of the grip rotation, by convention).
    +z: world-up with its +x component removed (i.e. opposite gravity, in the bending plane).
    +y: z cross x.
    """
    x = grip_R[:, 0] / np.linalg.norm(grip_R[:, 0])
    up = np.asarray(up, float)
    z = up - x * (up @ x)
    nz = np.linalg.norm(z)
    if nz < 1e-6:  # stick pointing straight up/down: fall back to the grip's own z
        z = grip_R[:, 2] - x * (grip_R[:, 2] @ x)
        nz = np.linalg.norm(z)
    z /= nz
    return np.column_stack([x, np.cross(z, x), z])


def to_stick_local(points_w: np.ndarray, grip_pos: np.ndarray, grip_R: np.ndarray, up=(0, 0, 1)) -> np.ndarray:
    F = stick_local_frame(grip_R, up)
    return (np.asarray(points_w) - grip_pos) @ F


def resample_by_arclen(node_s: np.ndarray, pts: np.ndarray, arclen: np.ndarray) -> np.ndarray:
    """Linear resampling of a node polyline (rest arc length node_s) at rest arc lengths `arclen`."""
    return np.stack([np.interp(arclen, node_s, pts[:, i]) for i in range(3)], axis=1)


def curvature_inplane(pts: np.ndarray, arclen: np.ndarray, window: float | None = 0.15) -> np.ndarray:
    """Signed curvature of the x-z projection, kappa = d(theta)/ds, positive = bending toward +z.

    theta is the tangent angle of each polyline segment (at segment midpoints). It is
    differentiated with a Savitzky-Golay filter (local quadratic fit over `window` metres of
    arc length), which unlike a smoothing spline imposes no boundary condition: the clamp end,
    where curvature is largest, is not forced to zero. `window` should span a few node spacings
    in sim and be widened for noisy video tracks.
    """
    d = np.diff(pts, axis=0)
    theta = np.unwrap(np.arctan2(d[:, 2], d[:, 0]))
    s_mid = 0.5 * (arclen[1:] + arclen[:-1])
    ds = np.diff(arclen).mean()
    if not np.allclose(np.diff(arclen), ds, rtol=1e-6):
        raise ValueError("curvature_inplane expects uniformly spaced arclen")
    n = theta.size
    w = max(5, int(round((window or 0.0) / ds)) | 1)  # odd, >= 5
    w = min(w, n if n % 2 else n - 1)
    k_mid = savgol_filter(theta, w, polyorder=2, deriv=1, delta=ds, mode="interp")
    return np.interp(arclen, s_mid, k_mid)  # constant extrapolation over the half-cell at each end


def interp_poses(t_src, pos, quat_wxyz, t_dst):
    """Linear position + slerp orientation, clamped at the ends."""
    t_dst = np.clip(t_dst, t_src[0], t_src[-1])
    p = np.stack([np.interp(t_dst, t_src, pos[:, i]) for i in range(3)], axis=1)
    rots = Rotation.from_quat(np.asarray(quat_wxyz)[:, [1, 2, 3, 0]])
    q = Slerp(t_src, rots)(t_dst).as_quat()[:, [3, 0, 1, 2]]
    return p, q
