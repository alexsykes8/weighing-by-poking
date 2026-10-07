"""2D stick tracker: RGB frames -> stick centreline on the push plane -> data-contract observation.

The real probe is painted in three solid colours, measured from the grip (the clamp exit): a yellow base band,
a blue body and an orange tip. Per frame:
  1. colour masks (HSV). The blue body is the largest blue component (nearest the previous fix when tracking)
  2. a provisional centreline: slice centroids of the blue blob along its principal axis, cubic fit, extended
     over the expected base and tip lengths. The end with yellow next to it is the grip end
  3. centre points: at every 1 px along the provisional curve, the mask profile across it (bilinear, 1/4 px)
     gives the centre (profile centroid) and the visible width per colour. One estimator for all three colours.
     Yellow/orange points are kept only while their width matches the needle's, walking outward from the blue:
     a yellow block, the hand or a box edge that touches the needle widens the profile and ends the walk
  4. anchors: the yellow->blue and blue->orange boundaries, where the blue width crosses half its median
     (sub-sample). They sit at known arc lengths (band_end, tip_start); the grip (s = 0, hidden in the hand,
     block or gripper) and the tip are reached by extrapolating the fit, never from the needle's visible ends
  5. back-projection onto the push plane and a cubic fit of the in-plane shape (a tip-loaded cantilever is
     exactly cubic); arc length scaled so the blue section has its measured length. A blue section >3% off
     means the stick has left the plane: the frame is dropped with that reason, not guessed
Then the shared contract code (stick-local frame, curvature) turns the sequence into a ProbeObservation.

Only numpy/scipy. Works on sim renders and Stray Scanner rgb.mp4 alike; colour thresholds are config.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy import ndimage
from scipy.signal import savgol_filter

from .camera import CameraConfig
from .geometry import curvature_inplane, mat_to_quat_wxyz, stick_local_frame
from .schema import GripTrajectory, ProbeObservation


@dataclass
class StickMarkings:
    """Where the colour boundaries sit along the stick, measured from the grip (the clamp exit), m."""
    length: float = 0.334     # grip -> tip
    band_end: float = 0.069   # yellow -> blue
    tip_start: float = 0.294  # blue -> orange


@dataclass
class HSVRange:
    h: tuple             # hue range, 0..1 (lo > hi wraps through red)
    s_min: float
    v_min: float

    def mask(self, h, s, v):
        lo, hi = self.h
        hue = (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)
        return hue & (s >= self.s_min) & (v >= self.v_min)


@dataclass
class ColourThresholds:
    """HSV (all 0..1), from the real needle under the real (warm) lamp: blue hue 0.560-0.584, sat 0.44-0.68;
    yellow 0.104-0.116, sat 0.56-0.67; orange 0.010-0.027, sat 0.55-0.71 (520 g clips, 2026-10-06). The sim
    needle is painted these measured colours. Under the warm lamp the drape and wall also pass the yellow test
    and the lamp glow comes close to orange: harmless, yellow/orange are only collected next to the blue body
    at the needle's width. Tune with `scripts/track_video.py --debug`."""
    base: HSVRange = field(default_factory=lambda: HSVRange((0.085, 0.16), 0.40, 0.30))  # yellow
    body: HSVRange = field(default_factory=lambda: HSVRange((0.53, 0.64), 0.35, 0.20))   # blue
    # orange (wraps red); upper hue 0.08 since the 2026-10-07 clips: the new needle's tip reads hue 0.04-0.08
    # (median 0.05), and the warm drape/wall at that hue stay below s 0.35. Stops short of yellow's 0.085
    tip: HSVRange = field(default_factory=lambda: HSVRange((0.95, 0.08), 0.45, 0.30))


@dataclass
class Plane:
    point: tuple = (0.0, 0.0, 0.0)
    normal: tuple = (0.0, 1.0, 0.0)  # sim: the push plane is y = const, camera looks along +y


@dataclass
class TrackerConfig:
    markings: StickMarkings = field(default_factory=StickMarkings)
    colours: ColourThresholds = field(default_factory=ColourThresholds)
    plane: Plane = field(default_factory=Plane)
    min_blob_px: int = 40
    roi_margin: int = 160           # px around the previous fix searched in the next frame
    step_px: float = 1.0            # spacing of centre samples along the curve
    width_tol: tuple = (0.5, 1.6)   # yellow/orange profile width accepted, x the blue body's median width
    boundary_gap: float = 2.5       # max blended strip between blue and its neighbour paint, x the needle width
    end_skip_w: float = 1.5         # drop this many widths at the occluded grip end and the rounded tip end
    poly_deg: int = 3               # in-plane shape model v(u): a tip-loaded cantilever is exactly cubic
    max_scale_dev: float = 0.03     # reject frames whose blue section measures >3% off: the stick has left the
                                    # push plane (e.g. buckled sideways off a fixed box), so its shape is unknowable


@dataclass
class FrameFix:
    ok: bool
    reason: str = ""
    px: np.ndarray | None = None        # (P,2) fitted curve in pixels, grip -> tip (full-frame coords)
    pts: np.ndarray | None = None       # (P,3) fitted curve, world points on the plane, same order
    s: np.ndarray | None = None         # (P,) arc length from grip (starts at 0)
    grip: np.ndarray | None = None      # (3,) world
    tangent: np.ndarray | None = None   # (3,) unit, stick axis at the grip
    scale: float = np.nan               # measured / nominal blue length (QC: plane depth, out-of-plane)
    width_px: float = np.nan            # visible needle width (QC)


# ---- image primitives ------------------------------------------------------------------------

def _blob(mask, prefer=None, min_px=12):
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    if prefer is not None:  # among decent blobs, prefer the one nearest the previous fix
        ok = np.where(sizes >= max(min_px, 0.3 * sizes.max()))[0]
        if len(ok) == 0:
            return None
        cents = np.atleast_2d(np.array(ndimage.center_of_mass(mask, lab, ok + 1)))[:, ::-1]
        k = ok[np.argmin(np.linalg.norm(cents - prefer, axis=1))]
    else:
        k = int(np.argmax(sizes))
    if sizes[k] < min_px:
        return None
    v, u = np.nonzero(lab == k + 1)
    return np.column_stack([u + 0.5, v + 0.5])  # pixel centres, OpenCV convention


def _axis(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]


def _provisional(pts, bin_px=3.0, deg=3):
    """Cubic centreline v(u) of an elongated blob in its principal-axis frame: (c, a, n, coef, u_lo, u_hi)."""
    c, a = _axis(pts)
    n = np.array([-a[1], a[0]])
    u, v = (pts - c) @ a, (pts - c) @ n
    edges = np.arange(u.min(), u.max() + bin_px, bin_px)
    idx = np.digitize(u, edges)
    cu, cv = [], []
    for k in np.unique(idx):
        sel = idx == k
        if sel.sum() >= 2:
            cu.append(u[sel].mean())
            cv.append(v[sel].mean())
    cu, cv = np.array(cu[1:-1]), np.array(cv[1:-1])  # ragged end slices
    if len(cu) < deg + 2:
        return None
    return c, a, n, np.polyfit(cu, cv, deg), u.min(), u.max()


def _profiles(masks, P, N, halfw, sub=0.25):
    """Across-curve profiles of each mask at curve points P with unit normals N.
    Returns per mask: (width in px, centre offset along N in px; nan where empty)."""
    offs = np.arange(-halfw, halfw + 1e-9, sub)
    Q = P[:, None, :] + offs[None, :, None] * N[:, None, :]           # (S,O,2)
    out = []
    for m in masks:
        val = ndimage.map_coordinates(m, [Q[..., 1] - 0.5, Q[..., 0] - 0.5], order=1, mode="constant", cval=0.0)
        w = val.sum(1) * sub
        with np.errstate(invalid="ignore", divide="ignore"):
            cen = (val * offs).sum(1) * sub / w
        out.append((w, np.where(w > 0, cen, np.nan)))
    return out


def _boundary(t, w, level, start, direction):
    """Walk from index `start` (inside, w >= level) in `direction` (+1/-1); return the interpolated t where w
    first drops below `level`, or None if it never does."""
    i = start
    while 0 <= i + direction < len(w):
        j = i + direction
        if w[j] < level:
            f = (w[i] - level) / (w[i] - w[j]) if w[i] != w[j] else 0.5
            return t[i] + f * (t[j] - t[i])
        i = j
    return None


# ---- geometry -----------------------------------------------------------------------------------

def backproject_to_plane(px, cam: CameraConfig, plane: Plane):
    R = cam.rotation_cv()
    rays_c = np.column_stack([(px[:, 0] - cam.cx) / cam.fx, (px[:, 1] - cam.cy) / cam.fy, np.ones(len(px))])
    rays = rays_c @ R.T
    C = np.asarray(cam.pos, float)
    n = np.asarray(plane.normal, float)
    t = ((np.asarray(plane.point) - C) @ n) / (rays @ n)
    return C + rays * t[:, None]


def _arclen(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


# ---- tracker ------------------------------------------------------------------------------------

class StickTracker:
    def __init__(self, cam: CameraConfig, cfg: TrackerConfig | None = None):
        self.cam, self.cfg = cam, cfg or TrackerConfig()
        self.prev = None  # previous pixel polyline, for the ROI and blob preference

    def _roi(self, shape):
        H, W = shape[:2]
        if self.prev is None:
            return 0, H, 0, W
        m = self.cfg.roi_margin
        lo, hi = self.prev.min(0) - m, self.prev.max(0) + m
        return max(0, int(lo[1])), min(H, int(hi[1])), max(0, int(lo[0])), min(W, int(hi[0]))

    def track(self, rgb: np.ndarray) -> FrameFix:
        fix = self._track(rgb)
        if not fix.ok and self.prev is not None:  # lost: retry on the whole frame
            self.prev = None
            fix = self._track(rgb)
        self.prev = fix.px if fix.ok else None
        return fix

    def _track(self, rgb):
        cfg, c, mk = self.cfg, self.cfg.colours, self.cfg.markings
        r0, r1, c0, c1 = self._roi(rgb.shape)
        img = rgb[r0:r1, c0:c1].astype(np.float32) / 255.0
        off = np.array([c0, r0], float)
        h, sat, val = np.moveaxis(rgb_to_hsv(img), -1, 0)
        yel, blu, ora = (r.mask(h, sat, val).astype(np.float32) for r in (c.base, c.body, c.tip))

        # 1-2. blue body -> provisional centreline, extended linearly past both ends
        pref = self.prev[len(self.prev) // 2] - off if self.prev is not None else None
        body = _blob(blu > 0, pref, cfg.min_blob_px)
        if body is None:
            return FrameFix(False, "blue body not found")
        prov = _provisional(body)
        if prov is None:
            return FrameFix(False, "blue body too short")
        cpt, a, n, coef, u0, u1 = prov
        L_body = u1 - u0
        width = len(body) / max(L_body, 1.0)
        blue_len = mk.tip_start - mk.band_end
        ext = 1.3 * max(mk.band_end, mk.length - mk.tip_start) / blue_len * L_body
        uu = np.arange(u0 - ext, u1 + ext, cfg.step_px)
        uc = np.clip(uu, u0, u1)
        dv = np.polyval(np.polyder(coef), uc)
        vv = np.polyval(coef, uc) + dv * (uu - uc)                    # linear beyond the blob's ends
        P = cpt + uu[:, None] * a + vv[:, None] * n
        T = a + dv[:, None] * n
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        N = np.column_stack([-T[:, 1], T[:, 0]])

        # 3. profiles across the curve, per colour
        (wy, cy), (wb, cb), (wo, co) = _profiles([yel, blu, ora], P, N, halfw=max(3.0 * width, 6.0))
        inner = (uu > u0 + 0.2 * L_body) & (uu < u1 - 0.2 * L_body)
        wmed = float(np.median(wb[inner])) if inner.any() else 0.0
        if wmed < 1.0:
            return FrameFix(False, "blue body too thin")
        lo, hi = uu < u0, uu > u1
        if (wy[lo].sum() - wo[lo].sum()) < (wy[hi].sum() - wo[hi].sum()):  # order samples grip -> tip
            P, N, wy, cy, wb, cb, wo, co, uu = (x[::-1] for x in (P, N, wy, cy, wb, cb, wo, co, uu))
        idx = np.arange(len(uu), dtype=float)

        # 4. anchors: where the blue width falls to half, walking out from the body's middle
        i_mid = int(np.argmin(np.abs(uu - 0.5 * (u0 + u1))))
        b0 = _boundary(idx, wb, 0.5 * wmed, i_mid, -1)
        b1 = _boundary(idx, wb, 0.5 * wmed, i_mid, +1)
        if b0 is None or b1 is None:
            return FrameFix(False, "colour boundary not found")
        at = lambda f: np.array([np.interp(f, idx, P[:, k]) for k in range(2)])  # noqa: E731
        ends_px = np.vstack([at(b0), at(b1)]) + off
        # the blue must end in its neighbouring paint: yellow just past b0, orange just past b1. A blue end with
        # anything else beyond it is shadow, occlusion or the frame edge, not a boundary we can anchor on.
        # Where two paints meet, video compression blends them into a strip that passes neither colour test
        # (measured 5-11 px on the 2026-10-07 clips, ~1-2 needle widths), so the neighbour paint is looked for
        # up to `boundary_gap` widths past the blue's end, then must hold for n_chk samples
        n_chk = max(3, int(np.ceil(1.5 * width / cfg.step_px)))
        gap_max = max(2, int(np.ceil(cfg.boundary_gap * width / cfg.step_px)))
        level = cfg.width_tol[0] * wmed
        for b, step, w_next, name in ((b0, -1, wy, "yellow/blue"), (b1, +1, wo, "blue/orange")):
            j0 = int(np.floor(b)) if step < 0 else int(np.ceil(b))
            hit = next((j0 + step * k for k in range(1, gap_max + 1)
                        if 0 <= j0 + step * k < len(uu) and w_next[j0 + step * k] >= level), None)
            js = [] if hit is None else [hit + step * k for k in range(n_chk) if 0 <= hit + step * k < len(uu)]
            if not js or np.median(w_next[js]) < level:
                return FrameFix(False, f"{name} boundary not visible (shadow, occlusion or frame edge)")

        # a boundary at the edge of the searched window is that edge, not the paint. At the image edge the needle
        # runs out of frame; at the edge of a tracking window, track() retries on the whole frame
        H, W = rgb.shape[:2]
        edge = max(3.0 * width, 6.0)
        lo_b, hi_b = np.array([c0, r0], float), np.array([c1, r1], float)
        if np.any((ends_px < lo_b + edge) | (ends_px > hi_b - edge)):
            full = (c0, r0, c1, r1) == (0, 0, W, H)
            return FrameFix(False, "needle runs out of frame" if full else "needle reaches the search window edge")

        # centre points: blue body, then yellow/orange walked outward while the profile has the needle's width
        gap = max(1, int(np.ceil(width / cfg.step_px)))
        tol_lo, tol_hi = cfg.width_tol[0] * wmed, cfg.width_tol[1] * wmed
        sel = [i for i in range(int(np.ceil(b0)) + gap, int(np.floor(b1)) - gap + 1)
               if tol_lo <= wb[i] <= tol_hi]
        centres = [P[i] + cb[i] * N[i] for i in sel]
        skip = int(np.ceil(cfg.end_skip_w * width / cfg.step_px))
        for start, step, w, cen in ((int(np.floor(b0)) - gap, -1, wy, cy), (int(np.ceil(b1)) + gap, +1, wo, co)):
            walk, i = [], start
            while 0 <= i < len(uu) and tol_lo <= w[i] <= tol_hi and wb[i] < 0.25 * wmed:
                walk.append(P[i] + cen[i] * N[i])
                i += step
            centres += walk[:-skip] if len(walk) > skip else []
        if len(centres) < cfg.poly_deg + 5:
            return FrameFix(False, "too few centre points")
        px_pts = np.array(centres) + off

        # 5. onto the plane, cubic in-plane fit, arc length anchored on the blue section
        Pw = backproject_to_plane(px_pts, self.cam, cfg.plane)
        E = backproject_to_plane(ends_px, self.cam, cfg.plane)
        nrm = np.asarray(cfg.plane.normal, float) / np.linalg.norm(cfg.plane.normal)
        e1 = (E[1] - E[0]) - nrm * ((E[1] - E[0]) @ nrm)
        u_ts = np.linalg.norm(e1)
        e1 /= u_ts
        e2 = np.cross(nrm, e1)
        u, v = (Pw - E[0]) @ e1, (Pw - E[0]) @ e2
        cf = np.polyfit(u / u_ts, v, cfg.poly_deg)
        pre = 1.6 * mk.band_end / blue_len * u_ts
        post = 1.6 * (mk.length - mk.tip_start) / blue_len * u_ts
        ud = np.linspace(-pre, u_ts + post, 2000)
        C = E[0] + np.outer(ud, e1) + np.outer(np.polyval(cf, ud / u_ts), e2)
        sig = _arclen(C)
        sig -= np.interp(0.0, ud, sig)                         # 0 at the yellow -> blue boundary
        scale = np.interp(u_ts, ud, sig) / blue_len
        if abs(scale - 1) > cfg.max_scale_dev:
            return FrameFix(False, "stick out of the push plane (length off by >3%)", scale=float(scale))
        s_c = mk.band_end + sig / scale
        if s_c[0] > 0 or s_c[-1] < mk.length:
            return FrameFix(False, "grip or tip outside the extrapolation range")
        s_out = np.linspace(0.0, mk.length, 600)
        u_out = np.interp(s_out, s_c, ud)
        C = E[0] + np.outer(u_out, e1) + np.outer(np.polyval(cf, u_out / u_ts), e2)
        d = np.polyval(np.polyder(cf), u_out[0] / u_ts) / u_ts
        tangent = (e1 + d * e2) / np.hypot(1.0, d)
        return FrameFix(True, "", self.cam.project(C), C, s_out, C[0], tangent, float(scale), float(wmed))


def fix_to_samples(fix: FrameFix, arclen: np.ndarray, markings: StickMarkings | None = None) -> np.ndarray:
    """World points (K,3) at contract arc lengths, from the fitted curve (which starts at the grip)."""
    sa = np.clip(arclen, fix.s[0], fix.s[-1])
    return np.column_stack([np.interp(sa, fix.s, fix.pts[:, i]) for i in range(3)])


def grip_rotation_from_fix(fix: FrameFix, plane: Plane, up=(0, 0, 1.0)) -> np.ndarray:
    """Grip frame matching the sim convention: x along the stick, y = push-plane normal, z = x cross y."""
    x = fix.tangent
    y = np.asarray(plane.normal, float)
    y = y - x * (x @ y)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    if z @ np.asarray(up) < 0:  # keep z on the 'up' side, as in the sim grip frame
        y, z = -y, -z
    return np.column_stack([x, y, z])


def fixes_to_observation(t, fixes, markings: StickMarkings, plane: Plane, label: str, meta: dict,
                         K: int = 41, curv_window: float = 0.15, temporal_window: int = 0):
    """Assemble valid frames into a ProbeObservation (+ the grip trajectory for replay).

    temporal_window: odd number of frames for a Savitzky-Golay smooth of the world-frame centreline
    over time (0 = off). Dropped frames are recorded in meta.
    """
    ok = np.array([f.ok for f in fixes])
    if ok.sum() < 2:
        raise RuntimeError(f"tracker failed on {np.sum(~ok)}/{len(fixes)} frames: "
                           f"{sorted({f.reason for f in fixes if not f.ok})}")
    t = np.asarray(t, float)[ok]
    good = [f for f in fixes if f.ok]
    arclen = np.linspace(0, markings.length, K)
    W = np.array([fix_to_samples(f, arclen, markings) for f in good])                  # (T,K,3) world
    Rg = np.array([grip_rotation_from_fix(f, plane) for f in good])
    if temporal_window and len(t) > temporal_window:
        W = savgol_filter(W, temporal_window, 2, axis=0)
        # re-derive grip and its rotation from the smoothed curve so the frame stays consistent
        tang = np.gradient(W[:, :3], arclen[:3], axis=1)[:, 0]
        for i, f in enumerate(good):
            f.tangent = tang[i] / np.linalg.norm(tang[i])
            Rg[i] = grip_rotation_from_fix(f, plane)
    grip = W[:, 0]
    cl = np.empty_like(W)
    kappa = np.empty(W.shape[:2])
    for i in range(len(t)):
        F = stick_local_frame(Rg[i])
        cl[i] = (W[i] - grip[i]) @ F
        kappa[i] = curvature_inplane(cl[i], arclen, curv_window)
    pose = np.hstack([grip, mat_to_quat_wxyz(Rg)])
    meta = {**meta, "source": "video", "frames_total": len(fixes), "frames_dropped": int((~ok).sum()),
            "drop_reasons": sorted({f.reason for f in fixes if not f.ok}),
            "length_scale_median": float(np.median([f.scale for f in good])),
            "tracker": {"markings": markings.__dict__, "plane": plane.__dict__, "temporal_window": temporal_window}}
    extras = {"centreline_world": W, "length_scale": np.array([f.scale for f in good])}
    obs = ProbeObservation(t - t[0], cl, cl[:, -1].copy(), kappa, arclen, pose, label, meta, extras).validate()
    traj = GripTrajectory(t - t[0], grip, pose[:, 3:]).validate()
    return obs, traj
