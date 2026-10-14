"""Mass + anchored estimator: a small GRU over (grip trajectory, stick curvature), conditioned on stick
parameters and friction. Outputs log-mass with a predicted variance (Gaussian NLL) and an anchored logit.

`sequence_features` is the single preprocessing path for sim rollouts AND tracked real video: each frame's
stick-local centreline is reduced by the same cubic fit the tracker uses, so sim and video inputs have the
same form. Training-time augmentation adds 0-2 mm iid noise to centreline points *before* that fit.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .geometry import quat_wxyz_to_mat

KAPPA_AT = (0.0, 0.25, 0.5, 0.75)   # fractions of the length where curvature is read off the cubic
SEQ_FEATURES = ("grip_dx", "grip_dz", "grip_vx", "grip_vz", "grip_pitch",
                "kappa_0", "kappa_25", "kappa_50", "kappa_75", "tip_x", "tip_z")
COND_FEATURES = ("log10_young", "length", "radius_mm", "log10_damping", "friction")
RATE_HZ = 30.0


def cubic_shape(cl: np.ndarray, arclen: np.ndarray):
    """Per-frame cubic fit x(s), z(s) of stick-local centrelines (T,K,3).
    Returns curvature at KAPPA_AT (T,4) and the fitted tip (T,2) [x - L, z]."""
    L = arclen[-1]
    u = arclen / L
    A = np.vander(u, 4, increasing=True)                      # (K,4): 1, u, u^2, u^3
    pinv = np.linalg.pinv(A)                                  # (4,K)
    cx = np.einsum("ck,tk->tc", pinv, cl[..., 0])             # (T,4)
    cz = np.einsum("ck,tk->tc", pinv, cl[..., 2])
    uu = np.asarray(KAPPA_AT)
    d1 = np.stack([np.zeros_like(uu), np.ones_like(uu), 2 * uu, 3 * uu**2], 1)   # d/du
    d2 = np.stack([np.zeros_like(uu), np.zeros_like(uu), 2 * np.ones_like(uu), 6 * uu], 1)
    x1, x2, z1, z2 = cx @ d1.T, cx @ d2.T, cz @ d1.T, cz @ d2.T
    kappa = (x1 * z2 - z1 * x2) / np.maximum((x1**2 + z1**2) ** 1.5, 1e-9) / L  # per metre
    tip = np.stack([cx.sum(1) - L, cz.sum(1)], 1)
    return kappa, tip


def grip_features(t, grip_pose):
    """Grip kinematics in the push plane: displacement/velocity along the initial horizontal stick
    direction and vertically, and the stick pitch at the grip (rad, + = pointing down)."""
    R0 = quat_wxyz_to_mat(grip_pose[0, 3:])
    fwd = R0[:, 0].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    d = grip_pose[:, :3] - grip_pose[0, :3]
    dx, dz = d @ fwd, d[:, 2]
    vx, vz = np.gradient(dx, t), np.gradient(dz, t)
    xs = quat_wxyz_to_mat(grip_pose[:, 3:])[:, :, 0]
    pitch = np.arcsin(np.clip(-xs[:, 2], -1, 1))
    return np.stack([dx, dz, vx, vz, pitch], 1)


def resample(t, *arrays, rate=RATE_HZ):
    tt = np.arange(0, t[-1] - t[0] + 1e-9, 1.0 / rate) + t[0]
    out = []
    for a in arrays:
        flat = a.reshape(len(t), -1)
        r = np.stack([np.interp(tt, t, flat[:, j]) for j in range(flat.shape[1])], 1)
        out.append(r.reshape((len(tt),) + a.shape[1:]))
    return tt, out


REST_WINDOW_S = 0.4  # track_video.py trims real clips to start at the needle's first 0.4 s still window
REST_MIN_FRAMES = 15  # dropped frames can leave 1-2 in that window, and single tip readings scatter 1-3 mm


def zero_rest_shape(t, centreline, window: float = REST_WINDOW_S, min_frames: int = REST_MIN_FRAMES):
    """Real clips only: subtract the needle's unloaded bend (stick-local z, median over the first `window` s,
    widened to at least `min_frames` tracked frames) from every frame. Tracked needles read 1-6 mm off straight
    at rest, always on the push-bend side, which the sim-trained estimator takes as force: +17-49% on light
    (100-160 g) boxes, ~0 on 345 g. Axial x is kept."""
    t, cl = np.asarray(t, float), np.array(centreline, float)
    n = max(int(np.sum(t < t[0] + window)), min_frames)
    rest = np.nanmedian(cl[:n, :, 2], axis=0)
    cl[:, :, 2] -= rest
    return cl


def sequence_features(t, grip_pose, centreline, arclen, noise_m: float = 0.0, rng=None, rate=RATE_HZ):
    """(T,F) model input from contract arrays (sim or video). noise_m: iid tracker noise on centreline points."""
    t = np.asarray(t, float)
    if abs(np.median(np.diff(t)) - 1.0 / rate) > 1e-3:
        t, (grip_pose, centreline) = resample(t, grip_pose, centreline, rate=rate)
    cl = np.asarray(centreline, float)
    if noise_m > 0:
        rng = rng or np.random.default_rng()
        cl = cl + rng.normal(0, noise_m, cl.shape)
        cl[:, 0] = 0.0  # the grip is the frame origin by definition
    kappa, tip = cubic_shape(cl, np.asarray(arclen, float))
    g = grip_features(t, grip_pose)
    return np.concatenate([g, kappa, tip], 1).astype(np.float32)


def cond_vector(young, length, radius, damping, friction) -> np.ndarray:
    return np.array([np.log10(young), length, radius * 1e3, np.log10(damping), friction], np.float32)


class Estimator(nn.Module):
    def __init__(self, n_seq=len(SEQ_FEATURES), n_cond=len(COND_FEATURES), hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.cond_in = nn.Linear(n_cond, hidden)
        self.gru = nn.GRU(n_seq + hidden, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(2 * hidden + hidden, 64), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(64, 3))

    def forward(self, x, lengths, cond):
        c = torch.tanh(self.cond_in(cond))                                   # (B,H)
        xin = torch.cat([x, c[:, None, :].expand(-1, x.shape[1], -1)], 2)    # condition every step
        packed = nn.utils.rnn.pack_padded_sequence(xin, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        mask = (torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]).float()[..., None]
        mean = (out * mask).sum(1) / mask.sum(1)
        z = self.head(torch.cat([h[-1], mean, c], 1))
        mu, log_var, logit = z[:, 0], z[:, 1].clamp(-8, 4), z[:, 2]
        return mu, log_var, logit


def loss_fn(mu, log_var, logit, y_logm, y_anch, is_free, w_anch=1.0):
    """Gaussian NLL on log-mass (free boxes only) + BCE on anchored (all samples)."""
    bce = nn.functional.binary_cross_entropy_with_logits(logit, y_anch)
    if is_free.any():
        m = is_free
        nll = (0.5 * (log_var[m] + (y_logm[m] - mu[m]) ** 2 / log_var[m].exp())).mean()
    else:
        nll = torch.zeros((), device=mu.device)
    return nll + w_anch * bce, nll, bce
