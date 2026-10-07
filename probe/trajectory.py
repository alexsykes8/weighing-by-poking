"""Synthetic push trajectories in the grip-trajectory CSV format.

Every push lives in the world x-z plane (y = const), i.e. perpendicular to the camera axis,
mirroring the constraint used when filming. Phases: approach -> contact -> push -> hold -> retract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .geometry import mat_to_quat_wxyz
from .scene import SceneConfig
from .schema import GripTrajectory
from .stick import StickParams


@dataclass
class PushParams:
    pitch_deg: float = 33.0        # stick angle below horizontal ("approach angle"); 33 = the real pushes
    contact_height: float = 0.03   # tip height above the table at contact (m); must be below the box top (80 mm)
    standoff: float = 0.08         # tip-to-face gap at the start of the approach (m)
    push_depth: float = 0.06       # how far the grip advances past first contact (m)
    approach_speed: float = 0.10   # m/s
    push_speed: float = 0.05       # m/s
    hold: float = 0.6              # s at full depth
    retract: float = 0.15          # m pulled back from the deepest point
    settle: float = 0.5            # s at rest before moving
    y: float = 0.0                 # sideways offset of the push line from the box centre (m)
    rate_hz: float = 60.0          # matches a 60 fps video track

    @staticmethod
    def sample(rng: np.random.Generator) -> "PushParams":
        return PushParams(pitch_deg=rng.uniform(30, 55), contact_height=rng.uniform(0.005, 0.065),
                          push_depth=rng.uniform(0.04, 0.20), push_speed=rng.uniform(0.03, 0.25),
                          approach_speed=rng.uniform(0.08, 0.25), hold=rng.uniform(0.4, 0.9),
                          y=rng.uniform(-0.015, 0.015))  # sideways miss of the face centre


def grip_rotation(pitch_deg: float) -> np.ndarray:
    """Grip frame: +x along the stick (pitched down toward +x world), +y = world +y."""
    a = np.radians(pitch_deg)
    x = np.array([np.cos(a), 0, -np.sin(a)])
    z = np.array([np.sin(a), 0, np.cos(a)])
    return np.column_stack([x, np.cross(z, x), z])


def make_push(pp: PushParams, stick: StickParams | None = None, scene: SceneConfig | None = None) -> GripTrajectory:
    stick = stick or StickParams()
    scene = scene or SceneConfig()
    R = grip_rotation(pp.pitch_deg)
    face_x = scene.object_pos[0] - scene.object_half[0]
    tip_contact = np.array([face_x - stick.radius, pp.y, scene.table_height + pp.contact_height])
    grip_contact = tip_contact - stick.length * R[:, 0]
    fwd = np.array([1.0, 0, 0])

    # waypoints (position, segment duration)
    p0 = grip_contact - pp.standoff * fwd
    p1 = grip_contact
    p2 = grip_contact + pp.push_depth * fwd
    p3 = p2 - pp.retract * fwd
    segs = [(p0, p0, pp.settle), (p0, p1, pp.standoff / pp.approach_speed),
            (p1, p2, pp.push_depth / pp.push_speed), (p2, p2, pp.hold),
            (p2, p3, pp.retract / pp.approach_speed), (p3, p3, 0.5)]
    ts, ps = [], []
    t = 0.0
    for a, b, dur in segs:
        n = max(2, int(round(dur * pp.rate_hz)))
        u = np.linspace(0, 1, n, endpoint=False)
        s = 0.5 - 0.5 * np.cos(np.pi * u)  # smooth (zero-velocity) ends, like a hand
        ts.append(t + u * dur)
        ps.append(a + s[:, None] * (b - a))
        t += dur
    ts.append([t])
    ps.append([segs[-1][1]])
    t_all = np.concatenate(ts)
    p_all = np.vstack(ps)
    q = np.tile(mat_to_quat_wxyz(R), (t_all.size, 1))
    return GripTrajectory(t_all, p_all, q).validate()


def push_meta(pp: PushParams) -> dict:
    return asdict(pp)


def first_still_index(t, pos, window: float = 0.4, tol: float = 4e-3, extra=None, extra_tol: float = 3e-3):
    """First sample from which `pos` (and optionally a scalar `extra`, e.g. tip height) stays within `tol`
    for `window` seconds: where a hand-held clip first comes to rest. None if it never does."""
    t, pos = np.asarray(t, float), np.asarray(pos, float)
    for i in range(len(t)):
        if t[i] + window > t[-1]:
            break
        w = (t >= t[i]) & (t < t[i] + window)
        if np.ptp(pos[w], axis=0).max() < tol and (extra is None or np.ptp(np.asarray(extra)[w]) < extra_tol):
            return i
    return None


def replay_variant(traj: GripTrajectory, time_scale: float = 1.0, y_offset: float = 0.0) -> GripTrajectory:
    """A recorded (video) grip trajectory made into a training push: trimmed to start once the hand is still
    (real clips begin with the needle being brought into shot), played `time_scale` times slower (>1) or faster
    (<1), and moved `y_offset` sideways off the box centre."""
    i0 = first_still_index(traj.t, traj.pos) or 0
    t = (traj.t[i0:] - traj.t[i0]) * time_scale
    pos = traj.pos[i0:].copy()
    pos[:, 1] += y_offset
    return GripTrajectory(t, pos, traj.quat[i0:].copy()).validate()
