"""Deformable stick model: a Kirchhoff-rod cable (MuJoCo `mujoco.elasticity.cable` plugin).

Route decision (see docs/design_notes.md): on MuJoCo 3.14 a 1D `flexcomp` gets *no* elastic
passive force (engine_passive.c skips flex_dim == 1), so the cable plugin is the only native
route that reproduces Euler-Bernoulli bending.

Conventions
-----------
* The stick lies along +x of its mount body. The mount body frame *is* the grip frame.
* `young` (Pa) is the single stiffness parameter. Shear modulus G = E / (2(1+nu)).
* `damping` is a Rayleigh-style time constant beta (s): joint damping = beta * EI / l_seg,
  so it is mesh-independent and scales with E instead of being a magic per-joint number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import mujoco
import numpy as np

CABLE_PLUGIN = "mujoco.elasticity.cable"
EXTENSION_XML = f'<extension><plugin plugin="{CABLE_PLUGIN}"/></extension>'

# Contact stiffness for stick, objects and table (MuJoCo's solref is mass-scaled: with the default 0.02 s the
# 2.4 g tip segment, loaded by ~5 N through the rod, sank ~20 mm into a box; with these it is < 1 mm and stable
# at dt = 1e-3). Shared by every contacting geom so pair priority can't silently pick a softer one.
CONTACT_SOLREF = "0.002 1"
CONTACT_SOLIMP = "0.95 0.99 0.001"

MOUNT_BODY = "stick_mount"
GRIP_SITE = "stick_grip"
PREFIX = "stick_"


@dataclass(frozen=True)
class StickParams:
    # real probe: 3 mm aluminium knitting needle + 2 layers of masking tape, 334 mm free length from the clamp.
    # EI = 0.31 N m^2 from hanging-weight tests at the tip (80 g: 32 mm, 242 g: 90 mm; large-deflection fit).
    length: float = 0.334      # m, clamp exit -> tip
    radius: float = 0.00175    # m (3.5 mm with tape)
    young: float = 4.21e10     # Pa: EI / (pi r^4 / 4) with EI = 0.31 N m^2 (the measured quantity)
    poisson: float = 0.3
    density: float = 2180.0    # kg/m^3: 7 g (taped needle, integer scale) over 334 mm at r = 1.75 mm
    n_elem: int = 24
    damping: float = 2e-3      # s, Rayleigh beta
    base_band: float = 0.069   # m, yellow grip band from the clamp exit
    tip_band: float = 0.040    # m, tip marker
    layout: str = "midpoint"   # "midpoint" (default, see vertex_arclen) | "uniform" (uncorrected, for the study)
    # needle-on-box sliding friction. None: the box's own friction wins every box contact (geom priority), so the
    # tip grips the box exactly as hard as the box grips the table. A value gives the needle priority 2 and this
    # friction: real straight pokes slide far more easily than that (see docs/design_notes.md)
    tip_friction: float | None = None

    @property
    def second_moment(self) -> float:
        return np.pi * self.radius**4 / 4.0

    @property
    def EI(self) -> float:
        return self.young * self.second_moment

    @property
    def shear(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def seg_len(self) -> float:
        return self.length / self.n_elem

    @property
    def joint_damping(self) -> float:
        return self.damping * self.EI / self.seg_len

    def with_(self, **kw) -> "StickParams":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


def vertex_arclen(p: StickParams) -> np.ndarray:
    """Cable vertices along the stick axis (midpoint-rule discretisation).

    The cable plugin lumps bending stiffness EI/l at each joint (= vertex). With a uniform
    mesh clamped at the first vertex, the clamp interval is dropped and tip compliance is
    under-predicted by ~3/(2N) (6% at N=24). Starting the cable l/2 *behind* the grip, with a
    rigid first body, puts the joints at the interval midpoints (i - 1/2) l, i = 1..N, which
    integrates the beam compliance with the midpoint rule: relative error 1/(4N^2).
    The rigid half-segment behind the grip is physically the part held in the hand.
    """
    l = p.seg_len
    if p.layout == "uniform":  # vertices 0, l, ..., L: clamped at vertex 0, first segment rigid
        return np.arange(p.n_elem + 1) * l
    if p.layout != "midpoint":
        raise ValueError(f"unknown layout {p.layout!r}")
    return np.concatenate([[-l / 2], (np.arange(1, p.n_elem + 1) - 0.5) * l, [p.length]])


def _vertex_str(p: StickParams) -> str:
    return " ".join(f"{x:.9g} 0 0" for x in vertex_arclen(p))


def stick_body_xml(p: StickParams, pos="0 0 0", quat="1 0 0 0") -> str:
    """MJCF fragment: a mount body carrying the cable. Insert inside any parent body.

    The caller must also add EXTENSION_XML at the top level of the model.
    """
    return f"""
<body name="{MOUNT_BODY}" pos="{pos}" quat="{quat}">
  <site name="{GRIP_SITE}" size="0.006" rgba="1 0 1 0.6" group="3"/>
  <composite prefix="{PREFIX}" type="cable" vertex="{_vertex_str(p)}" initial="none">
    <plugin plugin="{CABLE_PLUGIN}">
      <config key="bend" value="{p.young:.9g}"/>
      <config key="twist" value="{p.shear:.9g}"/>
      <config key="vmax" value="0"/>
    </plugin>
    <joint kind="main" damping="{p.joint_damping:.9g}" armature="0"/>
    <geom type="capsule" size="{p.radius}" density="{p.density}" rgba="0.95 0.95 0.9 1"
          condim="3" {_contact_attrs(p)} solref="{CONTACT_SOLREF}" solimp="{CONTACT_SOLIMP}" group="1"/>
  </composite>
</body>"""


def _contact_attrs(p: StickParams) -> str:
    if p.tip_friction is None:
        return 'friction="0.6 0.005 0.0001"'
    return f'friction="{p.tip_friction:.6g} 0.005 0.0001" priority="2"'


def build_stick_xml(length: float = StickParams.length, radius: float = StickParams.radius,
                    young: float = StickParams.young, n_elem: int = StickParams.n_elem,
                    damping: float = StickParams.damping, *, density: float = StickParams.density,
                    poisson: float = StickParams.poisson, timestep: float = 5e-4,
                    clamp_height: float = 1.0, gravity: bool = True, layout: str = "midpoint") -> str:
    """Standalone model: stick clamped horizontally to the world (calibration rig)."""
    p = StickParams(length=length, radius=radius, young=young, n_elem=n_elem,
                    damping=damping, density=density, poisson=poisson, layout=layout)
    g = "0 0 -9.81" if gravity else "0 0 0"
    return f"""<mujoco model="cantilever">
  {EXTENSION_XML}
  <compiler autolimits="true"/>
  <option timestep="{timestep}" integrator="implicitfast" solver="Newton" gravity="{g}">
    <flag island="enable"/>
  </option>
  <size memory="8M"/>
  <worldbody>
    <light pos="0 -1 2" dir="0 0.5 -1"/>
    <geom name="clamp" type="box" size="0.02 0.02 0.02" pos="-0.02 0 {clamp_height}" rgba="0.3 0.3 0.3 1"
          contype="0" conaffinity="0"/>
    {stick_body_xml(p, pos=f"0 0 {clamp_height}")}
  </worldbody>
</mujoco>"""


# ----------------------------------------------------------------------------------------------
# Model introspection: everything is discovered from the compiled model, no generated names.

@dataclass
class StickHandle:
    mount: int              # body id of the mount (grip frame)
    grip_site: int
    chain: np.ndarray       # body ids base -> tip
    tip_geom: int           # capsule geom on the last body
    length: float           # grip -> tip, m
    node_s: np.ndarray      # rest arc length of each node from grip, m

    def nodes_world(self, d: mujoco.MjData, m: mujoco.MjModel) -> np.ndarray:
        """(N+2, 3) world positions: grip origin, every bending joint, tip. Base -> tip."""
        pts = np.vstack([d.xpos[self.mount], d.xpos[self.chain[1:]]])
        g = self.tip_geom
        half = m.geom_size[g, 1]
        tip = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3)[:, 2] * half
        # capsule axis may point either way along the body; pick the end farther from base
        alt = d.geom_xpos[g] - d.geom_xmat[g].reshape(3, 3)[:, 2] * half
        if np.linalg.norm(alt - pts[-1]) > np.linalg.norm(tip - pts[-1]):
            tip = alt
        return np.vstack([pts, tip])

    def tip_world(self, d, m) -> np.ndarray:
        return self.nodes_world(d, m)[-1]


def find_stick(m: mujoco.MjModel) -> StickHandle:
    mount = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, MOUNT_BODY)
    if mount < 0:
        raise ValueError(f"no body named {MOUNT_BODY!r} in model")
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    chain, cur = [], mount
    while True:
        kids = [b for b in range(m.nbody) if m.body_parentid[b] == cur and b != cur]
        kids = [b for b in kids if m.body_plugin[b] >= 0 or (m.body_geomnum[b] and _is_capsule_body(m, b))]
        if not kids:
            break
        if len(kids) > 1:
            raise ValueError(f"stick chain branches at body {cur}")
        cur = kids[0]
        chain.append(cur)
    if len(chain) < 2:
        raise ValueError("stick chain not found under mount")
    tip_geom = next(g for g in range(m.body_geomadr[chain[-1]],
                                     m.body_geomadr[chain[-1]] + m.body_geomnum[chain[-1]])
                    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE)
    h = StickHandle(mount, site, np.array(chain), tip_geom, 0.0, np.zeros(0))
    # rest arc length of each node, measured from the grip, read off the compiled model
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    rest = h.nodes_world(d, m)
    R = d.xmat[mount].reshape(3, 3)
    h.node_s = (rest - d.xpos[mount]) @ R[:, 0]
    h.length = float(h.node_s[-1])
    return h


def _is_capsule_body(m, b) -> bool:
    return any(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
               for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]))


def stick_geoms(m: mujoco.MjModel, h: StickHandle) -> list[int]:
    return [g for b in h.chain for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])]


def _band_layout(h: StickHandle, p: StickParams, m: mujoco.MjModel):
    """(geom, s_start, s_end, colour) per stick geom; colour in {'yellow', 'blue', 'orange'}."""
    out, s = [], -h.node_s[1]  # first geom starts behind the grip
    for g in stick_geoms(m, h):
        seg = 2 * m.geom_size[g, 1]
        mid = s + seg / 2
        c = "yellow" if mid < p.base_band else "orange" if mid > p.length - p.tip_band else "blue"
        out.append((g, s, s + seg, c))
        s += seg
    return out


def colour_bands(m: mujoco.MjModel, h: StickHandle, p: StickParams) -> dict:
    """Paint the stick like the real one: yellow base band, blue body, orange tip (solid acrylic colours, as seen).

    Returns the arc lengths of the visible colour boundaries, which are quantised to segment
    boundaries in sim (e.g. the yellow band ends at 62.6 mm, not 69): {'band_end', 'tip_start'}.
    """
    # measured on the real needle under the real lamp (HSV 0.108/0.62/0.85, 0.574/0.60/0.72, 0.022/0.66/0.72)
    rgba = {"yellow": (0.85, 0.664, 0.323, 1), "blue": (0.288, 0.528, 0.72, 1), "orange": (0.72, 0.308, 0.245, 1)}
    layout = _band_layout(h, p, m)
    for g, _, _, c in layout:
        m.geom_rgba[g] = rgba[c]
    return {"band_end": max(e for _, _, e, c in layout if c == "yellow"),
            "tip_start": min(s0 for _, s0, _, c in layout if c == "orange")}
