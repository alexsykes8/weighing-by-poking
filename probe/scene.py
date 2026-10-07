"""Scene generation: UR5e + Robotiq 2F-85 holding the stick + table + three geometrically identical objects.

All physical/visual quantities are parameters (dataclasses below). `build_scene_xml()` returns
an MJCF string; `load_scene()` compiles it and applies the stick colour bands.

World frame: +x away from the robot (push direction), +z up, camera looks along +y, so pushes
in the x-z plane are perpendicular to the camera's optical axis.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .camera import CameraConfig  # noqa: F401  (re-exported: scene configs carry a camera)
from .schema import LABELS
from .stick import CONTACT_SOLIMP, CONTACT_SOLREF, EXTENSION_XML, GRIP_SITE, StickParams, colour_bands, find_stick, stick_body_xml

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
UR5E_DIR = REPO / "third_party" / "mujoco_menagerie" / "universal_robots_ur5e"
RQ_DIR = REPO / "third_party" / "mujoco_menagerie" / "robotiq_2f85"
RQ = "rq_"                   # prefix for every gripper name (both models have e.g. a body called 'base')
RQ_CLOSED = 255.0            # gripper actuator command: fully closed on the glued stick
# closed-pad geometry in the gripper base frame (measured with the fingers closed): pads span x = +/-11 mm and
# z = 125..163 mm; the stick passes through their centre, like the needle leaving the block halfway up its edge
PINCH_Z = 0.144
PAD_HALF_WIDTH = 0.011
PAD_TIP_ABOVE_PINCH = 0.0187
RQ_NAME_ATTRS = ("name", "class", "childclass", "mesh", "material", "body1", "body2", "joint", "joint1", "joint2",
                 "tendon", "site")
FLANGE_BODY = "wrist_3_link"
FLANGE_SITE = "attachment_site"
OBJECTS = ("fixed", "heavy", "light")
# elbow-up posture at the start of a 33 deg push with the gripper-mounted real needle; also the IK null-space bias.
# Found with a kinematic scan over base positions (robot_pos) x random IK seeds: from (-0.15, 0.25) the arm tracks
# 20-57 deg pushes at 15-65 mm contact height and 160 mm push depth within 1.3 mm, >= 33 mm clear of the table,
# and (from the default camera) no arm link sits behind the stick, so the stick is seen against the backdrop.
HOME_Q = np.array([2.002, -1.919, 2.147, -1.799, -1.571, -1.139])
# mirrored setup (robot base at (-0.15, -0.25)): same scan, other arm branch; also tracks within 1.3 mm, 33 mm clear
HOME_Q_MIRRORED = np.array([1.139, -1.223, -2.147, -1.343, 1.571, 1.139])


@dataclass
class ObjectClass:
    mass: tuple        # (lo, hi) kg, sampled per rollout
    friction: tuple    # (lo, hi) sliding friction
    rgba: tuple = (0.55, 0.55, 0.6, 1.0)  # identical appearance for every class


@dataclass
class SceneConfig:
    table_height: float = 0.75
    table_friction: float = 0.1   # low, so the object's own friction dominates (MuJoCo takes the max)
    robot_pos: tuple = (-0.15, 0.25, 0.75)  # behind the push plane (camera at -y) and left of the stick
    object_half: tuple = (0.0645, 0.0345, 0.040)     # real box 129 long (push, x) x 69 wide (pushed face) x 80 tall mm
    object_pos: tuple = (0.55, 0.0)                 # x, y of the active object's centre on the table
    parked_x: float = 0.85
    parked_y: tuple = (0.35, 0.55, 0.75)               # inactive objects sit behind the push plane
    timestep: float = 1e-3
    classes: dict = field(default_factory=lambda: {
        # real boxes: 100, 160, 230 g (light) and 345, 520 g (heavy), one box material on the grey drape:
        # tilt-tested static mu 0.55 (trials 0.51-0.58, same for the lightest and heaviest box)
        "heavy": ObjectClass(mass=(0.30, 0.60), friction=(0.50, 0.60)),  # mu*m*g ~ 1.5-3.5 N: the needle bows visibly
        "light": ObjectClass(mass=(0.08, 0.26), friction=(0.50, 0.60)),  # ~0.4-1.5 N
        "fixed": ObjectClass(mass=(1.0, 1.0), friction=(0.6, 0.8)),
    })
    camera: CameraConfig = field(default_factory=CameraConfig)
    # Real setup filmed from the other side of the push plane (needle enters from the right of the frame).
    # The sim push stays +x; the camera then sits at +y looking -y, and the robot, parked boxes and key light
    # are mirrored to -y so they stay behind the stick. None: inferred from the camera (looking along -y).
    mirrored: bool | None = None
    gripper: bool = True          # Robotiq 2F-85 on the flange, stick glued between its fingertips
    exit_angle_deg: float = 57.0  # stick axis vs the gripper's approach axis (measured on the real block)


    def side(self) -> int:
        """+1: camera at -y looking +y (default). -1: mirrored setup, camera at +y looking -y."""
        m = self.mirrored if self.mirrored is not None else self.camera.rotation_cv()[1, 2] < 0
        return -1 if m else 1


def home_q(cfg: "SceneConfig | None" = None) -> np.ndarray:
    """Start posture (and IK null-space bias) for the scene's robot placement."""
    return (HOME_Q if (cfg or SceneConfig()).side() > 0 else HOME_Q_MIRRORED).copy()


def _gripper_xml() -> ET.Element:
    """Robotiq 2F-85 MJCF with every name prefixed, mesh paths relative to the UR5e asset dir (the shared
    meshdir), and no collisions: like the real wooden block, the hand must never touch the box or table."""
    import os
    rq = ET.parse(RQ_DIR / "2f85.xml").getroot()
    for el in rq.iter():
        if el.tag == "mesh" and el.get("file"):
            el.set("name", Path(el.get("file")).stem)
            el.set("file", os.path.relpath(RQ_DIR / "assets" / el.get("file"), UR5E_DIR / "assets"))
        for a in RQ_NAME_ATTRS:
            if el.get(a) is not None:
                el.set(a, RQ + el.get(a))
        if el.tag == "default" and el.get("class") == RQ + "collision":
            el.find("geom").set("contype", "0")
            el.find("geom").set("conaffinity", "0")
        if el.tag == "body":
            el.set("gravcomp", "1")
    return rq


def gripper_stick_mount(exit_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Grip pose in the gripper base frame: (pos, quat wxyz).

    The stick lies in the plane between the fingers (normal = closing axis y), at `exit_angle_deg` from the
    approach axis z, and passes through the pinch point. The grip (s = 0) is where it leaves the pads: their
    side face (x = +/-11 mm) or their distal end, whichever comes first.
    """
    a = np.radians(exit_angle_deg)
    s, c = np.sin(a), np.cos(a)
    R = np.array([[s, 0, c], [0, -1, 0], [c, 0, -s]])  # columns: stick x, y (= -closing axis), z
    pinch = np.array([0.0, 0.0, PINCH_Z])
    t_side = PAD_HALF_WIDTH / s if s > 1e-9 else np.inf
    t_end = PAD_TIP_ABOVE_PINCH / c if c > 1e-9 else np.inf
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return pinch + min(t_side, t_end) * R[:, 0], q


def _object_xml(name: str, active: bool, idx: int, cfg: SceneConfig, mass: float, friction: float) -> str:
    hx, hy, hz = cfg.object_half
    if active:
        x, y = cfg.object_pos
    else:
        x, y = cfg.parked_x, cfg.side() * cfg.parked_y[idx]
    # the static (fixed) box is sunk 2 mm into the table: with any gap, a stick tip sliding down the face
    # wedges into it and ends up ~2 cm inside the box (found by the tracker validation)
    z = cfg.table_height + hz + (1e-4 if name != "fixed" else -2e-3)
    rgba = " ".join(map(str, cfg.classes[name].rgba))
    joint = "" if name == "fixed" else '<freejoint/>'  # fixed: a static body, welded to the world
    inertial = "" if name == "fixed" else f'<inertial pos="0 0 0" mass="{mass}" diaginertia="{_box_I(mass, cfg.object_half)}"/>'
    return f"""
    <body name="obj_{name}" pos="{x} {y} {z}">
      {joint}{inertial}
      <geom name="obj_{name}_geom" type="box" size="{hx} {hy} {hz}" rgba="{rgba}"
            friction="{friction} 0.005 0.0001" condim="4" priority="1" solref="{CONTACT_SOLREF}"
            solimp="{CONTACT_SOLIMP}"/>
    </body>"""


def object_front_face_centre(cfg: "SceneConfig | None" = None) -> np.ndarray:
    """Sim position of the centre of the active object's camera-facing face: the default ARKit anchor
    (a flat, matte face square to the camera gives the cleanest LiDAR return)."""
    cfg = cfg or SceneConfig()
    x, y = cfg.object_pos
    hx, hy, hz = cfg.object_half
    return np.array([x, y - cfg.side() * hy, cfg.table_height + hz])


def _box_I(m, h):
    a, b, c = (2 * np.asarray(h)) ** 2
    return f"{m*(b+c)/12:.6g} {m*(a+c)/12:.6g} {m*(a+b)/12:.6g}"


def build_scene_xml(stick: StickParams | None = None, cfg: SceneConfig | None = None,
                    active: str = "fixed", mass: float | None = None, friction: float | None = None,
                    meshdir: str | None = None) -> str:
    """Full scene MJCF. `active` in LABELS; for 'freespace' all objects are parked.

    meshdir: path the MJCF uses to find UR5e meshes (default: absolute; write_scene makes it relative).
    """
    if active not in LABELS:
        raise ValueError(f"active={active!r} not in {LABELS}")
    if active == "free":  # a movable box of explicitly given mass: the free-joint 'heavy' slot
        if mass is None:
            raise ValueError("active='free' needs an explicit mass")
        active = "heavy"
    stick = stick or StickParams()
    cfg = cfg or SceneConfig()

    ur = ET.parse(UR5E_DIR / "ur5e.xml").getroot()
    ur.find("compiler").set("meshdir", meshdir or str(UR5E_DIR / "assets"))
    for tag in ("option", "keyframe"):  # keyframe qpos size no longer matches once the stick is added
        for el in ur.findall(tag):
            ur.remove(el)
    # gravity compensation on the arm only (a human arm doesn't sag under its own weight either);
    # the stick bodies are generated without it, so the stick still sags.
    for b in ur.iter("body"):
        b.set("gravcomp", "1")
    for mat in ur.iter("material"):  # the UR5e's pale-blue covers share the real needle's hue: grey them out
        if mat.get("name") == "urblue":
            mat.set("rgba", "0.62 0.63 0.65 1")
    base = ur.find("worldbody/body")
    rx, ry, rz = cfg.robot_pos
    base.set("pos", f"{rx} {cfg.side() * ry} {rz}")

    flange = next(b for b in ur.iter("body") if b.get("name") == FLANGE_BODY)
    site = next(s for s in flange.iter("site") if s.get("name") == FLANGE_SITE)
    rq = None
    if cfg.gripper:
        # gripper on the flange site; stick glued to the gripper base between the fingertips
        rq = _gripper_xml()
        mount = ET.SubElement(flange, "body", name=RQ + "flange_mount", pos=site.get("pos"), quat=site.get("quat"))
        mount.append(rq.find("worldbody/body"))
        rq_base = next(b for b in mount.iter("body") if b.get("name") == RQ + "base")
        p, q = gripper_stick_mount(cfg.exit_angle_deg)
        rq_base.append(ET.fromstring(stick_body_xml(stick, pos=" ".join(f"{v:.9g}" for v in p),
                                                    quat=" ".join(f"{v:.9f}" for v in q))))
    else:
        # rigidly mount the stick at the flange site; stick +x along the flange's tool axis (+z)
        site_q = np.array([float(v) for v in site.get("quat").split()])
        site_q /= np.linalg.norm(site_q)
        mount_q = _qmul(site_q, np.array([np.cos(-np.pi / 4), 0, np.sin(-np.pi / 4), 0]))  # rot_y(-90): x->z
        flange.append(ET.fromstring(stick_body_xml(stick, pos=site.get("pos"),
                                                   quat=" ".join(f"{v:.9f}" for v in mount_q))))

    rng_obj = {n: (mass if (n == active and mass is not None) else float(np.mean(cfg.classes[n].mass)),
                   friction if (n == active and friction is not None) else float(np.mean(cfg.classes[n].friction)))
               for n in OBJECTS}
    parked = iter(range(len(cfg.parked_y)))
    objs = "".join(_object_xml(n, n == active, 0 if n == active else next(parked), cfg, *rng_obj[n])
                   for n in OBJECTS)

    T = cfg.table_height
    S = float(cfg.side())  # key light on the camera side
    world = f"""<mujoco>
  {EXTENSION_XML}
  <option timestep="{cfg.timestep}" integrator="implicitfast" solver="Newton" cone="elliptic"
          noslip_iterations="0">
    <flag island="enable"/>
  </option>
  <size memory="64M"/>
  <visual>
    <global offwidth="{cfg.camera.resolution[0]}" offheight="{cfg.camera.resolution[1]}"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.9 0.9 0.92" rgb2="0.6 0.62 0.65" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".45 .45 .47" rgb2=".4 .4 .42" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0"/>
    <material name="table" rgba="0.72 0.6 0.45 1"/>
  </asset>
  <worldbody>
    <light pos="0.3 {-S} 2.2" dir="0 {0.4 * S} -1" directional="true" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <light pos="0.3 {S} 2.0" dir="0 {-0.4 * S} -1" directional="true" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor"/>
    <geom name="table" type="box" pos="0.3 0 {T/2}" size="0.7 0.9 {T/2}" material="table"
          friction="{cfg.table_friction} 0.005 0.0001" solref="{CONTACT_SOLREF}" solimp="{CONTACT_SOLIMP}"/>
    {cfg.camera.to_mjcf()}
    {objs}
  </worldbody>
  <sensor>
    <force name="mount_force" site="{GRIP_SITE}"/>
    <torque name="mount_torque" site="{GRIP_SITE}"/>
  </sensor>
</mujoco>"""
    root = ET.fromstring(world)
    # merge: UR defaults/assets/worldbody/actuators into our world
    for child in ur:
        if child.tag == "worldbody":
            for b in child:
                if b.tag == "light":
                    continue
                root.find("worldbody").append(b)
        elif child.tag == "asset":
            for a in child:
                root.find("asset").append(a)
        else:
            root.insert(1, copy.deepcopy(child))
    if rq is not None:  # gripper defaults/assets, then its constraints and actuator (after the arm's)
        root.find("default").append(rq.find("default/default"))
        for a in rq.find("asset"):
            root.find("asset").append(a)
        for tag in ("contact", "tendon", "equality", "actuator"):
            root.append(rq.find(tag))
    root.set("model", f"probe_scene_{active}")
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def load_scene(stick: StickParams | None = None, cfg: SceneConfig | None = None, active="fixed",
               mass=None, friction=None):
    """Compile the scene. Returns (model, data, stick_handle)."""
    stick = stick or StickParams()
    xml = build_scene_xml(stick, cfg, active, mass, friction)
    m = mujoco.MjModel.from_xml_string(xml)
    h = find_stick(m)
    colour_bands(m, h, stick)
    d = mujoco.MjData(m)
    arm_qadr = arm_joint_qpos_adr(m)
    q0 = home_q(cfg)
    d.qpos[arm_qadr] = q0
    d.ctrl[arm_actuator_ids(m)] = q0
    close_gripper(m, d)
    mujoco.mj_forward(m, d)
    return m, d, h


def arm_joint_ids(m) -> np.ndarray:
    """Hinge joints that have an actuator (i.e. the arm), in actuator order."""
    return np.array([m.actuator_trnid[a, 0] for a in range(m.nu)
                     if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT])


def arm_actuator_ids(m) -> np.ndarray:
    """Actuators driving the arm joints (the gripper's tendon actuator excluded)."""
    return np.array([a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT])


def close_gripper(m, d):
    """Command the gripper shut and start its fingers closed (visual only: the stick is glued to its base)."""
    a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, RQ + "fingers_actuator")
    if a < 0:
        return
    d.ctrl[a] = RQ_CLOSED
    for side in ("left", "right"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{RQ}{side}_driver_joint")
        d.qpos[m.jnt_qposadr[j]] = 0.78


def arm_joint_qpos_adr(m) -> np.ndarray:
    return m.jnt_qposadr[arm_joint_ids(m)]


def arm_joint_dof_adr(m) -> np.ndarray:
    return m.jnt_dofadr[arm_joint_ids(m)]


def write_scene(path=MODELS / "scene.xml", **kw) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    rel = os.path.relpath(UR5E_DIR / "assets", path.parent.resolve())
    path.write_text(build_scene_xml(meshdir=rel, **kw))
    return path
