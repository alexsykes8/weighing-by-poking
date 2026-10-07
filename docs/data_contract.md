# Data contract

The video pipeline and the sim pipeline meet here. Both produce the same `ProbeObservation`,
so everything downstream of it (features, classifier, `compare.py`) runs unchanged on either source.
Code: `probe/schema.py` (dataclasses + `validate()`), `probe/io.py` (load/save; validates on the way in and out).
Violations raise `ContractError`, and nothing in this repo catches it.

Units are SI throughout: metres, seconds, kilograms, newtons, pascals.

## Grip trajectory: `data/trajectories/*.csv` (input)

This drives the arm. The sim writes it synthetically today (`scripts/make_synthetic_trajectory.py`);
tomorrow the video tracker writes it.

| column | meaning |
|---|---|
| `t` | seconds from clip start, strictly increasing |
| `x,y,z` | grip point position, world frame, m |
| `qw,qx,qy,qz` | grip orientation, unit quaternion (w first) |

The header line must be exactly `t,x,y,z,qw,qx,qy,qz`.

**Grip frame convention:** +x of the grip frame points along the stick, from grip toward tip. Rolling
about that axis is allowed, because the stick-local frame below doesn't depend on roll.
The grip point is where the stick leaves its clamp: the edge of the wooden block on the real needle (the
start of the yellow band), and the `stick_grip` site in sim (where it leaves the gripper's fingertip pads).

## Probe observation: `data/runs/*.npz` (output)

| key | shape | meaning |
|---|---|---|
| `t` | `(T,)` | s |
| `centreline` | `(T,K,3)` | K points base→tip, **stick-local frame**, m |
| `tip` | `(T,3)` | tip, stick-local; must equal `centreline[:,-1]`. This is the evaluation target |
| `curvature` | `(T,K)` | signed in-plane (x–z) curvature κ(s), 1/m; positive means bending toward +z |
| `arclen` | `(K,)` | arc length along the undeformed stick; `arclen[0]=0` at the grip, uniformly spaced |
| `grip_pose` | `(T,7)` | achieved grip pose, world, `x y z qw qx qy qz` |
| `label` | str | `fixed` \| `heavy` \| `light` \| `freespace` |
| `meta` | dict (JSON) | required: `young`, `radius`, `length`, `timestep`, `source` (`sim`\|`video`), `git_sha` |
| `extra_*` | `(T,...)` | optional. Sim adds `grip_cmd`, `f_contact_obj`, `f_contact_env`, `mount_force`, `mount_torque` (stick-local), `obj_pos` (world) |

### Stick-local frame

- origin at the grip point
- **+x** along the undeformed stick axis at the grip (the grip frame's +x)
- **+z** opposite gravity: world-up with its +x component removed
- **+y** = z × x

This frame is defined by the grip pose and the gravity direction alone. Video gets both without a precise
camera extrinsic (ARKit gives gravity), so sim and video are comparable without world-frame calibration.
When the stick points straight up or down, +z falls back to the grip frame's own z.

### What `validate()` enforces

- every shape above, finite values, strictly increasing `t`, and unit quaternions
- `tip == centreline[:,-1]`
- `centreline[:,0]` within 2 mm of the origin (the frame is grip-centred)
- the stick leaves the grip along +x: the chord from the grip to the sample at ~10% of the length has
  cos > 0.7. A soft stick jammed hard can genuinely bend ~30° near the grip, while a wrong frame gives cos ≈ 0. A chord is used rather than the first segment, so that mm-level tracker noise on short segments
  doesn't trip it, while a wrong frame still fails
- polyline length within 5% of `arclen[-1]` (catches unit or scale errors; the stick is inextensible). It is measured
  over ~10 coarse segments, so per-point tracker noise doesn't inflate the length
- `label` and `meta.source` come from the allowed sets, and the required meta keys are present

### Curvature

`probe.geometry.curvature_inplane` computes the tangent angle θ of each centreline segment in the
x–z plane, then κ = dθ/ds using a Savitzky–Golay filter (a local quadratic fit over a 0.15 m window).
Unlike a smoothing spline, this doesn't force κ to zero at the clamp, where curvature is largest.
Sim and video **must** use the same function. On a loaded cantilever it reproduces κ(s) = F(L−s)/EI
to within 1.1% RMS (`tests/test_extraction.py`).
