# Design notes

## Versions

- **MuJoCo 3.14.0** (pinned to `mujoco==3.14.*`; 3.15 is out but deliberately not used)
- Python 3.12, numpy 2.5, scipy 1.18, scikit-learn 1.9
- mujoco_menagerie at commit `f054586` (UR5e, unmodified; the scene generator edits a copy in memory)

## Deformable route: cable plugin (Route A), not 1D flex (Route B)

**Decision: `<composite type="cable">` with the `mujoco.elasticity.cable` plugin.**

Flex is where MuJoCo's deformable development has gone, so it was the expected winner. On 3.14,
though, a 1D flex has no elastic passive force at all. This is visible in the source: in
`engine_passive.c` (3.14.0), the flex-elasticity loop begins with `if (m->flex_dim[f] == 1 || m->flex_rigid[f]) continue;`,
and `mj_flexPassiveBend` returns early unless `dim == 2`. The empirical test agrees. A
`<flexcomp type="grid" dim="1">` cantilever with `<elasticity young="5e9">` and two pinned base vertices
free-falls: the tip moves 45.9 km in 6 s, against 4 mm expected from beam theory. Changing the integrator
changes nothing. The result is identical under `implicitfast` and `discrete`.

A second 3.14 change affects flex in general. Flex elasticity under `implicit`/`implicitfast` now raises a
runtime error and needs `integrator="discrete"` (changelog, "Breaking API changes"). If flex is ever
revisited (e.g. a 2D strip), the solver settings below would have to change.

The cable plugin uses `bend` = E and `twist` = G in pascals, with I = πr⁴/4 and J = πr⁴/2 taken from the capsule
radius (`plugin/elasticity/cable.cc`). E is therefore a true Young's modulus, and Euler–Bernoulli applies
directly.

### Discretisation: midpoint rule (the one non-obvious modelling choice)

The plugin lumps bending stiffness EI/ℓ at each joint, with ℓ the length of the preceding segment. With the
usual uniform mesh clamped at vertex 0, the first segment is rigid. That makes the chain a left-Riemann sum
of the beam compliance that drops the clamp interval. Tip compliance comes out low by exactly
(N−1)N(2N−1)/(2N³) ≈ 1 − 3/(2N), which is 0.9384 for N=24. The sim reproduced that value to 4 digits.

Rather than fudge E, the cable now starts ℓ/2 **behind** the grip. The rigid first body is the part held in
the hand, and the bending joints sit at the interval midpoints (i−½)ℓ. That is the midpoint rule, with error
1/(4N²) (0.04% at N=24). Measured sim/EB in the linear regime is 0.996–0.9995. `probe/stick.py:vertex_arclen`.

**Convergence study** (`scripts/discretisation_study.py`, `out/discretisation.png`). The study varies N from 4 to 96
for both layouts. It uses a weightless rod at α = FL²/EI = 0.06, and the error is measured against the
**exact elastica** at that α. Euler–Bernoulli is not the right reference even at this load, because the
true continuum answer sits 4.1e-4 below it. That gap first appeared as a fake error floor in the midpoint
fit (apparent order 1.15).

| N | uniform (uncorrected) error | midpoint error |
|---|---|---|
| 8 | 18.0% | 0.39% |
| 24 | 6.2% | 0.044% |
| 32 | 4.6% | 0.025% |
| 96 | 1.6% | 0.003% |
| **empirical order** | **0.98** | **1.99** |

So the uncorrected error does shrink, but only first-order. It needs N ≥ 32 to pass the 5% acceptance test and
N ≈ 150 to reach 1%. The midpoint layout beats that at N=4. Both layouts match their closed-form
discrete-chain predictions to ≤1.1e-4 at every N, so the error is explained entirely by the quadrature
and not by anything else in the plugin. Run time scales roughly linearly with N, which makes refining the
uniform mesh a poor substitute for fixing its quadrature.

### Calibration results (`scripts/calibrate.py`, `out/calibration.png`)

- **Linear regime** (total sag/L < 0.05, self-weight included): sim/Euler–Bernoulli = 0.9960–0.9995 over 10 (E, m) points.
- **Large deflection:** the sim matches the exact Kirchhoff elastica (BVP solve) within 0.11% up to δ/L = 0.87.
  The departure from linear theory is real beam physics, not sim error.
- **Self-weight matters for soft sticks.** At E=1e9 the unloaded stick already sags 12.6% of L, so even a 2 g
  load sits in the nonlinear regime (sim/EB = 0.936). For this reason the linear-regime test uses **total** sag.
- **Inverse:** `young_from_deflection(mass, delta, length, radius)` interpolates the stored dimensionless sim
  master curve δ/L = f(FL²/EI), valid for a weightless rod. Passing `density=` makes it root-find E by
  re-running the bench protocol in sim with self-weight. Both round-trip to <1% (`tests/test_cantilever.py`).

## Solver and timestep

`integrator="implicitfast"`, `solver="Newton"`, `cone="elliptic"`, `timestep=1e-3`, `<flag island="enable"/>`.

- dt=1e-3 is stable across E = 1e9…2e11 Pa and N = 8…40. The sweep checked dt ∈ {1e-3, 5e-4, 2e-4}, and the
  steady state is identical to 4 digits. The 5e-4 fallback was not needed. One early exploratory script did
  diverge at dt=1e-3, at t=0.043 s. It used a uniform mesh, E=5e9, and an `xfrc_applied` tip load on the last body.
  That divergence was **not root-caused**. The production model never diverged anywhere in the sweep or in the
  100-rollout dataset. If it does reappear, reduce dt; don't reduce E.
- Stick: N=24 elements, r=4 mm, L=0.6 m, ρ=1900 kg/m³ (glass-fibre ballpark), all placeholders until measured.
- Stick damping is a Rayleigh time constant β (s), with joint damping = β·EI/ℓ, so it is mesh-independent and
  scales with E. The default β=2 ms is untuned and should come from the real stick's ring-down.
- The calibration rig uses β=50 ms to settle quickly. Damping does not change statics.
- Rollouts run at ~1.4× real time on one core; a 5.1 s push takes 3.1–3.6 s wall.

## Arm, attachment, control

- **UR5e**; no fallback to Panda was needed.
- The stick mount body sits inside `wrist_3_link` at `attachment_site`, rotated so stick +x = flange +z.
  The first cable body has no joint (`initial="none"`), and the test checks this from the model: `body_jntnum==0`
  and the same `body_weldid` as the flange. `tests/test_attachment.py` asserts grip pose = flange pose to 1e-9 m
  with gravity off and the arm static.
- Arm bodies get `gravcomp="1"`; the stick bodies don't, so the stick still sags.
- **Controller** (`probe/control.py`): damped-least-squares IK (λ=0.03, null-space bias toward `HOME_Q`, joint
  limits clipped) on the **grip** site. The IK runs on a private kinematic copy (q_des), and the Menagerie
  position servos track it.
- **Tuned:** with the stock servos (kp=2000, kv=400, a ~0.2 s lag), grip tracking error was 29 mm. Adding
  velocity feedforward through the same servos, `ctrl = q_des + (kv/kp)·q̇_des`, brought it to 1–2 mm max and
  0.4–1.0 mm RMS. The *achieved* grip pose is what gets logged.
- **Robot placement:** the base sits at x=−0.6 m. At −0.35 m the grip was nearly over the shoulder and IK
  produced an awkward posture.
- A force/torque sensor at the mount (`mount_force`, `mount_torque`) and the summed stick–object contact
  force (`f_contact_obj`) are logged in the stick-local frame as ground truth for curvature→force.

## Scene

- Three identical 10 cm boxes with identical colour. `fixed` is a static body. `heavy` (0.4–0.7 kg, μ 0.5–0.7)
  and `light` (0.05–0.2 kg, μ 0.25–0.4) are free bodies. The table has μ=0.1, so the object's friction dominates.
- **Tuned:** `heavy` started at 0.8–1.5 kg with μ 0.6–0.8 (friction load ~8 N). That exceeds what this stick
  transmits before its tip slides or bows away (~5 N), so `heavy` was physically identical to `fixed` (it moved
  1.5 mm). The new range (μmg ≈ 2–4.8 N) is movable but loads the stick.
- Inactive objects are parked behind the push plane. Pushes stay in the world x–z plane, and the camera
  looks along +y.
- The camera (`probe/camera.py`) is a pinhole with **pixel intrinsics in OpenCV convention** (fx, fy, cx, cy,
  measured from the top-left), so ARKit/Stray values drop in unchanged. The default is an iPad-like 1920×1440 with
  fx=fy=1440 px. `to_mjcf()` emits `focalpixel`/`principalpixel`.
- **MuJoCo's `principalpixel` sign convention isn't stated in the docs, so it was established empirically.**
  The offset is (image centre − principal point) on **both** axes. My first reading (cx − w/2 for x) mirrored the
  image horizontally by 80 px. `tests/test_camera.py` renders a marker for three principal-point settings and
  three points each, and checks it lands within 1 px of the OpenCV projection.
- Stick colours: yellow base band, blue body, orange tip, as on the real needle (`probe.stick.colour_bands`).

## Contact stiffness (fixed after the tracker exposed it)

MuJoCo's `solref` is mass-scaled. With the default timeconst of 0.02 s, the 2.4 g tip segment, loaded by ~5 N
through the stiff rod, sank **~20 mm into the box**. The box geom's `priority=1` also meant the box's default
contact parameters silently overrode the stiffer ones set on the stick. All contacting geoms (stick, boxes, table)
now share `solref=0.002 1, solimp=0.95 0.99 0.001` (`probe/stick.py:CONTACT_SOLREF`). That is stable at dt=1e-3
and gives < 1 mm penetration against a free box.
**This changed the physics:** on the same push, the free box now slides 47 mm instead of 8 mm. Every earlier
number involving box motion was measured with the soft contact, including the 3-class dataset and the
E-sensitivity table below, and should be regenerated before it is quoted.

**Fixed box, deep push: physics, not an artifact.** On a deep push the tip slides down the face (the stick is
pitched 35°, tan 35° ≈ 0.7 ≈ μ) and pins in the box–table corner. The peak contact is the table pushing up
286 N on the tip, with 2.1 mm penetration. I first called this an artifact by comparing it with the
clamped-**free** buckling load (27 N), but once the tip is pinned the stick is clamped-**pinned**:
2.05π²EI/L² ≈ 226 N. So a stiff 8 mm rod jammed in a corner really can carry this. What's unrealistic is the
synthetic trajectory, which keeps driving the grip 6 cm into a jammed stick. A hand would stop well before
200 N. Under that load the position servos yield by up to 18 mm; `tests/test_pipeline.py` now requires < 5 mm
tracking at hand-like loads (peak < 20 N over the last 0.5 s) and < 30 mm overall. While jammed, the corner
contact chatters between 0 and ~160 N (stick-slip). That isn't investigated further; it only occurs in this
over-deep regime, which the estimator's training pushes avoid (see below). Sinking the static box 2 mm into the table (done)
removed a wedge gap the tip used to tunnel through.

## Stick tracker (`probe/tracker.py`, `scripts/track_video.py`, `scripts/validate_tracker.py`)

The real probe (a taped 3 mm knitting needle) is painted in solid colours, measured from the clamp exit: yellow
0-69 mm, blue 69-294 mm, orange 294-334 mm. The sim stick is painted the same (`probe.stick.colour_bands`).
Each frame goes through these steps:
1. HSV masks for the three colours. The blue body is the largest blue component (nearest the last fix).
2. A provisional cubic centreline through the blue blob's slice centroids, extended linearly past both ends.
   The end with yellow beside it is the grip end.
3. Every 1 px along that curve, the across-curve profile of each mask (bilinear, 1/4 px) gives the centre and
   the visible width. One estimator for all colours. Yellow and orange points are collected walking outward
   from the blue only while their width matches the needle's, so a yellow block, the hand or a box edge
   touching the needle ends the walk instead of dragging the fit.
4. Anchors: the yellow/blue and blue/orange boundaries (blue width crossing half its median, sub-sample).
   They sit at known arc lengths; the grip (hidden in the block/gripper) and the tip are reached by
   extrapolating the fit, never from the needle's visible ends.
5. Back-projection onto the push plane, a **cubic** in-plane fit, arc length scaled so the blue section has
   its measured length. If that scale is >3% off, the stick has left the push plane (a monocular camera
   can't recover it) and the frame is dropped with that reason.

**Validation** (sim at 1920x1440 through H.264; 153 frames per case; ~50 ms/frame, 155 ms on fixed cases):

| case | world centreline | grip | grip axis | **stick-local tip** mean (p95) | κ RMS (true max) | dropped |
|---|---|---|---|---|---|---|
| freespace, square cam | 0.5 mm | 0.8 mm | 0.17° | **1.2 mm** (1.7) | 0.018 /m (0.04) | 0 |
| light, square | 0.6 | 0.8 | 0.17° | **1.5** (2.5) | 0.020 (0.35) | 0 |
| heavy, square | 0.8 | 0.8 | 0.18° | **1.9** (5.3) | 0.031 (0.91) | 0 |
| heavy, oblique cam | 1.0 | 1.0 | 0.19° | **1.9** (6.4) | 0.034 (0.91) | 0 |
| fixed, square | 1.6 | 0.9 | 0.24° | **4.1** (19.4) | 0.094 (1.43) | 52 |
| fixed, oblique | 1.5 | 1.0 | 0.29° | **3.4** (16.4) | 0.111 (1.43) | 55 |

The heavy p95 comes from brief sideways swings of the thin needle at contact (up to 7.5 mm out of plane at
t = 3.2 s), which a single camera folds back into the plane. Against the fixed box the needle buckles
sideways and slides off the box face in about a third of the frames; those are the dropped ones. The real
anchored-box pushes should be gentle for the same reason. `tests/test_tracker.py` guards accuracy on rendered
frames, a yellow block touching the band, and the out-of-plane rejection.

**History:** the first tracker followed black/white stripes with a Hessian ridge trace (validation 1.5-2 mm
local tip on the old 600 mm rod). Its main lessons carried over: one centre estimator everywhere (blob
centroids vs ridge peaks disagreed by ~1 px, i.e. ~0.7° over 8 cm), a cubic rather than degree 4-5 fit, and
anchoring arc length on interior boundaries.

**For real video:** markings default to the measured needle; HSV thresholds are config. Tune them with
`--debug N`, which saves overlays with the length scale and visible width. Frames that are lost or out of plane
are dropped and counted, never guessed; a median length scale >3% off warns about the plane depth or markings.

## Matching the real camera (Stray Scanner export)

`probe/arkit.py` and `scripts/camera_from_stray.py`:
- **Format:** checked against Stray's own loader (StrayVisualizer). `camera_matrix.csv` holds K for the RGB
  frame. `odometry.csv` (`timestamp, frame, x, y, z, qx, qy, qz, qw`) is camera-to-world, with OpenCV camera
  axes. Depth is 256×192 uint16 in mm, and confidence levels are 0/1/2.
- **World frame:** assumed to be ARKit's (+y up). That's ARKit's documented convention, but Stray doesn't state
  it, so `gravity_check` verifies it on every export and the CLI warns if image-up doesn't point along world +y.
- **Tripod pose:** median position and mean rotation over frames. The report gives the maximum deviation, so a
  bumped tripod or ARKit drift is visible.
- **Alignment to sim:** z = ARKit up. The translation comes from one clicked pixel with known sim position
  (default: centre of the box's camera-facing face, back-projected through confident LiDAR depth). Yaw comes
  from an optional second pixel on the push line. Without it, the camera is assumed square to the plane and the
  CLI says so. The sign of the push direction is resolved from the camera direction, so it doesn't matter which
  side the second point is on.
- **Yaw sensitivity:** yaw comes from two LiDAR points, so their separation matters. With a synthetic export and
  only 1 mm depth quantisation, separations of 0.1 / 0.3 / 0.6 m gave 7.7 / 0.4 / 1.0 px maximum reprojection
  error at 1920×1440. **Put the axis mark ≥ 0.3 m from the anchor.**
- **Tested:** `tests/test_arkit.py` builds a Stray-format export from a known camera with ARKit yaw and offset,
  off-centre principal point, ray-cast 1 mm depth and tripod jitter. It recovers position to < 3 mm, rotation
  to < 0.3° and reprojection to < 3 px, and it checks that a wrong world convention gets flagged.
- **Not yet tested on a real export.**

## Extraction and features

- Node positions come from the model (`find_stick` walks the body tree under `stick_mount`), and nothing
  matches on generated names.
- **Didn't work:** smoothing-spline curvature. Its natural boundary conditions force κ=0 at the clamp, which
  is the maximum-curvature point, and at low smoothing it fits polyline kinks (23–45% error). It was replaced by
  Savitzky–Golay on the tangent angle (1.1% RMS error).
- **Tuned:** rollouts pre-settle for 1 s at the start pose before t=0. Otherwise the straight-to-sagged
  transient dominated the `load_slope` feature, which came out identical for every class.
- The features use only what video also provides (centreline + grip trajectory), never sim force channels.

## Classifier (`scripts/classify.py`, `out/confusion.png`)

100 rollouts (25 per class), randomised over push speed, contact height, pitch (25–45°), push depth, and
object mass/friction within class. Nine features feed a standardised logistic regression.

| input | held-out acc (n=30) | 5-fold CV |
|---|---|---|
| clean sim centreline | 96.7% | 98.0% ± 2.4% |
| + 1 mm iid tracker noise | 83.3% | 82.0% ± 7.5% |

Under noise, the errors are almost all light ↔ freespace. A 0.1 kg object bends the stick by only ~0.03 /m.

### Sim → real evaluation (`classify.py --real DIR`)

Train on all sim and test on every `meta.source == "video"` observation under DIR. The report gives real accuracy
with an exact (Clopper–Pearson) 95% CI, the gap (sim 5-fold CV − real accuracy), per-class accuracy, untested
classes, a confusion matrix, and a per-feature shift (real − sim mean, in sim std) showing *which* features
don't transfer. With 12 clips the CI is roughly ±30 points, and the report shows it.

**Smoke test on pseudo-real data.** These are 12 unseen sim rollouts with deliberate mismatch, kept out of
`data/` and `out/`. They separate the two failure sources:

| pseudo-real | real acc [95% CI] | gap |
|---|---|---|
| E 25% low, no noise | 75% [43–95%] | +23 pts |
| correct E, 1.5 mm iid tracker noise | 17% [2–48%] (chance 25%) | +81 pts |
| both | 17% [2–48%] | +81 pts |
| both, sim trained with 1.5 mm noise | 42% [15–72%] | +28 pts (sim CV drops to 70%) |

**Tracker noise, not physics mismatch, is what breaks transfer.** The derivative and near-zero features blow up
under noise: `load_slope` (dκ/dt) shifts by +77σ and `residual_bend` by +450σ (its sim spread is ~0), while
`peak_tip_drop` is unaffected. This is the main thing to fix before real data: smoothing in time before
differentiating, and noise-robust replacements for those two features. Note that iid per-point noise is a
pessimistic model of a real tracker, whose errors are temporally correlated.

**The smoke test found two contract checks that would have rejected real video,** and both are now fixed. The
"leaves the grip along +x" check used the first 15 mm segment, which 1.5 mm noise tilts by up to ~20°; it now
uses the chord to ~10% of the length. The length check summed every short segment, and noise inflates that sum
by ~2σ²/ds² (3.6% at 2 mm); it now uses ~10 coarse segments. Both checks still catch genuine frame or scale
errors, and `test_noisy_tracked_centreline_is_accepted` guards against regressions.

## Finding: what tip error does and doesn't test

Smoke test: the same push run with E = 2.0e10 vs 1.6e10 (−20%).

| condition | tip RMS diff | why |
|---|---|---|
| fixed | 1.6 mm | tip pinned against a rigid face: **displacement-controlled**, so shape is set by geometry; E only scales force (5.6 → 4.9 N) |
| heavy | 9.2 mm (max 20.6 mm) | friction-limited: a softer stick pushes the box 9.6 mm instead of 24.4 mm |
| freespace | 0.9 mm | quasi-static self-weight sag |

Tip error is therefore a strong fidelity metric for contact and geometry. **It is weak evidence about E
unless the clip includes force-limited contact (heavy/light objects) or dynamics.** The cantilever bench
measurement is what pins E.

## Not done or open

- `probe/residual.py` is an interface plus a zero baseline only, by design.
- The video tracker doesn't exist yet. It must write the trajectory CSV and the observation NPZ above.
- The features aren't noise-robust yet (see Sim → real). This should be fixed before reading anything into real-data accuracy.
- Stick damping β, density and radius are placeholders until measured.
