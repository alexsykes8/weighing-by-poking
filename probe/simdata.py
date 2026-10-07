"""Randomised sim rollouts for the mass/anchored estimator, cached to disk.

Every range lives in RANGES (logged verbatim to wandb.config). One rollout = one compressed npz:
compact float32 arrays at the video rate, in the same stick-local frame as the data contract, plus the
sampled parameters. Rollouts that diverge or are unreachable are recorded as failures, not silently dropped.
"""
from __future__ import annotations

import json
import time
import traceback
import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

RANGES = {
    # stick: randomised around the measured needle (334 mm, r 1.75 mm, EI 0.31 N m^2 -> E 4.2e10); damping unmeasured
    "young": ["loguniform", 3e10, 6e10],         # Pa (EI 0.22-0.44 N m^2 at r = 1.75 mm)
    "length": ["uniform", 0.32, 0.35],           # m
    "radius": ["uniform", 0.0016, 0.0019],       # m
    "damping": ["loguniform", 5e-4, 8e-3],       # s, Rayleigh beta
    "density": ["fixed", 2180.0],
    # box (real box 129 long x 69 wide x 80 tall mm; real fills 100-520 g, tilt-tested static mu 0.55)
    "mass": ["loguniform", 0.07, 0.8],           # kg; the needle cannot slide much more than ~0.6 kg at mu 0.55
    "friction": ["uniform", 0.40, 0.70],
    "p_anchored": 0.15,
    "p_freespace": 0.10,
    # push: covers the real pushes (~45 deg, 20-23 cm past contact, 50-200 mm/s, contact 2-34 mm on the first
    # 520 g clips) with margin; the later pushes aim for 10 cm at a fixed spot
    "pitch_deg": ["uniform", 30.0, 55.0],
    "push_speed": ["uniform", 0.03, 0.25],       # m/s
    "push_depth": ["uniform", 0.04, 0.20],       # m past first contact ("distance")
    "contact_frac": ["uniform", 0.06, 0.80],     # contact height as a fraction of box height (5-64 mm)
    "hit_offset": ["uniform", -0.015, 0.015],    # m, sideways miss of the face centre (camera can't see it;
                                                 # 10 mm already yaws the box 20-30 deg over a 12 cm push)
    "approach_speed": ["uniform", 0.08, 0.25],
    "hold": ["uniform", 0.3, 0.9],
    # replays: this fraction of free-box rollouts follows a recorded real grip trajectory (the person's actual
    # pushing style) instead of a synthetic push, time-scaled; box, needle and friction randomised as usual
    "p_replay": 0.3,
    "replay_glob": "data/runs/real/*.grip.csv",
    "replay_time_scale": ["loguniform", 0.6, 1.6],
    # recording
    "record_hz": 30.0,
    "K": 21,
}


# Straight pokes (2026-10-07): the needle held ~vertical, tip on the box's side face low down, so the face's
# push is almost all bending and tip friction runs along the needle. Matched to the real straight clips
# (pitch 83-89 deg, 25-50 mm/s, tip 1-25 mm above the table at peak). Boxes up to ~0.35 kg: heavier ones
# need ~2.8 N, ~110 mm of tip deflection, and the tip skids. The needle sits along the gripper's approach
# axis (exit angle 0) so the UR5e can hold it vertical. 30% of free-box rollouts replay a real straight-poke
# grip path from the TRAIN half of a by-clip split (data/replays/straight_split.json); the test clips never enter.
RANGES_STRAIGHT = {
    **RANGES,
    "mass": ["loguniform", 0.06, 0.45],
    "friction": ["uniform", 0.45, 0.65],
    "tip_friction": ["uniform", 0.10, 0.60],    # needle-on-box; barely matters straight down (calibrated)
    "pitch_deg": ["uniform", 80.0, 90.0],
    "push_speed": ["uniform", 0.02, 0.10],
    "push_depth": ["uniform", 0.06, 0.20],
    "contact_frac": ["uniform", 0.02, 0.35],    # 2-28 mm up the 80 mm face
    "approach_speed": ["uniform", 0.05, 0.15],
    "exit_angle_deg": 0.0,
    "p_replay": 0.3,
    "replay_glob": "data/replays/straight_train/*.grip.csv",
    # tracked grip heights carry the camera alignment's ~10 mm vertical error, and a straight poke's tip runs
    # 1-10 mm above the table, so some replays put the (unbent) tip below it. Lift those to this clearance
    "replay_min_tip_clearance": 0.003,
}
STYLES = {"angled": RANGES, "straight": RANGES_STRAIGHT}


def _draw(rng, spec):
    kind = spec[0]
    if kind == "uniform":
        return float(rng.uniform(spec[1], spec[2]))
    if kind == "loguniform":
        return float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
    if kind == "fixed":
        return float(spec[1])
    raise ValueError(kind)


def sample_params(seed: int, ranges=RANGES) -> dict:
    rng = np.random.default_rng(seed)
    u = rng.uniform()
    kind = "fixed" if u < ranges["p_anchored"] else "freespace" if u < ranges["p_anchored"] + ranges["p_freespace"] else "free"
    p = {k: _draw(rng, ranges[k]) for k in ("young", "length", "radius", "damping", "density", "mass", "friction",
                                              "pitch_deg", "push_speed", "push_depth", "contact_frac",
                                              "approach_speed", "hold", "hit_offset", "replay_time_scale")}
    replays = _replay_files(ranges)
    u_rep = rng.uniform()
    p["replay"] = (replays[int(rng.uniform() * len(replays))]
                   if kind == "free" and replays and u_rep < ranges.get("p_replay", 0.0) else "")
    if "tip_friction" in ranges:  # drawn last, so the angled ranges reproduce their old rollouts exactly
        p["tip_friction"] = _draw(rng, ranges["tip_friction"])
    if "exit_angle_deg" in ranges:
        p["exit_angle_deg"] = float(ranges["exit_angle_deg"])
    p.update(seed=seed, kind=kind, anchored=int(kind == "fixed"))
    return p


def _replay_files(ranges) -> list[str]:
    """Recorded grip trajectories (repo-relative, sorted: a seed always picks the same file for a given set)."""
    import glob
    repo = Path(__file__).resolve().parent.parent
    if not ranges.get("replay_glob"):
        return []
    return sorted(str(Path(f).relative_to(repo)) for f in glob.glob(str(repo / ranges["replay_glob"])))


def lift_to_clear_table(traj, stick, scene, clearance: float):
    """Raise a grip path just enough that the unbent stick tip stays `clearance` above the table throughout."""
    from .geometry import quat_wxyz_to_mat
    from .trajectory import GripTrajectory
    ax = np.stack([quat_wxyz_to_mat(q)[:, 0] for q in traj.quat])
    lift = scene.table_height + clearance - (traj.pos[:, 2] + stick.length * ax[:, 2]).min()
    if lift <= 0:
        return traj
    pos = traj.pos.copy()
    pos[:, 2] += lift
    return GripTrajectory(traj.t, pos, traj.quat).validate()


TIMESTEPS = (1e-3, 5e-4)  # on divergence retry with a smaller step; stiffness is never touched


def simulate(p: dict, ranges=RANGES) -> dict:
    """Run one rollout for parameter dict p, retrying at a smaller timestep if it diverges."""
    for i, dt in enumerate(TIMESTEPS):
        try:
            r = _simulate(p, ranges, dt)
            r["params"] = {**p, "timestep": dt}
            return r
        except FloatingPointError:
            if i == len(TIMESTEPS) - 1:
                raise


def _simulate(p: dict, ranges, timestep) -> dict:
    """One rollout for parameter dict p. Returns compact float32 arrays + p."""
    from .rollout import run_rollout
    from .scene import SceneConfig
    from .stick import StickParams
    from .io import load_trajectory
    from .trajectory import PushParams, make_push, replay_variant
    stick = StickParams(young=p["young"], length=p["length"], radius=p["radius"], damping=p["damping"],
                        density=p["density"], tip_friction=p.get("tip_friction"))
    scene = SceneConfig(timestep=timestep, **({"exit_angle_deg": p["exit_angle_deg"]} if "exit_angle_deg" in p else {}))
    pp = PushParams(pitch_deg=p["pitch_deg"], contact_height=p["contact_frac"] * 2 * scene.object_half[2],
                    push_depth=p["push_depth"], push_speed=p["push_speed"], approach_speed=p["approach_speed"],
                    hold=p["hold"], y=p.get("hit_offset", 0.0), rate_hz=60.0)
    label = {"fixed": "fixed", "freespace": "freespace", "free": "free"}[p["kind"]]
    if p.get("replay"):
        repo = Path(__file__).resolve().parent.parent
        traj = replay_variant(load_trajectory(repo / p["replay"]), p["replay_time_scale"], p["hit_offset"])
        if "replay_min_tip_clearance" in ranges:
            traj = lift_to_clear_table(traj, stick, scene, ranges["replay_min_tip_clearance"])
    else:
        traj = make_push(pp, stick, scene)
    obs = run_rollout(traj, label, stick, scene,
                      mass=p["mass"] if label == "free" else None, friction=p["friction"],
                      K=int(ranges["K"]), record_hz=float(ranges["record_hz"]))
    obj = obs.extras.get("obj_pos")
    disp = (obj[:, 0] - obj[0, 0]) if obj is not None else np.zeros(len(obs.t))
    f = lambda a: np.asarray(a, np.float32)  # noqa: E731
    return {"t": f(obs.t), "grip_pose": f(obs.grip_pose), "centreline": f(obs.centreline), "arclen": f(obs.arclen),
            "box_dx": f(disp), "f_contact": f(obs.extras["f_contact_obj"]), "params": p}


def _job(args):
    p, out_dir, ranges = args
    path = Path(out_dir) / f"{p['seed']:06d}.npz"
    if path.exists():
        return {"seed": p["seed"], "ok": True, "cached": True, "wall": 0.0}
    t0 = time.time()
    try:
        r = simulate(p, ranges)  # the style's ranges: replay lift, K, record rate
    except Exception as e:  # recorded, never silently dropped
        return {"seed": p["seed"], "ok": False, "error": f"{type(e).__name__}: {e}"[:300],
                "trace": traceback.format_exc()[-800:], "wall": time.time() - t0}
    p = r["params"]
    tmp = path.with_suffix(".tmp.npz")  # atomic: a worker dying mid-write must not leave a corrupt cache entry
    np.savez_compressed(tmp, **{k: v for k, v in r.items() if k != "params"}, params=np.array(json.dumps(p)))
    tmp.rename(path)
    peak = float(np.abs(np.diff(np.unwrap(np.arctan2(np.diff(r["centreline"][..., 2], axis=1),
                                                       np.diff(r["centreline"][..., 0], axis=1)), axis=1),
                                axis=1)).max() / np.diff(r["arclen"]).mean())
    return {"seed": p["seed"], "ok": True, "cached": False, "wall": time.time() - t0, "peak_kappa": peak,
            "timestep": p["timestep"],
            "box_moved_mm": float(np.abs(r["box_dx"]).max() * 1e3)}


HANG_S = 300.0              # no rollout finishing for this long = a hung pool: kill it and resubmit
TASKS_PER_WORKER = 20      # rollouts per worker before the pool is replaced. A long-lived worker once segfaulted
                           # inside libmujoco (not reproducible for any single seed in a fresh process), so state is
                           # not allowed to build up. Pools are rebuilt per batch: ProcessPoolExecutor's own
                           # max_tasks_per_child deadlocks on CPython 3.12 (it hung a 5000-rollout run)
MAX_CRASHES_IN_FLIGHT = 3  # a seed in flight during this many worker crashes is recorded as failed


def generate(n: int, out_dir, seed0: int = 0, workers: int = 30, progress=None, ranges=RANGES) -> list[dict]:
    """Generate (or reuse cached) rollouts seed0 .. seed0+n-1. Returns per-rollout status records.

    Runs in batches of workers x TASKS_PER_WORKER, each in a fresh process pool. Survives native worker crashes:
    the pool is rebuilt and unfinished seeds resubmitted (finished ones are cached on disk). Seeds that keep
    coinciding with crashes are recorded as failed, never silently dropped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ranges.json").write_text(json.dumps(ranges, indent=2))
    for stale in out_dir.glob("*.tmp.npz"):
        stale.unlink()
    pending = {seed0 + i: (sample_params(seed0 + i, ranges), str(out_dir), ranges) for i in range(n)}
    crashes: dict[int, int] = {}
    status = []
    ctx = mp.get_context("spawn")
    while pending:
        batch = dict(list(pending.items())[:workers * TASKS_PER_WORKER])
        ex = ProcessPoolExecutor(workers, mp_context=ctx)
        futs = {ex.submit(_job, job): seed for seed, job in batch.items()}
        hung = False
        try:
            not_done = set(futs)
            while not_done:
                done, not_done = wait(not_done, timeout=HANG_S, return_when=FIRST_COMPLETED)
                if not done:  # nothing finished in HANG_S: a worker hung (seen 2026-10-07; the seeds ran fine alone)
                    hung = True
                    for proc in list(ex._processes.values()):
                        proc.kill()
                    raise BrokenProcessPool(f"no rollout finished in {HANG_S:.0f} s")
                for fu in done:
                    s = fu.result()  # raises BrokenProcessPool if the pool died
                    pending.pop(futs[fu])
                    status.append(s)
                    if progress is not None:
                        progress(len(status), n, s)
        except BrokenProcessPool:
            running = [futs[f] for f in futs if futs[f] in pending and not f.done()]
            for seed in running:
                crashes[seed] = crashes.get(seed, 0) + 1
                if crashes[seed] >= MAX_CRASHES_IN_FLIGHT:
                    pending.pop(seed)
                    status.append({"seed": seed, "ok": False, "error": "worker crashed (native) while running",
                                   "wall": 0.0})
            print(f"  worker crashed; restarting the pool with {len(pending)} rollouts left", flush=True)
        finally:
            ex.shutdown(wait=not hung, cancel_futures=True)
    with open(out_dir / "status.jsonl", "a") as f:
        for s in status:
            f.write(json.dumps({k: v for k, v in s.items() if k != "trace"}) + "\n")
    return status


def load_cache(out_dir) -> list[dict]:
    """All cached rollouts as dicts of arrays + 'params'."""
    out = []
    for path in sorted(p for p in Path(out_dir).glob("[0-9]*.npz") if ".tmp" not in p.name):
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files if k != "params"}
            d["params"] = json.loads(str(z["params"]))
        out.append(d)
    return out
