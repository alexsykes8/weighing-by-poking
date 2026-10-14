"""Generate training-sim episodes for "push the heaviest/lightest box" and write them as LeRobot datasets.

Run with the VLA environment (LeRobot + SmolVLA + the same pinned MuJoCo):
    MUJOCO_GL=egl .venv-vla/bin/python scripts/gen_vla_episodes.py --n 1000 --out data/vla/heaviest_v1
    MUJOCO_GL=egl .venv-vla/bin/python scripts/gen_vla_episodes.py --style straight --n 1000 --out data/vla/straight_v1

Writes two datasets from the same episodes and images:
  <out>/full        state = grip pose + needle bending + per-box estimator summary (22)
  <out>/images_only state = grip pose only (7): the ablation (identical-looking boxes; the policy can only guess)
plus <out>/episodes.jsonl (masses, instruction, the expert's choice and whether it was right).
v2 options (straight_v2): --shuffle-order probes the boxes in a random order (the arm still starts at box 0);
--recovery-frac F makes that fraction recovery demos (probe unrecorded, drift to a hover pose 50-180 mm off,
record from there: go to the chosen box and push); --drop-wrong leaves out demos where the expert chose wrong.
v4 option: --replay-frac F replays one of my recorded straight pokes (REPLAY_GLOB, minus REPLAY_EXCLUDE)
in place of the synthetic probe, independently per box with probability F, played 0.6-1.0x real time.
Episodes run in parallel worker processes (fresh pool per batch); the main process writes the datasets.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

REPO = _bootstrap.REPO
CKPTS = {"angled": REPO / "checkpoints" / "estimator" / "gru-n5000-1006-1818" / "best.pt",
         "straight": REPO / "checkpoints" / "estimator" / "gru-n5000-1007-1935" / "best.pt"}
_EST = None
REPLAY_GLOB = "data/replays/straight_train/*.grip.csv"  # TRAIN half of the by-clip split only
REPLAY_EXCLUDE = ("160gS_18cb28b069",)  # untrimmed: includes the needle swinging in from above (pitch 48 deg)


def replay_files():
    import glob
    return sorted(str(Path(f).relative_to(REPO)) for f in glob.glob(str(REPO / REPLAY_GLOB))
                  if not any(x in f for x in REPLAY_EXCLUDE))


def hover_offset(rng):
    """Grip offset from the last retract pose to where a lost policy hovered (50-180 mm sideways/back, lifted)."""
    while True:
        dx, dy = rng.uniform(-0.12, 0.03), rng.uniform(-0.18, 0.18)
        if 0.05 <= np.hypot(dx, dy) <= 0.18:
            return dx, dy, rng.uniform(0.02, 0.11)


def _episode(args):
    seed, ckpt, style, shuffle, recovery_frac, replay_frac = args
    global _EST
    os.environ.setdefault("MUJOCO_GL", "egl")
    from probe.multibox import Estimator, run_episode, sample_episode
    if _EST is None:
        import torch
        torch.set_num_threads(1)
        _EST = Estimator(ckpt)
    try:
        cfg = sample_episode(seed, style)
        rng = np.random.default_rng(seed + 2 * 10**6)  # separate stream: sample_episode's draws are unchanged
        if shuffle:
            cfg.probe_order = tuple(int(i) for i in rng.permutation(3))
        if rng.uniform() < recovery_frac:
            ys = np.asarray(cfg.ys)
            while True:  # keep the hover over the row
                off = hover_offset(rng)
                if ys.min() - 0.1 <= ys[cfg.probe_order[-1]] + off[1] <= ys.max() + 0.1:
                    break
            cfg.recovery_offset = off
        if replay_frac > 0:
            files = replay_files()
            rr = np.random.default_rng(seed + 3 * 10**6)  # own stream: v1-v3 draws above are unchanged
            cfg.probe_replays = tuple((files[int(rr.integers(len(files)))], float(rr.uniform(0.6, 1.0)))
                                      if rr.uniform() < replay_frac else None for _ in range(3))
        return seed, run_episode(cfg, _EST, rng=np.random.default_rng(seed + 10**6)), None
    except Exception as e:  # recorded, never silently dropped
        return seed, None, f"{type(e).__name__}: {e}"[:300]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--style", default="angled", choices=tuple(CKPTS), help="push style (probe/multibox.py)")
    ap.add_argument("--ckpt", help="estimator checkpoint (default: the one trained for --style)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--shuffle-order", action="store_true", help="probe the boxes in a random order")
    ap.add_argument("--recovery-frac", type=float, default=0.0, help="fraction of recovery demos from a hover pose")
    ap.add_argument("--drop-wrong", action="store_true", help="leave out demos where the expert chose wrong")
    ap.add_argument("--replay-frac", type=float, default=0.0,
                    help="per-probe probability of replaying a recorded real straight poke (style straight)")
    a = ap.parse_args()
    a.ckpt = a.ckpt or str(CKPTS[a.style])

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from probe.multibox import ACTION_NAMES, STATE_NAMES
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    img = {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
    act = {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES}
    full = LeRobotDataset.create("local/needle_heaviest_full", fps=a.fps, root=out / "full", robot_type="ur5e_needle",
                                 features={"observation.images.camera1": img, "action": act,
                                           "observation.state": {"dtype": "float32", "shape": (22,),
                                                                 "names": STATE_NAMES}})
    ablate = LeRobotDataset.create("local/needle_heaviest_images_only", fps=a.fps, root=out / "images_only",
                                   robot_type="ur5e_needle",
                                   features={"observation.images.camera1": img, "action": act,
                                             "observation.state": {"dtype": "float32", "shape": (7,),
                                                                   "names": STATE_NAMES[:7]}})
    meta = open(out / "episodes.jsonl", "w")
    pending = list(range(a.seed0, a.seed0 + a.n))
    done, failed, correct, dropped, t0 = 0, 0, 0, 0, time.time()
    ctx = mp.get_context("spawn")
    while pending:
        batch, pending = pending[:a.workers * 4], pending[a.workers * 4:]
        try:
            with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
                jobs = [(s, a.ckpt, a.style, a.shuffle_order, a.recovery_frac, a.replay_frac) for s in batch]
                for seed, ep, err in ex.map(_episode, jobs):
                    if ep is None:
                        failed += 1
                        meta.write(json.dumps({"seed": seed, "ok": False, "error": err}) + "\n")
                        continue
                    if a.drop_wrong and not ep["correct"]:
                        dropped += 1
                        meta.write(json.dumps({"seed": seed, "ok": False, "dropped": "expert wrong",
                                               **{k: ep[k] for k in ("goal", "masses", "chosen", "truth")}}) + "\n")
                        meta.flush()
                        continue
                    for k in range(len(ep["state"])):
                        frame = {"observation.images.camera1": ep["image"][k], "action": ep["action"][k],
                                 "task": ep["task"]}
                        full.add_frame(frame | {"observation.state": ep["state"][k]})
                        ablate.add_frame(frame | {"observation.state": ep["state"][k][:7]})
                    full.save_episode()
                    ablate.save_episode()
                    done += 1
                    correct += int(ep["correct"])
                    meta.write(json.dumps({"seed": seed, "ok": True, "episode_index": done - 1,
                                           **{k: ep[k] for k in ("task", "goal", "masses", "friction", "chosen",
                                                                 "truth", "correct", "p_best", "probes",
                                                                 "probe_order", "recovery_offset")}}) + "\n")
                    meta.flush()
        except BrokenProcessPool:
            pending = batch + pending  # finished ones are already written; rerun the rest of the batch
            meta.flush()
            done_seeds = {json.loads(l)["seed"] for l in open(out / "episodes.jsonl")}
            pending = [s for s in pending if s not in done_seeds]
            print(f"  worker crashed; restarting with {len(pending)} episodes left", flush=True)
        el = time.time() - t0
        print(f"  {done}/{a.n} episodes ({failed} failed, {dropped} dropped) in {el:.0f}s, "
              f"expert correct {correct}/{max(done, 1)}",
              flush=True)
    full.finalize()
    ablate.finalize()
    meta.close()
    print(f"wrote {out}/full and {out}/images_only: {done} episodes, {failed} failed, {dropped} dropped, "
          f"expert accuracy {correct / max(done, 1):.1%}")


if __name__ == "__main__":
    main()
