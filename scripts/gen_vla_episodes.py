"""Generate training-sim episodes for "push the heaviest/lightest box" and write them as LeRobot datasets.

Run with the VLA environment (LeRobot + SmolVLA + the same pinned MuJoCo):
    MUJOCO_GL=egl .venv-vla/bin/python scripts/gen_vla_episodes.py --n 1000 --out data/vla/heaviest_v1

Writes two datasets from the same episodes and images:
  <out>/full        state = grip pose + needle bending + per-box estimator summary (22)
  <out>/images_only state = grip pose only (7): the ablation (identical-looking boxes; the policy can only guess)
plus <out>/episodes.jsonl (masses, instruction, the expert's choice and whether it was right).
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
CKPT = REPO / "checkpoints" / "estimator" / "gru-n5000-1006-1818" / "best.pt"
_EST = None


def _episode(args):
    seed, ckpt = args
    global _EST
    os.environ.setdefault("MUJOCO_GL", "egl")
    from probe.multibox import Estimator, run_episode, sample_episode
    if _EST is None:
        import torch
        torch.set_num_threads(1)
        _EST = Estimator(ckpt)
    try:
        return seed, run_episode(sample_episode(seed), _EST, rng=np.random.default_rng(seed + 10**6)), None
    except Exception as e:  # recorded, never silently dropped
        return seed, None, f"{type(e).__name__}: {e}"[:300]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()

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
    done, failed, correct, t0 = 0, 0, 0, time.time()
    ctx = mp.get_context("spawn")
    while pending:
        batch, pending = pending[:a.workers * 4], pending[a.workers * 4:]
        try:
            with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex:
                for seed, ep, err in ex.map(_episode, [(s, a.ckpt) for s in batch]):
                    if ep is None:
                        failed += 1
                        meta.write(json.dumps({"seed": seed, "ok": False, "error": err}) + "\n")
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
                                                                 "truth", "correct", "p_best", "probes")}}) + "\n")
                    meta.flush()
        except BrokenProcessPool:
            pending = batch + pending  # finished ones are already written; rerun the rest of the batch
            done_seeds = {json.loads(l)["seed"] for l in open(out / "episodes.jsonl")}
            pending = [s for s in pending if s not in done_seeds]
            print(f"  worker crashed; restarting with {len(pending)} episodes left", flush=True)
        el = time.time() - t0
        print(f"  {done}/{a.n} episodes ({failed} failed) in {el:.0f}s, expert correct {correct}/{max(done, 1)}",
              flush=True)
    full.finalize()
    ablate.finalize()
    meta.close()
    print(f"wrote {out}/full and {out}/images_only: {done} episodes, {failed} failed, "
          f"expert accuracy {correct / max(done, 1):.1%}")


if __name__ == "__main__":
    main()
