"""Write synthetic push trajectories in the grip-trajectory CSV format (docs/data_contract.md).

    python scripts/make_synthetic_trajectory.py                         # default push -> data/trajectories/synthetic_push.csv
    python scripts/make_synthetic_trajectory.py --random 5 --seed 1     # 5 randomised pushes
"""
import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from probe.io import save_trajectory
from probe.trajectory import PushParams, make_push, push_meta

OUT = _bootstrap.REPO / "data" / "trajectories"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", type=int, default=0, help="number of randomised pushes (0: one default push)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pitch", type=float, help="stick pitch below horizontal, deg")
    ap.add_argument("--push-depth", type=float, help="m past first contact")
    ap.add_argument("--push-speed", type=float, help="m/s")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.random:
        rng = np.random.default_rng(a.seed)
        jobs = [(f"synthetic_push_s{a.seed}_{i:03d}", PushParams.sample(rng)) for i in range(a.random)]
    else:
        kw = {k: v for k, v in dict(pitch_deg=a.pitch, push_depth=a.push_depth, push_speed=a.push_speed).items()
              if v is not None}
        jobs = [("synthetic_push", PushParams(**kw))]
    for name, pp in jobs:
        path = save_trajectory(f"{a.out}/{name}.csv", make_push(pp))
        path.with_suffix(".json").write_text(json.dumps(push_meta(pp), indent=2))
        print(path)


if __name__ == "__main__":
    main()
