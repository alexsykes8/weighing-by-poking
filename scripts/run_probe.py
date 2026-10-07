"""Drive the UR5e along a grip trajectory, let the stick deform, log a ProbeObservation (.npz).

    python scripts/run_probe.py --trajectory data/trajectories/synthetic_push.csv --object fixed \
        --out data/runs/fixed_demo.npz [--young 2e10] [--mass 0.5 --friction 0.6]
"""
import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np

from probe.io import load_trajectory, save_observation
from probe.rollout import run_rollout
from probe.schema import LABELS
from probe.stick import StickParams


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--object", required=True, choices=LABELS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--young", type=float, default=StickParams.young, help="Young's modulus, Pa")
    ap.add_argument("--length", type=float, default=StickParams.length)
    ap.add_argument("--radius", type=float, default=StickParams.radius)
    ap.add_argument("--density", type=float, default=StickParams.density)
    ap.add_argument("--mass", type=float, help="object mass (default: class mean)")
    ap.add_argument("--friction", type=float, help="object friction (default: class mean)")
    ap.add_argument("--k", type=int, default=41, help="centreline samples")
    a = ap.parse_args()

    stick = StickParams(young=a.young, length=a.length, radius=a.radius, density=a.density)
    t0 = time.time()
    obs = run_rollout(load_trajectory(a.trajectory), a.object, stick, mass=a.mass, friction=a.friction, K=a.k,
                      extra_meta={"trajectory": a.trajectory})
    save_observation(a.out, obs)
    f = np.linalg.norm(obs.extras["f_contact_obj"], axis=1)
    print(f"{a.out}: T={obs.t.size} K={obs.arclen.size} sim {obs.t[-1]:.2f}s in {time.time()-t0:.1f}s wall | "
          f"max |kappa| {np.abs(obs.curvature).max():.3f} 1/m | max tip drop {-obs.tip[:, 2].min()*1e3:.1f} mm | "
          f"max contact force {f.max():.2f} N")


if __name__ == "__main__":
    main()
