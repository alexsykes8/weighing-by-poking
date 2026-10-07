"""Drive the sim-IRL UR5e with the person's real grip paths, then score the sim against the video.

    MUJOCO_GL=egl python scripts/replay_real.py data/usable/straight_*/*/track.npz --out out/sim_irl

For each tracked clip: the grip trajectory recorded from the video (<clip>.grip.csv) is followed by the arm in
the sim-IRL (measured needle, the clip's box mass, measured friction 0.55), the needle and box respond
physically, and the simulated needle is compared with the tracked one frame by frame (scripts/compare.py: tip
error in the stick-local frame). Straight pokes (needle steeper than 70 deg) mount the needle along the
gripper's approach axis and are lifted, if needed, so the unbent tip clears the table by 3 mm (the tracked
grip height carries the camera alignment's ~10 mm vertical error; see probe/simdata.lift_to_clear_table).
Writes <out>/<clip>.sim.npz, <out>/<clip>/ (compare plots) and <out>/summary.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from probe.estimator import grip_features
from probe.io import load_observation, load_trajectory, save_observation
from probe.rollout import run_rollout
from probe.scene import SceneConfig
from probe.simdata import lift_to_clear_table
from probe.stick import StickParams


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obs", nargs="+", help="tracked clips (.npz); the grip path is the .grip.csv beside each")
    ap.add_argument("--friction", type=float, default=0.55, help="box-on-drape static mu (tilt test)")
    ap.add_argument("--clearance", type=float, default=0.003)
    ap.add_argument("--out", default=str(_bootstrap.REPO / "out" / "sim_irl"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in a.obs:
        path = Path(path).resolve()                      # usable/ links resolve to data/runs/real
        name = path.stem
        video = load_observation(path)
        traj = load_trajectory(path.with_suffix(".grip.csv"))
        pitch = float(np.median(np.degrees(grip_features(video.t, video.grip_pose)[:, 4])))
        straight = pitch > 70
        stick = StickParams()
        scene = SceneConfig(**({"exit_angle_deg": 0.0} if straight else {}))
        z0 = traj.pos[:, 2].copy()
        if straight:
            traj = lift_to_clear_table(traj, stick, scene, a.clearance)
        lift = float((traj.pos[:, 2] - z0).max())
        mass = video.meta["object_mass"]
        try:
            sim = run_rollout(traj, "free", stick, scene, mass=mass, friction=a.friction, K=len(video.arclen),
                              record_hz=1.0 / float(np.median(np.diff(video.t))),
                              extra_meta={"replay_of": str(path), "lift_m": lift})
        except Exception as e:  # recorded, never silently dropped
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            rows.append({"clip": name, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        sim_path = out / f"{name}.sim.npz"
        save_observation(sim_path, sim)
        subprocess.run([sys.executable, str(Path(__file__).with_name("compare.py")), str(path), str(sim_path),
                        "--out", str(out / name), "--names", "video", "sim-IRL"], check=True,
                       stdout=subprocess.DEVNULL)
        s = json.loads((out / name / "summary.json").read_text())
        peak = lambda o: float(np.percentile(np.abs(o.centreline[:, -1, 2]), 98) * 1e3)  # noqa: E731
        r = {"clip": name, "ok": True, "mass_kg": mass, "style": "straight" if straight else "angled",
             "pitch_deg": pitch, "lift_mm": 1e3 * lift, "tip_err_mm_mean": s["tip_err_mm"]["mean"],
             "tip_err_mm_rms": s["tip_err_mm"]["rms"], "tip_err_mm_p95": s["tip_err_mm"]["p95"],
             "centreline_err_mm_mean": s["centreline_err_mm_mean"],
             "peak_tip_dev_video_mm": peak(video), "peak_tip_dev_sim_mm": peak(sim), "n": s["n"]}
        rows.append(r)
        print(f"{name}: {1e3 * mass:.0f} g {r['style']} lift {r['lift_mm']:.0f} mm | tip error mean "
              f"{r['tip_err_mm_mean']:.1f} mm (p95 {r['tip_err_mm_p95']:.1f}) | peak tip deflection video "
              f"{r['peak_tip_dev_video_mm']:.0f} vs sim {r['peak_tip_dev_sim_mm']:.0f} mm", flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
