"""Validate the 2D stick tracker on rendered sim video against simulator ground truth.

    python scripts/validate_tracker.py                    # all cases, full 1920x1440, -> out/tracker_validation.*
    python scripts/validate_tracker.py --cases fixed:square --width 960

Each case renders a rollout through the scene camera to H.264 mp4 (as Stray Scanner records), records the
true stick state at exactly the rendered frames, runs the tracker on the decoded video, and reports:
  world centreline / tip / grip error on the push plane (mm), grip-axis angle error (deg),
  stick-local tip error (mm, the evaluation metric's frame), curvature error (1/m), frames dropped.
"""
import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import _bootstrap  # noqa: E402,F401
import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from probe.camera import CameraConfig  # noqa: E402
from probe.io import load_trajectory  # noqa: E402
from probe.rollout import extract_frame, run_rollout  # noqa: E402
from probe.scene import SceneConfig  # noqa: E402
from probe.stick import StickParams, colour_bands, find_stick  # noqa: E402
from probe.tracker import (Plane, StickMarkings, StickTracker, TrackerConfig, fix_to_samples,  # noqa: E402
                           fixes_to_observation)

OUT = _bootstrap.REPO / "out" / "tracker"
TRAJ = _bootstrap.REPO / "data" / "trajectories" / "synthetic_push.csv"
CAMERAS = {
    "square": CameraConfig(),  # default: square to the push plane, 1.2 m away
    # tripod off to one side, higher, pitched down, off-centre principal point: a realistic imperfect setup
    "oblique": CameraConfig(pos=(-0.05, -1.05, 1.15), lookat=(0.2, 0.0, 0.88), fx=1452.0, fy=1449.0,
                            cx=951.0, cy=728.0),
}
DEFAULT_CASES = ["fixed:square", "heavy:square", "light:square", "freespace:square", "fixed:oblique", "heavy:oblique"]


class Recorder:
    """frame_callback: renders the camera to mp4 at `fps` and records ground truth at the same instants."""

    def __init__(self, path, cam: CameraConfig, width, fps, arclen):
        self.cam, self.fps, self.arclen = cam, fps, arclen
        self.w, self.h = width, int(round(width * cam.resolution[1] / cam.resolution[0]))
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        self.r = self.hnd = None
        self.next_t = 0.0
        self.gt = {k: [] for k in ("t", "nodes", "grip", "grip_R", "cl_local", "kappa")}

    def __call__(self, m, d, t):
        if t + 1e-9 < self.next_t:
            return
        if self.r is None:
            self.r = mujoco.Renderer(m, self.h, self.w)
            self.hnd = find_stick(m)
        self.r.update_scene(d, camera="ipad")
        self.writer.append_data(self.r.render())
        h = self.hnd
        cl, kappa, grip, _ = extract_frame(m, d, h, self.arclen)
        self.gt["t"].append(t)
        self.gt["nodes"].append(h.nodes_world(d, m))
        self.gt["grip"].append(d.site_xpos[h.grip_site].copy())
        self.gt["grip_R"].append(d.site_xmat[h.grip_site].reshape(3, 3).copy())
        self.gt["cl_local"].append(cl)
        self.gt["kappa"].append(kappa)
        self.next_t += 1.0 / self.fps

    def close(self):
        self.writer.close()
        if self.r is not None:
            self.r.close()


def sim_markings(stick: StickParams, scene: SceneConfig) -> StickMarkings:
    from probe.scene import load_scene
    m, _, h = load_scene(stick, scene, active="freespace")
    b = colour_bands(m, h, stick)
    return StickMarkings(length=h.length, band_end=float(b["band_end"]), tip_start=float(b["tip_start"]))


def run_case(label, cam_name, width, fps, K, temporal):
    cam = CAMERAS[cam_name]
    stick, scene = StickParams(), SceneConfig(camera=cam)
    arclen = np.linspace(0, stick.length, K)
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / f"{label}_{cam_name}.mp4"
    rec = Recorder(video, cam, width, fps, arclen)
    try:
        run_rollout(load_trajectory(TRAJ), label, stick, scene, frame_callback=rec, record_hz=max(100.0, fps))
    finally:
        rec.close()
    gt = {k: np.array(v) for k, v in rec.gt.items()}
    node_s = rec.hnd.node_s

    mk = sim_markings(stick, scene)
    cam_px = cam.scaled(rec.w, rec.h)
    tracker = StickTracker(cam_px, TrackerConfig(markings=mk, plane=Plane()))
    t0 = time.time()
    fixes = [tracker.track(f) for f in imageio.get_reader(video)]
    track_s = (time.time() - t0) / len(fixes)
    n = min(len(fixes), len(gt["t"]))
    fixes, gt = fixes[:n], {k: v[:n] for k, v in gt.items()}

    res = {"case": f"{label}:{cam_name}", "frames": n, "resolution": [rec.w, rec.h], "ms_per_frame": 1e3 * track_s}
    ok = np.array([f.ok for f in fixes])
    res["frames_dropped"] = int((~ok).sum())
    # world-frame errors on the plane, per valid frame
    gt_w = np.array([np.column_stack([np.interp(arclen, node_s, nd[:, i]) for i in range(3)]) for nd in gt["nodes"]])
    tr_w = np.array([fix_to_samples(f, arclen) if f.ok else np.full((K, 3), np.nan) for f in fixes])
    e_cl = np.linalg.norm(tr_w - gt_w, axis=2) * 1e3
    e_tip = e_cl[:, -1]
    e_grip = np.array([np.linalg.norm(f.grip - g) * 1e3 if f.ok else np.nan for f, g in zip(fixes, gt["grip"])])
    e_ang = np.array([np.degrees(np.arccos(np.clip(f.tangent @ R[:, 0], -1, 1))) if f.ok else np.nan
                      for f, R in zip(fixes, gt["grip_R"])])
    scale = np.array([f.scale for f in fixes if f.ok])

    def stats(x):
        x = x[np.isfinite(x)]
        return {"mean": float(x.mean()), "rms": float(np.sqrt((x**2).mean())), "p95": float(np.percentile(x, 95)),
                "max": float(x.max())}

    res["world_centreline_mm"] = stats(e_cl.ravel())
    res["world_tip_mm"] = stats(e_tip)
    res["grip_pos_mm"] = stats(e_grip)
    res["grip_axis_deg"] = stats(e_ang)
    res["length_scale"] = {"median": float(np.median(scale)), "min": float(scale.min()), "max": float(scale.max())}

    # stick-local (evaluation frame) and curvature, through the full contract pipeline
    for tw in sorted({0, temporal}):
        obs, _ = fixes_to_observation(gt["t"], fixes, mk, Plane(), label,
                                      {"young": stick.young, "radius": stick.radius, "length": stick.length,
                                       "timestep": 1 / fps, "git_sha": "validation"}, K=K, temporal_window=tw)
        gl, gk = gt["cl_local"][ok], gt["kappa"][ok]
        key = "raw" if tw == 0 else f"temporal{tw}"
        res[f"local_tip_mm_{key}"] = stats(np.linalg.norm(obs.tip - gl[:, -1], axis=1) * 1e3)
        res[f"local_centreline_mm_{key}"] = stats(np.linalg.norm(obs.centreline - gl, axis=2).ravel() * 1e3)
        res[f"curvature_rms_per_m_{key}"] = float(np.sqrt(np.mean((obs.curvature - gk) ** 2)))
        res[f"gt_curvature_range_per_m"] = float(np.abs(gk).max())
        if tw == 0:
            res["_series"] = {"t": gt["t"][ok].tolist(), "local_tip_mm": (np.linalg.norm(obs.tip - gl[:, -1], axis=1)
                                                                          * 1e3).tolist(),
                              "world_tip_mm": e_tip[ok].tolist(), "gt_tip_drop_mm": (-gl[:, -1, 2] * 1e3).tolist()}
    # overlay on the most-bent frame for a visual check
    i = int(np.nanargmax(np.abs(gt["kappa"]).max(axis=1) * ok))
    frame = imageio.get_reader(video).get_data(i)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.imshow(frame)
    gt_px = cam_px.project(gt_w[i])
    ax.plot(gt_px[:, 0], gt_px[:, 1], "-", color="#1baf7a", lw=1.2, label="ground truth (projected)")
    if fixes[i].ok:
        ax.plot(fixes[i].px[:, 0], fixes[i].px[:, 1], "--", color="#eb6834", lw=1.2, label="tracker")
    lo, hi = np.minimum(gt_px.min(0), fixes[i].px.min(0)) - 40, np.maximum(gt_px.max(0), fixes[i].px.max(0)) + 40
    ax.set(xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]), title=f"{label}:{cam_name} t={gt['t'][i]:.2f}s  "
           f"world tip err {e_tip[i]:.1f} mm")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(OUT / f"{label}_{cam_name}_overlay.png", dpi=130)
    plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=DEFAULT_CASES, help="label:camera, camera in %s" % list(CAMERAS))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--k", type=int, default=41)
    ap.add_argument("--temporal", type=int, default=5, help="also evaluate a temporal smoothing window (frames)")
    a = ap.parse_args()

    results = []
    for c in a.cases:
        label, cam = c.split(":")
        r = run_case(label, cam, a.width, a.fps, a.k, a.temporal)
        results.append(r)
        print(f"{r['case']:18s} {r['resolution'][0]}x{r['resolution'][1]} drop {r['frames_dropped']}/{r['frames']} | "
              f"world: centreline {r['world_centreline_mm']['mean']:.1f} mm (p95 {r['world_centreline_mm']['p95']:.1f}), "
              f"tip {r['world_tip_mm']['mean']:.1f}, grip {r['grip_pos_mm']['mean']:.1f} mm, axis "
              f"{r['grip_axis_deg']['mean']:.2f} deg | local tip {r['local_tip_mm_raw']['mean']:.1f} mm "
              f"(p95 {r['local_tip_mm_raw']['p95']:.1f}; t{a.temporal}: {r[f'local_tip_mm_temporal{a.temporal}']['mean']:.1f}) | "
              f"kappa rms {r['curvature_rms_per_m_raw']:.3f}/m (gt max {r['gt_curvature_range_per_m']:.2f}) | "
              f"scale {r['length_scale']['median']:.3f} | {r['ms_per_frame']:.0f} ms/frame", flush=True)
    (OUT / "tracker_validation.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT/'tracker_validation.json'} and overlays in {OUT}")


if __name__ == "__main__":
    main()
