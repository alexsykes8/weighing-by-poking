"""Rerun eval_vla.run for chosen seeds, additionally saving per-frame (estimator summary, box pixel positions).

Same loop as scripts/eval_vla.py:run, plus a trace. Usage:
  MUJOCO_GL=egl .venv-vla/bin/python <this> --policy ... --seeds 100000 100002 --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import eval_vla as ev  # noqa: E402
from eval_vla import (GripIK, build_row_scene, extract_frame, arm_actuator_ids, arm_joint_qpos_adr,  # noqa: E402
                      needle_features, push_start_pose, stick_geoms, sample_episode, Estimator, load_policy)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def project(m, d, cam_id, pts, W, H):
    pos, R = d.cam_xpos[cam_id], d.cam_xmat[cam_id].reshape(3, 3)
    f = 0.5 * H / np.tan(np.deg2rad(m.cam_fovy[cam_id]) / 2)
    out = []
    for x in pts:
        c = R.T @ (np.asarray(x) - pos)
        out.append([W / 2 + f * c[0] / -c[2], H / 2 - f * c[1] / -c[2]])
    return out


def run(policy, pre, post, cfg, est, images_only, rate=15.0, max_t=30.0):
    m, d, h, boxes = build_row_scene(cfg)
    sc, st, style = cfg.scene, cfg.stick, cfg.style
    hx = sc.object_half[0]
    W, H = sc.camera.resolution
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, sc.camera.name)
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, W), max(m.vis.global_.offheight, H)
    rend = mujoco.Renderer(m, H, W)
    act_ids, qadr = arm_actuator_ids(m), arm_joint_qpos_adr(m)
    ik = GripIK(m, h.grip_site, home=d.qpos[qadr].copy())
    p0, R = push_start_pose(d.xpos[boxes[0]][0] - hx, cfg.ys[0], st, style, sc)
    q0 = np.zeros(4)
    mujoco.mju_mat2Quat(q0, R.flatten())
    ik.converge(p0, q0)
    d.qpos[qadr] = ik.q
    d.ctrl[act_ids] = ik.q
    mujoco.mj_forward(m, d)
    while d.time < 1.0:
        mujoco.mj_step(m, d)
    d.time = 0.0
    sgeo = set(stick_geoms(m, h))
    box_geo = [{g for g in range(m.ngeom) if m.geom_bodyid[g] == b} for b in boxes]
    arclen = np.linspace(0, st.length, 21)
    summary = np.zeros((3, 3))
    buf = {"t": [], "grip": [], "cl": []}
    contact_on, last_contact, contact_start = [False] * 3, [-1e9] * 3, [0.0] * 3
    contacts_after_probing = []
    target_prev = np.concatenate([p0, q0])
    target = target_prev.copy()
    t_cmd, next_est = 0.0, 0.0
    policy.reset()
    frames, trace = [], []
    while d.time < max_t:
        if d.time + 1e-9 >= t_cmd:
            mujoco.mj_forward(m, d)
            cl, _, gp, _ = extract_frame(m, d, h, arclen)
            state = np.concatenate([gp, needle_features(cl, arclen), summary.ravel()]).astype(np.float32)
            if images_only:
                state = state[:7]
            rend.update_scene(d, camera=sc.camera.name)
            img = rend.render()
            frames.append(img.copy())
            top = [d.xpos[b] + np.array([0, 0, sc.object_half[2]]) for b in boxes]
            trace.append({"t": float(d.time), "summary": summary.tolist(),
                          "chosen": contacts_after_probing[0] if contacts_after_probing else -1,
                          "box_px": project(m, d, cam_id, top, W, H)})
            obs = {"observation.images.camera1": torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0,
                   "observation.state": torch.from_numpy(state)[None], "task": [cfg.instruction]}
            with torch.inference_mode():
                a = post(policy.select_action(pre(obs)))
            target_prev, target = target, a[0].detach().cpu().numpy().astype(float)
            target[3:] /= np.linalg.norm(target[3:])
            t_cmd += 1.0 / rate
        u = np.clip(1.0 - (t_cmd - d.time) * rate, 0.0, 1.0)
        p = (1 - u) * target_prev[:3] + u * target[:3]
        d.ctrl[act_ids] = ik.solve(p, target[3:])
        if d.time + 1e-9 >= next_est:
            mujoco.mj_forward(m, d)
            cl, _, gp, _ = extract_frame(m, d, h, arclen)
            buf["t"].append(d.time); buf["grip"].append(gp); buf["cl"].append(cl)  # noqa: E702
            next_est = d.time + 1.0 / 30
        touching = [False] * 3
        for c in d.contact[:d.ncon]:
            g1, g2 = c.geom1, c.geom2
            for i in range(3):
                if (g1 in sgeo and g2 in box_geo[i]) or (g2 in sgeo and g1 in box_geo[i]):
                    touching[i] = True
        for i in range(3):
            if touching[i]:
                if not contact_on[i]:
                    contact_on[i], contact_start[i] = True, d.time
                    if summary[:, 0].all():
                        contacts_after_probing.append(i)
                last_contact[i] = d.time
            elif contact_on[i] and d.time - last_contact[i] > 0.3:
                contact_on[i] = False
                if not summary[i, 0]:
                    t = np.array(buf["t"])
                    sel = (t >= contact_start[i] - 1.5) & (t <= d.time)
                    if sel.sum() > 20:
                        logm, sd, _ = est(t[sel], np.array(buf["grip"])[sel], np.array(buf["cl"])[sel], arclen, st,
                                          cfg.friction)
                        summary[i] = (1.0, logm, sd)
        if contacts_after_probing and d.time - contact_start[contacts_after_probing[0]] > 2.0:
            break
        mujoco.mj_step(m, d)
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            break
    rend.close()
    truth = int(np.argmax(cfg.masses) if cfg.goal == "heaviest" else np.argmin(cfg.masses))
    chosen = contacts_after_probing[0] if contacts_after_probing else -1
    return frames, {"goal": cfg.goal, "task": cfg.instruction, "masses": list(cfg.masses), "chosen": chosen,
                    "truth": truth, "success": chosen == truth,
                    "t_choice": float(contact_start[chosen]) if chosen >= 0 else None, "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    policy, pre, post = load_policy(a.policy)
    est = Estimator(str(ev.CKPTS["straight"]))
    for s in a.seeds:
        frames, r = run(policy, pre, post, sample_episode(s, "straight"), est, False)
        np.save(out / f"seed{s}_frames.npy", np.stack(frames))
        (out / f"seed{s}.json").write_text(json.dumps(r))
        print(s, r["goal"], "chose", r["chosen"], "truth", r["truth"], "OK" if r["success"] else "MISS", flush=True)


if __name__ == "__main__":
    main()
