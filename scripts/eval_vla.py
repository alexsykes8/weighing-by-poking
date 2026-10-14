"""Closed-loop evaluation of a fine-tuned SmolVLA on "push the heaviest/lightest box".

By default the scenes come from the randomised training-sim (unseen seeds). --world sim-irl uses the measured
world instead: the real needle (StickParams defaults), friction 0.55, and three of the four real boxes
(100/160/230/345 g). --oracle replaces each probe's estimate with the box's true mass (a sensing ceiling).
--unseen-phrasings swaps the instruction for one never used in training (UNSEEN_PHRASINGS); the scenes are unchanged.

    MUJOCO_GL=egl .venv-vla/bin/python scripts/eval_vla.py --policy <run>/checkpoints/last/pretrained_model \
        --n 50 --out out/vla_eval/full [--images-only] [--videos 3]

The policy drives the arm: every 1/15 s it gets the camera image, the state and the instruction and returns the
next grip target, which the arm's IK servos to. The same perception module as the expert keeps the per-box
summary: when the needle's contact with a box ends, the estimator runs on the last few seconds of needle data and
that box's entry is filled in. Episodes use seeds disjoint from training.

Outcome per episode: the first box contacted after all three have been probed is the policy's choice; success if
it is the heaviest (lightest). Also reported: how many boxes it probed, and whether it ever chose at all.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import _bootstrap  # noqa: F401,E402
import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from probe.control import GripIK  # noqa: E402
from probe.multibox import (Estimator, build_row_scene, needle_features, push_start_pose, sample_episode)  # noqa: E402
from probe.rollout import extract_frame  # noqa: E402
from probe.scene import arm_actuator_ids, arm_joint_qpos_adr  # noqa: E402
from probe.stick import StickParams, stick_geoms  # noqa: E402
from probe.trajectory import grip_rotation  # noqa: E402

REAL_MASSES = (0.100, 0.160, 0.230, 0.345)  # the real box set (kg)
ORACLE_SD = 0.1  # log-mass sigma fed with --oracle: a typical confident estimate, not an out-of-range zero


# Instructions never used in training (training uses probe.multibox.INSTRUCTIONS): synonyms, comparatives and
# sentences without the words "heaviest"/"lightest".
UNSEEN_PHRASINGS = {
    "heaviest": ("push the box that weighs the most", "move the heaviest-feeling box", "which one is heavier than "
                 "the other two? push it", "test each box and shove the weightiest one", "push the box that is hardest to move"),
    "lightest": ("push the box that weighs the least", "move the lightest-feeling box", "which one is lighter than "
                 "the other two? push it", "test each box and shove the least weighty one", "push the least heavy box"),
}


def sim_irl(cfg, seed):
    """Turn a sampled episode into the measured world: real needle, measured friction, three real boxes."""
    rng = np.random.default_rng(seed + 4 * 10**6)
    cfg.stick = StickParams()
    cfg.friction = 0.55
    cfg.masses = tuple(float(m) for m in rng.permutation(REAL_MASSES)[:3])
    return cfg


CKPTS = {"angled": _bootstrap.REPO / "checkpoints" / "estimator" / "gru-n5000-1006-1818" / "best.pt",
         "straight": _bootstrap.REPO / "checkpoints" / "estimator" / "gru-n5000-1007-1935" / "best.pt"}


def load_policy(path, device="cuda"):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(path).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(path),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def run(policy, pre, post, cfg, est, images_only, rate=15.0, max_t=30.0, video=None, device="cuda", oracle=False):
    m, d, h, boxes = build_row_scene(cfg)
    sc, st, style = cfg.scene, cfg.stick, cfg.style
    hx = sc.object_half[0]
    W, H = sc.camera.resolution
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, W), max(m.vis.global_.offheight, H)
    rend = mujoco.Renderer(m, H, W)
    act_ids, qadr = arm_actuator_ids(m), arm_joint_qpos_adr(m)
    ik = GripIK(m, h.grip_site, home=d.qpos[qadr].copy())
    # same start as the expert: settled at the first approach pose
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
    contact_on = [False] * 3
    last_contact = [-1e9] * 3
    contact_start = [0.0] * 3
    contacts_after_probing = []
    box_start = np.array([d.xpos[b].copy() for b in boxes])
    box_at_probed, t_all_probed = None, None
    target_prev = np.concatenate([p0, q0])
    target = target_prev.copy()
    t_cmd, next_est = 0.0, 0.0
    policy.reset()
    frames = []
    while d.time < max_t:
        if d.time + 1e-9 >= t_cmd:  # policy step
            mujoco.mj_forward(m, d)
            cl, _, gp, _ = extract_frame(m, d, h, arclen)
            state = np.concatenate([gp, needle_features(cl, arclen), summary.ravel()]).astype(np.float32)
            if images_only:
                state = state[:7]
            rend.update_scene(d, camera=sc.camera.name)
            img = rend.render()
            if video is not None:
                frames.append(img.copy())
            obs = {"observation.images.camera1": torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0,
                   "observation.state": torch.from_numpy(state)[None], "task": [cfg.instruction]}
            with torch.inference_mode():
                a = post(policy.select_action(pre(obs)))
            target_prev, target = target, a[0].detach().cpu().numpy().astype(float)
            target[3:] /= np.linalg.norm(target[3:])
            t_cmd += 1.0 / rate
        # servo toward the latest target (linear blend over the control interval)
        u = np.clip(1.0 - (t_cmd - d.time) * rate, 0.0, 1.0)
        p = (1 - u) * target_prev[:3] + u * target[:3]
        d.ctrl[act_ids] = ik.solve(p, target[3:])
        if d.time + 1e-9 >= next_est:  # needle data for the perception module
            mujoco.mj_forward(m, d)
            cl, _, gp, _ = extract_frame(m, d, h, arclen)
            buf["t"].append(d.time); buf["grip"].append(gp); buf["cl"].append(cl)  # noqa: E702
            next_est = d.time + 1.0 / 30
        # contacts: probe bookkeeping and the choice
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
            elif contact_on[i] and d.time - last_contact[i] > 0.3:  # contact over: a probe
                contact_on[i] = False
                if not summary[i, 0]:
                    t = np.array(buf["t"])
                    sel = (t >= contact_start[i] - 1.5) & (t <= d.time)
                    if sel.sum() > 20:
                        logm, sd, _ = est(t[sel], np.array(buf["grip"])[sel], np.array(buf["cl"])[sel], arclen, st,
                                          cfg.friction)
                        if oracle:
                            logm, sd = float(np.log(cfg.masses[i])), ORACLE_SD
                        summary[i] = (1.0, logm, sd)
                        if summary[:, 0].all():
                            box_at_probed, t_all_probed = np.array([d.xpos[b].copy() for b in boxes]), float(d.time)
        if contacts_after_probing and d.time - contact_start[contacts_after_probing[0]] > 2.0:
            break  # the choice is made and being pushed
        mujoco.mj_step(m, d)
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            break
    rend.close()
    if video is not None and frames:
        imageio.mimsave(video, frames, fps=rate)
    truth = int(np.argmax(cfg.masses) if cfg.goal == "heaviest" else np.argmin(cfg.masses))
    chosen = contacts_after_probing[0] if contacts_after_probing else -1
    # what the expert's rule would pick from the same summary (probe/multibox.py), for policy-vs-estimator attribution
    rule_pick = -1
    if summary[:, 0].all():
        draws = np.random.default_rng(0).normal(summary[:, 1], np.maximum(summary[:, 2], 1e-3), size=(4000, 3))
        pick = draws.argmax(1) if cfg.goal == "heaviest" else draws.argmin(1)
        rule_pick = int(np.argmax(np.bincount(pick, minlength=3)))
    box_end = np.array([d.xpos[b].copy() for b in boxes])
    mujoco.mj_forward(m, d)
    _, _, gp_end, _ = extract_frame(m, d, h, arclen)
    return {"goal": cfg.goal, "task": cfg.instruction, "masses": list(cfg.masses), "probed": int(summary[:, 0].sum()),
            "chosen": chosen, "truth": truth, "success": chosen == truth, "t_end": float(d.time),
            "t_choice": float(contact_start[chosen]) if chosen >= 0 else None,
            "est_logm": summary[:, 1].tolist(), "est_sd": summary[:, 2].tolist(), "rule_pick": rule_pick,
            "rule_success": rule_pick == truth, "t_all_probed": t_all_probed, "box_start": box_start.tolist(),
            "box_at_probed": None if box_at_probed is None else box_at_probed.tolist(), "box_end": box_end.tolist(),
            "grip_end": np.asarray(gp_end)[:3].tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=100000)  # training episodes use seeds from 0
    ap.add_argument("--images-only", action="store_true")
    ap.add_argument("--style", default="angled", choices=tuple(CKPTS), help="push style the policy was trained on")
    ap.add_argument("--ckpt", help="estimator checkpoint (default: the one trained for --style)")
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--n-action-steps", type=int, help="replan after this many actions (default: the trained value)")
    ap.add_argument("--max-t", type=float, default=30.0, help="episode timeout [s]")
    ap.add_argument("--world", default="training", choices=("training", "sim-irl"),
                    help="training-sim (randomised) or the measured sim-IRL with the real box set")
    ap.add_argument("--oracle", action="store_true", help="feed true masses instead of the estimator's")
    ap.add_argument("--unseen-phrasings", action="store_true", help="instructions never seen in training")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.ckpt = a.ckpt or str(CKPTS[a.style])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    policy, pre, post = load_policy(a.policy)
    if a.n_action_steps:
        policy.config.n_action_steps = a.n_action_steps
    est = Estimator(a.ckpt)
    rows = []
    for k in range(a.n):
        cfg = sample_episode(a.seed0 + k, a.style)
        if a.world == "sim-irl":
            cfg = sim_irl(cfg, a.seed0 + k)
        if a.unseen_phrasings:
            ph = UNSEEN_PHRASINGS[cfg.goal]
            cfg.instruction = ph[k % len(ph)]
        r = run(policy, pre, post, cfg, est, a.images_only, max_t=a.max_t,
                video=out / f"ep{k:03d}.mp4" if k < a.videos else None, oracle=a.oracle)
        rows.append(r)
        print(f"  ep {k}: {r['goal']:8s} probed {r['probed']}/3 chose {r['chosen']} truth {r['truth']} "
              f"{'OK' if r['success'] else 'MISS'} (rule {r['rule_pick']})", flush=True)
    acc = float(np.mean([r["success"] for r in rows]))
    summary = {"policy": a.policy, "style": a.style, "images_only": a.images_only, "n": a.n, "success_rate": acc,
               "world": a.world, "oracle": a.oracle, "unseen_phrasings": a.unseen_phrasings,
               "n_action_steps": policy.config.n_action_steps, "max_t": a.max_t,
               "success_rate_30s": float(np.mean([r["success"] and r["t_choice"] < 30.0 for r in rows])),
               "probed_all_three": float(np.mean([r["probed"] == 3 for r in rows])),
               "made_a_choice": float(np.mean([r["chosen"] >= 0 for r in rows])),
               "rule_success_rate": float(np.mean([r["rule_success"] for r in rows])), "episodes": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
