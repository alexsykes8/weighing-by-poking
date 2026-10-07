"""Generate randomised sim rollouts, train the mass/anchored GRU estimator, log everything to W&B.

    python scripts/train_estimator.py --n 200 --smoke            # smoke test
    scripts/launch_estimator.sh 5000                             # full run in the background (nohup + log file)

One W&B run (job_type=train) covers generation, data-coverage histograms and training. Baselines
(predict-the-mean, old feature-based regressor/classifier) are separate runs in the same project and group.
Splits: random val from the in-range data, plus a held-out OOD split (E > ood_young, or a free box heavier
than ood_mass) that never enters training.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import wandb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probe import features as oldfeat
from probe.estimator import (COND_FEATURES, KAPPA_AT, SEQ_FEATURES, Estimator, cond_vector, cubic_shape, loss_fn,
                             sequence_features)
from probe.io import git_sha
from probe.schema import ProbeObservation
from probe.simdata import RANGES, STYLES, generate, load_cache

REPO = _bootstrap.REPO
INK2, GRID, BLUE, ORANGE = "#5f5e58", "#e4e3dc", "#2a78d6", "#eb6834"


# ---- data -----------------------------------------------------------------------------------------

def build_arrays(rollouts):
    P = [r["params"] for r in rollouts]
    X = [sequence_features(r["t"], r["grip_pose"], r["centreline"], r["arclen"]) for r in rollouts]
    C = np.stack([cond_vector(p["young"], p["length"], p["radius"], p["damping"], p["friction"]) for p in P])
    kind = np.array([p["kind"] for p in P])
    y_anch = (kind == "fixed").astype(np.float32)
    mass = np.array([p["mass"] for p in P])
    y_logm = np.where(kind == "free", np.log(mass), np.nan).astype(np.float32)
    return X, C, y_logm, y_anch, kind, P


def make_splits(P, kind, ood_young, ood_mass, val_frac, seed):
    E = np.array([p["young"] for p in P])
    m = np.array([p["mass"] for p in P])
    ood_E = E > ood_young
    ood_m = (kind == "free") & (m > ood_mass)
    ood = ood_E | ood_m
    rng = np.random.default_rng(seed)
    ind = np.where(~ood)[0]
    rng.shuffle(ind)
    n_val = int(round(val_frac * len(ind)))
    return {"train": np.sort(ind[n_val:]), "val": np.sort(ind[:n_val]), "ood": np.where(ood)[0],
            "ood_E_only": np.where(ood_E & ~ood_m)[0], "ood_mass_only": np.where(ood_m & ~ood_E)[0]}


class SeqData(torch.utils.data.Dataset):
    """Applies tracker-noise augmentation per item (fresh each epoch for train, fixed seed for eval)."""

    def __init__(self, rollouts, idx, C, y_logm, y_anch, kind, stats, noise_range, train, seed=0):
        self.r, self.idx, self.C, self.yl, self.ya, self.kind = rollouts, idx, C, y_logm, y_anch, kind
        self.stats, self.noise_range, self.train, self.seed = stats, noise_range, train, seed

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        rng = np.random.default_rng(None if self.train else self.seed * 100003 + int(j))
        noise = rng.uniform(*self.noise_range)
        r = self.r[j]
        x = sequence_features(r["t"], r["grip_pose"], r["centreline"], r["arclen"], noise_m=noise, rng=rng)
        x = (x - self.stats["x_mean"]) / self.stats["x_std"]
        c = (self.C[j] - self.stats["c_mean"]) / self.stats["c_std"]
        return (torch.from_numpy(x), torch.from_numpy(c), torch.tensor(np.nan_to_num(self.yl[j])),
                torch.tensor(self.ya[j]), torch.tensor(self.kind[j] == "free"), int(j))


def collate(batch):
    xs, cs, yl, ya, free, j = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    X = nn_pad(xs)
    return X, lengths, torch.stack(cs), torch.stack(yl), torch.stack(ya), torch.stack(free), torch.tensor(j)


def nn_pad(xs):
    return torch.nn.utils.rnn.pad_sequence(list(xs), batch_first=True)


# ---- metrics & plots ------------------------------------------------------------------------------

def metrics(pred, y_logm, y_anch, kind, prefix):
    mu, log_var, logit = pred["mu"], pred["log_var"], pred["logit"]
    free = kind == "free"
    out = {}
    if free.any():
        m_true, m_pred = np.exp(y_logm[free]), np.exp(mu[free])
        err = m_pred - m_true
        out[f"{prefix}/mass_mae_g"] = float(np.abs(err).mean() * 1e3)
        out[f"{prefix}/mass_mae_pct"] = float(np.mean(np.abs(err) / m_true) * 100)
        out[f"{prefix}/mass_median_ape_pct"] = float(np.median(np.abs(err) / m_true) * 100)
        z = (mu[free] - y_logm[free]) / np.exp(0.5 * log_var[free])
        out[f"{prefix}/z_std"] = float(z.std())  # ~1 when predicted uncertainty is calibrated
        out[f"{prefix}/nll_logm"] = float(np.mean(0.5 * (log_var[free] + (y_logm[free] - mu[free]) ** 2
                                                         / np.exp(log_var[free]))))
    p = 1 / (1 + np.exp(-logit))
    out[f"{prefix}/anchored_acc"] = float(((p > 0.5) == (y_anch > 0.5)).mean())
    out[f"{prefix}/anchored_base_rate_acc"] = float(max(y_anch.mean(), 1 - y_anch.mean()))
    if 0 < y_anch.sum() < len(y_anch):
        out[f"{prefix}/anchored_auc"] = float(roc_auc_score(y_anch, p))
    if (kind == "freespace").any():
        out[f"{prefix}/freespace_p_anchored"] = float(p[kind == "freespace"].mean())
    return out


def _bins(vals, edges, err):
    idx = np.digitize(vals, edges) - 1
    return [np.median(err[idx == b]) if (idx == b).sum() >= 3 else np.nan for b in range(len(edges) - 1)], \
           [int((idx == b).sum()) for b in range(len(edges) - 1)]


def plots(pred, y_logm, y_anch, kind, P, title):
    free = kind == "free"
    mu, sd = pred["mu"][free], np.exp(0.5 * pred["log_var"][free])
    mt = np.exp(y_logm[free])
    mp = np.exp(mu)
    ape = np.abs(mp - mt) / mt * 100
    figs = {}
    # predicted vs true
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.errorbar(mt, mp, yerr=[mp - np.exp(mu - sd), np.exp(mu + sd) - mp], fmt="o", ms=3, color=BLUE, alpha=0.5,
                ecolor=GRID, elinewidth=0.8)
    lim = [0.03, 6]
    ax.plot(lim, lim, "--", color=INK2, lw=1)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="true mass (kg)", ylabel="predicted mass (kg)",
           title=f"{title}: predicted vs true (±1σ)")
    figs["scatter"] = fig
    # error by bin
    E = np.array([p["young"] for p in P])[free]
    mu_f = np.array([p["friction"] for p in P])[free]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, vals, edges, lab, log in [
            (axs[0], mt, np.geomspace(RANGES["mass"][1], RANGES["mass"][2], 7), "true mass (kg)", True),
            (axs[1], E, np.geomspace(RANGES["young"][1], RANGES["young"][2], 7), "E (Pa)", True),
            (axs[2], mu_f, np.linspace(RANGES["friction"][1], RANGES["friction"][2], 7), "friction", False)]:
        med, n = _bins(vals, edges, ape)
        centres = np.sqrt(edges[:-1] * edges[1:]) if log else 0.5 * (edges[:-1] + edges[1:])
        ax.bar(range(len(med)), med, color=BLUE)
        ax.set_xticks(range(len(med)), [f"{c:.2g}\nn={k}" for c, k in zip(centres, n)], fontsize=7)
        ax.set(xlabel=lab, ylabel="median |mass error| (%)")
        ax.grid(axis="y", color=GRID)
    fig.suptitle(f"{title}: error by bin")
    figs["error_by_bin"] = fig
    # calibration
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    err = np.abs(mu - y_logm[free])
    edges = np.quantile(sd, np.linspace(0, 1, 9))
    idx = np.clip(np.digitize(sd, edges[1:-1]), 0, 7)
    xs = [sd[idx == b].mean() for b in range(8) if (idx == b).sum() > 2]
    ys = [np.sqrt((err[idx == b] ** 2).mean()) for b in range(8) if (idx == b).sum() > 2]
    ax.plot(xs, ys, "o-", color=BLUE, label="RMS error per σ bin")
    hi = max(max(xs, default=1), max(ys, default=1)) * 1.1
    ax.plot([0, hi], [0, hi], "--", color=INK2, lw=1, label="perfect calibration")
    ax.set(xlabel="predicted σ (log-mass)", ylabel="actual RMS error (log-mass)", title=f"{title}: calibration")
    ax.legend(frameon=False, fontsize=8)
    figs["calibration"] = fig
    out = {k: wandb.Image(f) for k, f in figs.items()}
    for f in figs.values():
        plt.close(f)
    return out


# ---- training -------------------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader):
    model.eval()
    mus, lvs, lgs, js, losses = [], [], [], [], []
    for X, lengths, C, yl, ya, free, j in loader:
        mu, lv, lg = model(X, lengths, C)
        losses.append(loss_fn(mu, lv, lg, yl, ya, free)[0].item() * len(j))
        mus.append(mu.numpy()), lvs.append(lv.numpy()), lgs.append(lg.numpy()), js.append(j.numpy())
    order = np.argsort(np.concatenate(js))
    cat = lambda a: np.concatenate(a)[order]  # noqa: E731
    return {"mu": cat(mus), "log_var": cat(lvs), "logit": cat(lgs), "idx": np.sort(np.concatenate(js)),
            "loss": float(np.sum(losses) / len(order))}


def run_baselines(cfg, rollouts, X, C, y_logm, y_anch, kind, P, splits):
    """Two baselines, each its own W&B run in the same project/group."""
    tr, evals = splits["train"], {k: splits[k] for k in ("val", "ood")}
    # 1) predict the mean
    free_tr = tr[kind[tr] == "free"]
    m0, v0 = float(np.mean(y_logm[free_tr])), float(np.var(y_logm[free_tr]))
    base_logit = float(np.log(y_anch[tr].mean() / (1 - y_anch[tr].mean())))
    # 2) old hand-crafted curvature features (+ the same conditioning) -> ridge on log-mass, logistic anchored
    def old_features(j):
        r = rollouts[j]
        arc = np.linspace(0, float(r["arclen"][-1]), len(r["arclen"]))
        from probe.geometry import curvature_inplane
        k = np.array([curvature_inplane(c.astype(float), arc) for c in r["centreline"]])
        o = ProbeObservation(r["t"].astype(float), r["centreline"].astype(float), r["centreline"][:, -1].astype(float),
                             k, arc, r["grip_pose"].astype(float), "free", {})
        return np.concatenate([oldfeat.extract(o), C[j]])
    F = np.array([old_features(j) for j in range(len(rollouts))])
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(F[free_tr], y_logm[free_tr])
    res_var = float(np.var(reg.predict(F[free_tr]) - y_logm[free_tr]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(F[tr], y_anch[tr])
    preds = {
        "predict-the-mean": lambda idx: {"mu": np.full(len(idx), m0), "log_var": np.full(len(idx), np.log(v0)),
                                         "logit": np.full(len(idx), base_logit)},
        "old-features": lambda idx: {"mu": reg.predict(F[idx]), "log_var": np.full(len(idx), np.log(res_var)),
                                     "logit": clf.decision_function(F[idx])},
    }
    summary = {}
    for name, fn in preds.items():
        run = wandb.init(project=cfg["project"], group=cfg["group"], job_type="baseline", name=f"{cfg['name']}-{name}",
                         config={**cfg, "baseline": name}, reinit="create_new")
        logs = {}
        for split, idx in evals.items():
            pr = fn(idx)
            logs.update(metrics(pr, y_logm[idx], y_anch[idx], kind[idx], split))
            imgs = plots(pr, y_logm[idx], y_anch[idx], kind[idx], [P[i] for i in idx], f"{name} {split}")
            logs.update({f"{split}/{k}": v for k, v in imgs.items()})
        run.log(logs)
        run.summary.update({k: v for k, v in logs.items() if not isinstance(v, wandb.Image)})
        summary[name] = {k: v for k, v in logs.items() if not isinstance(v, wandb.Image)}
        run.finish()
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed0", type=int, default=0)
    # v2: real needle + gripper scene + new ranges (v1 = old 0.6 m rod; the cache is keyed by seed only, so a
    # range change needs a new directory or old rollouts are silently reused)
    ap.add_argument("--style", choices=sorted(STYLES), default="angled",
                    help="push style of the training-sim (straight: ~vertical needle, boxes <= 0.45 kg)")
    ap.add_argument("--cache", help="default: data/sim_estimator/v3 (angled: wider pushes + replays) or "
                                    "v5_straight: straight pokes + by-clip real replays")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--noise-mm", nargs=2, type=float, default=(0.0, 2.0))
    ap.add_argument("--val-frac", type=float, default=0.15)
    # held out: the stiffest ~20% of the E range (3-6e10, log-uniform) and free boxes > 0.6 kg (~12% of them;
    # above the heaviest real box, 0.52 kg). The measured needle (4.2e10) and all real boxes stay in range.
    ap.add_argument("--ood-young", type=float, default=5.2e10)
    ap.add_argument("--ood-mass", type=float, help="default 0.6 (angled) / 0.40 (straight)")
    ap.add_argument("--plot-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--name")
    ap.add_argument("--no-baselines", action="store_true")
    a = ap.parse_args()
    global RANGES
    RANGES = STYLES[a.style]
    a.cache = a.cache or str(REPO / "data" / "sim_estimator" / ("v3" if a.style == "angled" else "v5_straight"))
    a.ood_mass = a.ood_mass if a.ood_mass is not None else (0.6 if a.style == "angled" else 0.40)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed0)

    name = a.name or (f"{'smoke' if a.smoke else 'gru'}-n{a.n}-{time.strftime('%m%d-%H%M')}")
    group = name
    cfg = {"project": "stick-probe", "group": group, "name": name, "n_rollouts": a.n, "seed0": a.seed0, "style": a.style,
           "ranges": RANGES, "noise_mm": list(a.noise_mm), "seq_features": SEQ_FEATURES, "kappa_at": KAPPA_AT,
           "cond_features": COND_FEATURES, "model": {"type": "GRU", "hidden": a.hidden, "layers": a.layers,
                                                    "dropout": a.dropout},
           "optim": {"lr": a.lr, "weight_decay": a.weight_decay, "batch": a.batch, "epochs": a.epochs,
                     "patience": a.patience, "sched": "ReduceLROnPlateau(0.5, 8)", "grad_clip": 1.0},
           "splits": {"val_frac": a.val_frac, "ood_young_gt": a.ood_young, "ood_mass_gt": a.ood_mass},
           "git_sha": git_sha(), "mujoco_version": mujoco.__version__, "torch_version": torch.__version__,
           "cache": a.cache, "loss": "GaussianNLL(log-mass, free only) + BCE(anchored)"}
    run = wandb.init(project="stick-probe", group=group, job_type="train", name=name, config=cfg,
                     tags=["smoke"] if a.smoke else ["full"])
    print(f"W&B run: {run.url}", flush=True)

    # ---- generate (cached) ----
    t0 = time.time()
    last = [t0]

    def progress(k, n, s):
        if k % 50 == 0 or k == n:
            el = time.time() - t0
            print(f"  gen {k}/{n}  {el:.0f}s  ({k/el:.1f}/s, eta {(n-k)/(k/el):.0f}s)", flush=True)
            run.log({"gen/done": k, "gen/rate_per_s": k / el, "gen/elapsed_s": el})
    status = generate(a.n, a.cache, a.seed0, a.workers, progress, ranges=RANGES)
    gen_s = time.time() - t0
    fails = [s for s in status if not s["ok"]]
    new = [s for s in status if s["ok"] and not s.get("cached")]
    run.summary.update({"gen/seconds": gen_s, "gen/failed": len(fails), "gen/new": len(new),
                        "gen/mean_rollout_wall_s": float(np.mean([s["wall"] for s in new])) if new else 0.0})
    print(f"generation: {len(status)} rollouts ({len(new)} new) in {gen_s:.0f}s, {len(fails)} failed", flush=True)
    for s in fails[:10]:
        print("  FAIL", s["seed"], s["error"], flush=True)

    seeds = {a.seed0 + i for i in range(a.n)}
    rollouts = [r for r in load_cache(a.cache) if r["params"]["seed"] in seeds]
    X, C, y_logm, y_anch, kind, P = build_arrays(rollouts)
    splits = make_splits(P, kind, a.ood_young, a.ood_mass, a.val_frac, a.seed0)
    run.summary.update({f"split/{k}": len(v) for k, v in splits.items()})
    print({k: len(v) for k, v in splits.items()}, flush=True)

    # ---- data coverage (once) ----
    peak = np.array([np.abs(cubic_shape(r["centreline"].astype(float), r["arclen"].astype(float))[0]).max()
                     for r in rollouts])
    moved = np.array([np.abs(r["box_dx"]).max() * 1e3 for r in rollouts])
    cov = {f"data/{k}": wandb.Histogram([p.get(k, 0.0) for p in P]) for k in
           ("young", "length", "radius", "damping", "friction", "pitch_deg", "push_speed", "push_depth", "contact_frac",
            "hit_offset")}
    cov["data/log10_mass_free"] = wandb.Histogram(np.log10([p["mass"] for p, k in zip(P, kind) if k == "free"]))
    cov["data/peak_bending_per_m"] = wandb.Histogram(peak)
    cov["data/box_moved_mm_free"] = wandb.Histogram(moved[kind == "free"])
    cov["data/kind"] = wandb.Table(data=[[k, int((kind == k).sum())] for k in ("free", "fixed", "freespace")],
                                   columns=["kind", "count"])
    run.log(cov)
    run.summary["data/frac_free_boxes_that_moved_gt_2mm"] = float((moved[kind == "free"] > 2).mean())

    # ---- baselines ----
    if not a.no_baselines:
        bl = run_baselines(cfg, rollouts, X, C, y_logm, y_anch, kind, P, splits)
        run.summary.update({f"baseline/{k}/{m}": v for k, d in bl.items() for m, v in d.items()})
        print("baselines:", json.dumps({k: {m: round(v, 3) for m, v in d.items() if 'mae' in m or 'acc' in m}
                                        for k, d in bl.items()}), flush=True)

    # ---- train ----
    tr = splits["train"]
    xcat = np.concatenate([X[i] for i in tr])
    stats = {"x_mean": xcat.mean(0), "x_std": xcat.std(0) + 1e-6, "c_mean": C[tr].mean(0), "c_std": C[tr].std(0) + 1e-6}
    ds = lambda idx, train: SeqData(rollouts, idx, C, y_logm, y_anch, kind, stats,  # noqa: E731
                                    np.array(a.noise_mm) * 1e-3, train, seed=a.seed0)
    dl = lambda idx, train: torch.utils.data.DataLoader(ds(idx, train), batch_size=a.batch, shuffle=train,  # noqa: E731
                                                        collate_fn=collate, num_workers=4, persistent_workers=True)
    loaders = {"train": dl(tr, True), "val": dl(splits["val"], False), "ood": dl(splits["ood"], False)}
    model = Estimator(hidden=a.hidden, layers=a.layers, dropout=a.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    ckpt_dir = REPO / "checkpoints" / "estimator" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best, bad_epochs = np.inf, 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        tl, tn, gnorms = 0.0, 0, []
        t_ep = time.time()
        for X_, L_, C_, yl, ya, free, _ in loaders["train"]:
            mu, lv, lg = model(X_, L_, C_)
            loss, nll, bce = loss_fn(mu, lv, lg, yl, ya, free)
            opt.zero_grad()
            loss.backward()
            gnorms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
            opt.step()
            tl += loss.item() * len(yl)
            tn += len(yl)
        log = {"epoch": epoch, "train/loss": tl / tn, "train/grad_norm": float(np.mean(gnorms)),
               "train/grad_norm_max": float(np.max(gnorms)), "train/lr": opt.param_groups[0]["lr"],
               "train/epoch_s": time.time() - t_ep}
        preds = {}
        for split in ("val", "ood"):
            pr = predict(model, loaders[split])
            preds[split] = pr
            idx = pr["idx"]
            log[f"{split}/loss"] = pr["loss"]
            log.update(metrics(pr, y_logm[idx], y_anch[idx], kind[idx], split))
            for sub in ("ood_E_only", "ood_mass_only"):
                if split == "ood" and len(splits[sub]):
                    sel = np.isin(idx, splits[sub])
                    sp = {k: v[sel] for k, v in pr.items() if isinstance(v, np.ndarray)}
                    log.update(metrics(sp, y_logm[idx][sel], y_anch[idx][sel], kind[idx][sel], sub))
        if epoch % a.plot_every == 0 or epoch == 1:
            for split in ("val", "ood"):
                idx = preds[split]["idx"]
                imgs = plots(preds[split], y_logm[idx], y_anch[idx], kind[idx], [P[i] for i in idx], f"{split} ep{epoch}")
                log.update({f"{split}/{k}": v for k, v in imgs.items()})
        sched.step(log["val/loss"])
        if log["val/loss"] < best - 1e-4:
            best, bad_epochs = log["val/loss"], 0
            torch.save({"model": model.state_dict(), "stats": stats, "cfg": cfg, "epoch": epoch,
                        "val_loss": best}, ckpt_dir / "best.pt")
            log["val/best_loss"] = best
        else:
            bad_epochs += 1
        run.log(log)
        print(f"ep {epoch:3d} train {log['train/loss']:.3f} val {log['val/loss']:.3f} ood {log['ood/loss']:.3f} | "
              f"val MAE {log.get('val/mass_mae_pct', float('nan')):.0f}% ({log.get('val/mass_mae_g', float('nan')):.0f} g) "
              f"anch acc {log['val/anchored_acc']:.3f} auc {log.get('val/anchored_auc', float('nan')):.3f} "
              f"z_std {log.get('val/z_std', float('nan')):.2f} | lr {log['train/lr']:.1e} | {log['train/epoch_s']:.0f}s",
              flush=True)
        if bad_epochs >= a.patience:
            print(f"early stop at epoch {epoch} (best val loss {best:.3f})", flush=True)
            break

    # ---- final: best checkpoint on val and OOD ----
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", weights_only=False)["model"])
    final = {}
    for split in ("val", "ood"):
        pr = predict(model, loaders[split])
        idx = pr["idx"]
        final.update(metrics(pr, y_logm[idx], y_anch[idx], kind[idx], f"final_{split}"))
        imgs = plots(pr, y_logm[idx], y_anch[idx], kind[idx], [P[i] for i in idx], f"final {split}")
        run.log({f"final_{split}/{k}": v for k, v in imgs.items()})
    run.summary.update(final)
    (ckpt_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
    art = wandb.Artifact(f"estimator-{name}", type="model")
    art.add_file(str(ckpt_dir / "best.pt"))
    run.log_artifact(art)
    print("FINAL", json.dumps(final, indent=1), flush=True)
    run.finish()


if __name__ == "__main__":
    main()
