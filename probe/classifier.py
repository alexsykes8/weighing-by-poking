"""Object-class classifier on curvature features: in-domain (sim) evaluation and sim->real transfer."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features
from .io import load_observation
from .schema import CLASSES as LABELS


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def feature_table(paths, noise: float = 0.0, seed: int = 0):
    """(X, y, sources) for observation files; `noise` (m) adds simulated tracker noise first."""
    rng = np.random.default_rng(seed)
    X, y, src = [], [], []
    for p in paths:
        o = load_observation(p)
        if noise > 0:
            cl, k = features.add_tracking_noise(o, noise, rng)
            X.append(features.extract(o, cl, k))
        else:
            X.append(features.extract(o))
        y.append(o.label)
        src.append(o.meta["source"])
    return np.array(X), np.array(y), np.array(src)


def evaluate_in_domain(X, y, seed=0) -> dict:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    clf = model().fit(Xtr, ytr)
    cv = cross_val_score(model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return {"n_train": len(ytr), "n_test": len(yte), "heldout_acc": float((clf.predict(Xte) == yte).mean()),
            "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
            "confusion": confusion_matrix(yte, clf.predict(Xte), labels=list(LABELS)).tolist(),
            "feature_weight_norm": dict(zip(features.FEATURES,
                                            np.linalg.norm(clf[-1].coef_, axis=0).round(3).tolist()))}


def exact_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson (exact) interval for k successes in n: honest error bars for tiny real sets."""
    if n == 0:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return (float(lo), float(hi))


def feature_shift(X_sim, X_real) -> dict:
    """Per feature: standardised mean shift (real - sim)/std_sim, and fraction of real samples
    outside the sim range. Large values say *which* part of the signature doesn't transfer."""
    mu, sd = X_sim.mean(0), X_sim.std(0) + 1e-12
    lo, hi = X_sim.min(0), X_sim.max(0)
    out = {}
    for j, name in enumerate(features.FEATURES):
        out[name] = {"z_shift": float((X_real[:, j].mean() - mu[j]) / sd[j]),
                     "frac_outside_sim_range": float(((X_real[:, j] < lo[j]) | (X_real[:, j] > hi[j])).mean())}
    return out


def evaluate_transfer(X_sim, y_sim, X_real, y_real, names_real=None, seed=0) -> dict:
    """Train on all sim, test on real. The gap is sim 5-fold CV accuracy minus real accuracy."""
    cv = cross_val_score(model(), X_sim, y_sim, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    clf = model().fit(X_sim, y_sim)
    pred = clf.predict(X_real)
    proba = clf.predict_proba(X_real)
    k, n = int((pred == y_real).sum()), len(y_real)
    acc = k / n
    per_class = {}
    for c in LABELS:
        m = y_real == c
        if m.any():
            kc = int((pred[m] == c).sum())
            per_class[c] = {"n": int(m.sum()), "acc": kc / int(m.sum()), "ci95": exact_ci(kc, int(m.sum()))}
    absent = [c for c in LABELS if c not in per_class]
    if absent:
        warnings.warn(f"real set has no examples of {absent}: their accuracy is untested", stacklevel=2)
    names = names_real if names_real is not None else [str(i) for i in range(n)]
    return {
        "sim_cv_acc": float(cv.mean()), "sim_cv_std": float(cv.std()), "n_sim": int(len(y_sim)),
        "real_acc": acc, "real_ci95": exact_ci(k, n), "n_real": n,
        "gap": float(cv.mean() - acc),
        "chance": 1.0 / len(LABELS),
        "per_class": per_class, "classes_untested": absent,
        "confusion": confusion_matrix(y_real, pred, labels=list(LABELS)).tolist(), "labels": list(LABELS),
        "feature_shift": feature_shift(X_sim, X_real),
        "predictions": [{"file": nm, "true": t, "pred": p, "p": dict(zip(clf.classes_, pr.round(3).tolist()))}
                        for nm, t, p, pr in zip(names, y_real, pred, proba)],
    }


def real_paths(root) -> tuple[list[Path], list[str]]:
    """Observation files under `root` whose meta.source is 'video'. Returns (paths, skipped-reasons)."""
    root = Path(root)
    if not root.exists():
        return [], [f"{root} does not exist"]
    keep, skipped = [], []
    for p in sorted(root.rglob("*.npz")):
        try:
            o = load_observation(p)
        except Exception as e:  # report and skip; one bad file shouldn't hide the rest
            skipped.append(f"{p.name}: invalid ({e})")
            continue
        if o.meta["source"] != "video":
            skipped.append(f"{p.name}: source={o.meta['source']!r}, not video")
            continue
        keep.append(p)
    return keep, skipped
