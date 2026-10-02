"""Portable, leakage-free linear probes and annotation decisions."""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix

CUT = 0.57
DELTA = 0.05
BUDGETS = (10, 20, 30, 40, 50, 60, 80)
THRESHOLDS = np.round(np.arange(0.40, 0.96, 0.005), 3)


def probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, tol=1e-4),
    )


def cv_predictions(X, y, seed=0, return_folds=False):
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2:
        return None
    splits = list(
        StratifiedKFold(min(5, int(counts.min())), shuffle=True, random_state=seed).split(X, y)
    )
    p = np.full(len(y), np.nan)
    for train, test in splits:
        model = probe().fit(X[train], y[train])
        p[test] = model.predict_proba(X[test])[:, 1]
    assert np.isfinite(p).all()
    return (p, splits) if return_folds else p


def cv_auc(X, y, seed=0):
    p = cv_predictions(X, y, seed)
    return float(roc_auc_score(y, p)) if p is not None else float("nan")


def balanced_sample(y, n, rng):
    counts = np.bincount(y, minlength=2)
    if n % 2 or counts.min() < n // 2:
        raise ValueError("A balanced pilot requires n/2 existing labels per class")
    return np.concatenate(
        [rng.choice(np.flatnonzero(y == k), n // 2, replace=False) for k in (1, 0)]
    )


def best_cut(a, saturated, weights=None):
    a, saturated = np.asarray(a), np.asarray(saturated)
    if weights is None:
        weights = np.ones(len(a))
    acc = np.average((a[:, None] <= THRESHOLDS) == saturated[:, None], axis=0, weights=weights)
    return float(THRESHOLDS[np.argmax(acc)])


def metrics(y, stop, score=None, weights=None):
    y, stop = np.asarray(y, int), np.asarray(stop, int)
    tn, fp, fn, tp = confusion_matrix(y, stop, labels=[0, 1], sample_weight=weights).ravel()

    def ratio(a, b):
        return float(a / b) if b else None

    result = dict(
        n=int(len(y)),
        true_stop=float(tp),
        false_stop=float(fp),
        true_continue=float(tn),
        false_continue=float(fn),
        accuracy=ratio(tp + tn, tp + tn + fp + fn),
        saturation_sensitivity=ratio(tp, tp + fn),
        nonsaturation_specificity=ratio(tn, tn + fp),
        false_stop_rate=ratio(fp, tp + fp),
        missed_opportunity_rate=ratio(fp, fp + tn),
        stop_fraction=ratio(tp + fp, tp + tn + fp + fn),
        majority_accuracy=ratio(max(tp + fn, tn + fp), tp + tn + fp + fn),
    )
    if score is not None:
        result["auroc"] = (
            float(roc_auc_score(y, score, sample_weight=weights))
            if len(np.unique(y)) == 2
            else None
        )
    return result


def oof_interval(y, p, seed=0, n_boot=200):
    if p is None:
        return None
    rng = np.random.RandomState(seed)
    counts = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=n_boot)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    wins = (p[pos, None] > p[None, neg]).astype(float) + 0.5 * (p[pos, None] == p[None, neg])
    numerator = np.einsum("bi,ij,bj->b", counts[:, pos], wins, counts[:, neg])
    denominator = counts[:, pos].sum(1) * counts[:, neg].sum(1)
    vals = numerator[denominator > 0] / denominator[denominator > 0]
    return list(map(float, np.quantile(vals, [0.025, 0.975]))) if len(vals) else None


def decide(auc, interval=None, cut=CUT):
    if interval is None or not np.isfinite(auc):
        return "insufficient"
    if interval[1] <= cut:
        return "stop"
    if interval[0] > cut:
        return "continue"
    return "insufficient"
