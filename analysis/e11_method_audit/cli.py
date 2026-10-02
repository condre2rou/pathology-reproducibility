"""Assess one labelled pilot; run a synthetic demonstration without patient data."""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import argparse, json
import numpy as np
from sklearn.metrics import roc_auc_score
from core import cv_predictions, cv_auc, oof_interval, decide, CUT


def assess(X, y, seed=0):
    X = np.asarray(X, float)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or (len(y) != 40):
        raise ValueError(
            "Supply exactly 40 analysis-unit rows (directory records or participants), with X[40,d] and y[40]"
        )
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("X must be finite and y must contain binary 0/1 labels")
    y = y.astype(int)
    p = cv_predictions(X, y, seed)
    a = float(roc_auc_score(y, p)) if p is not None else None
    ci = oof_interval(y, p, seed)
    action = decide(a if a is not None else np.nan, ci)
    rng = np.random.RandomState(seed)
    order = rng.permutation(40)
    curve = []
    for n in (10, 20, 30, 40):
        ix = order[:n]
        v = (a if a is not None else float("nan")) if n == 40 else cv_auc(X[ix], y[ix], seed)
        curve.append(dict(n=n, auroc=float(v) if np.isfinite(v) else None))
    return dict(
        n=40,
        positives=int(y.sum()),
        auroc=a,
        conditional_pair_resampling_interval=ci,
        conditional_oof_interval=ci,
        action=action,
        cut=CUT,
        learning_curve=curve,
        interpretation={
            "stop": "Exploratory stop classification only: the single-pilot study did not support this rule as a stopping recommendation.",
            "continue": "The pilot is above the cut; this does not guarantee further gain.",
            "insufficient": "The conditional pair interval crosses the historical cut or the score cannot be estimated.",
        }[action],
        scope="Conditional pair-resampling percentile interval with fold models fixed; no unconditional coverage or future-gain guarantee. The 0.57 threshold was historically data-selected. conditional_oof_interval is a compatibility alias. No full-data labels used.",
    )
