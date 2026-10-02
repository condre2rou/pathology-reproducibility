# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "e2_train"))
from data import load_manifest  # noqa: E402

RUNS = os.path.join(ROOT, "e2_train", "runs")
LABELS_EC = os.path.join(ROOT, "e1_label_engine", "outputs", "labels_ec.csv")
NOISE_RATES = [0.0, 0.10, 0.20, 0.30, 0.40]
DETECTOR_ACC = 0.70
SEEDS = 5


def run_once(X, Y_true, flip_mask, prior, seed):
    """Preserved computation; see docs/experiments.md for its protocol."""
    Y_train = Y_true.copy()
    Y_train[flip_mask] = 1 - Y_train[flip_mask]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs_ce, aucs_snap = [], []
    for tr, te in skf.split(X, Y_true):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
        clf.fit(Xtr, Y_train[tr])
        aucs_ce.append(roc_auc_score(Y_true[te], clf.predict_proba(Xte)[:, 1]))

        w = np.where(prior[tr], 0.2, 1.0)
        clf2 = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
        clf2.fit(Xtr, Y_train[tr], sample_weight=w)
        aucs_snap.append(roc_auc_score(Y_true[te], clf2.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs_ce)), float(np.mean(aucs_snap))


def main():
    manifest = load_manifest()
    lab = pd.read_csv(LABELS_EC, dtype=str)
    lab["pid"] = lab["year"] + "_" + lab["folder"]
    cache = os.path.join(RUNS, "feats_phikon.npz")
    feats = np.load(cache, allow_pickle=True)["feats"].item()

    rows = [
        (r["pid"], 1 if r["grade"] == "3" else 0)
        for _, r in lab.iterrows()
        if r["grade"] in ("1", "2", "3") and r["pid"] in feats
    ]
    X = np.stack([feats[p] for p, _ in rows])
    Y = np.array([y for _, y in rows])
    n = len(Y)
    print(f"grade3: N={n}, 正类={int(Y.sum())}\n")

    out = {"task": "grade3", "n": n, "detector_acc": DETECTOR_ACC, "results": {}}
    print(f"{'噪声率':>7} {'CE':>10} {'SNAP':>10} {'Δ':>8}")
    print("-" * 40)
    for rate in NOISE_RATES:
        ce_list, snap_list = [], []
        for s in range(SEEDS):
            rng = np.random.RandomState(1000 + s)
            flip = rng.rand(n) < rate

            detected = np.where(flip, rng.rand(n) < DETECTOR_ACC, rng.rand(n) < (1 - DETECTOR_ACC))
            ce, snap = run_once(X, Y, flip, detected, s)
            ce_list.append(ce)
            snap_list.append(snap)
        ce_m, snap_m = np.mean(ce_list), np.mean(snap_list)
        out["results"][f"{rate:.2f}"] = {
            "ce": round(float(ce_m), 3),
            "snap": round(float(snap_m), 3),
            "delta": round(float(snap_m - ce_m), 3),
        }
        print(f"{rate:>7.0%} {ce_m:>10.3f} {snap_m:>10.3f} {snap_m - ce_m:>+8.3f}")

    with open(os.path.join(RUNS, "snap_synthetic.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n已写 runs/snap_synthetic.json")


if __name__ == "__main__":
    main()
