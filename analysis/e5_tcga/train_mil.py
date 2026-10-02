# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(OUT))
CACHE = os.path.join(ROOT, "models")
POOLED = os.path.join(OUT, "pooled_features.npz")


def find_h5(pid, files):
    """Preserved computation; see docs/experiments.md for its protocol."""
    hits = []
    for f in files:
        if os.path.basename(f).startswith(pid + "-"):
            p = glob.glob(
                os.path.join(CACHE, "datasets--W8Yi--tcga-wsi-uni2h-features", "snapshots", "*", f)
            )
            if p:
                hits.append(p[0])
    return hits


def pool_features(paths, mode="mean"):
    """Preserved computation; see docs/experiments.md for its protocol."""
    feats = []
    for p in paths:
        with h5py.File(p, "r") as f:
            X = f["features"][:]  # (1, N, D)
            X = X.reshape(-1, X.shape[-1])
            if mode == "mean":
                v = X.mean(axis=0)
            elif mode == "max":
                v = X.max(axis=0)
            elif mode == "meanstd":
                v = np.concatenate([X.mean(axis=0), X.std(axis=0)])
            else:
                raise ValueError(mode)
            feats.append(v)
    return np.mean(feats, axis=0)


def cv_eval(X, Y, clf_name="lr", seeds=5):
    aucs, auprs = [], []
    for s in range(seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        for tr, te in skf.split(X, Y):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            if clf_name == "lr":
                clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
            else:
                clf = MLPClassifier(
                    hidden_layer_sizes=(256,), max_iter=500, early_stopping=True, random_state=s
                )
            clf.fit(Xtr, Y[tr])
            prob = clf.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(Y[te], prob))
            auprs.append(average_precision_score(Y[te], prob))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(auprs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="mean")
    args = ap.parse_args()

    sub = pd.read_csv(os.path.join(OUT, "tcga_subset.csv"), dtype=str)
    sub["label"] = sub["label"].astype(int)
    files = json.load(open(os.path.join(OUT, "ucec_files.json")))

    if os.path.exists(POOLED):
        z = np.load(POOLED, allow_pickle=True)
        X, Y, pids = z["X"], z["Y"], z["pids"]
        print(f"复用池化特征缓存: {X.shape}")
    else:
        Xs, Ys, P = [], [], []
        miss = 0
        for _, r in sub.iterrows():
            hs = find_h5(r["patient_id"], files)
            if not hs:
                miss += 1
                continue
            try:
                Xs.append(pool_features(hs, args.pool))
                Ys.append(r["label"])
                P.append(r["patient_id"])
            except Exception as e:
                miss += 1
        X, Y, pids = np.stack(Xs), np.array(Ys), np.array(P)
        np.savez(POOLED, X=X, Y=Y, pids=pids)
        print(f"池化完成: {X.shape}, 缺失 {miss}")

    print(
        f"\n数据: {len(Y)} 例 (dMMR {int(Y.sum())} / pMMR {int((Y == 0).sum())}), 特征维度 {X.shape[1]}"
    )

    out = {
        "cohort": "TCGA-UCEC",
        "model": "UNI2-h",
        "pool": args.pool,
        "n": len(Y),
        "pos": int(Y.sum()),
    }
    print(f"\n{'分类器':<10} {'AUROC':>16} {'AUPRC':>9}")
    print("-" * 40)
    for name in ["lr", "mlp"]:
        m, s, ap_ = cv_eval(X, Y, name)
        out[name] = {"auroc": round(m, 3), "std": round(s, 3), "auprc": round(ap_, 3)}
        print(f"{name:<10} {m:.3f}±{s:.3f}      {ap_:.3f}")

    print(f"\n对照 — 本院照片级数据 dMMR AUROC: 0.503 (Phikon)")
    with open(os.path.join(OUT, "tcga_mil_result.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"已写 tcga_mil_result.json")


if __name__ == "__main__":
    main()
