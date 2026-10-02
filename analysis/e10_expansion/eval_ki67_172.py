# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "4")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
Z = os.path.join(HERE, "ki67_all172.npz")


def main():
    z = np.load(Z, allow_pickle=True)
    X, Y, pids = z["X"], z["Y"], z["pids"]
    n = len(Y)
    print(
        f"患者 {n} 例；Ki67 阳性率 mean {Y.mean():.2f}%  sd {Y.std():.2f}%  "
        f"range [{Y.min():.2f}, {Y.max():.2f}]"
    )
    print(f"≥20% 的患者: {int((Y >= 20).sum())} 例")

    out = {
        "meta": {
            "n_patients": n,
            "source": "e10_expansion/ki67_all172.npz",
            "label": "Ki67 IHC patch 比色法阳性率（DAB 棕/组织面积）的患者级均值",
            "backbone": "Phikon ViT-B/16 CLS, 患者级 mean 池化",
            "comparison": "论文既有结果基于 40 例（按 patch 数最多的患者选）",
        },
        "label_distribution": {
            "mean": round(float(Y.mean()), 3),
            "sd": round(float(Y.std()), 3),
            "min": round(float(Y.min()), 3),
            "max": round(float(Y.max()), 3),
            "n_ge_20pct": int((Y >= 20).sum()),
            "n_ge_10pct": int((Y >= 10).sum()),
        },
    }

    preds = np.zeros(n)
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")

        from sklearn.linear_model import Ridge

        rg = Ridge(alpha=1.0).fit(sc.transform(X[tr]), Y[tr])
        preds[i] = rg.predict(sc.transform(X[i : i + 1]))[0]
    rho, p = spearmanr(Y, preds)
    r, _ = pearsonr(Y, preds)
    out["patient_loocv_regression"] = {
        "spearman_rho": round(float(rho), 3),
        "spearman_p": round(float(p), 4),
        "pearson_r": round(float(r), 3),
        "note": "Ridge 回归，留一交叉验证",
    }

    for thr in [10, 20]:
        yb = (Y >= thr).astype(int)
        if min(np.bincount(yb)) < 5:
            out[f"patient_binary_ge{thr}pct"] = {
                "n_pos": int(yb.sum()),
                "skipped": "正类 <5，无法评估",
            }
            continue
        aucs = []
        for s in range(10):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, yb):
                if len(set(yb[te])) < 2:
                    continue
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
                clf.fit(sc.transform(X[tr]), yb[tr])
                aucs.append(roc_auc_score(yb[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
        out[f"patient_binary_ge{thr}pct"] = {
            "n_pos": int(yb.sum()),
            "n_neg": int((1 - yb).sum()),
            "auroc": round(float(np.mean(aucs)), 3),
            "std": round(float(np.std(aucs)), 3),
            "n_folds": len(aucs),
        }

    Xp = z["X"] if "Xp" not in z else z["Xp"]
    out["note_patch_level"] = (
        "patch 级评估需 patch 级特征；本文件只存患者级，"
        "patch 级结论见 e7_external/ki67_result_v2.json（40 例）"
    )

    json.dump(
        out,
        open(os.path.join(HERE, "ki67_172_result.json"), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    print()
    for k, v in out.items():
        if isinstance(v, dict) and k not in ("meta", "label_distribution"):
            print(f"  {k}: {v}")
    print(f"\n→ {os.path.join(HERE, 'ki67_172_result.json')}")


if __name__ == "__main__":
    main()
