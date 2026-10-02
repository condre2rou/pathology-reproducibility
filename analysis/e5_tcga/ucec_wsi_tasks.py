# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

OUT = os.path.dirname(os.path.abspath(__file__))
POOLED = os.path.join(OUT, "pooled_features.npz")
LABELS = os.path.join(OUT, "tcga_ucec_labels.csv")


def cbioportal(study, attr):
    import urllib.request

    for dtype in ["PATIENT", "SAMPLE"]:
        url = (
            f"https://www.cbioportal.org/api/studies/{study}/clinical-data"
            f"?clinicalDataType={dtype}&attributeId={attr}&projection=SUMMARY"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=120).read())
            if d:
                return {r["patientId"]: r["value"] for r in d}
        except Exception:
            pass
    return {}


def cv(X, Y, seeds=10):
    aucs, auprs = [], []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, Y):
            if len(np.unique(Y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(X[tr]), Y[tr])
            pr = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(Y[te], pr))
            auprs.append(average_precision_score(Y[te], pr))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(auprs))


def main():
    z = np.load(POOLED, allow_pickle=True)
    X, Ydmmr, pids = z["X"], z["Y"], z["pids"]
    print(f"UCEC WSI: {X.shape}, dMMR={int(Ydmmr.sum())}/{len(Ydmmr)}")

    dm, ds, dap = cv(X, Ydmmr)
    print(
        f"[分子] dMMR        n={len(Ydmmr)}  正类={int(Ydmmr.sum())}  "
        f"AUROC={dm:.3f}±{ds:.3f}  AUPRC={dap:.3f}"
    )

    grade = cbioportal("ucec_tcga_pan_can_atlas_2018", "GRADE")
    print(f"  GRADE 拉取 {len(grade)} 例")
    idx, gy = [], []
    for i, p in enumerate(pids):
        g = grade.get(p)
        if not g:
            continue
        gs = str(g)
        if "GX" in gs:
            continue
        idx.append(i)
        gy.append(1 if ("G3" in gs or "High" in gs) else 0)
    gy = np.array(gy)
    res = {
        "dMMR": {
            "n": len(Ydmmr),
            "pos": int(Ydmmr.sum()),
            "auroc": round(dm, 3),
            "std": round(ds, 3),
            "auprc": round(dap, 3),
        }
    }
    if len(idx) >= 40 and len(set(gy)) == 2 and min(gy.sum(), (gy == 0).sum()) >= 10:
        gm, gsd, gap = cv(X[idx], gy)
        res["grade"] = {
            "n": len(idx),
            "pos": int(gy.sum()),
            "auroc": round(gm, 3),
            "std": round(gsd, 3),
            "auprc": round(gap, 3),
        }
        print(
            f"[形态] grade(G3 vs G1/2)  n={len(idx)}  正类={int(gy.sum())}  "
            f"AUROC={gm:.3f}±{gsd:.3f}  AUPRC={gap:.3f}"
        )
    else:
        print(f"[形态] grade 不可用（{len(idx)} 例, 类别 {sorted(set(gy)) if len(gy) else '无'}）")

    print("\n=== 同任务 × 两模态对照 ===")
    print(f"{'任务':<16} {'照片级(本院)':>14} {'WSI(TCGA-UCEC)':>16}")
    print("-" * 50)
    print(f"{'grade (形态学)':<16} {'0.809':>14} {res.get('grade', {}).get('auroc', 'N/A'):>16}")
    print(f"{'dMMR (分子)':<16} {'0.503':>14} {res['dMMR']['auroc']:>16}")

    with open(os.path.join(OUT, "ucec_wsi_tasks.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\n已写 ucec_wsi_tasks.json")


if __name__ == "__main__":
    main()
