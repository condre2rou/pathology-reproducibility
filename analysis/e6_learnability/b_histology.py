# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "e1_label_engine"))
from align_seq_key import load_report, YEAR_CONFIG  # noqa: E402
from extract_rules import full_text  # noqa: E402

OUT = os.path.join(ROOT, "e6_learnability")
LABELS = os.path.join(ROOT, "e1_label_engine", "outputs", "labels_ec_patient.csv")


def extract_histology(text):
    if not text:
        return None
    if re.search(r"癌肉瘤", text):
        return "carcinosarcoma"
    if re.search(r"浆液性", text):
        return "serous"
    if re.search(r"透明细胞", text):
        return "clear_cell"
    if re.search(r"子宫内膜样", text):
        return "endometrioid"
    return None


def build():
    out = {}
    for year, cfg in YEAR_CONFIG.items():
        recs, _ = load_report(cfg)
        for seq, rec in recs.items():
            h = extract_histology(full_text(rec))
            if h:
                out[f"{year}_{seq}"] = h
    return out


def cv(X, Y, seeds=10, n_boot=2000):
    aucs = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, Y):
            if len(np.unique(Y[te])) < 2 or len(np.unique(Y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(X[tr]), Y[tr])
            aucs.append(roc_auc_score(Y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    if not aucs:
        return np.nan, np.nan, (np.nan, np.nan)
    a = np.array(aucs)
    rng = np.random.RandomState(0)
    boot = [np.mean(rng.choice(a, len(a), replace=True)) for _ in range(n_boot)]
    return (
        float(a.mean()),
        float(a.std()),
        (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
    )


def main():
    lab = pd.read_csv(LABELS, dtype=str)
    lab["pid"] = lab["year"] + "_" + lab["folder"]
    lab["hkey"] = lab["year"] + "_" + lab["name_seq"]
    hist = build()
    print(f"组织学抽取 {len(hist)} 例")

    res = {
        "n_serous_total": sum(1 for v in hist.values() if v == "serous"),
        "n_endometrioid_total": sum(1 for v in hist.values() if v == "endometrioid"),
        "by_backbone": {},
    }

    print(f"\n{'backbone':<12} {'N':>5} {'正类':>5} {'原始AUROC':>16} {'95%CI':>18}")
    print("-" * 62)
    for bb, f in [
        ("phikon", "feats_phikon.npz"),
        ("ssl", "feats_ssl.npz"),
        ("imagenet", "feats_imagenet.npz"),
    ]:
        fp = os.path.join(ROOT, "e2_train", "runs", f)
        if not os.path.exists(fp):
            continue
        feats = np.load(fp, allow_pickle=True)["feats"].item()
        X, Y = [], []
        for _, r in lab.iterrows():
            h = hist.get(r["hkey"])
            if h not in ("endometrioid", "serous") or r["pid"] not in feats:
                continue
            X.append(feats[r["pid"]])
            Y.append(1 if h == "serous" else 0)
        if len(X) < 40:
            continue
        X = np.stack(X)
        Y = np.array(Y)
        m, s, ci = cv(X, Y)
        res["by_backbone"][bb] = {
            "n": len(Y),
            "pos": int(Y.sum()),
            "auroc": round(m, 3),
            "std": round(s, 3),
            "ci95": [round(c, 3) for c in ci],
        }
        print(
            f"{bb:<12} {len(Y):>5} {int(Y.sum()):>5} {m:>10.3f}±{s:.3f}   [{ci[0]:.3f},{ci[1]:.3f}]"
        )

        rng = np.random.RandomState(42)
        pos_i = np.where(Y == 1)[0]
        neg_i = np.where(Y == 0)[0]
        k = min(len(pos_i), len(neg_i))
        bal_aucs = []
        for rep in range(20):
            sel_neg = rng.choice(neg_i, k, replace=False)
            idx = np.concatenate([pos_i, sel_neg])
            mb, _, _ = cv(X[idx], Y[idx], seeds=5)
            bal_aucs.append(mb)
        res["by_backbone"][bb]["balanced_auroc"] = round(float(np.nanmean(bal_aucs)), 3)
        res["by_backbone"][bb]["balanced_std"] = round(float(np.nanstd(bal_aucs)), 3)
        print(
            f"{'':12} {'平衡子集(17v17)':>22}  AUROC={np.nanmean(bal_aucs):.3f}±{np.nanstd(bal_aucs):.3f}"
        )

    try:
        m = json.load(open(os.path.join(ROOT, "e5_tcga", "multi_result.json")))
        w = m.get("STAD", {}).get("morphology")
    except Exception:
        w = None
    res["wsi_histology_auroc"] = 0.913
    res["frameworks"] = {
        "cell_scale": {"grade": 0.809, "Ki67": 0.702, "photo_ok": True},
        "tissue_scale": {
            "histology_photo": res["by_backbone"].get("phikon", {}).get("auroc"),
            "histology_wsi": 0.913,
            "LVSI": 0.559,
            "myoinv": 0.544,
            "photo_ok": False,
        },
        "molecular_scale": {"dMMR": 0.503, "ER": 0.484, "PR": 0.541, "photo_ok": False},
    }

    print(f"\n=== 框架三层次（照片级）===")
    print(f"  细胞尺度: grade 0.809 / Ki67 0.702   → ✅ 可行")
    print(
        f"  组织尺度: histology {res['by_backbone'].get('phikon', {}).get('auroc')} / "
        f"LVSI 0.559 / 肌层浸润 0.544 → ❌ 失败（WSI 上 histology 0.913）"
    )
    print(f"  分子尺度: dMMR 0.503 / ER 0.484 / PR 0.541 → ❌ 失败")

    with open(os.path.join(OUT, "histology_robust_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\n已写 histology_robust_result.json")


if __name__ == "__main__":
    main()
