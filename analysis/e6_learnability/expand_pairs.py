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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "e1_label_engine"))
T5 = os.path.join(ROOT, "e5_tcga")
OUT = os.path.join(ROOT, "e6_learnability")

from align_seq_key import load_report, YEAR_CONFIG  # noqa: E402
from extract_rules import full_text  # noqa: E402


PHOTO = {
    "grade3": 0.809,
    "Ki67_high30": 0.702,
    "p53_abn": 0.566,
    "PR_pos": 0.541,
    "dMMR": 0.503,
    "ER_pos": 0.484,
    "histology": None,
}


def extract_histology(text):
    """Preserved computation; see docs/experiments.md for its protocol."""
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


def build_photo_histology():
    """Preserved computation; see docs/experiments.md for its protocol."""
    out = {}
    for year, cfg in YEAR_CONFIG.items():
        recs, _ = load_report(cfg)
        for seq, rec in recs.items():
            h = extract_histology(full_text(rec))
            if h:
                out[f"{year}_{seq}"] = h
    return out


def cv(X, Y, seeds=10):
    aucs, auprs = [], []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, Y):
            if len(np.unique(Y[te])) < 2 or len(np.unique(Y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(X[tr]), Y[tr])
            pr = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(Y[te], pr))
            auprs.append(average_precision_score(Y[te], pr))
    return (
        (float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(auprs)))
        if aucs
        else (np.nan,) * 3
    )


def eval_photo_histology(ph):
    """Preserved computation; see docs/experiments.md for its protocol."""
    lab = pd.read_csv(
        os.path.join(ROOT, "e1_label_engine", "outputs", "labels_ec_patient.csv"), dtype=str
    )
    lab["pid"] = lab["year"] + "_" + lab["folder"]
    lab["hkey"] = lab["year"] + "_" + lab["name_seq"]
    fp = os.path.join(ROOT, "e2_train", "runs", "feats_phikon.npz")
    if not os.path.exists(fp):
        return None
    feats = np.load(fp, allow_pickle=True)["feats"].item()
    X, Y = [], []
    for _, r in lab.iterrows():
        h = ph.get(r["hkey"])
        if h not in ("endometrioid", "serous"):
            continue
        if r["pid"] not in feats:
            continue
        X.append(feats[r["pid"]])
        Y.append(1 if h == "serous" else 0)
    if len(X) < 40 or min(Y.count(0), Y.count(1)) < 5:
        return None
    m, s, ap = cv(np.stack(X), np.array(Y))
    return {
        "n": len(Y),
        "pos": Y.count(1),
        "auroc": round(m, 3),
        "std": round(s, 3),
        "auprc": round(ap, 3),
    }


def main():

    ph = build_photo_histology()
    from collections import Counter

    print(f"照片级组织学抽取：{len(ph)} 例")
    print(f"  {Counter(ph.values()).most_common()}")

    photo_hist = eval_photo_histology(ph)
    if photo_hist:
        PHOTO["histology"] = photo_hist["auroc"]
        print(
            f"  [照片级] 组织学亚型(内膜样 vs 浆液性) n={photo_hist['n']} "
            f"pos={photo_hist['pos']} AUROC={photo_hist['auroc']:.3f}±{photo_hist['std']:.3f}"
        )
    else:
        print("  [照片级] 组织学亚型样本不足，跳过")

    ex = pd.read_csv(os.path.join(T5, "tcga_extra_labels.csv"), dtype=str)
    onco = dict(zip(ex.patient_id, ex.oncotree))
    z = np.load(os.path.join(T5, "pooled_features.npz"), allow_pickle=True)
    X, pids = z["X"], list(z["pids"])
    print(f"\nWSI 特征：{X.shape}（{len(pids)} 患者）")

    idx, y = [], []
    for i, p in enumerate(pids):
        o = onco.get(p)
        if o == "UEC":
            idx.append(i)
            y.append(0)
        elif o == "USC":
            idx.append(i)
            y.append(1)
    print(f"  组织学亚型任务：UEC={y.count(0)} / USC={y.count(1)}")
    wsi_hist = None
    if len(idx) >= 40 and min(y.count(0), y.count(1)) >= 10:
        m, s, ap = cv(X[idx], np.array(y))
        wsi_hist = {
            "n": len(idx),
            "pos": y.count(1),
            "auroc": round(m, 3),
            "std": round(s, 3),
            "auprc": round(ap, 3),
        }
        print(f"  [WSI] 组织学亚型(UEC vs USC)  AUROC={m:.3f}±{s:.3f}")

    u = json.load(open(os.path.join(T5, "ucec_wsi_tasks.json")))
    m2 = json.load(open(os.path.join(T5, "multi_result.json")))
    wsi_tasks = {
        "grade3": {"auroc": u["grade"]["auroc"], "n": u["grade"]["n"], "cohort": "TCGA-UCEC"},
        "dMMR": {"auroc": u["dMMR"]["auroc"], "n": u["dMMR"]["n"], "cohort": "TCGA-UCEC"},
        "ER_pos": {
            "auroc": m2["BRCA_IDC"]["molecular"]["auroc"],
            "n": m2["BRCA_IDC"]["molecular"]["n"],
            "cohort": "TCGA-BRCA",
        },
        "PR_pos": {
            "auroc": m2["BRCA_IDC_PR"]["molecular"]["auroc"],
            "n": m2["BRCA_IDC_PR"]["molecular"]["n"],
            "cohort": "TCGA-BRCA",
        },
    }
    if wsi_hist:
        wsi_tasks["histology"] = {
            "auroc": wsi_hist["auroc"],
            "n": wsi_hist["n"],
            "cohort": "TCGA-UCEC",
        }

    rows = []
    for t, pv in PHOTO.items():
        if pv is None:
            continue
        w = wsi_tasks.get(t)
        if not w:
            continue
        rows.append(
            {
                "task": t,
                "photo_auroc": pv,
                "wsi_auroc": w["auroc"],
                "wsi_cohort": w["cohort"],
                "wsi_n": w["n"],
                "gap": round(w["auroc"] - pv, 3),
                "category": (
                    "morphology"
                    if t in ("grade3", "histology")
                    else ("proliferation" if t == "Ki67_high30" else "molecular/IHC")
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values("gap", ascending=False)
    df.to_csv(os.path.join(OUT, "pairing_table.csv"), index=False, encoding="utf-8-sig")

    print(f"\n=== 跨模态配对表（{len(df)} 对）===")
    print(f"{'任务':<16} {'类别':<16} {'照片级':>8} {'WSI':>8} {'差距':>8}")
    print("-" * 62)
    for _, r in df.iterrows():
        print(
            f"{r['task']:<16} {r['category']:<16} {r['photo_auroc']:>8.3f} {r['wsi_auroc']:>8.3f} {r['gap']:>+8.3f}"
        )

    with open(os.path.join(OUT, "pairing_table.json"), "w") as f:
        json.dump(
            {"pairs": df.to_dict("records"), "photo_histology_dist": dict(Counter(ph.values()))},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("\n已写 pairing_table.csv / .json")


if __name__ == "__main__":
    main()
