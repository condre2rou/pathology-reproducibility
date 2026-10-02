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


def extract_lvsi(text):
    """Preserved computation; see docs/experiments.md for its protocol."""
    if not text:
        return None
    if re.search(r"未见.{0,10}脉管.{0,10}(癌栓|瘤栓)", text):
        return "neg"
    if re.search(r"脉管.{0,8}(癌栓|瘤栓)", text):
        return "pos"
    return None


def extract_myoinv(text):
    """Preserved computation; see docs/experiments.md for its protocol."""
    if not text:
        return None
    if re.search(r"(侵及|浸润|达).{0,14}(深肌层|＞1/2|>1/2)", text):
        return "deep"
    if re.search(r"(侵及|浸润|达).{0,14}(浅肌层|<1/2|＜1/2|≤1/2)", text):
        return "inner"
    return None


def build_labels():
    """Preserved computation; see docs/experiments.md for its protocol."""
    recs_all = {}
    for year, cfg in YEAR_CONFIG.items():
        recs, _ = load_report(cfg)
        for seq, rec in recs.items():
            t = full_text(rec)
            recs_all[f"{year}_{seq}"] = {
                "lvsi": extract_lvsi(t),
                "myoinv": extract_myoinv(t),
            }
    return recs_all


def cv(X, Y, seeds=10):
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
        return np.nan, np.nan, np.nan
    a = np.array(aucs)

    rng = np.random.RandomState(0)
    boot = [np.mean(rng.choice(a, len(a), replace=True)) for _ in range(1000)]
    return (
        float(a.mean()),
        float(a.std()),
        (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
    )


def main():
    lab = pd.read_csv(LABELS, dtype=str)
    lab["pid"] = lab["year"] + "_" + lab["folder"]
    lab["hkey"] = lab["year"] + "_" + lab["name_seq"]
    extra = build_labels()
    from collections import Counter

    print("抽取分布：")
    print(f"  LVSI: {Counter(v['lvsi'] for v in extra.values() if v['lvsi'])}")
    print(f"  肌层浸润: {Counter(v['myoinv'] for v in extra.values() if v['myoinv'])}")

    def hist_of(text):
        if not text:
            return None
        if re.search(r"浆液性", text):
            return "serous"
        if re.search(r"子宫内膜样", text):
            return "endometrioid"
        return None

    results = {}
    backbones = [
        ("phikon", "feats_phikon.npz"),
        ("ssl", "feats_ssl.npz"),
        ("imagenet", "feats_imagenet.npz"),
    ]

    print(f"\n{'任务':<22} {'backbone':<10} {'N':>5} {'正类':>5} {'AUROC':>16} {'95%CI':>18}")
    print("-" * 82)

    for bb, fname in backbones:
        fp = os.path.join(ROOT, "e2_train", "runs", fname)
        if not os.path.exists(fp):
            continue
        feats = np.load(fp, allow_pickle=True)["feats"].item()

        for task, key in [("LVSI(组织尺度)", "lvsi"), ("肌层浸润深度(组织尺度)", "myoinv")]:
            X, Y = [], []
            for _, r in lab.iterrows():
                v = extra.get(r["hkey"], {}).get(key)
                if v is None or r["pid"] not in feats:
                    continue
                X.append(feats[r["pid"]])
                Y.append(1 if v in ("pos", "deep") else 0)
            if len(X) < 60 or min(Y.count(0), Y.count(1)) < 15:
                continue
            m, s, ci = cv(np.stack(X), np.array(Y))
            results[f"{bb}_{key}"] = {
                "n": len(Y),
                "pos": Y.count(1),
                "auroc": round(m, 3),
                "std": round(s, 3),
                "ci95": [round(c, 3) for c in ci],
            }
            print(
                f"{task:<22} {bb:<10} {len(Y):>5} {Y.count(1):>5} {m:>10.3f}±{s:.3f} "
                f"  [{ci[0]:.3f},{ci[1]:.3f}]"
            )

    print(f"\n--- 对照：细胞尺度任务（应成功）---")
    ref = {"phikon": (0.809, 0.702), "ssl": (0.730, 0.597), "imagenet": (0.726, 0.545)}
    for bb, (g, k) in ref.items():
        print(f"  {bb:<10} grade={g:.3f}  Ki67={k:.3f}")

    with open(os.path.join(OUT, "validate_scale_result.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n已写 validate_scale_result.json")


if __name__ == "__main__":
    main()
