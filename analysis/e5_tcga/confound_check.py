# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import json
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

OUT = os.path.dirname(os.path.abspath(__file__))
POOLED = os.path.join(OUT, "pooled_features.npz")
STUDY = "ucec_tcga_pan_can_atlas_2018"
N_SEEDS = 10
ATTRS = [
    "GRADE",
    "TISSUE_SOURCE_SITE_CODE",
    "ANEUPLOIDY_SCORE",
    "FRACTION_GENOME_ALTERED",
    "MSI_SCORE_MANTIS",
    "MUTATION_COUNT",
]


def fetch(attr, dtype="PATIENT"):
    url = (
        f"https://www.cbioportal.org/api/studies/{STUDY}/clinical-data"
        f"?clinicalDataType={dtype}&attributeId={attr}&projection=SUMMARY"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def get_confounders(pids):

    cache = os.path.join(OUT, "tcga_confound.csv")
    if os.path.exists(cache):
        c = pd.read_csv(cache)
        need = {"patient_id", "GRADE", "TISSUE_SOURCE_SITE_CODE"} | set(ATTRS)
        if need <= set(c.columns):
            print(f"  使用缓存 {os.path.basename(cache)}（{len(c)} 行），跳过网络拉取")
            return c[[x for x in c.columns if x != "y" and x != "oof" and x != "tss"]]
    data = {}
    for a in ATTRS:
        for dtype in ["PATIENT", "SAMPLE"]:
            try:
                recs = fetch(a, dtype)
                if not recs:
                    continue
                for r in recs:
                    pid = r["patientId"]
                    if pid in pids:
                        data.setdefault(pid, {})[a] = r["value"]
                print(f"  {a} ({dtype}): {len(recs)} 条")
                break
            except Exception as e:
                print(f"  {a} ({dtype}): ERR {str(e)[:60]}")
    return pd.DataFrame([{"patient_id": k, **v} for k, v in data.items()])


def oof_predictions(X, Y, seeds=10):
    """Preserved computation; see docs/experiments.md for its protocol."""

    aucs, oof_sum, per_seed = [], np.zeros(len(Y)), []
    for s in range(seeds):
        o = np.zeros(len(Y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, Y):
            if len(np.unique(Y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(X[tr]), Y[tr])
            pr = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(Y[te], pr))
            oof_sum[te] += pr
            o[te] = pr
        per_seed.append(o)
    return oof_sum / seeds, float(np.mean(aucs)), float(np.std(aucs)), len(aucs), per_seed


def main():
    z = np.load(POOLED, allow_pickle=True)
    X, Y, pids = z["X"], z["Y"], z["pids"]
    print(f"数据: {X.shape}, dMMR {int(Y.sum())}/{len(Y)}")

    print("\n拉取混淆因子...")
    conf = get_confounders(set(pids))
    df = pd.DataFrame({"patient_id": pids, "y": Y})
    df = df.merge(conf, on="patient_id", how="left")
    oof, fm, fs, nf, oof_seeds = oof_predictions(X, Y, seeds=N_SEEDS)
    df["oof"] = oof

    res = {
        "n": len(df),
        "n_seeds": N_SEEDS,
        "auroc_all": round(fm, 3),
        "auroc_all_std": round(fs, 3),
        "n_folds": nf,
        "auroc_oof_aggregate": round(roc_auc_score(Y, df["oof"]), 3),
        "auroc_note": (
            "auroc_all = 逐折 AUROC 均值（与 ucec_wsi_tasks.py 同协议）；"
            "auroc_oof_aggregate = 种子平均 OOF 的整体 AUROC，二者为不同估计量，"
            "论文统一引用前者"
        ),
    }
    print(f"\n总体 AUROC = {fm:.4f} ± {fs:.4f}（{nf} 折，折均值，= ucec_wsi_tasks 口径）")
    print(f"          OOF 聚合值 = {res['auroc_oof_aggregate']:.4f}")

    print(f"\n{'混淆因子':<28} {'可用':>5} {'与OOF预测相关':>16} {'p值':>10}")
    print("-" * 64)
    res["correlations"] = {}
    for c in ["ANEUPLOIDY_SCORE", "FRACTION_GENOME_ALTERED", "MUTATION_COUNT", "MSI_SCORE_MANTIS"]:
        if c not in df.columns:
            continue
        sub = df[df[c].notna()].copy()
        if len(sub) < 30:
            continue
        v = pd.to_numeric(sub[c], errors="coerce")
        ok = v.notna()
        rho, p = spearmanr(sub["oof"][ok], v[ok])
        res["correlations"][c] = {
            "n": int(ok.sum()),
            "spearman_rho": round(float(rho), 3),
            "p": float(p),
        }
        print(f"{c:<28} {int(ok.sum()):>5} {rho:>16.3f} {p:>10.2e}")

    if "GRADE" in df.columns:
        print(f"\n[分级分层] GRADE 分布: {df['GRADE'].value_counts().to_dict()}")
        res["by_grade"] = {}
        for g in sorted(df["GRADE"].dropna().unique()):
            sub = df[df["GRADE"] == g]
            if len(sub) >= 30 and 0 < sub["y"].sum() < len(sub):
                a = roc_auc_score(sub["y"], sub["oof"])
                res["by_grade"][g] = {
                    "n": len(sub),
                    "pos": int(sub["y"].sum()),
                    "auroc": round(a, 3),
                }
                print(f"  仅 {g}: n={len(sub)}, dMMR={int(sub['y'].sum())}, AUROC={a:.3f}")

        gnum = df["GRADE"].map({"G1": 1, "G2": 2, "G3": 3})
        ok = gnum.notna()
        if ok.sum() > 30:
            a_grade = roc_auc_score(df["y"][ok], gnum[ok])
            res["grade_only_auroc"] = round(a_grade, 3)
            print(f"  ★ GRADE-only 基线 AUROC = {a_grade:.3f}（对照模型 {res['auroc_all']}）")

            r, p = spearmanr(gnum[ok], df["y"][ok])
            res["grade_vs_dmmr"] = {"spearman": round(float(r), 3), "p": float(p)}
            print(f"  GRADE 与 dMMR 相关: rho={r:.3f}, p={p:.2e}")

    tss_col = "TISSUE_SOURCE_SITE_CODE" if "TISSUE_SOURCE_SITE_CODE" in df.columns else None
    if tss_col:
        df["tss"] = df[tss_col]
        print(f"\n[TSS 批次] 中心数: {df['tss'].nunique()}")
        # leave-one-TSS-out
        aucs = []
        for t in df["tss"].dropna().unique():
            te = df["tss"] == t
            tr = ~te
            if te.sum() < 15 or len(set(df["y"][tr])) < 2:
                continue
            if df["y"][te].sum() == 0 or df["y"][te].sum() == te.sum():
                continue
            sc = StandardScaler().fit(X[tr.values])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(X[tr.values]), df["y"][tr].values)
            pr = clf.predict_proba(sc.transform(X[te.values]))[:, 1]
            aucs.append(roc_auc_score(df["y"][te].values, pr))
        if aucs:
            res["leave_one_tss_out"] = {
                "n_folds": len(aucs),
                "mean_auroc": round(float(np.mean(aucs)), 3),
                "std": round(float(np.std(aucs)), 3),
                "folds": [round(a, 3) for a in aucs],
            }
            print(
                f"  Leave-one-TSS-out: {len(aucs)} 折, AUROC {np.mean(aucs):.3f}±{np.std(aucs):.3f}"
            )

        tss_r = df.groupby("tss")["oof"].mean()
        res["tss_pred_variance"] = round(float(tss_r.var() / df["oof"].var()), 4)
        print(f"  TSS 间预测均值方差占比（{N_SEEDS} seeds 平均）: {res['tss_pred_variance']:.4f}")

        tss_arr = df["tss"].values
        ratios = [float(pd.Series(o).groupby(tss_arr).mean().var() / np.var(o)) for o in oof_seeds]
        res["tss_pred_variance_singlesplit"] = {
            "mean": round(float(np.mean(ratios)), 4),
            "std": round(float(np.std(ratios)), 4),
            "min": round(min(ratios), 4),
            "max": round(max(ratios), 4),
            "per_seed": [round(x, 4) for x in ratios],
        }
        print(
            f"  TSS 间预测均值方差占比（单次划分，{N_SEEDS} 个种子）: "
            f"{np.mean(ratios):.4f} ± {np.std(ratios):.4f} "
            f"(范围 {min(ratios):.4f}–{max(ratios):.4f})"
        )

    print("\n=== 小结 ===")
    print(f"总体 AUROC: {res['auroc_all']}")
    if "grade_only_auroc" in res:
        print(f"GRADE-only 基线: {res['grade_only_auroc']}")
    if "by_grade" in res:
        for g, v in res["by_grade"].items():
            print(f"  分层 {g}: AUROC {v['auroc']} (n={v['n']})")
    if "leave_one_tss_out" in res:
        print(
            f"Leave-one-TSS-out: {res['leave_one_tss_out']['mean_auroc']}±{res['leave_one_tss_out']['std']}"
        )

    with open(os.path.join(OUT, "confound_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    df.to_csv(os.path.join(OUT, "tcga_confound.csv"), index=False, encoding="utf-8-sig")
    print("\n已写 confound_result.json 与 tcga_confound.csv")


if __name__ == "__main__":
    main()
