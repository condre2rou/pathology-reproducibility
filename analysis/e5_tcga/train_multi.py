# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json
import glob
import argparse
import urllib.request

import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get(
    "PATHOLOGY_HF_CACHE", os.path.join(os.path.dirname(__file__), "..", "models")
)
POOLED_DIR = os.path.join(OUT, "pooled")

CANCERS = {
    "COAD": dict(
        hf_dir="TCGA-COAD",
        study="coadread_tcga_pan_can_atlas_2018",
        mol_attr="SUBTYPE",
        mol_pos="MSI",
        mol_name="MSI",
    ),
    "STAD": dict(
        hf_dir="TCGA-STAD",
        study="stad_tcga_pan_can_atlas_2018",
        mol_attr="SUBTYPE",
        mol_pos="MSI",
        mol_name="MSI",
    ),
    "BRCA_IDC": dict(
        hf_dir="TCGA-BRCA_IDC",
        study="brca_tcga",
        mol_attr="ER_STATUS_BY_IHC",
        mol_pos="Positive",
        mol_name="ER",
    ),
    "BRCA_IDC_PR": dict(
        hf_dir="TCGA-BRCA_IDC",
        study="brca_tcga",
        mol_attr="PR_STATUS_BY_IHC",
        mol_pos="Positive",
        mol_name="PR",
    ),
}


def cbioportal(study, attr):
    """Preserved computation; see docs/experiments.md for its protocol."""
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
        except Exception as e:
            print(f"  [{attr}/{dtype}] 拉取失败: {str(e)[:60]}")
    return {}


def find_h5(hf_dir, pid):
    pat = os.path.join(
        CACHE,
        "datasets--W8Yi--tcga-wsi-uni2h-features",
        "snapshots",
        "*",
        hf_dir,
        "features",
        f"{pid}-*.h5",
    )
    return sorted(glob.glob(pat))


def pool(paths):
    vs = []
    for p in paths:
        with h5py.File(p, "r") as f:
            X = f["features"][:]
            X = X.reshape(-1, X.shape[-1])
            vs.append(X.mean(axis=0))
    return np.mean(vs, axis=0)


def cv(X, Y, seeds=5):
    aucs, auprs = [], []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, Y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(X[tr]), Y[tr])
            pr = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(Y[te], pr))
            auprs.append(average_precision_score(Y[te], pr))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(auprs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", default="")
    args = ap.parse_args()
    os.makedirs(POOLED_DIR, exist_ok=True)

    sub = pd.read_csv(os.path.join(OUT, "multi_subset.csv"), dtype=str)
    sub["label"] = sub["label"].astype(int)
    keys = [args.cancer] if args.cancer else list(CANCERS)

    all_res = {}
    for key in keys:
        cfg = CANCERS[key]
        s = sub[sub["cancer"] == key]
        if s.empty:
            print(f"[{key}] 无子集，跳过")
            continue
        print(f"\n=== {key} ===")

        pcache = os.path.join(POOLED_DIR, f"{key}.npz")
        if os.path.exists(pcache):
            z = np.load(pcache, allow_pickle=True)
            X, Ymol, pids = z["X"], z["Y"], z["pids"]
        else:
            Xs, Ys, P, miss = [], [], [], 0
            for _, r in s.iterrows():
                hs = find_h5(cfg["hf_dir"], r["patient_id"])
                if not hs:
                    miss += 1
                    continue
                try:
                    Xs.append(pool(hs))
                    Ys.append(r["label"])
                    P.append(r["patient_id"])
                except Exception:
                    miss += 1
            if not Xs:
                print("  无已下载特征，跳过")
                continue
            X, Ymol, pids = np.stack(Xs), np.array(Ys), np.array(P)
            np.savez(pcache, X=X, Y=Ymol, pids=pids)
            print(f"  池化 {len(Ymol)} 例（缺 {miss}）")

        ser = pd.Series(Ymol, index=pids)

        m, sd, ap_ = cv(X, Ymol)
        print(
            f"  [分子] {cfg['mol_name']:<4} n={len(Ymol):>4} 正类={int(Ymol.sum()):>3}  "
            f"AUROC={m:.3f}±{sd:.3f}  AUPRC={ap_:.3f}"
        )

        grade = cbioportal(cfg["study"], "GRADE")
        g_rows, g_y = [], []
        for i, p in enumerate(pids):
            g = grade.get(p)
            if not g:
                continue
            gs = str(g)
            if "GX" in gs or not gs:
                continue
            g_rows.append(i)
            g_y.append(1 if ("G3" in gs or "High" in gs) else 0)
        g_res = None
        if len(g_rows) >= 40 and len(set(g_y)) == 2 and min(g_y.count(0), g_y.count(1)) >= 10:
            Xg = X[g_rows]
            yg = np.array(g_y)
            gm, gsd, gap = cv(Xg, yg)
            g_res = {
                "n": len(g_rows),
                "pos": int(yg.sum()),
                "auroc": round(gm, 3),
                "std": round(gsd, 3),
                "dist": {"G3": int(yg.sum()), "G1G2": int((yg == 0).sum())},
            }
            print(
                f"  [形态] GRADE(G3 vs G1/2)  n={g_res['n']:>4} 正类={int(yg.sum()):>3}  "
                f"AUROC={gm:.3f}±{gsd:.3f}"
            )
        else:
            print(
                f"  [形态] GRADE 不可用（可用 {len(g_rows)} 例，类别 "
                f"{sorted(set(g_y)) if g_y else '无'}）"
            )

        all_res[key] = {
            "molecular": {
                "task": cfg["mol_name"],
                "n": len(Ymol),
                "pos": int(Ymol.sum()),
                "auroc": round(m, 3),
                "std": round(sd, 3),
                "auprc": round(ap_, 3),
            },
            "morphology": g_res,
        }

    with open(os.path.join(OUT, "multi_result.json"), "w") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print("\n已写 multi_result.json")

    print(f"\n{'癌种':<14} {'分子任务':<8} {'AUROC':>8}   {'形态(GRADE)':>12}")
    print("-" * 50)
    for k, v in all_res.items():
        mol = v["molecular"]
        mor = v.get("morphology")
        mo = f"{mor['auroc']:.3f}" if mor else "N/A"
        print(f"{k:<14} {mol['task']:<8} {mol['auroc']:>8.3f}   {mo:>12}")


if __name__ == "__main__":
    main()
