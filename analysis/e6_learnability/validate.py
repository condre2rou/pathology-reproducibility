# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json
import glob
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS2 = os.path.join(ROOT, "e2_train", "runs")
T5 = os.path.join(ROOT, "e5_tcga")
LABELS_EC = os.path.join(ROOT, "e1_label_engine", "outputs", "labels_ec.csv")

N_LIST = [10, 20, 40]
SEEDS = 10


def _pca_project(X, Y, max_comp=20):
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    k = int(min(max_comp, max(2, len(Y) // 3), Xs.shape[1], len(Y) - 1))
    pca = PCA(n_components=k, random_state=0).fit(Xs)
    return pca.transform(Xs)


def m_fisher(Z, Y):
    """Preserved computation; see docs/experiments.md for its protocol."""
    mu0, mu1 = Z[Y == 0].mean(0), Z[Y == 1].mean(0)
    S_B = float(((mu1 - mu0) ** 2).sum())
    S_W = float(Z[Y == 0].var(0).sum() + Z[Y == 1].var(0).sum())
    return S_B / (S_W + 1e-9)


def m_centroid(Z, Y):
    """Preserved computation; see docs/experiments.md for its protocol."""
    mu0, mu1 = Z[Y == 0].mean(0), Z[Y == 1].mean(0)
    return float(np.sqrt(((mu1 - mu0) ** 2).sum()))


def m_knn_loo(X, Y, k=1):
    """Preserved computation; see docs/experiments.md for its protocol."""
    n = len(Y)
    if n < 6:
        return np.nan
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    correct = 0
    for i in range(n):
        nn = np.argsort(D[i])[:k]
        pred = np.round(Y[nn].mean())
        correct += int(pred == Y[i])
    return correct / n


def m_silhouette_ratio(Z, Y):
    """Preserved computation; see docs/experiments.md for its protocol."""
    try:
        sc = StandardScaler().fit(Z)
        Zi = sc.transform(Z)
        mu0, mu1 = Zi[Y == 0].mean(0), Zi[Y == 1].mean(0)
        d = np.sqrt(((mu1 - mu0) ** 2).sum())

        r0 = np.sqrt(((Zi[Y == 0] - mu0) ** 2).sum(1)).mean() if (Y == 0).sum() else 1
        r1 = np.sqrt(((Zi[Y == 1] - mu1) ** 2).sum(1)).mean() if (Y == 1).sum() else 1
        return float(d / (r0 + r1 + 1e-9))
    except Exception:
        return np.nan


def m_fewshot_auroc(X, Y, seed=0):
    """Preserved computation; see docs/experiments.md for its protocol."""
    try:
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        loo = LeaveOneOut()
        ys, ps = [], []
        for tr, te in loo.split(Xs):
            if len(set(Y[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
            clf.fit(Xs[tr], Y[tr])
            ps.append(clf.predict_proba(Xs[te])[0, 1])
            ys.append(Y[te][0])
        if len(set(ys)) < 2:
            return np.nan
        return roc_auc_score(ys, ps)
    except Exception:
        return np.nan


METRICS = {
    "fisher": lambda X, Y: m_fisher(_pca_project(X, Y), Y),
    "centroid": lambda X, Y: m_centroid(_pca_project(X, Y), Y),
    "silhouette_ratio": lambda X, Y: m_silhouette_ratio(_pca_project(X, Y), Y),
    "knn_loo": lambda X, Y: m_knn_loo(X, Y, k=1),
    "knn_loo3": lambda X, Y: m_knn_loo(X, Y, k=3),
    "fewshot_auroc": lambda X, Y: m_fewshot_auroc(X, Y),
}


def tasks_micro(labels_ec, feats):
    """Preserved computation; see docs/experiments.md for its protocol."""
    lab = labels_ec.drop_duplicates("folder")
    lab = lab.set_index(lab["year"] + "_" + lab["folder"])
    defs = {
        "grade3": lambda r: (
            (1 if r["grade"] == "3" else 0) if r["grade"] in ("1", "2", "3") else None
        ),
        "Ki67_high30": lambda r: (
            (1 if float(r["ki67"]) >= 30 else 0)
            if str(r["ki67"]).replace(".", "", 1).isdigit()
            else None
        ),
        "p53_abn": lambda r: (1 if r["p53"] == "abn" else 0) if r["p53"] in ("wt", "abn") else None,
        "dMMR": lambda r: (
            (1 if r["mmr"] == "dMMR" else 0) if r["mmr"] in ("pMMR", "dMMR") else None
        ),
        "ER_pos": lambda r: (1 if r["er"] == "pos" else 0) if r["er"] in ("pos", "neg") else None,
        "PR_pos": lambda r: (1 if r["pr"] == "pos" else 0) if r["pr"] in ("pos", "neg") else None,
    }
    out = {}
    for name, fn in defs.items():
        idx = [p for p in lab.index if p in feats]
        ys = [fn(lab.loc[p]) for p in idx]
        keep = [i for i, y in enumerate(ys) if y is not None]
        if len(keep) < 60:
            continue
        X = np.stack([feats[idx[i]] for i in keep])
        Y = np.array([ys[i] for i in keep])
        if len(set(Y)) < 2 or min(np.bincount(Y)) < 15:
            continue
        out[name] = (X, Y)
    return out


def tasks_wsi():
    """Preserved computation; see docs/experiments.md for its protocol."""
    out = {}
    # UCEC
    p = os.path.join(T5, "pooled_features.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        out["WSI-UCEC-dMMR"] = (z["X"], z["Y"])
    # COAD / STAD / BRCA
    for key, f in [
        ("COAD", "COAD.npz"),
        ("STAD", "STAD.npz"),
        ("BRCA_IDC", "BRCA_IDC.npz"),
        ("BRCA_IDC_PR", "BRCA_IDC_PR.npz"),
    ]:
        p = os.path.join(T5, "pooled", f)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            out[f"WSI-{key}"] = (z["X"], z["Y"])
    return out


def true_auroc(X, Y, seeds=5):
    aucs = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, Y):
            if len(np.unique(Y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(X[tr]), Y[tr])
            aucs.append(roc_auc_score(Y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs))


def sample_balanced(Y, n, rng):
    pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
    hp, hn = min(len(pos), n // 2), min(len(neg), n // 2)
    return np.concatenate([rng.choice(pos, hp, replace=False), rng.choice(neg, hn, replace=False)])


def main():
    labels_ec = pd.read_csv(LABELS_EC, dtype=str)
    datasets = {}

    for bb, f in [("phikon", "feats_phikon.npz"), ("ssl", "feats_ssl.npz")]:
        p = os.path.join(RUNS2, f)
        if os.path.exists(p):
            feats = np.load(p, allow_pickle=True)["feats"].item()
            for t, (X, Y) in tasks_micro(labels_ec, feats).items():
                datasets[f"{bb}@{t}"] = (X, Y)

    datasets.update(tasks_wsi())

    print(f"数据点 {len(datasets)} 个\n")
    rows = []
    for name, (X, Y) in datasets.items():
        ta = true_auroc(X, Y)
        rows.append({"dataset": name, "n": len(Y), "pos": int(Y.sum()), "true_auroc": round(ta, 3)})
        print(f"  {name:<26} n={len(Y):>4} pos={int(Y.sum()):>4}  true AUROC={ta:.3f}")

    df = pd.DataFrame(rows)

    print(f"\n计算少样本可分性度量（seeds={SEEDS}）...")
    for n in N_LIST:
        for mname, fn in METRICS.items():
            vals = []
            for name, (X, Y) in datasets.items():
                if len(Y) < n + 20:
                    vals.append(np.nan)
                    continue
                v = []
                for s in range(SEEDS):
                    rng = np.random.RandomState(s)
                    idx = sample_balanced(Y, n, rng)
                    if len(set(Y[idx])) < 2:
                        continue
                    try:
                        v.append(fn(X[idx], Y[idx]))
                    except Exception:
                        pass
                vals.append(np.nanmean(v) if v else np.nan)
            df[f"{mname}_n{n}"] = vals

    print(f"\n{'度量':<20} {'n':>4} {'Spearman':>10} {'p':>10} {'Pearson':>10}")
    print("-" * 60)
    res = {}
    for n in N_LIST:
        for mname in METRICS:
            col = f"{mname}_n{n}"
            sub = df[[col, "true_auroc"]].dropna()
            if len(sub) < 8:
                continue
            rho, p = spearmanr(sub[col], sub["true_auroc"])
            r, _ = pearsonr(sub[col], sub["true_auroc"])
            res[col] = {
                "n": len(sub),
                "spearman": round(float(rho), 3),
                "p": float(p),
                "pearson": round(float(r), 3),
            }
            flag = "★" if abs(rho) > 0.7 else (" " if abs(rho) > 0.5 else " ")
            print(f"{mname:<20} {n:>4} {rho:>10.3f} {p:>10.2e} {r:>10.3f} {flag}")

    df.to_csv(
        os.path.join(ROOT, "e6_learnability", "learnability_data.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    with open(os.path.join(ROOT, "e6_learnability", "validate_result.json"), "w") as f:
        json.dump(
            {"n_datasets": len(datasets), "correlations": res}, f, indent=2, ensure_ascii=False
        )
    print("\n已写 learnability_data.csv 与 validate_result.json")


if __name__ == "__main__":
    main()
