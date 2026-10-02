# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate import tasks_micro, tasks_wsi, true_auroc, sample_balanced  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS2 = os.path.join(ROOT, "e2_train", "runs")
LABELS_EC = os.path.join(ROOT, "e1_label_engine", "outputs", "labels_ec.csv")
OUT = os.path.join(ROOT, "e6_learnability")

N_LIST = [10, 20, 40]
N_SEEDS = 20


def auroc_of(X, Y):
    try:
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        ns = min(5, int(min(np.bincount(Y))))
        if ns < 2:
            return np.nan
        ys, ps = [], []
        for tr, te in StratifiedKFold(ns, shuffle=True, random_state=0).split(Xs, Y):
            if len(set(Y[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced")
            clf.fit(Xs[tr], Y[tr])
            ps.extend(clf.predict_proba(Xs[te])[:, 1])
            ys.extend(Y[te])
        return roc_auc_score(ys, ps) if len(set(ys)) == 2 else np.nan
    except Exception:
        return np.nan


def sel_random(X, Y, n, rng):
    return sample_balanced(Y, n, rng)


def sel_coreset(X, Y, n, rng):
    """Preserved computation; see docs/experiments.md for its protocol."""
    sel = []
    for cls in (0, 1):
        idx_c = np.where(Y == cls)[0]
        k = max(2, n // 2)
        if len(idx_c) <= k:
            sel.extend(idx_c)
            continue
        start = rng.choice(idx_c)
        chosen = [start]
        d = pairwise_distances(X[idx_c], X[[start]]).ravel()
        for _ in range(k - 1):
            nxt = idx_c[np.argmax(d)]
            chosen.append(nxt)
            d = np.minimum(d, pairwise_distances(X[idx_c], X[[nxt]]).ravel())
        sel.extend(chosen)
    return np.array(sel)


def sel_cluster(X, Y, n, rng):
    """Preserved computation; see docs/experiments.md for its protocol."""
    sel = []
    for cls in (0, 1):
        idx_c = np.where(Y == cls)[0]
        k = max(2, n // 2)
        if len(idx_c) <= k:
            sel.extend(idx_c)
            continue
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X[idx_c])
        for c in range(k):
            members = idx_c[km.labels_ == c]
            if len(members) == 0:
                continue
            dist = pairwise_distances(X[members], km.cluster_centers_[[c]]).ravel()
            sel.append(members[np.argmin(dist)])
    return np.array(sel)


def sel_uncertain(X, Y, n, rng):
    """Preserved computation; see docs/experiments.md for its protocol."""
    try:
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
        clf.fit(sc.transform(X), Y)
        p = clf.predict_proba(sc.transform(X))[:, 1]
        marg = np.abs(p - 0.5)
        sel = []
        for cls in (0, 1):
            idx_c = np.where(Y == cls)[0]
            k = max(2, n // 2)
            order = idx_c[np.argsort(marg[idx_c])]
            sel.extend(order[:k])
        return np.array(sel)
    except Exception:
        return sample_balanced(Y, n, rng)


def sel_strat_feat(X, Y, n, rng):
    """Preserved computation; see docs/experiments.md for its protocol."""
    try:
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        p1 = PCA(n_components=1, random_state=0).fit_transform(Xs).ravel()
        sel = []
        for cls in (0, 1):
            idx_c = np.where(Y == cls)[0]
            k = max(2, n // 2)
            if len(idx_c) <= k:
                sel.extend(idx_c)
                continue
            qs = np.quantile(p1[idx_c], np.linspace(0, 1, k + 2)[1:-1])
            for q in qs:
                j = idx_c[np.argmin(np.abs(p1[idx_c] - q))]
                sel.append(j)
        return np.array(sorted(set(sel)))
    except Exception:
        return sample_balanced(Y, n, rng)


STRATEGIES = {
    "random": sel_random,
    "coreset": sel_coreset,
    "cluster": sel_cluster,
    "uncertain": sel_uncertain,
    "strat_feat": sel_strat_feat,
}


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
    names = list(datasets)
    true_auc = np.array([true_auroc(*datasets[k]) for k in names])

    report = {"n_list": N_LIST, "n_seeds": N_SEEDS, "results": {}}
    for n in N_LIST:
        print(f"===== n = {n} =====")
        est = {s: [] for s in STRATEGIES}
        for k in names:
            X, Y = datasets[k]
            for sname, fn in STRATEGIES.items():
                vals = []
                for b in range(N_SEEDS):
                    rng = np.random.RandomState(b)
                    try:
                        idx = fn(X, Y, n, rng)
                    except Exception:
                        continue
                    if len(set(Y[idx])) < 2:
                        continue
                    vals.append(auroc_of(X[idx], Y[idx]))
                est[sname].append(np.nanmean(vals))
            print(f"  {k} done", flush=True)

        print(f"\n{'策略':<12} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'Pearson':>8} {'Spearman':>9}")
        print("-" * 56)
        for sname in STRATEGIES:
            pv = np.array(est[sname])
            ok = ~np.isnan(pv)
            if ok.sum() < 8:
                continue
            yt = true_auc[ok]
            pv2 = pv[ok]
            mae = float(np.mean(np.abs(pv2 - yt)))
            rmse = float(np.sqrt(np.mean((pv2 - yt) ** 2)))
            r2 = float(1 - np.sum((pv2 - yt) ** 2) / np.sum((yt - yt.mean()) ** 2))
            r, _ = pearsonr(pv2, yt)
            rho, _ = spearmanr(pv2, yt)
            report["results"][f"{sname}_n{n}"] = {
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "r2": round(r2, 3),
                "pearson": round(float(r), 3),
                "spearman": round(float(rho), 3),
                "n": int(ok.sum()),
            }
            print(f"{sname:<12} {mae:>7.3f} {rmse:>7.3f} {r2:>7.3f} {r:>8.3f} {rho:>9.3f}")
        print()

    with open(os.path.join(OUT, "active_select_result.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("已写 active_select_result.json")


if __name__ == "__main__":
    main()
