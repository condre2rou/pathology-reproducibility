# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate import tasks_micro, tasks_wsi, true_auroc, sample_balanced  # noqa: E402
from calibrate import m_knn_loo, m_fisher_norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS2 = os.path.join(ROOT, "e2_train", "runs")
LABELS_EC = os.path.join(ROOT, "e1_label_engine", "outputs", "labels_ec.csv")
OUT = os.path.join(ROOT, "e6_learnability")

N_GRID = [10, 20, 40, 80]
N_SEEDS = 30


def fewshot_auroc(X, Y):
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


def power_law(n, a_inf, c, alpha):
    return a_inf - c * np.power(n, -alpha)


def fit_curve(ns, aucs):
    """Preserved computation; see docs/experiments.md for its protocol."""
    ns = np.asarray(ns, float)
    aucs = np.asarray(aucs, float)
    ok = ~np.isnan(aucs)
    if ok.sum() < 3:
        return np.nan, None
    try:
        p0 = [min(1.0, aucs.max() + 0.1), 0.5, 0.5]
        popt, _ = curve_fit(
            power_law,
            ns[ok],
            aucs[ok],
            p0=p0,
            maxfev=20000,
            bounds=([0.4, 0, 0.01], [1.0, 5.0, 3.0]),
        )
        return float(popt[0]), popt
    except Exception:
        try:
            f = lambda n, a_inf, c: power_law(n, a_inf, c, 0.5)
            popt, _ = curve_fit(
                f,
                ns[ok],
                aucs[ok],
                p0=[aucs.max() + 0.1, 0.5],
                maxfev=20000,
                bounds=([0.4, 0], [1.0, 5.0]),
            )
            return float(popt[0]), None
        except Exception:
            return np.nan, None


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

    cache = os.path.join(OUT, "curve_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        curves, extra = z["curves"].item(), z["extra"].item()
        print("复用学习曲线缓存")
    else:
        curves, extra = {}, {}
        for k in names:
            X, Y = datasets[k]
            cur = {}
            for n in N_GRID:
                if 2 * n > len(Y):
                    continue
                vals, ksep, fsep = [], [], []
                for b in range(N_SEEDS):
                    rng = np.random.RandomState(b)
                    idx = sample_balanced(Y, n, rng)
                    if len(set(Y[idx])) < 2:
                        continue
                    vals.append(fewshot_auroc(X[idx], Y[idx]))
                    ksep.append(m_knn_loo(X[idx], Y[idx]))
                    fsep.append(m_fisher_norm(X[idx], Y[idx]))
                cur[n] = (float(np.nanmean(vals)), float(np.nanmean(ksep)), float(np.nanmean(fsep)))
            curves[k] = cur
            print(f"  {k} done", flush=True)
        np.savez(cache, curves=curves, extra=extra)

    rows = []
    for k in names:
        cur = curves[k]
        ns = sorted(cur)
        aucs = [cur[n][0] for n in ns]
        a_inf, _ = fit_curve(ns, aucs)
        rows.append(
            {
                "dataset": k,
                "true": None,
                "n_max": max(ns),
                "auc_nmax": cur[max(ns)][0],
                "a_inf": a_inf,
                "knn_nmax": cur[max(ns)][1],
                "fisher_nmax": cur[max(ns)][2],
                "curve": {int(n): round(cur[n][0], 3) for n in ns},
            }
        )
    df = pd.DataFrame(rows)
    df["true"] = true_auc

    print(f"\n{'数据集':<26} {'真实':>7} {'n_max估计':>10} {'外推A∞':>9} {'偏差':>8}")
    print("-" * 64)
    for _, r in df.iterrows():
        b = r["a_inf"] - r["true"]
        print(
            f"{r['dataset']:<26} {r['true']:>7.3f} {r['auc_nmax']:>10.3f} {r['a_inf']:>9.3f} {b:>+8.3f}"
        )

    print(f"\n{'方法':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'Pearson':>8}")
    print("-" * 56)
    res = {}
    cands = {
        "naive (max n)": df["auc_nmax"].values,
        "extrapolation (A∞)": df["a_inf"].values,
        "kNN sep (max n)": df["knn_nmax"].values,
        "Fisher sep (max n)": df["fisher_nmax"].values,
    }
    for nm, pr in cands.items():
        ok = ~np.isnan(pr)
        if ok.sum() < 8:
            continue
        yt, pv = df["true"].values[ok], pr[ok]
        mae = float(np.mean(np.abs(pv - yt)))
        rmse = float(np.sqrt(np.mean((pv - yt) ** 2)))
        r2 = float(1 - np.sum((pv - yt) ** 2) / np.sum((yt - yt.mean()) ** 2))
        r, _ = pearsonr(pv, yt)
        rho, _ = spearmanr(pv, yt)
        res[nm] = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 3),
            "pearson": round(float(r), 3),
            "spearman": round(float(rho), 3),
            "n": int(ok.sum()),
        }
        print(f"{nm:<22} {mae:>7.3f} {rmse:>7.3f} {r2:>7.3f} {r:>8.3f}")

    with open(os.path.join(OUT, "learning_curve_result.json"), "w") as f:
        json.dump(
            {
                "n_grid": N_GRID,
                "n_seeds": N_SEEDS,
                "results": res,
                "per_dataset": df.drop(columns=["curve"]).to_dict("records"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    df.to_csv(os.path.join(OUT, "learning_curve_data.csv"), index=False, encoding="utf-8-sig")
    print("\n已写 learning_curve_result.json / learning_curve_data.csv")


if __name__ == "__main__":
    main()
