#!/usr/bin/env python3
"""Reviewer sensitivities; read E11 caches, write only E13 aggregates."""

import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
from pathlib import Path
import sys, json, argparse, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
E11 = ROOT / "e11_method_audit"
sys.path.insert(0, str(E11))
from data import load_tasks
from core import probe, metrics, decide
from run import test_membership, save
from summarize import reference_frame, decision_stats


def grouped_meta(a, y, groups, weights=None):
    a = np.asarray(a)
    y = np.asarray(y)
    groups = np.asarray(groups)
    p = np.full(len(y), np.nan)
    folds = []
    valid = np.isfinite(a)
    for group in np.unique(groups):
        tr = (groups != group) & valid
        te = (groups == group) & valid
        assert not np.any(tr & te)
        w = np.ones(tr.sum()) if weights is None else np.asarray(weights)[tr]
        # Weighted scaling and fitting use training cohorts only. Class weights
        # follow the same sample-weighted class counts, avoiding repetition bias.
        counts = np.bincount(y[tr], weights=w, minlength=2)
        cls = {i: float(counts.sum() / (2 * counts[i])) for i in (0, 1)}
        m = make_pipeline(
            StandardScaler(), LogisticRegression(C=1, class_weight=cls, max_iter=3000)
        )
        m.fit(
            a[tr, None], y[tr], standardscaler__sample_weight=w, logisticregression__sample_weight=w
        )
        p[te] = m.predict_proba(a[te, None])[:, 1]
        folds.append(
            dict(
                held_out=str(group),
                train_rows=int(tr.sum()),
                test_rows=int(te.sum()),
                training_groups=int(len(set(groups[tr]))),
            )
        )
    return p, folds


def resampled_oof(X, y, rng):
    # Resample records, then group all bootstrap copies in the same fold.
    sampled = np.concatenate(
        [rng.choice(np.flatnonzero(y == c), int((y == c).sum()), replace=True) for c in (0, 1)]
    )
    unique, counts = np.unique(sampled, return_counts=True)
    uy = y[unique]
    nf = min(5, int(np.bincount(uy, minlength=2).min()))
    if nf < 2:
        return None
    pred = np.empty(len(unique))
    for tr, te in StratifiedKFold(nf, shuffle=True, random_state=int(rng.randint(2**31 - 1))).split(
        unique, uy
    ):
        train = np.repeat(unique[tr], counts[tr])
        test = unique[te]
        assert not set(train) & set(test)
        pred[te] = probe().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return float(roc_auc_score(uy, pred, sample_weight=counts))


def paired_gain_interval(y, expanded, pilot, seed, n_boot=200):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    def wins(p):
        return (p[pos, None] > p[None, neg]).astype(float) + 0.5 * (p[pos, None] == p[None, neg])

    delta = wins(expanded) - wins(pilot)
    rng = np.random.RandomState(seed)
    wp = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)), size=n_boot)
    wn = rng.multinomial(len(neg), np.full(len(neg), 1 / len(neg)), size=n_boot)
    vals = np.sum((wp @ delta) * wn, axis=1) / (len(pos) * len(neg))
    return list(map(float, np.quantile(vals, [0.025, 0.975])))


def task_analysis(task):
    target = HERE / "outputs/reconstructed" / f"{task.name}.json"
    if target.exists():
        return task.name, "cached"
    old = json.loads((E11 / "outputs/operation" / f"{task.name}.json").read_text())
    rows = sorted([r for r in old["rows"] if r["sampling"] == "natural"], key=lambda r: r["seed"])
    diagnostic = {r["seed"] for r in rows[:3]}
    out = []
    for r in rows:
        seed = r["seed"]
        test = test_membership(task, seed)
        pool = ~test
        X, y = task.X[pool], task.y[pool]
        Xt, yt = task.X[test], task.y[test]
        ix = np.random.RandomState(10000 + seed).choice(len(y), 40, replace=False)
        pilot = (
            probe().fit(X[ix], y[ix]).predict_proba(Xt)[:, 1]
            if len(set(y[ix])) == 2
            else np.full(len(yt), 0.5)
        )
        expanded = probe().fit(X, y).predict_proba(Xt)[:, 1]
        assert abs(roc_auc_score(yt, pilot) - r["pilot_test_auc"]) < 1e-8
        assert abs(roc_auc_score(yt, expanded) - r["expanded_test_auc"]) < 1e-8
        record = dict(
            dataset=task.name,
            cohort=task.cohort,
            seed=seed,
            test_n=len(yt),
            test_positive=int(yt.sum()),
            test_negative=int((yt == 0).sum()),
            gain=r["gain"],
            gain_interval=paired_gain_interval(yt, expanded, pilot, 60000 + seed),
            diagnostic=seed in diagnostic,
        )
        if seed in diagnostic:
            vals = []
            if r["a40"] is not None:
                rng = np.random.RandomState(70000 + seed)
                vals = [
                    v for _ in range(100) if (v := resampled_oof(X[ix], y[ix], rng)) is not None
                ]
            interval = (
                list(map(float, np.quantile(vals, [0.025, 0.975]))) if len(vals) >= 50 else None
            )
            record.update(
                refit_interval=interval,
                valid_refits=len(vals),
                refit_draws=vals,
                conditional_pair_interval=r["ci"],
                refit_action=decide(r["a40"] if r["a40"] is not None else np.nan, interval),
                pair_action=r["action"],
            )
        out.append(record)
    save(target, dict(dataset=task.name, rows=out))
    return task.name, "computed"


def cached_analyses():
    ref = reference_frame(E11 / "outputs")
    a = ref.a40.values
    y = (ref.full.values - a <= 0.05).astype(int)
    p, folds = grouped_meta(a, y, ref.cohort.values)
    out = {"reference_grouped_meta": dict(**metrics(y, p > 0.5, p), folds=folds)}
    raw = pd.read_csv(E11 / "outputs/single_budget_draws.csv")
    raw = raw[raw.sampling == "natural"].copy().reset_index(drop=True)
    raw["weight"] = 1 / raw.groupby("dataset").dataset.transform("size")
    p, folds = grouped_meta(raw.a40, raw.saturated, raw.cohort, raw.weight)
    valid = np.isfinite(p)
    m = decision_stats(raw, valid & (p > 0.5), valid)
    m["auroc"] = float(
        roc_auc_score(raw.saturated[valid], p[valid], sample_weight=raw.weight[valid])
    )
    m["folds"] = folds
    out["operation_grouped_meta"] = m
    pd.DataFrame(
        dict(dataset=raw.dataset, cohort=raw.cohort, seed=raw.seed, score=p, stop=(p > 0.5) & valid)
    ).to_csv(HERE / "outputs/meta_predictions.csv", index=False)
    save(HERE / "outputs/grouped_meta.json", out)
    print(
        "grouped_meta",
        {k: {m: v[m] for m in ("auroc", "accuracy", "false_stop_rate")} for k, v in out.items()},
        flush=True,
    )


def summarise():
    rows = [
        row
        for p in sorted((HERE / "outputs/reconstructed").glob("*.json"))
        for row in json.loads(p.read_text())["rows"]
    ]
    assert len(list((HERE / "outputs/reconstructed").glob("*.json"))) == 134
    base = pd.read_csv(E11 / "outputs/single_budget_draws.csv")
    base = base[base.sampling == "natural"].copy()
    details = pd.DataFrame(rows)
    df = base.merge(
        details,
        on=["dataset", "cohort", "seed"],
        validate="one_to_one",
        suffixes=("", "_reconstructed"),
    )
    assert len(df) == len(base) == len(details), "Every cached repetition must be retained"
    assert np.array_equal(df.test_n, df.test_n_reconstructed)
    assert np.allclose(df.gain, df.gain_reconstructed, rtol=0, atol=1e-12)
    df = df.drop(columns=["test_n_reconstructed", "gain_reconstructed"])
    decisions = pd.read_csv(E11 / "outputs/single_budget_decisions.csv")
    decisions = decisions[decisions.sampling == "natural"]

    def evaluate(frame):
        frame = frame.copy()
        frame["weight"] = 1 / frame.groupby("dataset").dataset.transform("size")
        methods = {}
        for method in decisions.method.unique():
            d = frame.merge(
                decisions[decisions.method == method][["dataset", "seed", "stop", "decisive"]],
                on=["dataset", "seed"],
                validate="one_to_one",
            )
            methods[method] = decision_stats(d, d.stop.to_numpy(bool), d.decisive.to_numpy(bool))
        valid = frame.a40.notna()
        auc = (
            float(
                roc_auc_score(
                    frame.saturated[valid], -frame.a40[valid], sample_weight=frame.weight[valid]
                )
            )
            if len(set(frame.saturated[valid])) == 2
            else None
        )
        return dict(
            n_repetitions=len(frame),
            n_tasks=int(frame.dataset.nunique()),
            n_cohorts=int(frame.cohort.nunique()),
            score_auroc=auc,
            methods=methods,
        )

    masks = {
        "all": np.ones(len(df), bool),
        "test30_min5": (df.test_n >= 30) & (df.test_positive >= 5) & (df.test_negative >= 5),
        "test50_min10": (df.test_n >= 50) & (df.test_positive >= 10) & (df.test_negative >= 10),
        "gain_margin001": abs(df.gain - 0.05) > 0.01,
    }
    profiles = {key: evaluate(df[mask]) for key, mask in masks.items()}
    deletion = {g: evaluate(df[df.cohort != g]) for g in sorted(df.cohort.unique())}
    diag = df[df.diagnostic].copy()
    diag["weight"] = 1 / diag.groupby("dataset").dataset.transform("size")
    diagnostic = {}
    for key in ("pair", "refit"):
        diagnostic[key] = decision_stats(
            diag,
            diag[key + "_action"].eq("stop").to_numpy(),
            diag[key + "_action"].ne("insufficient").to_numpy(),
        )
    diagnostic.update(
        n_repetitions=len(diag),
        n_tasks=int(diag.dataset.nunique()),
        refits=int(diag.valid_refits.sum()),
        changed_decisions=int((diag.pair_action != diag.refit_action).sum()),
        median_valid_refits=float(diag.valid_refits.median()),
    )
    for key, column in [("pair", "conditional_pair_interval"), ("refit", "refit_interval")]:
        vals = [x[1] - x[0] for x in diag[column] if isinstance(x, list)]
        diagnostic[key]["median_interval_width"] = float(np.median(vals))
    widths = np.array([x[1] - x[0] for x in df.gain_interval])
    cross = np.array([x[0] <= 0.05 < x[1] for x in df.gain_interval])
    counts = {
        c: {"min": int(df[c].min()), "median": float(df[c].median()), "max": int(df[c].max())}
        for c in ("test_n", "test_positive", "test_negative")
    }
    uncertainty = dict(
        counts=counts,
        median_gain_interval_width=float(np.median(widths)),
        intervals_spanning_cutoff=int(cross.sum()),
        n_intervals=len(cross),
        scope="200 stratified paired resamples, percentile intervals conditional on fitted test predictions",
    )
    save(
        HERE / "outputs/sensitivity.json",
        dict(
            profiles=profiles,
            delete_one_cohort=deletion,
            refit_diagnostic=diagnostic,
            test_uncertainty=uncertainty,
        ),
    )
    df.drop(columns=["refit_draws"]).to_csv(HERE / "outputs/test_sensitivities.csv", index=False)
    print("refit_diagnostic", diagnostic, flush=True)
    print("test_uncertainty", uncertainty, flush=True)
    print(
        "profiles",
        {
            k: (v["n_repetitions"], v["score_auroc"], v["methods"]["fixed_057"]["false_stop_rate"])
            for k, v in profiles.items()
        },
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cached", "refit", "summary"])
    args = ap.parse_args()
    if args.mode == "cached":
        cached_analyses()
    elif args.mode == "summary":
        summarise()
    else:
        start = time.time()
        tasks = load_tasks()
        with ProcessPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(task_analysis, t) for t in tasks]
            for i, job in enumerate(as_completed(jobs), 1):
                name, status = job.result()
                if i % 10 == 0 or i == 134:
                    print(i, name, status, "seconds", round(time.time() - start), flush=True)


if __name__ == "__main__":
    main()
