#!/usr/bin/env python3
"""Aggregate E11 results with explicit statistical units and cohort resampling."""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from core import metrics, best_cut, CUT, BUDGETS
from run import save

HERE = Path(__file__).resolve().parent


def reference_frame(root):
    rows = []
    for p in sorted((root / "reference").glob("*.json")):
        r = json.loads(p.read_text())
        row = {
            k: r[k] for k in ("dataset", "cohort", "endpoint", "config", "n", "positives", "full")
        }
        for n, b in r["budgets"].items():
            row["a" + n] = b["mean"]
            row["sd" + n] = b["sd"]
        draws = r["budgets"]["40"]["draws"]
        half = len(draws) // 2
        row["a40_half1"] = np.mean(draws[:half])
        row["a40_half2"] = np.mean(draws[half:])
        rows.append(row)
    df = pd.DataFrame(rows)
    assert len(df) == 134, f"Incomplete reference run: {len(df)}"
    df["source_group"] = df.cohort.map(lambda x: "YT" if x.startswith("YT-") else x)
    return df


def grouped_threshold(a, y, groups, weights=None):
    cuts = np.empty(len(a))
    records = {}
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        c = best_cut(a[tr], y[tr], None if weights is None else weights[tr])
        cuts[te] = c
        records[g] = c
    return a <= cuts, records


def cohort_ci(df, fn, n_boot=1000, seed=20260923):
    groups = {g: np.flatnonzero(df.cohort.values == g) for g in df.cohort.unique()}
    keys = list(groups)
    rng = np.random.RandomState(seed)
    out = {}
    for _ in range(n_boot):
        ids = np.concatenate([groups[keys[i]] for i in rng.randint(len(keys), size=len(keys))])
        result = fn(ids)
        for k, v in result.items():
            if v is not None and np.isfinite(v):
                out.setdefault(k, []).append(v)
    return {
        k: {
            "interval": list(map(float, np.quantile(v, [0.025, 0.975]))),
            "valid_bootstraps": len(v),
        }
        for k, v in out.items()
    }


def reference_summary(df):
    a = df.a40.values
    y = (df.full.values - a <= 0.05).astype(int)
    primary = metrics(y, a <= CUT, -a)
    primary["ci"] = cohort_ci(
        df,
        lambda ix: {
            k: v
            for k, v in metrics(y[ix], a[ix] <= CUT, -a[ix]).items()
            if k in ("auroc", "accuracy", "false_stop_rate", "missed_opportunity_rate")
        },
    )
    grouped = {}
    for column in ("cohort", "source_group", "endpoint", "config"):
        pred, cuts = grouped_threshold(a, y, df[column].values)
        grouped[column] = dict(n_groups=len(cuts), cuts=cuts, **metrics(y, pred, -a))
        if column == "cohort":
            grouped[column]["ci"] = cohort_ci(
                df,
                lambda ix: {
                    k: v
                    for k, v in metrics(y[ix], pred[ix], -a[ix]).items()
                    if k in ("accuracy", "false_stop_rate")
                },
            )
    # Secondary estimator, deliberately kept separate from the fixed rule.
    probs = []
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        m = make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)
        )
        m.fit(a[tr, None], y[tr])
        probs.append(m.predict_proba(a[i : i + 1, None])[0, 1])
    secondary = metrics(y, np.array(probs) > 0.5, probs)
    delta = {}
    for d in (0.03, 0.05, 0.10):
        yy = (df.full.values - a <= d).astype(int)
        pred, cuts = grouped_threshold(a, yy, df.cohort.values)
        delta[str(d)] = dict(
            fixed=metrics(yy, a <= CUT, -a),
            cohort_recalibrated=metrics(yy, pred, -a),
            saturated=int(yy.sum()),
        )
    budgets = {}
    common = df.dropna(subset=["a" + str(n) for n in BUDGETS])
    commons = {}
    for label, frame, store in [("all", df, budgets), ("common", common, commons)]:
        for n in BUDGETS:
            sub = frame.dropna(subset=["a" + str(n)])
            aa = sub["a" + str(n)].values
            yy = (sub.full.values - aa <= 0.05).astype(int)
            pred, cuts = grouped_threshold(aa, yy, sub.cohort.values)
            store[str(n)] = dict(
                n_points=len(sub),
                n_saturated=int(yy.sum()),
                fixed=metrics(yy, aa <= CUT, -aa),
                cohort_recalibrated=metrics(yy, pred, -aa),
                median_cut=float(np.median(list(cuts.values()))),
            )
    cross_y = (df.full - df.a40_half2 <= 0.05).astype(int).values
    cross = metrics(cross_y, df.a40_half1.values <= CUT, -df.a40_half1.values)
    return dict(
        n_points=len(df),
        n_cohorts=df.cohort.nunique(),
        n_saturated=int(y.sum()),
        fixed_rule=primary,
        secondary_loo_logreg=secondary,
        grouped=grouped,
        delta_sensitivity=delta,
        budget_sensitivity=budgets,
        common_subset=dict(n_points=len(common), budgets=commons),
        disjoint_draw_sensitivity=cross,
        ci_scope="Cohort resampling conditional on per-task estimates; fitted decisions held fixed",
    )


def decision_stats(df, stop, coverage):
    # Each task contributes total weight one, regardless of eligible repetitions.
    weights = df["weight"].values
    yy = df.saturated.values
    m = metrics(yy, stop, weights=weights)

    def avg(x):
        return float(np.average(x, weights=weights))

    remaining = df.pool_n.values - 40
    wrong = stop & (yy == 0)
    m.update(
        decision_coverage=avg(coverage),
        mean_labels_saved=avg(stop * remaining),
        annotation_saving_fraction=float(
            np.sum(weights * stop * remaining) / np.sum(weights * df.pool_n)
        ),
        missed_gain_per_decision=avg(wrong * np.maximum(df.gain.values, 0)),
        mean_gain_when_falsely_stopped=float(
            np.average(df.gain.values[wrong], weights=weights[wrong])
        )
        if wrong.any()
        else None,
    )
    return m


def operational_summary(root):
    entries = [json.loads(p.read_text()) for p in sorted((root / "operation").glob("*.json"))]
    assert len(entries) == 134, f"Incomplete operation run: {len(entries)}"
    allrows = [row for e in entries for row in e["rows"]]
    df = pd.DataFrame(allrows)
    excluded = [r for e in entries for r in e["excluded"]]
    out = {}
    decisions = []
    for sampling in ("natural", "balanced"):
        sub = df[df.sampling == sampling].copy().reset_index(drop=True)
        sub["weight"] = 1 / sub.groupby("dataset").dataset.transform("size")
        valid = sub.a40.notna().values
        a = sub.a40.fillna(1).values
        y = sub.saturated.values
        learned = np.ones(len(sub))
        cuts = {}
        for g in sub.cohort.unique():
            te = sub.cohort.values == g
            tr = (~te) & valid
            cuts[g] = best_cut(a[tr], y[tr], sub.weight.values[tr])
            learned[te] = cuts[g]
        methods = {
            "always_continue": (np.zeros(len(sub), bool), np.ones(len(sub), bool)),
            "always_stop": (np.ones(len(sub), bool), np.ones(len(sub), bool)),
            "fixed_057": ((a <= CUT) & valid, valid),
            "cohort_recalibrated": ((a <= learned) & valid, valid),
            "flat_curve": (
                ((sub.a40 - sub.a20).values <= 0.02) & valid & sub.a20.notna().values,
                valid & sub.a20.notna().values,
            ),
            "three_way": (sub.action.values == "stop", sub.action.values != "insufficient"),
        }
        methods_out = {}
        for name, (stop, coverage) in methods.items():
            stat = decision_stats(sub, stop, coverage)
            stat["ci"] = cohort_ci(
                sub,
                lambda ix: {
                    k: v
                    for k, v in decision_stats(sub.iloc[ix], stop[ix], coverage[ix]).items()
                    if k
                    in (
                        "accuracy",
                        "false_stop_rate",
                        "missed_opportunity_rate",
                        "annotation_saving_fraction",
                        "decision_coverage",
                    )
                },
            )
            methods_out[name] = stat
            if name in ("fixed_057", "three_way", "cohort_recalibrated"):
                repeatability = (
                    pd.DataFrame(dict(dataset=sub.dataset, stop=stop))
                    .groupby("dataset")
                    .stop.mean()
                )
                stat["tasks_with_discordant_decisions"] = int(
                    ((repeatability > 0) & (repeatability < 1)).sum()
                )
            for i, row in sub.iterrows():
                decisions.append(
                    dict(
                        dataset=row.dataset,
                        cohort=row.cohort,
                        seed=int(row.seed),
                        sampling=sampling,
                        method=name,
                        stop=int(stop[i]),
                        decisive=int(coverage[i]),
                        saturated=int(row.saturated),
                        gain=row.gain,
                        pool_n=int(row.pool_n),
                    )
                )
        # Discrimination is evaluated on single draws, not their task means.
        vv = sub[valid]
        discrimination = float(roc_auc_score(vv.saturated, -vv.a40, sample_weight=vv.weight))
        out[sampling] = dict(
            n_repetitions=len(sub),
            n_tasks=sub.dataset.nunique(),
            n_cohorts=sub.cohort.nunique(),
            invalid_pilot_count=int((~valid).sum()),
            single_draw_score_auroc=discrimination,
            methods=methods_out,
            recalibrated_cuts=cuts,
            mean_pool_size=float(np.average(sub.pool_n, weights=sub.weight)),
            mean_test_size=float(np.average(sub.test_n, weights=sub.weight)),
        )
    out["exclusions"] = dict(
        n_events=len(excluded),
        by_reason=pd.Series([r["reason"] for r in excluded]).value_counts().to_dict(),
    )
    return out, df, pd.DataFrame(decisions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE / "outputs")
    args = ap.parse_args()
    df = reference_frame(args.root)
    ref = reference_summary(df)
    print(
        "REFERENCE",
        json.dumps(
            {k: ref["fixed_rule"][k] for k in ("auroc", "accuracy", "false_stop_rate")}, indent=2
        ),
        flush=True,
    )
    op, rows, decisions = operational_summary(args.root)
    save(args.root / "corrected_reference_result.json", ref)
    save(args.root / "single_budget_result.json", op)
    df.to_csv(args.root / "corrected_points.csv", index=False)
    rows.to_csv(args.root / "single_budget_draws.csv", index=False)
    decisions.to_csv(args.root / "single_budget_decisions.csv", index=False)
    for s in ("natural", "balanced"):
        print(s, json.dumps(op[s], indent=2), flush=True)


if __name__ == "__main__":
    main()
