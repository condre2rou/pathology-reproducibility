#!/usr/bin/env python3
"""Cohort uncertainty for valid single-pilot score AUROC; no patient features."""

import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
from pathlib import Path
import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score
from summarize import cohort_ci
from run import save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = ap.parse_args()
    df = pd.read_csv(args.root / "single_budget_draws.csv")
    out = {}
    for sampling in ("natural", "balanced"):
        s = df[df.sampling == sampling].copy()
        s["weight"] = 1 / s.groupby("dataset").dataset.transform("size")
        v = s[s.a40.notna()].reset_index(drop=True)
        ci = cohort_ci(
            v,
            lambda ix: dict(
                auroc=float(
                    roc_auc_score(
                        v.saturated.values[ix], -v.a40.values[ix], sample_weight=v.weight.values[ix]
                    )
                )
            ),
        )
        out[sampling] = dict(
            valid_pilots=len(v),
            interval=ci["auroc"],
            weighting="Original task weights retained, restricted to valid OOF scores",
        )
    save(args.root / "operational_auc_ci.json", out)


if __name__ == "__main__":
    main()
