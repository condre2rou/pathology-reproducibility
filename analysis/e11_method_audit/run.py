#!/usr/bin/env python3
"""E11 cached-feature analyses. New outputs only; resumable per-task jobs."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from pathlib import Path
import hashlib
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from core import (
    probe,
    cv_auc,
    cv_predictions,
    balanced_sample,
    oof_interval,
    decide,
    BUDGETS,
    DELTA,
)
from data import load_tasks, manifest

HERE = Path(__file__).resolve().parent
warnings.filterwarnings("error", category=ConvergenceWarning)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False)
    if path.exists():
        if path.read_text() != value:
            raise FileExistsError(f"Refusing to overwrite different results: {path}")
        return
    path.write_text(value)


def fingerprint(task, args, mode):
    h = hashlib.sha256()
    for path in ("core.py", "run.py", "data.py", "PROTOCOL.md"):
        h.update((HERE / path).read_bytes())
    h.update(task.X.tobytes())
    h.update(task.y.tobytes())
    h.update(str((args.repeats, mode)).encode())
    return h.hexdigest()


def reference(task, repeats):
    full = [cv_auc(task.X, task.y, seed) for seed in range(5)]
    points = {}
    for n in BUDGETS:
        if len(task.y) < 2 * n or np.bincount(task.y).min() < n // 2:
            continue
        vals = []
        for seed in range(repeats):
            idx = balanced_sample(task.y, n, np.random.RandomState(seed))
            vals.append(cv_auc(task.X[idx], task.y[idx]))
        points[str(n)] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)), draws=vals)
    return dict(
        dataset=task.name,
        cohort=task.cohort,
        endpoint=task.endpoint,
        config=task.config,
        n=len(task.y),
        positives=int(task.y.sum()),
        full=float(np.mean(full)),
        full_draws=full,
        budgets=points,
    )


def test_membership(task, seed):
    return np.array(
        [
            int.from_bytes(
                hashlib.sha256(f"{task.cohort}|{pid}|{seed}".encode()).digest()[:8], "big"
            )
            / 2**64
            < 0.25
            for pid in task.ids
        ]
    )


def operation(task, repeats):
    rows = []
    excluded = []
    for seed in range(repeats):
        test = test_membership(task, seed)
        pool = ~test
        xt, yt = task.X[test], task.y[test]
        xp, yp = task.X[pool], task.y[pool]
        if (
            len(yp) < 60
            or len(yt) < 10
            or min(np.bincount(yt, minlength=2)) < 2
            or min(np.bincount(yp, minlength=2)) < 2
        ):
            excluded.append(
                dict(
                    seed=seed,
                    reason="pool/test size or class availability",
                    pool_n=len(yp),
                    test_n=len(yt),
                )
            )
            continue
        expanded = probe().fit(xp, yp).predict_proba(xt)[:, 1]
        full_auc = float(roc_auc_score(yt, expanded))
        for sampling in ("natural", "balanced"):
            rng = np.random.RandomState(10000 + seed)
            if sampling == "balanced" and np.bincount(yp, minlength=2).min() < 20:
                excluded.append(dict(seed=seed, sampling=sampling, reason="pool minority <20"))
                continue
            ix = (
                rng.choice(len(yp), 40, replace=False)
                if sampling == "natural"
                else balanced_sample(yp, 40, rng)
            )
            x, y = xp[ix], yp[ix]
            predictions = cv_predictions(x, y, seed)
            a40 = float(roc_auc_score(y, predictions)) if predictions is not None else np.nan
            interval = oof_interval(y, predictions, seed=20000 + seed)
            pilot = (
                probe().fit(x, y).predict_proba(xt)[:, 1]
                if len(np.unique(y)) == 2
                else np.full(len(yt), 0.5)
            )
            pilot_auc = float(roc_auc_score(yt, pilot))
            small = (
                balanced_sample(y, 20, np.random.RandomState(30000 + seed))
                if sampling == "balanced"
                else np.random.RandomState(30000 + seed).choice(40, 20, replace=False)
            )
            a20 = cv_auc(x[small], y[small], seed)
            rows.append(
                dict(
                    dataset=task.name,
                    cohort=task.cohort,
                    seed=seed,
                    sampling=sampling,
                    pool_n=len(yp),
                    test_n=len(yt),
                    pilot_positives=int(y.sum()),
                    a40=a40,
                    a20=a20,
                    ci=interval,
                    action=decide(a40, interval),
                    pilot_test_auc=pilot_auc,
                    expanded_test_auc=full_auc,
                    gain=full_auc - pilot_auc,
                    saturated=int(full_auc - pilot_auc <= DELTA),
                )
            )
    return dict(dataset=task.name, cohort=task.cohort, rows=rows, excluded=excluded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reference", "operation", "inventory"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    args = ap.parse_args()
    if not 1 <= args.workers <= 4:
        ap.error("Use 1..4 workers to respect the project CPU limit")
    tasks = load_tasks()
    save(args.out / "manifest.json", manifest(tasks))
    if args.mode == "inventory":
        print(json.dumps({k: v for k, v in manifest(tasks).items() if k != "tasks"}, indent=2))
        return
    if args.limit:
        tasks = tasks[: args.limit]
    print(
        f"{args.mode}: {len(tasks)} tasks, {args.repeats} repetitions, {args.workers} workers",
        flush=True,
    )
    start = time.time()

    def work(task):
        path = args.out / args.mode / (task.name + ".json")
        digest = fingerprint(task, args, args.mode)
        if path.exists():
            result = json.loads(path.read_text())
            if result["fingerprint"] != digest:
                raise ValueError("Stale cache; choose a NEW output directory: " + str(path))
            return task.name, "cached"
        result = (
            reference(task, args.repeats)
            if args.mode == "reference"
            else operation(task, args.repeats)
        )
        result["fingerprint"] = digest
        save(path, result)
        return task.name, "computed"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(work, t) for t in tasks]
        for i, job in enumerate(as_completed(jobs), 1):
            name, status = job.result()
            print(
                f"{i}/{len(tasks)} {name} {status} elapsed={time.time() - start:.0f}s", flush=True
            )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
