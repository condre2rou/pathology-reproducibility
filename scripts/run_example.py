"""Refit the real TCGA-COAD manuscript task and compare its saved outputs."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.data.exists():
        ap.error("Real example data are missing; follow docs/tutorial.md to prepare them")
    sys.path[:0] = [str(args.root / "e11_method_audit"), str(args.root / "e13_review")]
    from data import Task
    from run import reference, operation, save
    from reanalyse import task_analysis
    from compare_results import compare

    with np.load(args.data, allow_pickle=False) as z:
        task = Task(
            "WSI-COAD",
            "TCGA-COAD",
            "MSI",
            "UNI2-h",
            np.asarray(z["X"], np.float64),
            z["y"],
            z["ids"],
        )
    args.out.mkdir(parents=True, exist_ok=False)
    comparisons = {}
    for name, function in [("reference", reference), ("operation", operation)]:
        result = function(task, 30)
        expected = json.loads(
            (args.root / "e11_method_audit/outputs" / name / "WSI-COAD.json").read_text()
        )
        differences = compare(result, expected)
        comparisons[name] = {"matches": not differences, "differences": differences}
        save(args.out / f"{name}.json", result)
    target = args.root / "e13_review/outputs/reconstructed/WSI-COAD.json"
    expected = json.loads(target.read_text())
    target.unlink()
    task_analysis(task)
    reconstructed = json.loads(target.read_text())
    differences = compare(reconstructed, expected)
    comparisons["refit_and_paired_gain"] = {"matches": not differences, "differences": differences}
    save(args.out / "refitting.json", reconstructed)
    summary = {
        "dataset": task.name,
        "records": len(task.y),
        "positive": int(task.y.sum()),
        "feature_dimension": task.X.shape[1],
        "repetitions": 30,
        "public_subset_only": True,
        "comparisons": comparisons,
        "all_match": all(v["matches"] for v in comparisons.values()),
    }
    save(args.out / "verification.json", summary)
    print(json.dumps(summary, indent=2))
    if not summary["all_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
