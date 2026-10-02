"""Compare numerical outputs, excluding portable paths and code fingerprints."""

import argparse
import json
import math
from pathlib import Path

IGNORED = {
    "fingerprint",
    "seconds",
    "elapsed_seconds",
    "notes",
    "note",
    "meta",
    "protocol",
    "scope",
    "source",
}


def compare(actual, expected, path="", differences=None):
    differences = [] if differences is None else differences
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            differences.append(path + ": type mismatch")
            return differences
        for key, value in expected.items():
            if key in IGNORED:
                continue
            if key not in actual:
                differences.append(path + "/" + key + ": missing")
            else:
                compare(actual[key], value, path + "/" + key, differences)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            differences.append(path + ": length mismatch")
        else:
            for i, (a, e) in enumerate(zip(actual, expected)):
                compare(a, e, path + f"/{i}", differences)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(
            actual, expected, rel_tol=0, abs_tol=1e-10
        ):
            differences.append(f"{path}: {actual!r} != {expected!r}")
    elif expected is None or isinstance(expected, bool):
        if actual != expected:
            differences.append(f"{path}: {actual!r} != {expected!r}")
    elif isinstance(expected, str) and path.rsplit("/", 1)[-1] in {
        "dataset",
        "cohort",
        "sampling",
        "action",
        "refit_action",
        "pair_action",
        "held_out",
    }:
        if actual != expected:
            differences.append(f"{path}: categorical result mismatch")
    return differences


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actual", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--files", nargs="+")
    args = ap.parse_args()
    files = args.files or [
        str(p.relative_to(args.reference)) for p in sorted(args.reference.rglob("*.json"))
    ]
    rows = []
    for file in files:
        a, e = args.actual / file, args.reference / file
        differences = (
            ["output missing"]
            if not a.exists()
            else compare(json.loads(a.read_text()), json.loads(e.read_text()))
        )
        rows.append({"file": file, "matches": not differences, "differences": differences[:20]})
    result = {
        "comparison": "numeric values and scientific decision categories",
        "absolute_tolerance": 1e-10,
        "does_not_prove_refitting": True,
        "files": rows,
        "all_match": all(x["matches"] for x in rows),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{sum(x['matches'] for x in rows)}/{len(rows)} result files match")
    if not result["all_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
