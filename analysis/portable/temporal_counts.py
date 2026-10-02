"""Audit temporal record/image counts without exporting clinical identifiers."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def audit(manifest, splits):
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {row["patient_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Manifest directory identifiers must be unique")
    temporal = json.loads(splits.read_text())["temporal"]
    partitions = {}
    for name in ("train", "test_2023", "test_2024"):
        counts = dict(
            split_directory_records=len(temporal[name]),
            records_with_usable_images=0,
            positive_records_with_usable_images=0,
            usable_images=0,
            positive_images=0,
            missing_or_empty_paths=0,
        )
        for identifier in temporal[name]:
            if identifier not in index:
                continue
            row = index[identifier]
            label = int(float(row["label_bin"]))
            paths = [Path(p) for p in row["image_paths"].split("|") if p]
            # Match PatientImageDataset's existence and nonempty-file checks.
            usable = sum(p.exists() and p.stat().st_size > 0 for p in paths)
            counts["records_with_usable_images"] += int(usable > 0)
            counts["positive_records_with_usable_images"] += int(usable > 0) * label
            counts["usable_images"] += usable
            counts["positive_images"] += usable * label
            counts["missing_or_empty_paths"] += len(paths) - usable
        partitions[name] = counts
    return {
        "scope": "Aggregate audit of the retained manifest and current usable paths; not a recovered run-specific image inventory or person linkage.",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "splits_sha256": hashlib.sha256(splits.read_bytes()).hexdigest(),
        "partitions": partitions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.manifest, args.splits)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("Wrote aggregate counts only; no identifiers or image paths exported.")
