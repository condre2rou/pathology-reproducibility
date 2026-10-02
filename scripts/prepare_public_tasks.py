"""Restore selected public manuscript tasks from verified caches or pinned H5 files."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tasks", nargs="+", help="Exact task names, or all (85 public tasks)")
    ap.add_argument("--cache-root", type=Path)
    ap.add_argument("--features-root", type=Path)
    ap.add_argument(
        "--download", action="store_true", help="Explicitly fetch missing large upstream H5s"
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    manifest = json.loads((ROOT / "data/public_tasks.json").read_text())
    if args.list:
        for t in manifest["tasks"]:
            print(f"{t['name']:44s} n={t['n']:4d} positive={t['positives']:4d} {t['cohort']}")
        return
    if not args.tasks or args.out is None or bool(args.cache_root) == bool(args.features_root):
        ap.error("Supply --tasks, --out and exactly one of --cache-root / --features-root")
    if args.download and args.features_root is None:
        ap.error("--download requires --features-root")
    names = {t["name"] for t in manifest["tasks"]}
    if args.tasks != ["all"] and not set(args.tasks) <= names:
        ap.error("Unknown task name; use --list")
    selected = [t for t in manifest["tasks"] if args.tasks == ["all"] or t["name"] in args.tasks]
    if args.out.exists():
        ap.error("Choose a new output directory")
    args.out.mkdir(parents=True)
    done, pooled = [], {}
    for t in selected:
        destination = args.out / t["cache_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_y = np.array([p["label"] for p in t["participants"]], dtype=int)
        expected_ids = np.array([p["participant_id"] for p in t["participants"]])
        if args.cache_root:
            cache = args.cache_root / t["cache_path"]
            if sha(cache) != t["original_cache_sha256"]:
                raise ValueError(f"Original cache checksum mismatch: {t['name']}")
            with np.load(cache, allow_pickle=False) as z:
                X, y, ids = z["X"], z["Y"], z["pids"]
            if not np.array_equal(y, expected_y) or not np.array_equal(
                ids.astype(str), expected_ids
            ):
                raise ValueError("Public task membership does not match")
        else:
            import h5py

            vectors = []
            for row in t["participants"]:
                files = tuple(row["feature_files"])
                if not files:
                    raise ValueError(f"No retained slide inventory for {row['participant_id']}")
                if files not in pooled:
                    slide_means = []
                    for name in files:
                        path = args.features_root / name
                        if not path.exists() and args.download:
                            print("Downloading", name, flush=True)
                            fetch(manifest["resolve_base"] + "/" + name, path)
                        if not path.exists():
                            raise FileNotFoundError(
                                f"Missing {name}; obtain upstream data or use --download"
                            )
                        with h5py.File(path, "r") as h5:
                            values = h5["features"][:]
                            values = values.reshape(-1, values.shape[-1])
                            if (
                                values.shape[1] != t["feature_dimension"]
                                or not np.isfinite(values).all()
                            ):
                                raise ValueError(f"Unexpected feature data: {name}")
                            slide_means.append(values.mean(axis=0))
                    pooled[files] = np.mean(slide_means, axis=0)
                vectors.append(pooled[files])
            X = np.stack(vectors).astype(t["feature_dtype"])
        if hashlib.sha256(X.tobytes()).hexdigest() != t["feature_values_sha256"]:
            raise ValueError(
                f"{t['name']}: pooled features differ from the evaluated cache. Do not label this an exact reproduction; inspect slide selection and upstream version."
            )
        np.savez_compressed(destination, X=X, Y=expected_y, pids=expected_ids)
        done.append(
            dict(
                task=t["name"],
                file=t["cache_path"],
                sha256=sha(destination),
                feature_values_match=True,
            )
        )
        print(f"Prepared {t['name']}: {len(expected_y)} participants", flush=True)
    (args.out / "public_preparation.json").write_text(json.dumps(done, indent=2) + "\n")


if __name__ == "__main__":
    main()
