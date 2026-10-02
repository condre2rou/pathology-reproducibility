#!/usr/bin/env python3
"""Prepare the observed TCGA-COAD example from a cache or upstream H5 files."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pooled-cache", type=Path, help="Existing verified X/Y/pids or X/y/ids NPZ"
    )
    source.add_argument(
        "--features-dir", type=Path, help="Directory containing TCGA-COAD/features/*.h5"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download missing H5s from their source URLs"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/example")
    args = parser.parse_args()
    if args.download and args.features_dir is None:
        parser.error("--download requires --features-dir")
    manifest = json.loads((ROOT / "data/example/source_manifest.json").read_text())
    expected_ids = np.array([row["participant_id"] for row in manifest["participants"]])
    expected_y = np.array([row["label"] for row in manifest["participants"]], dtype=int)
    pool_path, pilot_path = (
        args.out / "tcga_coad_msi_pool.npz",
        args.out / "tcga_coad_msi_pilot40.npz",
    )
    if pool_path.exists() or pilot_path.exists():
        parser.error("Example output already exists; choose a new --out directory")
    if args.pooled_cache:
        digest = file_hash(args.pooled_cache)
        allowed = [manifest["original_cache_sha256"], manifest.get("exported_pool_sha256")]
        if digest not in allowed:
            parser.error("Cache hash does not match the documented public TCGA example")
        with np.load(args.pooled_cache, allow_pickle=False) as archive:
            X = archive["X"]
            y = archive["y"] if "y" in archive else archive["Y"]
            ids = archive["ids"] if "ids" in archive else archive["pids"]
    else:
        import h5py

        vectors = []
        args.features_dir.mkdir(parents=True, exist_ok=True)
        for row in manifest["participants"]:
            slides = []
            for remote_path in row["feature_files"]:
                path = args.features_dir / remote_path
                if not path.exists() and args.download:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    url = manifest["feature_resolve_base"] + "/" + remote_path
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "pathology-pilot/0.1"}
                    )
                    temporary = path.with_suffix(".h5.part")
                    try:
                        with (
                            urllib.request.urlopen(request, timeout=120) as response,
                            temporary.open("wb") as target,
                        ):
                            while block := response.read(1024 * 1024):
                                target.write(block)
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
                if not path.exists():
                    parser.error(
                        f"Missing feature file: {remote_path}; download it or add --download"
                    )
                with h5py.File(path, "r") as archive:
                    values = archive["features"][:]
                    values = values.reshape(-1, values.shape[-1])
                    if (
                        values.shape[1] != manifest["feature_dimension"]
                        or not np.isfinite(values).all()
                    ):
                        parser.error(f"Unexpected or nonfinite features in {remote_path}")
                    slides.append(values.mean(axis=0))
            vectors.append(np.mean(slides, axis=0))
        X, y, ids = np.stack(vectors), expected_y, expected_ids
    if (
        not np.array_equal(ids, expected_ids)
        or not np.array_equal(y, expected_y)
        or X.shape != (len(y), manifest["feature_dimension"])
    ):
        parser.error("Feature shape, labels or participant order do not match source metadata")
    if not np.isfinite(X).all():
        parser.error("Features contain nonfinite values")
    indices = np.asarray(manifest["pilot_row_indices"], dtype=int)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pool_path, X=X, y=y, ids=ids)
    np.savez_compressed(pilot_path, X=X[indices], y=y[indices], ids=ids[indices])
    summary = {
        "source": "TCGA-COAD",
        "pool_n": len(y),
        "pool_positive": int(y.sum()),
        "pilot_n": len(indices),
        "pilot_positive": int(y[indices].sum()),
        "files": {p.name: file_hash(p) for p in (pool_path, pilot_path)},
        "feature_values_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
    }
    (args.out / "prepared_data.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
