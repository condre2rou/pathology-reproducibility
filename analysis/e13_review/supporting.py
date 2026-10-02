#!/usr/bin/env python3
"""Recover supporting seed summaries and audit inputs without exporting records."""

import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
from pathlib import Path
import json, sys, hashlib, argparse
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def write(name, data):
    (HERE / "outputs" / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def noise():
    sys.path.insert(0, str(ROOT / "e2_train"))
    from snap_synthetic import run_once

    labels = pd.read_csv(ROOT / "e1_label_engine/outputs/labels_ec.csv", dtype=str)
    labels["pid"] = labels.year + "_" + labels.folder
    feats = np.load(ROOT / "e2_train/runs/feats_phikon.npz", allow_pickle=True)["feats"].item()
    rows = [
        (r.pid, int(r.grade == "3"))
        for r in labels.itertuples()
        if r.grade in ("1", "2", "3") and r.pid in feats
    ]
    X = np.stack([feats[p] for p, y in rows])
    y = np.array([y for p, y in rows])
    out = {}
    old = json.loads((ROOT / "e2_train/runs/snap_synthetic.json").read_text())
    for rate in (0.0, 0.1, 0.2, 0.3, 0.4):
        vals = []
        for seed in range(5):
            rng = np.random.RandomState(1000 + seed)
            flip = rng.rand(len(y)) < rate
            detected = np.where(flip, rng.rand(len(y)) < 0.7, rng.rand(len(y)) < 0.3)
            vals.append(run_once(X, y, flip, detected, seed))
        a = np.array(vals)
        delta = a[:, 1] - a[:, 0]
        row = dict(
            seed_means=vals,
            ce_mean=float(a[:, 0].mean()),
            weighted_mean=float(a[:, 1].mean()),
            ce_sd=float(a[:, 0].std(ddof=1)),
            weighted_sd=float(a[:, 1].std(ddof=1)),
            delta_mean=float(delta.mean()),
            delta_sd=float(delta.std(ddof=1)),
        )
        for key, new in [("ce", "ce_mean"), ("snap", "weighted_mean"), ("delta", "delta_mean")]:
            assert round(row[new], 3) == old["results"][f"{rate:.2f}"][key]
        out[f"{rate:.2f}"] = row
    write(
        "synthetic_noise.json", dict(n=len(y), positive=int(y.sum()), seeds=5, folds=5, results=out)
    )
    print("Synthetic noise: all original rounded point estimates reproduced; seed SD recovered.")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance():
    sys.path.insert(0, str(ROOT / "e11_method_audit"))
    from data import load_tasks, manifest

    tasks = load_tasks()
    manifest(tasks)
    lab = pd.read_csv(ROOT / "e1_label_engine/outputs/labels_ec_patient.csv", dtype=str)
    paths = {
        p
        for ps in lab.image_paths.dropna()
        for p in ps.split("|")
        if p and Path(p).is_file() and Path(p).stat().st_size > 0
    }
    byid = (
        {r.patient_id: set(str(r.image_paths).split("|")) & paths for r in lab.itertuples()}
        if "patient_id" in lab
        else {
            str(r.year) + "_" + str(r.folder): set(str(r.image_paths).split("|")) & paths
            for r in lab.itertuples()
        }
    )
    overlap = []
    for task in tasks:
        if task.config == "phikon" and task.cohort == "YT-EC":
            images = set().union(*(byid.get(p, set()) for p in task.ids))
            overlap.append(
                dict(
                    task=task.endpoint,
                    records=len(task.ids),
                    records_with_pretraining_images=sum(bool(byid.get(p)) for p in task.ids),
                    images=len(images),
                    images_in_pretraining=len(images & paths),
                )
            )
    sys.path.insert(0, str(ROOT / "e6_learnability"))
    from b_histology import build

    hist = build()
    feats = np.load(ROOT / "e2_train/runs/feats_ssl.npz", allow_pickle=True)["feats"].item()
    subtype = [
        str(r.year) + "_" + str(r.folder)
        for r in lab.itertuples()
        if hist.get(str(r.year) + "_" + str(r.name_seq)) in ("serous", "endometrioid")
        and str(r.year) + "_" + str(r.folder) in feats
    ]
    images = set().union(*(byid.get(p, set()) for p in subtype))
    overlap.append(
        dict(
            task="serous_subtype",
            records=len(subtype),
            records_with_pretraining_images=sum(bool(byid.get(p)) for p in subtype),
            images=len(images),
            images_in_pretraining=len(images & paths),
        )
    )
    for row in overlap:
        row["records_in_current_loader"] = row.pop("records_with_pretraining_images")
        row["images_in_current_loader"] = row.pop("images_in_pretraining")
    cached = [
        ROOT / "e2_train/runs/feats_phikon.npz",
        ROOT / "e2_train/runs/feats_ssl.npz",
        ROOT / "e2_train/runs/feats_imagenet.npz",
        ROOT / "e6_learnability/res_feats_ssl.npz",
        ROOT / "e5_tcga/pooled_features.npz",
    ]
    cached += list((ROOT / "e5_tcga/pooled").glob("*.npz")) + list(
        (ROOT / "e10_expansion/pooled").glob("*.npz")
    )
    files = [
        dict(file=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(set(cached))
        if p.exists()
    ]
    snapshots = []
    for base in (ROOT / "models", ROOT.parent / "models"):
        for p in base.glob("*/snapshots/*"):
            snapshots.append(dict(repository_cache=p.parent.parent.name, revision_directory=p.name))
    counts = {}
    for task in tasks:
        c = counts.setdefault(task.cohort, set())
        c.update(task.ids)
    write(
        "input_provenance.json",
        dict(
            audit_date="2026-09-23",
            current_loader_image_count=len(paths),
            current_manifest_path_count=sum(
                len(str(ps).split("|")) for ps in lab.image_paths.dropna()
            ),
            simclr_overlap=overlap,
            cohort_units={
                g: dict(
                    n_unique_units=len(ids),
                    unit="TCGA participant (12-character)"
                    if g.startswith("TCGA")
                    else "directory record",
                )
                for g, ids in counts.items()
            },
            feature_caches=files,
            observed_snapshot_directories=snapshots,
            limitations=[
                "Snapshot directory names do not establish that every file downloaded through a mutable main URL belongs to that revision.",
                "Historical notes cite 861 archive images; the current manifest lists 814 paths, 813 usable. The original run-specific pretraining image list was not retained. Overlap counts describe the current all-image loader, not authenticated historical training membership.",
                "Original download timestamps, upstream UNI2-h weight revision, tiling/tissue-filter parameters and full upstream licensing provenance are not recoverable from pooled feature caches.",
                "Institutional longitudinal person linkage and human label adjudication are unavailable.",
                "Ki67 filename prefixes do not establish patient identity, specimen pairing, source institution or clinical ground truth.",
            ],
        ),
    )
    print("Input hashes and unit checks saved; SimCLR overlap:", overlap)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["noise", "provenance"])
    a = p.parse_args()
    noise() if a.mode == "noise" else provenance()
