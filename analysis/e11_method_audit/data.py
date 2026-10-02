"""Local data adapter. Analysis-unit identifiers and feature vectors are not exported."""

from pathlib import Path
from dataclasses import dataclass
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "e6_learnability"))
from validate import tasks_micro


@dataclass
class Task:
    name: str
    cohort: str
    endpoint: str
    config: str
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray


def cohort_of(name):
    if "@" in name:
        return "YT-EC"
    if name.startswith("OV-"):
        return "YT-OV"
    c = name.split("-")[1]
    if c.startswith("BRCA"):
        return "TCGA-BRCA"
    return "TCGA-" + c


def endpoint_of(name):
    if "@" in name:
        return name.split("@")[1]
    legacy = {
        "WSI-UCEC-dMMR": "dMMR",
        "WSI-COAD": "MSI",
        "WSI-STAD": "MSI",
        "WSI-BRCA_IDC": "ER_pos",
        "WSI-BRCA_IDC_PR": "PR_pos",
    }
    if name in legacy:
        return legacy[name]
    return name.split("-", 2)[-1] if name.startswith("WSI-") else name[3:]


def load_tasks():
    wanted = pd.read_csv(ROOT / "e10_expansion/b_upside_data_v2.csv").dataset.tolist()
    labels = pd.read_csv(ROOT / "e1_label_engine/outputs/labels_ec.csv", dtype=str)
    datasets = {}

    def photos(prefix, features):
        # Pass the same ordered feature dictionary through the historical label
        # selector to recover exact record ordering without exporting raw IDs.
        index_features = {k: np.array([i]) for i, k in enumerate(features)}
        ordered = np.array(list(features))
        selected = tasks_micro(labels, features)
        index_selected = tasks_micro(labels, index_features)
        for endpoint, (X, y) in selected.items():
            name = prefix + "@" + endpoint
            ix, yi = index_selected[endpoint]
            assert np.array_equal(y, yi)
            datasets[name] = (X, y, ordered[ix[:, 0].astype(int)])

    for prefix, filename in [("phikon", "feats_phikon.npz")]:
        photos(
            prefix, np.load(ROOT / "e2_train/runs" / filename, allow_pickle=True)["feats"].item()
        )
    for resolution, features in (
        np.load(ROOT / "e6_learnability/res_feats_ssl.npz", allow_pickle=True)["feats"]
        .item()
        .items()
    ):
        photos("res" + str(resolution), features)
    files = {"WSI-UCEC-dMMR": ROOT / "e5_tcga/pooled_features.npz"}
    for tag in ("COAD", "STAD", "BRCA_IDC", "BRCA_IDC_PR"):
        files["WSI-" + tag] = ROOT / "e5_tcga/pooled" / f"{tag}.npz"
    files.update({p.stem: p for p in (ROOT / "e10_expansion/pooled").glob("*.npz")})
    for name, p in files.items():
        if name in wanted:
            z = np.load(p, allow_pickle=True)
            datasets[name] = (z["X"], z["Y"].astype(int), z["pids"].astype(str))
    out = []
    for name in wanted:
        X, y, ids = datasets[name]
        group = cohort_of(name)
        # TCGA first 12 characters are participant, not slide, IDs.
        if group.startswith("TCGA-"):
            ids = np.array([str(x)[:12] for x in ids])
        assert len(ids) == len(set(ids)), f"duplicate analysis units: {name}"
        assert np.isfinite(X).all() and set(y) == {0, 1}
        out.append(
            Task(
                name,
                group,
                endpoint_of(name),
                name.split("@")[0]
                if "@" in name
                else ("phikon-OV" if name.startswith("OV-") else "UNI2-h"),
                np.asarray(X, np.float64),
                y,
                ids,
            )
        )
    assert len(out) == 134
    return out


def manifest(tasks):
    records = [
        dict(
            dataset=t.name,
            cohort=t.cohort,
            endpoint=t.endpoint,
            config=t.config,
            n=len(t.y),
            positives=int(t.y.sum()),
            dimension=t.X.shape[1],
        )
        for t in tasks
    ]
    # Check that no participant ID crosses distinct TCGA cohort groups.
    seen = {}
    for t in tasks:
        for pid in t.ids:
            key = str(pid) if t.cohort.startswith("TCGA-") else t.cohort + ":" + str(pid)
            if key in seen:
                assert seen[key] == t.cohort, "Participant crosses cohort groups"
            seen[key] = t.cohort
    return dict(
        n_combinations=len(tasks),
        n_cohorts=len({t.cohort for t in tasks}),
        n_tcga_cohorts=len({t.cohort for t in tasks if t.cohort.startswith("TCGA-")}),
        cohorts=sorted({t.cohort for t in tasks}),
        local_person_linkage="Directory records; cross-year person linkage unavailable",
        cross_cohort_tcga_overlap=0,
        tasks=records,
    )
