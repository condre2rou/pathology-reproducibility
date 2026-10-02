# -*- coding: utf-8 -*-
"""Load image rows after splitting institutional directory records, not linked people."""

import os
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # snap-cpath
MANIFEST = os.path.join(ROOT, "e2_data", "manifest.csv")
SPLITS = os.path.join(ROOT, "e2_data", "splits.json")


def load_manifest():
    return pd.read_csv(MANIFEST, dtype=str)


def load_splits():
    with open(SPLITS) as f:
        return json.load(f)


def build_index(manifest):
    """patient_id -> {label_bin, image_paths, year, ...}"""
    return manifest.set_index("patient_id")


class PatientImageDataset(Dataset):
    """Expand directory records into image/label rows; the class name is historical.

    The manifest's patient_id field is a directory key. Its separation does not
    establish person-level independence, and evaluation operates on image rows.
    """

    def __init__(self, patient_ids, manifest, transform=None, limit_per_patient=None):
        idx = build_index(manifest)
        self.samples = []
        for pid in patient_ids:
            if pid not in idx.index:
                continue
            row = idx.loc[pid]
            paths = [p for p in str(row["image_paths"]).split("|") if p]
            if limit_per_patient:
                paths = paths[:limit_per_patient]
            label = int(float(row["label_bin"]))
            for p in paths:
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    self.samples.append((p, label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def imread_np(path, size=None):
    """Preserved computation; see docs/experiments.md for its protocol."""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size))
    import numpy as np

    return np.asarray(img)
