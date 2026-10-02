# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_manifest, ROOT  # noqa: E402

RUNS = os.path.join(ROOT, "e2_train", "runs")
LABELS = os.path.join(ROOT, "e1_label_engine", "outputs", "labels_ec_patient.csv")
RESOLUTIONS = [0, 512, 384, 256, 192, 128, 96]


class DownsampledDataset(Dataset):
    """Preserved computation; see docs/experiments.md for its protocol."""

    def __init__(self, items, target, size=224):
        self.items = items
        self.target = target
        self.tf = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pid, img = self.items[i]
        if self.target and min(img.size) > self.target:
            w, h = img.size
            scale = self.target / min(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        return self.tf(img), pid


def get_backbone(kind):
    if kind == "ssl":
        ck = torch.load(os.path.join(RUNS, "simclr_backbone.pt"), map_location="cpu")
        m = models.resnet50()
        m.fc = nn.Identity()
        m.load_state_dict(ck["encoder"])
    else:
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
    return m.eval().cuda()


@torch.no_grad()
def features_at(model, items, target, batch=32):
    ds = DownsampledDataset(items, target)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)
    feats = {}
    for x, pids in dl:
        f = model(x.cuda()).cpu().numpy()
        for i, pid in enumerate(pids):
            feats.setdefault(pid, []).append(f[i])
    return {k: np.mean(v, axis=0) for k, v in feats.items()}


def build_tasks(labels, pids):
    idx = labels.drop_duplicates("folder").set_index(labels["year"] + "_" + labels["folder"])
    tasks = {}

    def add(name, fn):
        y = {}
        for p in pids:
            if p in idx.index:
                v = fn(idx.loc[p])
                if v is not None:
                    y[p] = v
        if (
            len(y) >= 60
            and len(set(y.values())) == 2
            and min(pd.Series(list(y.values())).value_counts()) >= 15
        ):
            tasks[name] = y

    add(
        "grade3",
        lambda r: (1 if r["grade"] == "3" else 0) if r["grade"] in ("1", "2", "3") else None,
    )
    add(
        "dMMR", lambda r: (1 if r["mmr"] == "dMMR" else 0) if r["mmr"] in ("pMMR", "dMMR") else None
    )
    add(
        "Ki67_high30",
        lambda r: (
            (1 if float(r["ki67"]) >= 30 else 0)
            if str(r["ki67"]).replace(".", "", 1).isdigit()
            else None
        ),
    )
    add("p53_abn", lambda r: (1 if r["p53"] == "abn" else 0) if r["p53"] in ("wt", "abn") else None)
    return tasks


def cv_eval(feats, y, seed=42):
    pids = [p for p in y if p in feats]
    X = np.stack([feats[p] for p in pids])
    Y = np.array([y[p] for p in pids])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, Y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
        clf.fit(sc.transform(X[tr]), Y[tr])
        aucs.append(roc_auc_score(Y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return round(float(np.mean(aucs)), 3), round(float(np.std(aucs)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="ssl", choices=["ssl", "imagenet"])
    args = ap.parse_args()

    manifest = load_manifest()
    labels = pd.read_csv(LABELS, dtype=str)
    pids = list(manifest["patient_id"])
    idx = manifest.set_index("patient_id")
    tasks = build_tasks(labels, pids)

    print("加载图像到内存...")
    items = []
    for pid in pids:
        if pid not in idx.index:
            continue
        for p in str(idx.loc[pid, "image_paths"]).split("|"):
            if p and os.path.exists(p) and os.path.getsize(p) > 0:
                try:
                    items.append((pid, Image.open(p).convert("RGB")))
                except Exception:
                    pass
    print(f"  {len(items)} 张图")

    model = get_backbone(args.backbone)
    out = {"backbone": args.backbone, "resolutions": RESOLUTIONS, "tasks": {}}

    for name, y in tasks.items():
        out["tasks"][name] = {"n": len(y), "pos": int(sum(y.values())), "curve": {}}

    for res in RESOLUTIONS:
        label = "native" if res == 0 else str(res)
        feats = features_at(model, items, res)
        line = f"[res {label:>6}] "
        for name, y in tasks.items():
            m, s = cv_eval(feats, y)
            out["tasks"][name]["curve"][label] = {"auroc": m, "std": s}
            line += f"{name}={m:.3f} "
        print(line)

    with open(os.path.join(RUNS, f"resolution_ladder_{args.backbone}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n已写 runs/resolution_ladder_{args.backbone}.json")


if __name__ == "__main__":
    main()
