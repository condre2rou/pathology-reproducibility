# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "e2_train"))
from data import load_manifest  # noqa: E402

RUNS = os.path.join(ROOT, "e2_train", "runs")
LABELS = os.path.join(ROOT, "e1_label_engine", "outputs", "labels_ec_patient.csv")
MODEL_ID = "owkin/phikon"


class ImgDataset(Dataset):
    def __init__(self, items, processor):
        self.items = items
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pid, p = self.items[i]
        img = Image.open(p).convert("RGB")
        px = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return px, pid


@torch.no_grad()
def extract(model, items, processor, batch=16):
    dl = DataLoader(ImgDataset(items, processor), batch_size=batch, shuffle=False, num_workers=4)
    feats, model = {}, model.cuda().eval()
    for x, pids in tqdm(dl, desc="phikon"):
        out = model(pixel_values=x.cuda())
        f = out.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
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
    add("ER_pos", lambda r: (1 if r["er"] == "pos" else 0) if r["er"] in ("pos", "neg") else None)
    add("PR_pos", lambda r: (1 if r["pr"] == "pos" else 0) if r["pr"] in ("pos", "neg") else None)
    return tasks


def cv(X, Y, seeds=10):
    aucs = []
    for s in range(seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        for tr, te in skf.split(X, Y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(X[tr]), Y[tr])
            aucs.append(roc_auc_score(Y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    manifest = load_manifest()
    labels = pd.read_csv(LABELS, dtype=str)
    idx = manifest.set_index("patient_id")
    pids = [p for p in manifest["patient_id"] if p in idx.index]
    items = [
        (p, img)
        for p in pids
        for img in str(idx.loc[p, "image_paths"]).split("|")
        if img and os.path.exists(img) and os.path.getsize(img) > 0
    ]
    print(f"{len(pids)} 患者 / {len(items)} 图")

    cache = os.path.join(RUNS, "feats_phikon.npz")
    if os.path.exists(cache):
        feats = np.load(cache, allow_pickle=True)["feats"].item()
        print("复用 Phikon 特征缓存")
    else:
        processor = AutoImageProcessor.from_pretrained(
            MODEL_ID, cache_dir=os.path.join(ROOT, "models")
        )
        model = AutoModel.from_pretrained(MODEL_ID, cache_dir=os.path.join(ROOT, "models"))
        feats = extract(model, items, processor)
        np.savez(cache, feats=feats)

    tasks = build_tasks(labels, pids)
    ref_ssl = {
        "grade3": 0.730,
        "dMMR": 0.550,
        "Ki67_high30": 0.597,
        "p53_abn": 0.537,
        "ER_pos": 0.532,
        "PR_pos": 0.569,
    }

    print(f"\n{'任务':<12} {'N':>5} {'正类':>5} {'Phikon AUROC':>16}   SSL对照")
    print("-" * 62)
    out = {"model": MODEL_ID, "tasks": {}}
    for name, y in tasks.items():
        ps = [p for p in y if p in feats]
        X = np.stack([feats[p] for p in ps])
        Y = np.array([y[p] for p in ps])
        m, s = cv(X, Y)
        out["tasks"][name] = {
            "n": len(Y),
            "pos": int(Y.sum()),
            "auroc": round(m, 3),
            "std": round(s, 3),
        }
        print(
            f"{name:<12} {len(Y):>5} {int(Y.sum()):>5} {m:.3f}±{s:.3f}      {ref_ssl.get(name, '')}"
        )

    with open(os.path.join(RUNS, "tasks_phikon.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n已写 runs/tasks_phikon.json")


if __name__ == "__main__":
    main()
