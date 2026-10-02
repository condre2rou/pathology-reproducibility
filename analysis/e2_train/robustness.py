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
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_manifest, PatientImageDataset, ROOT  # noqa: E402

RUNS = os.path.join(ROOT, "e2_train", "runs")
LABELS = os.path.join(ROOT, "e1_label_engine", "outputs", "labels_ec_patient.csv")
N_SEEDS = 10


def get_backbone(kind):
    if kind == "ssl":
        ck = torch.load(os.path.join(RUNS, "simclr_backbone.pt"), map_location="cpu")
        m = models.resnet50()
        m.fc = nn.Identity()
        m.load_state_dict(ck["encoder"])
    else:
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
    return m.eval()


@torch.no_grad()
def extract(model, manifest, pids, size=224, batch=32):
    tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = PatientImageDataset(pids, manifest, tf)
    idx = manifest.set_index("patient_id")
    p2p = {
        p: pid
        for pid in pids
        if pid in idx.index
        for p in str(idx.loc[pid, "image_paths"]).split("|")
        if p
    }
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)
    feats, model, ptr = {}, model.cuda(), 0
    for x, _ in dl:
        f = model(x.cuda()).cpu().numpy()
        for i in range(len(f)):
            pid = p2p.get(ds.samples[ptr + i][0])
            if pid:
                feats.setdefault(pid, []).append(f[i])
        ptr += len(f)
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


def cv_aucs(X, Y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, Y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
        clf.fit(sc.transform(X[tr]), Y[tr])
        aucs.append(roc_auc_score(Y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="ssl", choices=["ssl", "imagenet"])
    args = ap.parse_args()

    manifest = load_manifest()
    labels = pd.read_csv(LABELS, dtype=str)
    pids = list(manifest["patient_id"])
    tasks = build_tasks(labels, pids)

    cache = os.path.join(RUNS, f"feats_{args.backbone}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        feats = z["feats"].item()
        print(f"复用特征缓存 {cache}")
    else:
        print("提取特征...")
        feats = extract(get_backbone(args.backbone), manifest, pids)
        np.savez(cache, feats=feats)

    out = {"backbone": args.backbone, "n_seeds": N_SEEDS, "tasks": {}}
    fold_auc = {}
    print(f"\n{'任务':<12} {'N':>5} {'正类':>5} {'AUROC (10 seeds)':>20} {'95% CI':>16}")
    print("-" * 68)
    for name, y in tasks.items():
        ps = [p for p in y if p in feats]
        X = np.stack([feats[p] for p in ps])
        Y = np.array([y[p] for p in ps])
        all_folds = []
        means = []
        for s in range(N_SEEDS):
            a = cv_aucs(X, Y, s)
            all_folds += a
            means.append(np.mean(a))
        fold_auc[name] = means
        lo, hi = np.percentile(all_folds, [2.5, 97.5])
        out["tasks"][name] = {
            "n": len(Y),
            "pos": int(Y.sum()),
            "auroc_mean": round(float(np.mean(all_folds)), 3),
            "auroc_std": round(float(np.std(all_folds)), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "seed_means": [round(float(m), 3) for m in means],
        }
        print(
            f"{name:<12} {len(Y):>5} {int(Y.sum()):>5} {np.mean(all_folds):.3f}±{np.std(all_folds):.3f}"
            f"      [{lo:.3f}, {hi:.3f}]"
        )

    if "grade3" in fold_auc and "dMMR" in fold_auc:
        stat, p = wilcoxon(fold_auc["grade3"], fold_auc["dMMR"])
        out["grade3_vs_dMMR_wilcoxon_p"] = float(p)
        print(f"\ngrade3 vs dMMR 配对 Wilcoxon p = {p:.2e}")

    with open(os.path.join(RUNS, f"robustness_{args.backbone}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"已写 runs/robustness_{args.backbone}.json")


if __name__ == "__main__":
    main()
