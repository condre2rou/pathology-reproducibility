# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_manifest, load_splits, PatientImageDataset  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "e2_train", "runs")
os.makedirs(OUT, exist_ok=True)


def get_transform(size=224, train=True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
            labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    if len(np.unique(labels)) < 2:
        return {
            "n": len(labels),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "pos": int(labels.sum()),
            "neg": int((labels == 0).sum()),
        }
    return {
        "n": len(labels),
        "auroc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
        "pos": int(labels.sum()),
        "neg": int((labels == 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", type=str, default="resnet50")
    ap.add_argument("--freeze", action="store_true", help="冻结 backbone，只训线性分类头")
    ap.add_argument(
        "--ssl", type=str, default="", help="SimCLR backbone checkpoint 路径（覆盖 ImageNet）"
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} backbone={args.backbone} size={args.size}")

    manifest = load_manifest()
    splits = load_splits()
    t = splits["temporal"]

    train_ds = PatientImageDataset(t["train"], manifest, get_transform(args.size, True))
    test23_ds = PatientImageDataset(t["test_2023"], manifest, get_transform(args.size, False))
    test24_ds = PatientImageDataset(t["test_2024"], manifest, get_transform(args.size, False))
    print(f"train images={len(train_ds)} | test2023={len(test23_ds)} | test2024={len(test24_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    test23_loader = DataLoader(test23_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    test24_loader = DataLoader(test24_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    pos = sum(1 for _, y in train_ds if y == 1)
    neg = len(train_ds) - pos
    pos_weight = torch.tensor([neg / pos]).to(device)
    print(f"train pos(dMMR)={pos} neg={neg} pos_weight={neg / pos:.2f}")

    if args.ssl:
        model = models.resnet50() if args.backbone == "resnet50" else models.resnet18()
        ckpt = torch.load(args.ssl, map_location="cpu")
        model.fc = nn.Identity()
        model.load_state_dict(ckpt["encoder"])
        feat_dim = ckpt["feat_dim"]
    else:
        weights = (
            models.ResNet50_Weights.IMAGENET1K_V1
            if args.backbone == "resnet50"
            else models.ResNet18_Weights.IMAGENET1K_V1
        )
        model = (
            models.resnet50(weights=weights)
            if args.backbone == "resnet50"
            else models.resnet18(weights=weights)
        )
        feat_dim = model.fc.in_features
    model.fc = nn.Linear(feat_dim, 1)
    if args.freeze:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(args.epochs):
        model.train()
        tot, run = 0, 0.0
        for x, y in tqdm(train_loader, desc=f"ep{ep + 1}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device).float()
            opt.zero_grad()
            loss = crit(model(x).ravel(), y)
            loss.backward()
            opt.step()
            tot += len(y)
            run += loss.item() * len(y)
        sched.step()
        r23 = evaluate(model, test23_loader, device)
        r24 = evaluate(model, test24_loader, device)
        print(
            f"[ep{ep + 1}] loss={run / tot:.4f} | test2023 AUROC={r23['auroc']:.3f} AUPRC={r23['auprc']:.3f} "
            f"| test2024 AUROC={r24['auroc']:.3f} AUPRC={r24['auprc']:.3f}"
        )

    final = {
        "test_2023": r23,
        "test_2024": r24,
        "train_pos": pos,
        "train_neg": neg,
        "backbone": args.backbone,
        "size": args.size,
        "seed": args.seed,
    }
    with open(os.path.join(OUT, "baseline_result.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("\n完成 → e2_train/runs/baseline_result.json")


if __name__ == "__main__":
    main()
