# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT  # noqa: E402

RUNS = os.path.join(ROOT, "e2_train", "runs")
os.makedirs(RUNS, exist_ok=True)


class TwoViewDataset(Dataset):
    """Preserved computation; see docs/experiments.md for its protocol."""

    def __init__(self, paths, transform):
        self.paths = [p for p in paths if os.path.exists(p) and os.path.getsize(p) > 0]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.transform(img)


class SimCLR(nn.Module):
    def __init__(self, backbone="resnet50", proj_dim=128):
        super().__init__()
        w = (
            models.ResNet50_Weights.IMAGENET1K_V1
            if backbone == "resnet50"
            else models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.encoder = (
            models.resnet50(weights=w) if backbone == "resnet50" else models.resnet18(weights=w)
        )
        self.feat_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = F.normalize(self.proj(h), dim=1)
        return z


def nt_xent(z1, z2, temp=0.5):
    z = torch.cat([z1, z2], dim=0)  # 2N x d
    sim = z @ z.T / temp
    N = z1.size(0)
    mask = torch.eye(2 * N, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(z.device)
    return F.cross_entropy(sim, labels)


def get_ssl_transform(size=224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--backbone", type=str, default="resnet50")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} backbone={args.backbone} epochs={args.epochs}")

    lab = pd.read_csv(
        os.path.join(ROOT, "e1_label_engine", "outputs", "labels_ec_patient.csv"), dtype=str
    )
    paths = [p for ps in lab["image_paths"].dropna() for p in ps.split("|") if p]
    ds = TwoViewDataset(paths, get_ssl_transform(args.size))
    print(f"SSL 图像数 {len(ds)}（来自 {lab['image_paths'].notna().sum()} 患者）")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    model = SimCLR(args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(args.epochs):
        model.train()
        run = 0.0
        n = 0
        for x1, x2 in tqdm(loader, desc=f"ssl ep{ep + 1}/{args.epochs}", leave=False):
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)
            loss = nt_xent(z1, z2, args.temp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(x1)
            n += len(x1)
        sched.step()
        print(f"[ssl ep{ep + 1}] loss={run / n:.4f}")

    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "feat_dim": model.feat_dim,
            "backbone": args.backbone,
            "size": args.size,
        },
        os.path.join(RUNS, "simclr_backbone.pt"),
    )
    print("已存 → e2_train/runs/simclr_backbone.pt")


if __name__ == "__main__":
    main()
