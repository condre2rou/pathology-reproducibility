# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "e2_train"))
from data import load_manifest  # noqa: E402

OUT = os.path.join(ROOT, "e6_learnability")
RUNS = os.path.join(ROOT, "e2_train", "runs")
RESOLUTIONS = [0, 512, 384, 256, 192, 128, 96]


class DS(Dataset):
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
            s = self.target / min(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
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
def extract(model, items, target, batch=32):
    dl = DataLoader(DS(items, target), batch_size=batch, shuffle=False, num_workers=4)
    feats = {}
    for x, pids in dl:
        f = model(x.cuda()).cpu().numpy()
        for i, pid in enumerate(pids):
            feats.setdefault(pid, []).append(f[i])
    return {k: np.mean(v, axis=0) for k, v in feats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="ssl", choices=["ssl", "imagenet"])
    args = ap.parse_args()
    out_path = os.path.join(OUT, f"res_feats_{args.backbone}.npz")
    if os.path.exists(out_path):
        print(f"已存在 {out_path}，跳过")
        return

    manifest = load_manifest()
    idx = manifest.set_index("patient_id")
    print("加载图像...")
    items = []
    for pid in manifest["patient_id"]:
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
    all_feats = {}
    for res in RESOLUTIONS:
        key = "native" if res == 0 else str(res)
        all_feats[key] = extract(model, items, res)
        print(
            f"  res={key}: {len(all_feats[key])} 患者 × {len(next(iter(all_feats[key].values())))} 维",
            flush=True,
        )

    np.savez(out_path, feats=all_feats)
    print(f"已写 {out_path}")


if __name__ == "__main__":
    main()
