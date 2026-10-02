# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.dirname(HERE)
OV = os.path.join(SNAP, "e2_data", "ovarian_manifest.csv")
OUT = os.path.join(HERE, "pooled")
CACHE = os.path.join(HERE, "ovarian_features.npz")
MODEL_ID = "owkin/phikon"

MIN_MINORITY, MIN_TOTAL = 20, 80


class ImgDataset(Dataset):
    def __init__(self, items, processor):
        self.items, self.processor = items, processor

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
    for x, pids in dl:
        out = model(pixel_values=x.cuda())
        f = out.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
        for i, pid in enumerate(pids):
            feats.setdefault(pid, []).append(f[i])
    return {k: np.mean(v, axis=0) for k, v in feats.items()}


TASKS = {
    "ER_pos": lambda r: (1 if r["er"] == "pos" else 0) if r["er"] in ("pos", "neg") else None,
    "PR_pos": lambda r: (1 if r["pr"] == "pos" else 0) if r["pr"] in ("pos", "neg") else None,
    "p53_abn": lambda r: (1 if r["p53"] == "abn" else 0) if r["p53"] in ("wt", "abn") else None,
    "dMMR": lambda r: (1 if r["mmr"] == "dMMR" else 0) if r["mmr"] in ("pMMR", "dMMR") else None,
    "Ki67_high20": lambda r: (
        (1 if float(r["ki67"]) >= 20 else 0)
        if str(r["ki67"]).replace(".", "", 1).isdigit()
        else None
    ),
    "grade3": lambda r: (1 if r["grade"] == "3" else 0) if r["grade"] in ("1", "2", "3") else None,
}


def main():
    df = pd.read_csv(OV, dtype=str)
    items = [
        (r["patient_id"], p)
        for _, r in df.iterrows()
        for p in str(r["image_paths"]).split("|")
        if p and os.path.exists(p)
    ]
    print(f"卵巢癌 {len(df)} 患者, {len(items)} 张图")

    if os.path.exists(CACHE):
        print("复用 Phikon 特征缓存")
        feats = {k: v for k, v in np.load(CACHE, allow_pickle=True)["feats"].item().items()}
    else:
        from transformers import AutoImageProcessor, AutoModel
        import glob

        hits = glob.glob(
            os.path.join(SNAP, "models", "models--owkin--phikon", "snapshots", "*", "")
        )
        local = hits[0] if hits else MODEL_ID
        print(f"Phikon 本地路径: {local}")
        proc = AutoImageProcessor.from_pretrained(local, local_files_only=True)
        model = AutoModel.from_pretrained(local, local_files_only=True)
        feats = extract(model, items, proc)
        np.savez(CACHE, feats=feats)
        print(f"特征已缓存 → {CACHE}")

    summary = []

    kv = {}
    for _, r in df.iterrows():
        s = str(r["ki67"])
        if s.replace(".", "", 1).isdigit():
            kv[r["patient_id"]] = float(s)
    if len(kv) >= MIN_TOTAL:
        med = float(np.median(list(kv.values())))
        tasks_all = dict(TASKS)
        tasks_all["Ki67_median"] = (
            lambda m: (
                lambda r: (
                    (1 if float(r["ki67"]) > m else 0)
                    if str(r["ki67"]).replace(".", "", 1).isdigit()
                    else None
                )
            )
        )(med)
        print(f"Ki67 队列内中位数 = {med}")
    for name, fn in tasks_all.items():
        y = {r["patient_id"]: fn(r) for _, r in df.iterrows()}
        y = {k: v for k, v in y.items() if v is not None and k in feats}
        if not y:
            summary.append(dict(tag=f"OV-{name}", kept=False, reason="无可用样本"))
            continue
        ps = sorted(y)
        X = np.stack([feats[p] for p in ps]).astype(np.float32)
        Y = np.array([y[p] for p in ps], dtype=int)
        n_pos, n_neg = int(Y.sum()), int((1 - Y).sum())
        minor = min(n_pos, n_neg)
        if len(Y) < MIN_TOTAL or minor < MIN_MINORITY:
            r = dict(
                tag=f"OV-{name}",
                cohort="OV",
                kept=False,
                n_total=len(Y),
                n_pos=n_pos,
                n_neg=n_neg,
                reason=f"n={len(Y)} 少数类={minor} 低于阈值",
            )
        else:
            np.savez(os.path.join(OUT, f"OV-{name}.npz"), X=X, Y=Y, pids=np.array(ps))
            r = dict(
                tag=f"OV-{name}",
                cohort="OV",
                kept=True,
                n_total=len(Y),
                n_pos=n_pos,
                n_neg=n_neg,
                dim=int(X.shape[1]),
            )
        summary.append(r)
        print(f"  [{r['tag']}] {r}")

    sp = os.path.join(HERE, "ovarian_summary.json")
    json.dump(summary, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    kept = [s for s in summary if s.get("kept")]
    print(f"\n卵巢癌新增 {len(kept)} 个点 → {OUT}")


if __name__ == "__main__":
    main()
