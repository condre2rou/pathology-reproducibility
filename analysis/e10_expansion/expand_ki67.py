# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import io
import sys
import json
import time
import zipfile
import argparse
from collections import defaultdict

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "4")

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.dirname(HERE)
K = os.path.join(SNAP, "e7_external", "ki67_probe")
ZA, ZB = os.path.join(K, "trainA.zip"), os.path.join(K, "trainB.zip")
PHIKON = os.path.join(SNAP, "models", "models--owkin--phikon", "snapshots")


def ki67_positive_rate(img):
    """Preserved computation; see docs/experiments.md for its protocol."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    tissue = ((r + g + b) / 3) < 220
    if tissue.sum() < 500:
        return None
    brown = (r > g) & (g >= b) & ((r - b) > 25) & tissue
    return float(brown.sum() / tissue.sum() * 100)


def index_zip(path):
    """{patient: [entry names sorted]}。"""
    by = defaultdict(list)
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith(".png"):
                by[os.path.basename(n).split("_")[0]].append(n)
    return {p: sorted(v) for p, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-patches",
        type=int,
        default=60,
        help="每患者最多取多少对 patch（特征提取用；标签用全部 B patch）",
    )
    args = ap.parse_args()

    t0 = time.time()
    print("索引 zip …", flush=True)
    ia, ib = index_zip(ZA), index_zip(ZB)
    pats = sorted(set(ia) & set(ib))
    tot_a = sum(len(v) for v in ia.values())
    print(f"  trainA {len(ia)} 患者 / {tot_a} png；trainB {len(ib)} 患者", flush=True)
    print(f"  共同患者 {len(pats)}", flush=True)

    print("\n计算 Ki67 阳性率（全部 B patch）…", flush=True)
    y_by_p, nb_by_p = {}, {}
    with zipfile.ZipFile(ZB) as z:
        for pi, p in enumerate(pats, 1):
            vals = []
            for n in ib[p]:
                try:
                    r = ki67_positive_rate(Image.open(io.BytesIO(z.read(n))))
                    if r is not None:
                        vals.append(r)
                except Exception:
                    pass
            y_by_p[p] = float(np.mean(vals)) if vals else np.nan
            nb_by_p[p] = len(vals)
            if pi % 30 == 0 or pi == len(pats):
                print(f"  {pi}/{len(pats)}  {time.time() - t0:.0f}s", flush=True)

    from transformers import AutoImageProcessor, AutoModel
    import glob

    hits = glob.glob(os.path.join(PHIKON, "*", ""))
    local = hits[0] if hits else "owkin/phikon"
    proc = AutoImageProcessor.from_pretrained(local, local_files_only=True)
    model = AutoModel.from_pretrained(local, local_files_only=True).cuda().eval()
    print(f"\nPhikon: {local}", flush=True)

    src = {p: sorted(ia[p])[: args.max_patches] for p in pats}
    print(f"提取特征：{sum(len(v) for v in src.values())} 张 A patch", flush=True)

    feats_by_p, ypatch, ppatch = {}, [], []
    B = 32
    with torch.no_grad(), zipfile.ZipFile(ZA) as z:
        done = 0
        for p in pats:
            names = src[p]
            vs = []
            for i in range(0, len(names), B):
                imgs = [Image.open(io.BytesIO(z.read(n))).convert("RGB") for n in names[i : i + B]]
                px = proc(images=imgs, return_tensors="pt")["pixel_values"].cuda()
                f = model(pixel_values=px).last_hidden_state[:, 0, :].cpu().numpy()
                vs.append(f)
                done += len(imgs)
            feats_by_p[p] = np.mean(np.concatenate(vs), axis=0) if vs else None

            ypatch.append(np.array([y_by_p[p]] * len(names)))
            ppatch.append(np.array([p] * len(names)))
            if done % 2000 < B:
                print(f"  {done} 张  {time.time() - t0:.0f}s", flush=True)

    keep = [p for p in pats if feats_by_p[p] is not None and not np.isnan(y_by_p[p])]
    Xp = np.stack([feats_by_p[p] for p in keep])
    Yp = np.array([y_by_p[p] for p in keep])
    pid = np.array(keep)
    print(f"\n可用患者 {len(keep)}")

    np.savez(
        os.path.join(HERE, "ki67_all172.npz"),
        X=Xp,
        Y=Yp,
        pids=pid,
        y_by_p=np.array([y_by_p[p] for p in pats]),
        nb_by_p=np.array([nb_by_p[p] for p in pats]),
        pats=np.array(pats),
    )
    json.dump(
        {
            "n_patients": len(keep),
            "y_mean": float(Yp.mean()),
            "y_std": float(Yp.std()),
            "y_min": float(Yp.min()),
            "y_max": float(Yp.max()),
            "n_ge_20pct": int((Yp >= 20).sum()),
            "max_patches_per_patient": args.max_patches,
            "seconds": round(time.time() - t0),
        },
        open(os.path.join(HERE, "ki67_all172_meta.json"), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    print(f"→ {os.path.join(HERE, 'ki67_all172.npz')}")


if __name__ == "__main__":
    main()
