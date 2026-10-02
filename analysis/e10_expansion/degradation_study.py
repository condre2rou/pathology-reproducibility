# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import io
import sys
import json
import time
from collections import defaultdict

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.dirname(HERE)
MANIFEST = os.path.join(SNAP, "e2_data", "manifest.csv")
PHIKON = os.path.join(SNAP, "models", "models--owkin--phikon", "snapshots")

TASKS = {
    "grade3": lambda r: (1 if r["grade"] == "3" else 0) if r["grade"] in ("1", "2", "3") else None,
    "Ki67_high30": lambda r: (
        (1 if float(r["ki67"]) >= 30 else 0)
        if str(r["ki67"]).replace(".", "", 1).isdigit()
        else None
    ),
    "p53_abn": lambda r: (1 if r["p53"] == "abn" else 0) if r["p53"] in ("wt", "abn") else None,
    "dMMR": lambda r: (1 if r["mmr"] == "dMMR" else 0) if r["mmr"] in ("pMMR", "dMMR") else None,
    "ER_pos": lambda r: (1 if r["er"] == "pos" else 0) if r["er"] in ("pos", "neg") else None,
    "PR_pos": lambda r: (1 if r["pr"] == "pos" else 0) if r["pr"] in ("pos", "neg") else None,
}

SCALE = {
    "grade3": "cell",
    "Ki67_high30": "cell",
    "p53_abn": "molecular",
    "dMMR": "molecular",
    "ER_pos": "molecular",
    "PR_pos": "molecular",
}


def deg_none(im):
    return im


def deg_down(f):
    def _f(im):
        w, h = im.size
        s = 1.0 / f
        return im.resize((max(8, int(w * s)), max(8, int(h * s))), Image.LANCZOS)

    return _f


def deg_blur(sigma):
    return lambda im: im.filter(ImageFilter.GaussianBlur(sigma))


def deg_jpeg(q):
    def _f(im):
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    return _f


def deg_contrast(c):
    return lambda im: ImageEnhance.Contrast(im).enhance(c)


def deg_brightness(b):
    return lambda im: ImageEnhance.Brightness(im).enhance(b)


LEVELS = [
    ("native", deg_none, "无降质（基线）"),
    ("down2", deg_down(2), "降采样 ÷2"),
    ("down4", deg_down(4), "降采样 ÷4"),
    ("down8", deg_down(8), "降采样 ÷8"),
    ("blur1", deg_blur(1.0), "高斯模糊 σ=1"),
    ("blur2", deg_blur(2.0), "高斯模糊 σ=2"),
    ("blur4", deg_blur(4.0), "高斯模糊 σ=4"),
    ("jpeg50", deg_jpeg(50), "JPEG q50"),
    ("jpeg20", deg_jpeg(20), "JPEG q20"),
    ("jpeg5", deg_jpeg(5), "JPEG q5"),
    ("contrast0.5", deg_contrast(0.5), "对比度 ×0.5"),
    ("bright0.6", deg_brightness(0.6), "亮度 ×0.6"),
]


def extract(model, proc, paths, tf, batch=32):
    """Preserved computation; see docs/experiments.md for its protocol."""
    feats = defaultdict(list)
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs, pids = [], []
            for pid, p in paths[i : i + batch]:
                try:
                    im = tf(Image.open(p).convert("RGB"))
                except Exception:
                    continue
                imgs.append(im)
                pids.append(pid)
            if not imgs:
                continue
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].cuda()
            f = model(pixel_values=px).last_hidden_state[:, 0, :].cpu().numpy()
            for j, pid in enumerate(pids):
                feats[pid].append(f[j])
    return {k: np.mean(v, axis=0) for k, v in feats.items()}


def cv_auroc(X, Y, seeds=10):
    aucs = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, Y):
            if len(set(Y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(X[tr]), Y[tr])
            aucs.append(roc_auc_score(Y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    df = pd.read_csv(MANIFEST, dtype=str)
    paths = [
        (r["patient_id"], p)
        for _, r in df.iterrows()
        for p in str(r["image_paths"]).split("|")
        if p and os.path.exists(p)
    ]
    print(f"共 {len(df)} 患者 / {len(paths)} 张图", flush=True)

    from transformers import AutoImageProcessor, AutoModel
    import glob

    hits = glob.glob(os.path.join(PHIKON, "*", ""))
    local = hits[0] if hits else "owkin/phikon"
    proc = AutoImageProcessor.from_pretrained(local, local_files_only=True)
    model = AutoModel.from_pretrained(local, local_files_only=True).cuda().eval()
    print(f"Phikon: {local}\n", flush=True)

    out = {
        "meta": {
            "n_patients": len(df),
            "n_images": len(paths),
            "backbone": "Phikon ViT-B/16 CLS, 患者级 mean 池化",
            "protocol": "10 seeds × 5-fold stratified CV, LR(C=0.1, balanced)",
            "scale_expectation": SCALE,
        },
        "curves": {},
    }

    for name, tf, desc in LEVELS:
        t0 = time.time()
        feats = extract(model, proc, paths, tf)
        row = {}
        for tname, fn in TASKS.items():
            y = {r["patient_id"]: fn(r) for _, r in df.iterrows()}
            y = {k: v for k, v in y.items() if v is not None and k in feats}
            if len(y) < 60 or len(set(y.values())) < 2:
                continue
            ps = sorted(y)
            X = np.stack([feats[p] for p in ps])
            Y = np.array([y[p] for p in ps])
            if min(np.bincount(Y)) < 8:
                continue
            m, s = cv_auroc(X, Y)
            row[tname] = {
                "n": len(Y),
                "pos": int(Y.sum()),
                "auroc": round(m, 4),
                "std": round(s, 4),
            }
        out["curves"][name] = {"desc": desc, "tasks": row, "seconds": round(time.time() - t0)}
        cells = "  ".join(f"{t}={row[t]['auroc']:.3f}" for t in TASKS if t in row)
        print(f"  [{name:<12}] {cells}   ({time.time() - t0:.0f}s)", flush=True)

    json.dump(
        out,
        open(os.path.join(HERE, "degradation_result.json"), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    print(f"\n{'档位':<14}" + "".join(f"{t:>12}" for t in TASKS))
    for name, _, _ in LEVELS:
        row = out["curves"][name]["tasks"]
        print(
            f"{name:<14}"
            + "".join(f"{row[t]['auroc']:>12.3f}" if t in row else f"{'—':>12}" for t in TASKS)
        )
    print(f"\n→ {os.path.join(HERE, 'degradation_result.json')}")


if __name__ == "__main__":
    main()
