# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os, sys, json

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "e2_train"))
from data import load_manifest

feats = np.load(os.path.join(ROOT, "e2_train", "runs", "feats_phikon.npz"), allow_pickle=True)[
    "feats"
].item()
labels = pd.read_csv(
    os.path.join(ROOT, "e1_label_engine", "outputs", "labels_ec_patient.csv"), dtype=str
)
man = load_manifest()
idx = labels.drop_duplicates("folder").set_index(labels["year"] + "_" + labels["folder"])


try:
    from scipy.ndimage import laplace

    def lap_var(g):
        return float(laplace(g.astype(np.float32)).var())
except ImportError:
    _K = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    def lap_var(g):
        gp = np.pad(g.astype(np.float32), 1, mode="edge")
        h, w = g.shape
        out = np.zeros((h, w), dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                out += _K[dy, dx] * gp[dy : dy + h, dx : dx + w]
        return float(out.var())


def image_quality(path):
    """Preserved computation; see docs/experiments.md for its protocol."""

    with Image.open(path) as im:
        im.load()
        rgb = im.convert("RGB")
        g = np.asarray(rgb.convert("L"), dtype=np.float32)
        arr = np.asarray(rgb, dtype=np.float32)
        h, w = g.shape

        q = im.quantization
        qmean = float(np.mean([v for t in q.values() for v in t])) if q else float("nan")

    bh, bw = max(1, h // 3), max(1, w // 3)
    blocks = [g[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw] for i in range(3) for j in range(3)]
    block_means = np.array([b.mean() for b in blocks if b.size])
    illum = float(block_means.std() / (block_means.mean() + 1e-6))

    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = float(np.mean((mx - mn) / (mx + 1e-6)))
    cast = float(np.std(arr.reshape(-1, 3).mean(axis=0)))

    return {
        "sharp": lap_var(g),
        "bytes_per_px": os.path.getsize(path) / float(h * w),
        "jpeg_q": qmean,
        "contrast": float(g.std() / 255.0),
        "illum_uneven": illum,
        "sat_mean": sat,
        "colour_cast": cast,
        "h": h,
        "w": w,
    }


QUAL_KEYS = [
    "sharp",
    "bytes_per_px",
    "jpeg_q",
    "contrast",
    "illum_uneven",
    "sat_mean",
    "colour_cast",
]
per_patient = {}
for _, r in man.iterrows():
    pid = r["patient_id"]
    if pid not in idx.index or pid not in feats:
        continue
    ps = [
        p
        for p in str(r["image_paths"]).split("|")
        if p and os.path.exists(p) and os.path.getsize(p) > 0
    ]
    if not ps:
        continue
    rows = []
    for p in ps:
        try:
            rows.append(image_quality(p))
        except Exception as e:
            print(f"  [warn] {p}: {e}")
    if not rows:
        continue
    rec = {k: float(np.mean([x[k] for x in rows])) for k in QUAL_KEYS}
    rec["n_images"] = len(rows)
    per_patient[pid] = rec

print(f"已计算质量指标的患者数: {len(per_patient)}")


def build(task_fn, subset=None):
    y = {}
    for pid, q in per_patient.items():
        if subset is not None and pid not in subset:
            continue
        v = task_fn(idx.loc[pid])
        if v is not None:
            y[pid] = v
    ps = list(y)
    return np.stack([feats[p] for p in ps]), np.array([y[p] for p in ps])


def cv(X, Y, seeds=10):
    a = []
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, Y):
            if len(set(Y[tr])) < 2 or len(set(Y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(X[tr]), Y[tr])
            a.append(roc_auc_score(Y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return (float(np.mean(a)), float(np.std(a)), len(Y), int(Y.sum())) if a else (float("nan"),) * 4


TASKS = {
    "grade3": lambda r: (1 if r["grade"] == "3" else 0) if r["grade"] in ("1", "2", "3") else None,
    "dMMR": lambda r: (1 if r["mmr"] == "dMMR" else 0) if r["mmr"] in ("pMMR", "dMMR") else None,
}


BETTER = {
    "sharp": "high",
    "bytes_per_px": "high",
    "jpeg_q": "low",
    "contrast": "high",
    "illum_uneven": "low",
    "sat_mean": "high",
    "colour_cast": "low",
}

out = {"n_patients_with_quality": len(per_patient), "tasks": {}}

for tname, tfn in TASKS.items():
    X, Y = build(tfn)
    m, s, n, pos = cv(X, Y)
    print(f"\n[{tname}] 全体 n={n} pos={pos} AUROC={m:.3f}±{s:.3f}")
    out["tasks"][tname] = {
        "ALL": {"auroc": round(m, 3), "sd": round(s, 3), "n": n, "pos": pos},
        "strata": {},
    }

    for k in QUAL_KEYS:
        vals = {
            pid: q[k]
            for pid, q in per_patient.items()
            if pid in idx.index and pid in feats and np.isfinite(q[k])
        }
        if len(vals) < 60:
            continue
        v = pd.Series(vals)
        try:
            tert = pd.qcut(v, 3, labels=["T1", "T2", "T3"])
        except ValueError:
            continue

        fav = "T3" if BETTER[k] == "high" else "T1"
        worst = "T1" if BETTER[k] == "high" else "T3"
        rec = {}
        for lab in ["T1", "T2", "T3"]:
            sub = set(v.index[tert == lab])
            Xs, Ys = build(tfn, sub)
            if len(Ys) < 40 or len(set(Ys)) < 2:
                rec[lab] = {"note": "样本不足", "n": int(len(Ys))}
                continue
            mm, ss, nn, pp = cv(Xs, Ys)
            rec[lab] = {
                "auroc": round(mm, 3),
                "sd": round(ss, 3),
                "n": nn,
                "pos": pp,
                "range": [round(float(v[list(sub)].min()), 4), round(float(v[list(sub)].max()), 4)],
            }
            print(f"   {k:<14} {lab} n={nn:>4} pos={pp:>3} AUROC={mm:.3f}±{ss:.3f}")
        rec["favourable_stratum"] = fav
        rec["least_favourable_stratum"] = worst
        out["tasks"][tname]["strata"][k] = rec
        print(f"   -> {k}: 最有利档 {fav}")


print("\n=== 混杂检查：质量指标单独预测标签的能力（AUROC）===")
out["quality_predicts_label"] = {}
for tname, tfn in TASKS.items():
    _, Y = build(tfn)
    pids = [p for p in per_patient if p in idx.index and p in feats and tfn(idx.loc[p]) is not None]
    Q = np.array([[per_patient[p][k] for k in QUAL_KEYS] for p in pids], dtype=float)
    Yq = np.array([tfn(idx.loc[p]) for p in pids])
    ok = np.isfinite(Q).all(axis=1)
    Q, Yq = Q[ok], Yq[ok]
    res = {}
    for i, k in enumerate(QUAL_KEYS):
        col = Q[:, i : i + 1]
        if len(set(Yq)) < 2:
            continue
        mm, ss, nn, pp = cv(col, Yq)
        res[k] = {"auroc": round(mm, 3), "sd": round(ss, 3), "n": nn, "pos": pp}
        print(f"   [{tname}] {k:<14} AUROC={mm:.3f}±{ss:.3f}")
    res["_multivariate_all_quality"] = dict(zip(["auroc", "sd", "n", "pos"], cv(Q, Yq)))
    print(f"   [{tname}] 全部质量指标联合   AUROC={res['_multivariate_all_quality']['auroc']:.3f}")
    out["quality_predicts_label"][tname] = res

out["notes"] = (
    "质量指标全部在原生分辨率上计算，未做任何 resize（Laplacian 方差与尺度强相关）。"
    "分层用三分位；favourable_stratum 是按质量假设『更好的一档信号应更强』指定的那一档。"
    "特征用缓存的患者级 Phikon 特征，分层只改变患者子集，不重新提特征。"
)

with open(os.path.join(HERE, "quality_strat_result.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n已写 e9_revision/quality_strat_result.json")
