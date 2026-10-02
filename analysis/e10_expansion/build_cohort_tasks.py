# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import sys
import json
import argparse
import warnings

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.dirname(HERE)  # .../snap-cpath
T5 = os.path.join(SNAP, "e5_tcga")
OUT = os.path.join(HERE, "pooled")
HF_CACHE = os.environ.get(
    "PATHOLOGY_HF_CACHE", os.path.join(os.path.dirname(__file__), "..", "models")
)  # datasets--W8Yi--tcga-wsi-uni2h-features

os.makedirs(OUT, exist_ok=True)


MIN_MINORITY = 20
MIN_TOTAL = 80


def load_cached_cohort(cohort):
    """Preserved computation; see docs/experiments.md for its protocol."""
    paths = {
        "UCEC": os.path.join(T5, "pooled_features.npz"),
        "COAD": os.path.join(T5, "pooled", "COAD.npz"),
        "STAD": os.path.join(T5, "pooled", "STAD.npz"),
        "BRCA_IDC": os.path.join(T5, "pooled", "BRCA_IDC.npz"),
    }
    p = paths.get(cohort)
    if not p or not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {str(k): v for k, v in zip(z["pids"], z["X"])}


def find_h5(hf_dir, pid):
    """Preserved computation; see docs/experiments.md for its protocol."""
    import glob

    pat = os.path.join(
        HF_CACHE,
        "datasets--W8Yi--tcga-wsi-uni2h-features",
        "snapshots",
        "*",
        hf_dir,
        "features",
        f"{pid}-*.h5",
    )
    return sorted(glob.glob(pat))


def pool_h5(paths):
    """Preserved computation; see docs/experiments.md for its protocol."""

    import h5py

    vs = []
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                X = f["features"][0]  # (N, 1536)
                vs.append(X.mean(axis=0))
        except Exception:
            continue
    if not vs:
        return None
    return np.mean(vs, axis=0)


def _load_hf_cohort_feats(cohort):
    """Preserved computation; see docs/experiments.md for its protocol."""
    cache = os.path.join(HERE, f"feats_{cohort}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {str(k): v for k, v in zip(z["pids"], z["X"])}
    avail = load_hf_cohort(f"TCGA-{cohort}")
    if not avail:
        return None
    print(f"  [{cohort}] 本地切片 {len(avail)} 例，池化中…", flush=True)
    feats = {}
    for pid, paths in avail.items():
        v = pool_h5(paths)
        if v is not None:
            feats[pid] = v
    pids = sorted(feats)
    np.savez(cache, X=np.stack([feats[p] for p in pids]).astype(np.float32), pids=np.array(pids))
    return feats


def load_hf_cohort(hf_dir, pids=None):
    """Preserved computation; see docs/experiments.md for its protocol."""
    import glob

    cands = [os.path.join(HERE, "h5", hf_dir, "features")]
    snap = glob.glob(
        os.path.join(
            HF_CACHE,
            "datasets--W8Yi--tcga-wsi-uni2h-features",
            "snapshots",
            "*",
            hf_dir,
            "features",
        )
    )
    cands += snap
    feats_dir = next(
        (c for c in cands if os.path.isdir(c) and any(f.endswith(".h5") for f in os.listdir(c))),
        None,
    )
    if feats_dir is None:
        return None
    avail = {}
    for fn in os.listdir(feats_dir):
        if fn.endswith(".h5"):
            avail.setdefault(fn.split("-01Z")[0], []).append(os.path.join(feats_dir, fn))
    if pids is not None:
        avail = {k: v for k, v in avail.items() if k in pids}
    return avail


def task_t1_ucec_subtype(feats):
    """Preserved computation; see docs/experiments.md for its protocol."""
    d = pd.read_csv(os.path.join(T5, "tcga_extra_labels.csv"), dtype=str)
    m = {"UEC": 1, "USC": 0}
    d = d[d.oncotree.isin(m)]
    return [(str(r.patient_id), m[r.oncotree]) for r in d.itertuples()]


def task_t1_ucec_cn(feats):
    """Preserved computation; see docs/experiments.md for its protocol."""
    d = pd.read_csv(os.path.join(T5, "tcga_ucec_labels.csv"), dtype=str)
    m = {"UCEC_CN_HIGH": 1, "UCEC_CN_LOW": 0}
    d = d[d.SUBTYPE.isin(m)]
    return [(str(r.patient_id), m[r.SUBTYPE]) for r in d.itertuples()]


def task_t1_ucec_grade(feats):
    """Preserved computation; see docs/experiments.md for its protocol."""
    d = pd.read_csv(os.path.join(T5, "tcga_ucec_clinical.csv"), dtype=str)
    g = d.tumor_grade.fillna("")
    keep = g.isin(["G1", "G2", "G3"])
    d = d[keep]
    return [(str(r.submitter_id), 1 if r.tumor_grade == "G3" else 0) for r in d.itertuples()]


def _subtype_pair(cancer, pos, neg):
    """Preserved computation; see docs/experiments.md for its protocol."""
    d = pd.read_csv(os.path.join(T5, "multi_subset.csv"), dtype=str)
    d = d[d.cancer == cancer]
    d = d[d.raw.isin([pos, neg])]
    return [(str(r.patient_id), 1 if r.raw == pos else 0) for r in d.itertuples()]


T1_TASKS = {
    "WSI-UCEC-UECvsUSC": ("UCEC", task_t1_ucec_subtype),
    "WSI-UCEC-CNhighvsCNlow": ("UCEC", task_t1_ucec_cn),
    "WSI-UCEC-grade3": ("UCEC", task_t1_ucec_grade),
    "WSI-COAD-MSIvsCIN": ("COAD", lambda f: _subtype_pair("COAD", "COAD_MSI", "COAD_CIN")),
    "WSI-STAD-MSIvsCIN": ("STAD", lambda f: _subtype_pair("STAD", "STAD_MSI", "STAD_CIN")),
}


STUDY_OF = {
    "UCEC": "ucec_tcga_pan_can_atlas_2018",
    "COAD": "coadread_tcga_pan_can_atlas_2018",
    "STAD": "stad_tcga_pan_can_atlas_2018",
    "BRCA_IDC": "brca_tcga_pan_can_atlas_2018",
}
GENES_OF = {
    "UCEC": ["TP53", "PTEN", "PIK3CA", "ARID1A", "CTNNB1", "PPP2R1A", "KRAS", "FBXW7", "MSH6"],
    "COAD": ["TP53", "KRAS", "APC", "PIK3CA", "SMAD4", "BRAF", "FBXW7"],
    "STAD": ["TP53", "ARID1A", "PIK3CA", "KRAS", "SMAD4", "RHOA"],
    "BRCA_IDC": ["TP53", "PIK3CA", "GATA3", "CDH1", "MAP3K1"],
}


PANEL = [
    "TP53",
    "KRAS",
    "PIK3CA",
    "PTEN",
    "ARID1A",
    "APC",
    "BRAF",
    "CTNNB1",
    "SMAD4",
    "RB1",
    "ATM",
    "CDH1",
    "GATA3",
    "MAP3K1",
    "FBXW7",
    "PPP2R1A",
    "MSH6",
    "NF1",
    "PIK3R1",
    "STK11",
    "EGFR",
    "ERBB2",
    "IDH1",
    "VHL",
]

STUDY_OF_EXTRA = {
    "LGG": "lgg_tcga_pan_can_atlas_2018",
    "LUAD": "luad_tcga_pan_can_atlas_2018",
    "LUSC": "lusc_tcga_pan_can_atlas_2018",
    "HNSC": "hnsc_tcga_pan_can_atlas_2018",
    "BLCA": "blca_tcga_pan_can_atlas_2018",
    "PRAD": "prad_tcga_pan_can_atlas_2018",
    "LIHC": "lihc_tcga_pan_can_atlas_2018",
    "KIRP": "kirp_tcga_pan_can_atlas_2018",
    "KIRC": "kirc_tcga_pan_can_atlas_2018",
    "CESC": "cesc_tcga_pan_can_atlas_2018",
    "TGCT": "tgct_tcga_pan_can_atlas_2018",
    "PAAD": "paad_tcga_pan_can_atlas_2018",
    "PCPG": "pcpg_tcga_pan_can_atlas_2018",
    "THYM": "thym_tcga_pan_can_atlas_2018",
    "ESCA": "esca_tcga_pan_can_atlas_2018",
    "THCA": "thca_tcga_pan_can_atlas_2018",
    "SKCM": "skcm_tcga_pan_can_atlas_2018",
    "SARC": "sarc_tcga_pan_can_atlas_2018",
    "GBM": "gbm_tcga_pan_can_atlas_2018",
    "READ": "coadread_tcga_pan_can_atlas_2018",
    "OV": "ov_tcga_pan_can_atlas_2018",
    "UCS": "ucec_tcga_pan_can_atlas_2018",
    "KICH": "kich_tcga_pan_can_atlas_2018",
    "ACC": "acc_tcga_pan_can_atlas_2018",
    "UVM": "uvm_tcga_pan_can_atlas_2018",
    "DLBC": "dlbc_tcga_pan_can_atlas_2018",
    "CHOL": "chol_tcga_pan_can_atlas_2018",
    "MESO": "meso_tcga_pan_can_atlas_2018",
    "BRCA_OTHERS": "brca_tcga_pan_can_atlas_2018",
}
STUDY_OF.update(STUDY_OF_EXTRA)


def local_cohorts():
    """Preserved computation; see docs/experiments.md for its protocol."""
    import glob

    bases = [os.path.join(HERE, "h5")]
    bases += glob.glob(
        os.path.join(HF_CACHE, "datasets--W8Yi--tcga-wsi-uni2h-features", "snapshots", "*")
    )
    out = set()
    for b in bases:
        if not os.path.isdir(b):
            continue
        for d in os.listdir(b):
            if d.startswith("TCGA-") and os.path.isdir(os.path.join(b, d, "features")):
                out.add(d.replace("TCGA-", ""))
    return sorted(out)


def _cbio_json(url, tries=3):
    """Preserved computation; see docs/experiments.md for its protocol."""
    import urllib.request
    import time as _t

    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=180).read())
        except Exception as e:
            last = e
            _t.sleep(2 * (i + 1))
    raise last


def entrez_of(symbol):
    try:
        return _cbio_json(f"https://www.cbioportal.org/api/genes/{symbol}")["entrezGeneId"]
    except Exception:
        return None


def sequenced_patients(study):
    """Preserved computation; see docs/experiments.md for its protocol."""

    try:
        ids = _cbio_json(
            f"https://www.cbioportal.org/api/sample-lists/{study}_sequenced/sample-ids"
        )
    except Exception:
        return None
    out = set()
    for s in ids:
        parts = str(s).split("-")
        if len(parts) >= 3:
            out.add("-".join(parts[:3]))
    return out


def mutated_patients(study, gene):
    """Preserved computation; see docs/experiments.md for its protocol."""
    ez = entrez_of(gene)
    if ez is None:
        return None
    url = (
        f"https://www.cbioportal.org/api/molecular-profiles/{study}_mutations/mutations"
        f"?sampleListId={study}_sequenced&entrezGeneId={ez}&projection=SUMMARY"
    )
    try:
        return set(r["patientId"] for r in _cbio_json(url))
    except Exception:
        return None


def task_mutation(cohort, gene, feats):
    """Preserved computation; see docs/experiments.md for its protocol."""
    study = STUDY_OF[cohort]
    mut = mutated_patients(study, gene)
    seq = sequenced_patients(study)
    if mut is None or seq is None:
        return None
    return [(pid, 1 if pid in mut else 0) for pid in feats if pid in seq]


CONT_ATTRS = ["ANEUPLOIDY_SCORE", "FRACTION_GENOME_ALTERED", "MUTATION_COUNT"]


def cbio_attr(study, attr):
    """Preserved computation; see docs/experiments.md for its protocol."""
    for dt in ["PATIENT", "SAMPLE"]:
        try:
            d = _cbio_json(
                f"https://www.cbioportal.org/api/studies/{study}/clinical-data"
                f"?clinicalDataType={dt}&attributeId={attr}&projection=SUMMARY"
            )
        except Exception:
            continue
        if d:
            return {r["patientId"]: r["value"] for r in d}
    return {}


def task_median_split(cohort, attr, feats):
    """Preserved computation; see docs/experiments.md for its protocol."""
    vals = cbio_attr(STUDY_OF[cohort], attr)
    xs = []
    for pid in feats:
        v = vals.get(pid)
        if v in (None, "", "NA"):
            continue
        try:
            xs.append((pid, float(v)))
        except ValueError:
            continue
    if len(xs) < MIN_TOTAL:
        return None, None
    arr = np.array([x[1] for x in xs])
    med = float(np.median(arr))
    return [(pid, 1 if v > med else 0) for pid, v in xs], med


def assemble(tag, cohort, pairs, feats):
    """Preserved computation; see docs/experiments.md for its protocol."""
    X, Y, P, missing = [], [], [], 0
    for pid, y in pairs:
        if pid in feats:
            X.append(feats[pid])
            Y.append(y)
            P.append(pid)
        else:
            missing += 1
    if not X:
        return None
    X = np.stack(X).astype(np.float32)
    Y = np.array(Y, dtype=int)
    n_pos, n_neg = int(Y.sum()), int((1 - Y).sum())
    minor = min(n_pos, n_neg)
    if len(Y) < MIN_TOTAL or minor < MIN_MINORITY:
        return dict(
            tag=tag,
            cohort=cohort,
            kept=False,
            reason=f"n={len(Y)} 少数类={minor} 低于阈值（n≥{MIN_TOTAL}, 少数类≥{MIN_MINORITY}）",
            n_total=len(Y),
            n_pos=n_pos,
            n_neg=n_neg,
        )
    np.savez(os.path.join(OUT, f"{tag}.npz"), X=X, Y=Y, pids=np.array(P))
    return dict(
        tag=tag,
        cohort=cohort,
        kept=True,
        n_total=len(Y),
        n_pos=n_pos,
        n_neg=n_neg,
        n_missing_feats=missing,
        dim=int(X.shape[1]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["t1", "mut", "cont", "t2"], required=True)
    ap.add_argument(
        "--cohort", nargs="*", default=[], help="T2 模式：要池化的 HF 队列目录名，如 TCGA-LUAD"
    )
    ap.add_argument(
        "--extra-cohorts",
        nargs="*",
        default=[],
        help="cont 模式：额外纳入的新下载队列短名，如 LGG LUAD",
    )
    ap.add_argument(
        "--tasks", default=os.path.join(HERE, "t2_tasks.json"), help="T2 模式：任务定义 JSON"
    )
    args = ap.parse_args()

    summary = []
    if args.mode == "t1":
        cache = {}
        for tag, (cohort, fn) in T1_TASKS.items():
            if cohort not in cache:
                cache[cohort] = load_cached_cohort(cohort)
            feats = cache[cohort]
            if feats is None:
                summary.append(dict(tag=tag, cohort=cohort, kept=False, reason="无已缓存特征"))
                continue
            pairs = fn(feats)
            r = assemble(tag, cohort, pairs, feats)
            summary.append(
                r if r else dict(tag=tag, cohort=cohort, kept=False, reason="无可用样本")
            )
            print(f"  [{tag}] {summary[-1]}")
    elif args.mode == "mut":
        targets = list(GENES_OF.keys()) + [c for c in local_cohorts() if c not in GENES_OF]
        if args.extra_cohorts:
            targets = list(dict.fromkeys(targets + args.extra_cohorts))
        for cohort in targets:
            if cohort not in STUDY_OF:
                continue
            feats = (
                load_cached_cohort(cohort) if cohort in GENES_OF else _load_hf_cohort_feats(cohort)
            )
            if feats is None:
                print(f"  [{cohort}] 无特征，跳过", flush=True)
                continue
            print(f"\n=== {cohort}（{len(feats)} 例）===", flush=True)
            for gene in GENES_OF.get(cohort, PANEL):
                tag = f"WSI-{cohort}-{gene}"
                pairs = task_mutation(cohort, gene, feats)
                if pairs is None:
                    summary.append(
                        dict(tag=tag, cohort=cohort, kept=False, reason="cBioPortal 取突变失败")
                    )
                    print(f"  [{tag}] 取突变失败")
                    continue
                r = assemble(tag, cohort, pairs, feats)
                summary.append(
                    r if r else dict(tag=tag, cohort=cohort, kept=False, reason="样本量不足")
                )
                print(f"  [{tag}] {summary[-1]}")
    elif args.mode == "cont":
        targets = list(GENES_OF.keys()) + [c for c in local_cohorts() if c not in GENES_OF]
        if args.extra_cohorts:
            targets = list(dict.fromkeys(targets + args.extra_cohorts))
        seen = set()
        for cohort in targets:
            if cohort in seen:
                continue
            seen.add(cohort)
            feats = (
                load_cached_cohort(cohort) if cohort in GENES_OF else _load_hf_cohort_feats(cohort)
            )
            if feats is None:
                continue
            for attr in CONT_ATTRS:
                tag = f"WSI-{cohort}-{attr}"
                pairs, med = task_median_split(cohort, attr, feats)
                if pairs is None:
                    summary.append(
                        dict(tag=tag, cohort=cohort, kept=False, reason="取数失败或样本不足")
                    )
                    continue
                r = assemble(tag, cohort, pairs, feats)
                if r:
                    r["median"] = med
                summary.append(r or dict(tag=tag, cohort=cohort, kept=False, reason="样本量不足"))
                print(f"  [{tag}] {summary[-1]}")
    else:
        if not os.path.exists(args.tasks):
            sys.exit(f"缺少任务定义：{args.tasks}")
        spec = json.load(open(args.tasks, encoding="utf-8"))
        for cohort in args.cohort:
            key = cohort.replace("TCGA-", "")
            cfg = spec.get(key)
            if cfg is None:
                print(f"  [{cohort}] 任务定义里没有，跳过")
                continue
            avail = load_hf_cohort(cohort)
            if not avail:
                summary.append(dict(tag=cohort, kept=False, reason="HF 快照里没有该队列"))
                print(f"  [{cohort}] 未下载，跳过")
                continue
            print(f"  [{cohort}] 本地切片 {len(avail)} 例，开始池化…")
            feats = {}
    for pid, paths in avail.items():
        v = pool_h5(paths)
        if v is not None:
            feats[pid] = v
            for tname, rule in cfg.items():
                pairs = build_t2_pairs(rule, feats)
                tag = f"WSI-{key}-{tname}"
                r = assemble(tag, key, pairs, feats)
                summary.append(r if r else dict(tag=tag, kept=False, reason="无可用样本"))
                print(f"    [{tag}] {summary[-1]}")

    sp = os.path.join(HERE, "build_summary.json")
    json.dump(summary, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    kept = [s for s in summary if s.get("kept")]
    print(f"\n新增 {len(kept)} 个点 → {OUT}")
    print(f"汇总 → {sp}")


def build_t2_pairs(rule, feats):
    """Preserved computation; see docs/experiments.md for its protocol."""
    import urllib.request

    def cbio(study, attr):
        for dt in ["PATIENT", "SAMPLE"]:
            url = (
                f"https://www.cbioportal.org/api/studies/{study}/clinical-data"
                f"?clinicalDataType={dt}&attributeId={attr}&projection=SUMMARY"
            )
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                d = json.loads(urllib.request.urlopen(req, timeout=120).read())
                if d:
                    return {r["patientId"]: r["value"] for r in d}
            except Exception:
                continue
        return {}

    vals = cbio(rule["study"], rule["attr"])
    pos, neg = set(rule["pos"]), set(rule["neg"])
    out = []
    for pid, v in vals.items():
        if v in pos:
            out.append((pid, 1))
        elif v in neg:
            out.append((pid, 0))
    return out


if __name__ == "__main__":
    main()
