# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import json
import argparse
import urllib.request

import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = "W8Yi/tcga-wsi-uni2h-features"


CONFIGS = {
    "COAD": ("TCGA-COAD", "coadread_tcga_pan_can_atlas_2018", "SUBTYPE", "MSI"),
    "STAD": ("TCGA-STAD", "stad_tcga_pan_can_atlas_2018", "SUBTYPE", "MSI"),
    "BRCA_IDC": ("TCGA-BRCA_IDC", "brca_tcga", "ER_STATUS_BY_IHC", "Positive"),
    "BRCA_IDC_PR": ("TCGA-BRCA_IDC", "brca_tcga", "PR_STATUS_BY_IHC", "Positive"),
}
N_PER_CLASS = 150


def cbioportal(study, attr, dtype="PATIENT"):
    url = (
        f"https://www.cbioportal.org/api/studies/{study}/clinical-data"
        f"?clinicalDataType={dtype}&attributeId={attr}&projection=SUMMARY"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def hf_files(cancer_dir):
    """Preserved computation; see docs/experiments.md for its protocol."""
    cache = os.path.join(OUT, f"files_{cancer_dir}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    import subprocess

    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{cancer_dir}/features"
    r = subprocess.run(["curl", "-s", "--max-time", "120", url], capture_output=True, text=True)
    d = json.loads(r.stdout)
    files = [x["path"] for x in d if x["type"] == "file"]
    json.dump(files, open(cache, "w"))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--n", type=int, default=N_PER_CLASS)
    args = ap.parse_args()

    plan_all = []
    for key, (cdir, study, attr, posval) in CONFIGS.items():
        print(f"\n=== {key} ({study}.{attr}) ===")
        try:
            recs = cbioportal(study, attr)
        except Exception as e:
            print(f"  标签拉取失败: {str(e)[:80]}")
            continue
        labels = {r["patientId"]: r["value"] for r in recs}
        files = hf_files(cdir)
        print(f"  标签 {len(labels)} 例，特征文件 {len(files)}")

        def pid_of(fn):
            b = os.path.basename(fn)
            return "-".join(b.split("-")[:3]) if len(b.split("-")) >= 3 else None

        by_pid = {}
        for f in files:
            p = pid_of(f)
            if p:
                by_pid.setdefault(p, []).append(f)

        rows = []
        for pid, fl in by_pid.items():
            v = labels.get(pid)
            if v is None:
                continue

            if posval in ("Positive", "Negative"):
                if v == posval:
                    y = 1
                elif v == "Negative":
                    y = 0
                else:
                    continue
            else:
                vs = str(v)
                if "Indeterminate" in vs:
                    continue
                y = 1 if posval in vs else 0
            rows.append({"patient_id": pid, "label": y, "raw": v, "files": sorted(fl)})
        df = pd.DataFrame(rows)
        if df.empty:
            print("  无可用匹配")
            continue
        n_pos = int((df.label == 1).sum())
        n_neg = int((df.label == 0).sum())
        print(f"  可匹配: 阳性 {n_pos} / 阴性 {n_neg}")

        n_take = min(args.n, n_pos, n_neg)
        pick = []
        for y in [1, 0]:
            sub = df[df.label == y].sort_values("files", key=lambda s: s.map(len))
            pick.append(sub.head(n_take))
        sel = pd.concat(pick).assign(cancer=key)
        print(
            f"  选中 {len(sel)} 例 (阳性 {int((sel.label == 1).sum())} / 阴性 {int((sel.label == 0).sum())})"
        )
        plan_all.append(sel)

    if plan_all:
        allsel = pd.concat(plan_all)
        allsel.to_csv(os.path.join(OUT, "multi_subset.csv"), index=False, encoding="utf-8-sig")
        print(f"\n合计选中 {len(allsel)} 例 → multi_subset.csv")

    if args.download and plan_all:
        import time
        from huggingface_hub import hf_hub_download

        cache_dir = os.environ.get(
            "PATHOLOGY_HF_CACHE", os.path.join(os.path.dirname(__file__), "..", "models")
        )

        uniq = sorted({f for _, r in allsel.iterrows() for f in r["files"]})
        print(f"去重后需下载 {len(uniq)} 个文件")
        done = 0
        t0 = time.time()
        for f in uniq:
            try:
                hf_hub_download(REPO, f, repo_type="dataset", cache_dir=cache_dir)
                done += 1
                if done % 20 == 0:
                    el = (time.time() - t0) / 60
                    print(
                        f"  {done}/{len(uniq)} 文件, {el:.1f} 分钟, {done / el:.1f} 文件/分",
                        flush=True,
                    )
            except Exception as e:
                print(f"  ERR {f}: {str(e)[:70]}", flush=True)
        print(f"下载完成: {done} 文件, 用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
