# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import json
import argparse
import urllib.request

import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = "W8Yi/tcga-wsi-uni2h-features"
N_PER_CLASS = 100


def list_feature_files():
    """Preserved computation; see docs/experiments.md for its protocol."""
    cache = os.path.join(OUT, "ucec_files.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    from huggingface_hub import HfApi

    api = HfApi(endpoint=os.environ["HF_ENDPOINT"])
    files = [
        f
        for f in api.list_repo_files(REPO, repo_type="dataset")
        if f.startswith("TCGA-UCEC/features/") and f.endswith(".h5")
    ]
    json.dump(files, open(cache, "w"))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--n", type=int, default=N_PER_CLASS)
    args = ap.parse_args()

    labels = pd.read_csv(os.path.join(OUT, "tcga_ucec_labels.csv"), dtype=str)
    files = list_feature_files()
    print(f"特征文件 {len(files)} 个，标签 {len(labels)} 例")

    def pid_of(fn):
        b = os.path.basename(fn)
        parts = b.split("-")
        return "-".join(parts[:3]) if len(parts) >= 3 else None

    file_by_pid = {}
    for f in files:
        pid = pid_of(f)
        if pid:
            file_by_pid.setdefault(pid, []).append(f)

    lab = labels.set_index("patient_id")["SUBTYPE"].to_dict()

    # MSI → dMMR；POLE/CN_LOW/CN_HIGH → pMMR
    def bin_of(st):
        if st is None:
            return None
        return 1 if st == "UCEC_MSI" else 0

    rows = []
    for pid, fl in file_by_pid.items():
        y = bin_of(lab.get(pid))
        if y is None:
            continue
        rows.append(
            {
                "patient_id": pid,
                "label": y,
                "subtype": lab.get(pid),
                "files": sorted(fl),
                "n_files": len(fl),
            }
        )
    df = pd.DataFrame(rows)
    print(f"\n可匹配病例: {len(df)}")
    print(df["label"].value_counts().to_string())
    print(f"亚型分布:\n{df['subtype'].value_counts().to_string()}")

    pick = []
    for y in [1, 0]:
        sub = df[df.label == y].sort_values("n_files")
        pick.append(sub.head(args.n))
    sel = pd.concat(pick)
    print(
        f"\n选中子集: {len(sel)} 例 (dMMR {int((sel.label == 1).sum())} / pMMR {int((sel.label == 0).sum())})"
    )
    sel.to_csv(os.path.join(OUT, "tcga_subset.csv"), index=False, encoding="utf-8-sig")
    print("已写 tcga_subset.csv")

    if args.download:
        from huggingface_hub import hf_hub_download
        import time

        cache = os.path.join(os.path.dirname(os.path.dirname(OUT)), "models")
        done = 0
        t0 = time.time()
        for _, r in sel.iterrows():
            for f in r["files"]:
                try:
                    hf_hub_download(REPO, f, repo_type="dataset", cache_dir=cache)
                    done += 1
                    if done % 10 == 0:
                        el = time.time() - t0
                        print(
                            f"  {done} 文件, {el / 60:.1f} 分钟, {done / el * 60:.1f} 文件/分",
                            flush=True,
                        )
                except Exception as e:
                    print(f"  ERR {f}: {str(e)[:80]}", flush=True)
        print(f"下载完成: {done} 文件, 用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
