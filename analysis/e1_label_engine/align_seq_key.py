# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import json
import glob
from itertools import product

import pandas as pd

try:
    from pypinyin import pinyin as pypinyin_full, Style
except ImportError:
    raise SystemExit("缺少 pypinyin：pip install pypinyin")

DATA = os.environ.get(
    "PATHOLOGY_PRIVATE_IMAGES", os.path.join(os.path.dirname(__file__), "..", "private_images")
)

YEAR_CONFIG = {
    "2021": {
        "xls": f"{DATA}/子宫内膜癌/2021子宫内膜癌(含分子分型)/2021子宫.xls",
        "img_root": f"{DATA}/子宫内膜癌/2021子宫内膜癌(含分子分型)",
        "seq_mode": "column",
    },
    "2022": {
        "xls": f"{DATA}/子宫内膜癌/2022子宫内膜癌数据（含分子分型）/2022子宫.xls",
        "img_root": f"{DATA}/子宫内膜癌/2022子宫内膜癌数据（含分子分型）",
        "seq_mode": "column",
    },
    "2023": {
        "xls": f"{DATA}/子宫内膜癌/2023/2023.xls",
        "img_root": f"{DATA}/子宫内膜癌/2023",
        "seq_mode": "row+2",
    },
    "2024": {
        "xls": f"{DATA}/子宫内膜癌/2024/2024子宫内膜癌（病理图片）/2024子宫.xls",
        "img_root": f"{DATA}/子宫内膜癌/2024/2024子宫内膜癌（病理图片）",
        "seq_mode": "column",
    },
}

TEXT_COLS = [
    "病理诊断",
    "诊断及建议",
    "免疫组化",
    "免疫组化结果",
    "免疫组化临床诊断",
    "免疫组化病理诊断",
]


MANUAL_READINGS = {"逄": "pf"}


def initials_options(name: str):
    """Preserved computation; see docs/experiments.md for its protocol."""
    name = str(name).strip()
    py_lists = pypinyin_full(name, heteronym=True, style=Style.NORMAL)
    first_letters = []
    for i, plist in enumerate(py_lists):
        letters = {p[0] for p in plist if p}

        ch = name[i] if i < len(name) else ""
        letters |= set(MANUAL_READINGS.get(ch, ""))
        first_letters.append(letters if letters else {""})
    return {"".join(c) for c in product(*first_letters)}


def load_report(cfg):
    """Preserved computation; see docs/experiments.md for its protocol."""
    df = pd.read_excel(cfg["xls"], engine="xlrd")
    recs = {}
    by_row = {}
    for i, row in df.iterrows():
        seq = int(row["序号"]) if cfg["seq_mode"] == "column" else (i + 2)
        rec = {
            "seq": seq,
            "name": str(row.get("姓名", "")).strip(),
            "report_row": i,
            "检查号": str(row.get("检查号", "")).strip(),
            "病理号": str(row.get("病理号", "")).strip(),
            "检查日期": str(row.get("检查日期", "")).strip(),
        }
        for c in TEXT_COLS:
            v = row.get(c)
            rec[c] = "" if pd.isna(v) else str(v)
        recs[seq] = rec
        by_row[i] = rec
    return recs, by_row


def scan_folders(root):
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        sp = d.split("_")
        if not sp[0].isdigit():
            continue
        seq = int(sp[0])
        py = "".join(sp[1:])
        jpgs = sorted(glob.glob(os.path.join(p, "*.jpg")) + glob.glob(os.path.join(p, "*.JPG")))
        out.append({"seq": seq, "py": py, "folder": d, "n_images": len(jpgs), "paths": jpgs})
    return out


def resolve(folder, recs, by_row):
    """Preserved computation; see docs/experiments.md for its protocol."""
    seq, py = folder["seq"], folder["py"]
    rec = recs.get(seq)

    if rec is None:
        cand = [r for r in by_row.values() if py and py in initials_options(r["name"])]
        if len(cand) == 1:
            return cand[0], "manual_review", "seq_missing_pinyin_unique"
        return None, "manual_review", "seq_missing"

    if not py:
        return rec, "manual_review", "no_pinyin"

    if py in initials_options(rec["name"]):
        return rec, "exact", ""

    row = rec["report_row"]
    for dr in range(-3, 4):
        r = by_row.get(row + dr)
        if r and py in initials_options(r["name"]):
            return r, "shifted", f"seq_offset_{dr}"

    return rec, "manual_review", "pinyin_unexplained"


def main():
    all_rows = []
    summary = {}
    for year, cfg in YEAR_CONFIG.items():
        recs, by_row = load_report(cfg)
        folders = scan_folders(cfg["img_root"])
        stat = {"folders": len(folders), "report_rows": len(recs)}
        for f in folders:
            rec, status, note = resolve(f, recs, by_row)
            stat[status] = stat.get(status, 0) + 1
            all_rows.append(
                {
                    "year": year,
                    "folder_seq": f["seq"],
                    "folder_pinyin": f["py"],
                    "folder": f["folder"],
                    "name": rec["name"] if rec else "",
                    "name_seq": rec["seq"] if rec else "",
                    "match_status": status,
                    "note": note,
                    "n_images": f["n_images"],
                    "image_paths": "|".join(f["paths"]),
                    "report_row": rec["report_row"] if rec else "",
                    "检查号": rec.get("检查号", "") if rec else "",
                    "病理号": rec.get("病理号", "") if rec else "",
                }
            )
        summary[year] = stat
        print(
            f"[{year}] 报告 {len(recs)} | 目录 {len(folders)} | "
            f"exact {stat.get('exact', 0)} | shifted {stat.get('shifted', 0)} | "
            f"需复核 {stat.get('manual_review', 0)}"
        )

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    idx = pd.DataFrame(all_rows)
    idx.to_csv(os.path.join(out_dir, "image_index.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "alignment_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\n已写 outputs/image_index.csv（{len(idx)} 行）")
    bad = idx[idx["match_status"] == "manual_review"]
    if len(bad):
        print(f"\n需人工复核 {len(bad)} 条：")
        for _, r in bad.iterrows():
            print(
                f"  {r['year']} {r['folder']} -> 按序号={r['name']}({r['name_seq']}) 状态={r['note']}"
            )


if __name__ == "__main__":
    main()
