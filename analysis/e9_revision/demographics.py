# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os, sys, json, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "e1_label_engine"))
from align_seq_key import YEAR_CONFIG

E1 = os.path.join(ROOT, "e1_label_engine", "outputs")

labels = pd.read_csv(os.path.join(E1, "labels_ec_patient.csv"), dtype=str)
index = pd.read_csv(os.path.join(E1, "image_index.csv"), dtype={"year": str})


main = labels[
    (labels["match_status"].isin(["exact", "shifted"])) & (labels["mmr"] != "unknown")
].copy()
n_p, n_d = int((main.mmr == "pMMR").sum()), int((main.mmr == "dMMR").sum())
print(f"主分析集: {len(main)} 例 (pMMR {n_p} / dMMR {n_d})")
assert (len(main), n_p, n_d) == (563, 433, 130), (
    f"主分析集口径不符，应为 563/433/130，实得 {len(main)}/{n_p}/{n_d}"
)

# (year, folder) -> report_row
key2row = {}
for _, r in index.iterrows():
    key2row[(str(r["year"]), str(r["folder"]))] = r


rows = []
for year, cfg in YEAR_CONFIG.items():
    p = cfg["xls"]
    if not os.path.exists(p):
        print(f"  [warn] {year} 报告表不存在: {p}")
        continue
    d = pd.read_excel(p)
    d.columns = [str(c).strip() for c in d.columns]
    for _, r in main[main["year"] == year].iterrows():
        k = (str(year), str(r["folder"]))
        if k not in key2row:
            rows.append({"year": year, "folder": r["folder"], "matched": False})
            continue
        rr = key2row[k]
        ri = int(rr["report_row"])
        if not (0 <= ri < len(d)):
            rows.append({"year": year, "folder": r["folder"], "matched": False})
            continue
        rec = d.iloc[ri]
        row = {
            "year": year,
            "folder": r["folder"],
            "matched": True,
            "age": rec.get("年龄"),
            "sex": rec.get("性别"),
            "diagnosis": str(rec.get("病理诊断") or ""),
            "mmr": r["mmr"],
            "grade": r["grade"],
        }

        for c in [
            "诊断及建议",
            "临床诊断",
            "病史摘要",
            "妇科信息",
            "手术所见",
            "检查所见",
            "大体描述",
        ]:
            row[c] = str(rec.get(c) or "")
        rows.append(row)

df = pd.DataFrame(rows)
ok = df[df["matched"] == True].copy()
print(f"回连成功: {len(ok)} / {len(df)}")


def num(x):
    try:
        v = float(x)
        return v if 0 < v < 120 else np.nan
    except (TypeError, ValueError):
        return np.nan


ok["age"] = ok["age"].map(num)
age = ok["age"].dropna()
print(f"\n年龄可获取 {len(age)} / {len(ok)}")
print(
    f"  mean {age.mean():.1f}  sd {age.std():.1f}  median {age.median():.0f}  "
    f"range {age.min():.0f}-{age.max():.0f}  IQR [{age.quantile(0.25):.0f}, {age.quantile(0.75):.0f}]"
)

sex = ok["sex"].astype(str).str.strip().value_counts().to_dict()
print(f"  性别: {sex}")


SUBTYPE = {
    "endometrioid": ["子宫内膜样", "内膜样腺癌", "腺癌"],
    "serous": ["浆液性", "浆液"],
    "clear_cell": ["透明细胞"],
    "mucinous": ["黏液性", "粘液性"],
    "mixed": ["混合"],
    "carcinosarcoma": ["癌肉瘤", "恶性中胚叶混合瘤"],
}


def subtype(txt):
    hit = []
    for name, keys in SUBTYPE.items():
        if any(k in txt for k in keys):
            hit.append(name)
    if not hit:
        return "unspecified"

    for pref in ["carcinosarcoma", "clear_cell", "serous", "mucinous", "mixed", "endometrioid"]:
        if pref in hit:
            return pref
    return hit[0]


ok["subtype"] = ok["diagnosis"].map(subtype)
sub = ok["subtype"].value_counts().to_dict()
print(f"\n组织学亚型（由病理诊断文本判定）: {sub}")


STAGE_RE = re.compile(r"(?:FIGO\s*)?([ⅠⅡⅢⅣIVX]+|[1-4])\s*[aAbBcC]?\s*期")
TEXT_COLS = [
    "病理诊断",
    "诊断及建议",
    "临床诊断",
    "病史摘要",
    "妇科信息",
    "手术所见",
    "检查所见",
    "大体描述",
]
stage_scan = {}
for col in TEXT_COLS:
    vals = ok[col].astype(str) if col in ok.columns else pd.Series([], dtype=str)
    hits = vals[vals.str.contains(STAGE_RE, na=False)]
    stage_scan[col] = int(len(hits))
n_stage = stage_scan.get("病理诊断", 0)
print("\n分期关键词逐列扫描（无该列记 0）:")
for c, n in stage_scan.items():
    print(f"  {c:<8} {n}")

print(f"病理诊断列明写分期: {n_stage} / {len(ok)} → 判定为不可用")


tx_cols = [c for c in ok.columns if any(k in c for k in ("治疗", "化疗", "放疗"))]
print(f"治疗相关列: {tx_cols if tx_cols else '无'}（病理报告表不记录治疗）")

out = {
    "n_main_analysis": int(len(main)),
    "n_reconnected": int(len(ok)),
    "age": {
        "n": int(len(age)),
        "mean": round(float(age.mean()), 1),
        "sd": round(float(age.std()), 1),
        "median": float(age.median()),
        "iqr": [float(age.quantile(0.25)), float(age.quantile(0.75))],
        "range": [float(age.min()), float(age.max())],
        "by_year": {
            y: {
                "n": int(g["age"].notna().sum()),
                "mean": round(float(g["age"].mean()), 1),
                "median": float(g["age"].median()),
            }
            for y, g in ok.groupby("year")
            if g["age"].notna().any()
        },
    },
    "sex": sex,
    "subtype_from_diagnosis_text": sub,
    "stage_written_in_text": {
        "n": int(n_stage),
        "pct": round(100 * n_stage / max(len(ok), 1), 1),
        "per_column_scan": stage_scan,
        "verdict": "FIGO stage is not recorded in these report tables; not reportable",
    },
    "treatment_information": "no treatment-related column exists in the report tables",
    "mmr_by_grade": pd.crosstab(ok["grade"], ok["mmr"]).to_dict()
    if ok["grade"].notna().any()
    else {},
    "notes": (
        "年龄/性别为报告表直接字段（非抽取），回连键为 (year, report_row)，report_row 由 align_seq_key 落地。"
        "组织学亚型由病理诊断自由文本关键词判定，非病理医生复核，仅作队列描述。"
        "分期对全部文本列逐一扫描确认缺失（非只查一列），未做任何推断。"
        "治疗信息：报告表中无任何含'治疗/化疗/放疗'的列——病理报告本就不记录治疗。"
    ),
}

with open(os.path.join(HERE, "demographics_result.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n已写 e9_revision/demographics_result.json")
