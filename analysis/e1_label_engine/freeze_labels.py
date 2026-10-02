# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import json
import pandas as pd

from align_seq_key import load_report, YEAR_CONFIG
from extract_rules import (
    full_text,
    extract_mmr,
    extract_p53,
    extract_hormone,
    extract_ki67,
    extract_grade,
    AB_ER,
    AB_PR,
)

OUT = os.path.join(os.path.dirname(__file__), "outputs")
IDX = os.path.join(OUT, "image_index.csv")


W = {
    "base": 0.03,
    "discordant": 0.20,
    "partial": 0.12,
    "incomplete": 0.05,
    "shifted": 0.10,
    "manual_review": 0.30,
}


def noise_prior(mmr_statuses, match_status):
    """Preserved computation; see docs/experiments.md for its protocol."""
    eps = W["base"]
    flags = []
    lost = [k for k, v in mmr_statuses.items() if v == "lost"]
    partial = [k for k, v in mmr_statuses.items() if v == "partial"]
    found = sum(1 for v in mmr_statuses.values() if v != "unknown")
    n_total = len(mmr_statuses)

    if len(lost) == 1:
        eps += W["discordant"]
        flags.append(f"孤立{lost[0]}缺失")
    if partial:
        eps += W["partial"]
        flags.append("弱/灶表达")
    if found < n_total:
        eps += W["incomplete"]
        flags.append("抗体不全")
    if match_status == "manual_review":
        eps += W["manual_review"]
        flags.append("对齐待复核")
    elif match_status == "shifted":
        eps += W["shifted"]
        flags.append("对齐差一")

    eps = min(eps, 0.95)
    return round(eps, 3), round(1 - eps, 3), "|".join(flags)


def main():
    idx = pd.read_csv(IDX, dtype=str)
    rows = []
    for year, cfg in YEAR_CONFIG.items():
        recs, _ = load_report(cfg)
        sub = idx[idx["year"] == year]
        for _, r in sub.iterrows():
            seq = int(r["name_seq"]) if r["name_seq"] else None
            rec = recs.get(seq)
            if rec is None:
                continue
            txt = full_text(rec)
            mmr, statuses, _, _ = extract_mmr(txt)
            p53, _ = extract_p53(txt)
            er, _ = extract_hormone(txt, AB_ER)
            pr, _ = extract_hormone(txt, AB_PR)
            ki67, _ = extract_ki67(txt)
            grade, _ = extract_grade(txt)
            eps, conf, flags = noise_prior(statuses, r["match_status"])
            rows.append(
                {
                    "year": year,
                    "folder": r["folder"],
                    "name_seq": r["name_seq"],
                    "match_status": r["match_status"],
                    "mmr": mmr,
                    "mmr_conf": "low" if "弱/灶表达" in flags else "high",
                    "mmr_statuses": json.dumps(statuses, ensure_ascii=False),
                    "noise_eps": eps,
                    "confidence": conf,
                    "noise_flags": flags,
                    "p53": p53,
                    "er": er,
                    "pr": pr,
                    "ki67": ki67,
                    "grade": grade,
                    "image_paths": r["image_paths"],
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "labels_ec_patient.csv"), index=False, encoding="utf-8-sig")

    main = out[(out["match_status"].isin(["exact", "shifted"])) & (out["mmr"] != "unknown")]
    summary = {
        "total_patients": len(out),
        "mmr_distribution": out["mmr"].value_counts().to_dict(),
        "mmr_main_analysis": len(main),
        "mmr_main_pMMR": int((main["mmr"] == "pMMR").sum()),
        "mmr_main_dMMR": int((main["mmr"] == "dMMR").sum()),
        "discordant_isolated_loss": int(out["noise_flags"].str.contains("孤立", na=False).sum()),
        "weak_focal": int(out["noise_flags"].str.contains("弱/灶", na=False).sum()),
        "manual_review_alignment": int(
            out["noise_flags"].str.contains("对齐待复核", na=False).sum()
        ),
        "noise_eps_distribution": {
            "low(<0.1)": int((out["noise_eps"] < 0.1).sum()),
            "medium(0.1-0.3)": int(((out["noise_eps"] >= 0.1) & (out["noise_eps"] < 0.3)).sum()),
            "high(>=0.3)": int((out["noise_eps"] >= 0.3).sum()),
        },
    }
    with open(os.path.join(OUT, "noise_prior_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"冻结患者级主表 {len(out)} 行 → outputs/labels_ec_patient.csv")
    print(f"\nMMR 分布: {summary['mmr_distribution']}")
    print(
        f"MMR 主分析集（对齐可信+有结论）: {summary['mmr_main_analysis']} 例 "
        f"(pMMR {summary['mmr_main_pMMR']} / dMMR {summary['mmr_main_dMMR']})"
    )
    print(f"\n噪声先验分布:")
    for k, v in summary["noise_eps_distribution"].items():
        print(f"  {k}: {v}")
    print(
        f"不协调(孤立缺失) {summary['discordant_isolated_loss']} | 弱/灶表达 {summary['weak_focal']} | 对齐待复核 {summary['manual_review_alignment']}"
    )


if __name__ == "__main__":
    main()
