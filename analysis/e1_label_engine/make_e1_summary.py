# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import re
import json
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "outputs")


def num(s):
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else None


def main():
    align = json.load(open(os.path.join(OUT, "alignment_summary.json")))
    noise = json.load(open(os.path.join(OUT, "noise_prior_summary.json")))
    gold = json.load(open(os.path.join(OUT, "e1_gold_metrics.json")))
    labels = pd.read_csv(os.path.join(OUT, "labels_ec_patient.csv"), dtype=str)
    llm = pd.read_csv(os.path.join(OUT, "llm_extraction.csv"), dtype=str)

    def clean(s):
        s = str(s).strip().lower()
        return s if s in ["pmmr", "dmmr", "wt", "abn", "pos", "neg"] else "unknown"

    llm_agree = {}
    for f in ["mmr", "p53", "er", "pr"]:
        a = llm[f + "_rule"].str.strip().str.lower()
        b = llm[f + "_llm"].map(clean)
        both = (a != "unknown") & (b != "unknown")
        llm_agree[f] = {
            "agree_rate": round((a[both] == b[both]).mean(), 3),
            "both_known": int(both.sum()),
            "llm_unknown": int((b == "unknown").sum()),
        }
    k_r, k_l = llm["ki67_rule"].map(num), llm["ki67_llm"].map(num)
    both = k_r.notna() & k_l.notna()
    llm_agree["ki67"] = {
        "agree_rate": round((k_r[both] == k_l[both]).mean(), 3),
        "both_known": int(both.sum()),
    }

    summary = {
        "version": "v1.0",
        "alignment": align,
        "labels": {
            "total_patients": len(labels),
            "mmr_distribution": labels["mmr"].value_counts().to_dict(),
            "main_analysis": {
                "n": noise["mmr_main_analysis"],
                "pMMR": noise["mmr_main_pMMR"],
                "dMMR": noise["mmr_main_dMMR"],
            },
            "p53_distribution": labels["p53"].value_counts().to_dict(),
        },
        "gold_evaluation_rule_table": gold,
        "llm_baseline": {
            "n_sample": len(llm),
            "agree_rule_vs_llm": llm_agree,
            "note": "MMR 主终点 100% 一致；p53 LLM 更保守(62 unknown)；ki67 已按数值归一",
        },
        "noise_prior": noise,
        "pending": "病理医生金标验证（TODO 1.7，暂缓）",
    }
    with open(os.path.join(OUT, "e1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("已写 outputs/e1_summary.json")
    print(f"  对齐精确率: {align['2021']['exact'] / align['2021']['folders']:.1%} (2021)")
    print(f"  MMR 主分析集: {noise['mmr_main_analysis']} 例")
    print(f"  LLM 一致性(MMR): {llm_agree['mmr']['agree_rate']:.1%}")


if __name__ == "__main__":
    main()
