# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import re
import pandas as pd

from align_seq_key import load_report, YEAR_CONFIG, initials_options

DATA = os.environ.get(
    "PATHOLOGY_PRIVATE_IMAGES", os.path.join(os.path.dirname(__file__), "..", "private_images")
)
IDX = os.path.join(os.path.dirname(__file__), "outputs", "image_index.csv")


AB_MMR = {
    "MLH1": r"MLH[\s\-]?1",
    "PMS2": r"PMS[\s\-]?2",
    "MSH2": r"MSH[\s\-]?2",
    "MSH6": r"MSH[\s\-]?6",
}
AB_P53 = r"(?:P53|P\-?53|TP53)"
AB_ER = r"(?:(?<!H)ER|雌激素受体)"
AB_PR = r"(?:(?<!V)PR|孕激素受体)"
AB_KI67 = r"(?:[Kk][Ii][\s\-]?67|Ki67|KI67)"

KI67_THRESHOLD = 30.0


def normalize(text: str) -> str:
    """Preserved computation; see docs/experiments.md for its protocol."""
    if not isinstance(text, str):
        return ""
    t = text

    t = t.translate(str.maketrans("（）＋－％：，；、＝＜＞", "()+-%:,;,=<>"))

    t = t.replace("_x000D_", "").replace("\r", "").replace("\n", "").replace("\t", "")
    t = re.sub(r"\s+", "", t)
    return t


def full_text(rec) -> str:
    """Preserved computation; see docs/experiments.md for its protocol."""
    parts = []
    for c in [
        "病理诊断",
        "诊断及建议",
        "免疫组化",
        "免疫组化结果",
        "免疫组化临床诊断",
        "免疫组化病理诊断",
    ]:
        v = rec.get(c, "")
        if v:
            parts.append(str(v))
    return normalize("。".join(parts))


# ---------------- MMR ----------------
def _antibody_status(text, ab_pat):
    """Preserved computation; see docs/experiments.md for its protocol."""
    m = re.search(ab_pat + r"\s*[（(]?\s*([^）)\s;；。]{0,12})", text)
    if not m:
        return "unknown", ""
    token = m.group(1)
    token = token.strip("（）() ,，、。")
    evidence = m.group(0)[:30]
    if re.search(r"缺失|不表达|阴性|表达缺失|无表达", token) or ("-" in token and "+" not in token):
        return "lost", evidence
    if "灶" in token or "部分" in token or "斑驳" in token or "弱" in token:
        return "partial", evidence
    if "+" in token:
        return "intact", evidence
    if "-" in token:
        return "lost", evidence
    return "unknown", evidence


def extract_mmr(text):
    statuses, ev = {}, {}
    for ab, pat in AB_MMR.items():
        s, e = _antibody_status(text, pat)
        statuses[ab] = s
        ev[ab] = e
    n_intact = sum(v == "intact" for v in statuses.values())
    n_lost = sum(v == "lost" for v in statuses.values())
    n_partial = sum(v == "partial" for v in statuses.values())
    n_unknown = sum(v == "unknown" for v in statuses.values())

    if n_lost > 0:
        label = "dMMR"
    elif n_intact + n_partial == 4 and n_unknown == 0:
        label = "pMMR"
    else:
        label = "unknown"

    lowconf = n_partial > 0
    conf = "low" if lowconf else "high"
    return (
        label,
        statuses,
        ev,
        f"intact={n_intact},lost={n_lost},partial={n_partial},unknown={n_unknown},conf={conf}",
    )


# ---------------- p53 ----------------
def extract_p53(text):
    m = re.search(AB_P53 + r"[（(]?\s*([^）)\s;；。]{0,24})", text)
    if not m:
        return "unknown", ""
    token = m.group(1).strip("（）() ,，、。")
    ev = m.group(0)[:40]

    if token in ("-", "－"):
        return "abn", ev

    if re.search(
        r"野生|斑驳|无突变|未突变|弱\+|弱-中\+|散在|少量|部分\+|少许|强弱不等|强弱不一|异质性",
        token,
    ):
        return "wt", ev

    if re.search(
        r"(?<!无)(?<!未)突变|(?<!部分)弥漫|全阴|完全阴性|无义|9[0-9]%\+|8[0-9]%[，,]?强\+", token
    ):
        return "abn", ev
    return "unknown", ev


# ---------------- ER / PR ----------------
def extract_hormone(text, ab_pat):
    m = re.search(ab_pat + r"[（(]?\s*([^）)\s;；。]{0,16})", text)
    if not m:
        return "unknown", ""
    token = m.group(1).strip("（）() ,，、。")
    ev = m.group(0)[:40]

    if re.search(r"^[-－]|0\s*%|阴性", token) and "+" not in token:
        return "neg", ev

    pct = re.search(r"(\d+(?:\.\d+)?)\s*%", token)
    if "+" in token or (pct and float(pct.group(1)) >= 1):
        return "pos", ev
    if re.search(r"^[-－]|0\s*%|阴性", token):
        return "neg", ev
    return "unknown", ev


# ---------------- Ki67 ----------------
def extract_ki67(text):
    m = re.search(AB_KI67 + r"[^0-9%]{0,12}(?:约|阳性率|指数)?\s*(\d+(?:\.\d+)?)\s*%", text)
    if not m:
        return "unknown", ""
    val = float(m.group(1))
    ev = m.group(0)[:40]
    level = "high" if val >= KI67_THRESHOLD else "low"
    return f"{val}", ev + f"|{level}"


# ---------------- grade ----------------
def extract_grade(text):
    m = re.search(r"(高|中|低)[\-\s]?分化", text)
    if m:
        return {"高": "1", "中": "2", "低": "3"}[m.group(1)], m.group(0)
    m = re.search(r"\bG([123])\b", text)
    if m:
        return m.group(1), m.group(0)
    return "unknown", ""


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
            mmr, mmr_s, mmr_ev, mmr_note = extract_mmr(txt)
            p53, p53_ev = extract_p53(txt)
            er, er_ev = extract_hormone(txt, AB_ER)
            pr, pr_ev = extract_hormone(txt, AB_PR)
            ki67, ki67_ev = extract_ki67(txt)
            grade, grade_ev = extract_grade(txt)
            rows.append(
                {
                    "year": year,
                    "folder": r["folder"],
                    "name": r["name"],
                    "name_seq": r["name_seq"],
                    "match_status": r["match_status"],
                    "mmr": mmr,
                    "mmr_evidence": mmr_note
                    + " | "
                    + " ".join(f"{k}={v}" for k, v in mmr_s.items()),
                    "p53": p53,
                    "p53_evidence": p53_ev,
                    "er": er,
                    "er_evidence": er_ev,
                    "pr": pr,
                    "pr_evidence": pr_ev,
                    "ki67": ki67,
                    "ki67_evidence": ki67_ev,
                    "grade": grade,
                    "grade_evidence": grade_ev,
                }
            )

    out = pd.DataFrame(rows)
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    out.to_csv(os.path.join(out_dir, "labels_ec.csv"), index=False, encoding="utf-8-sig")

    print(f"患者级标签 {len(out)} 行")
    for c in ["mmr", "p53", "er", "pr", "grade"]:
        print(f"\n[{c}]")
        print(out[c].value_counts(dropna=False).to_string())
    ki = out[out["ki67"].str.match(r"^\d", na=False)]
    if len(ki):
        vals = ki["ki67"].astype(float)
        print(
            f"\n[ki67] 抽取到 {len(ki)} 例，median={vals.median():.0f}%  high(>=30%)={(vals >= 30).sum()}"
        )


if __name__ == "__main__":
    main()
