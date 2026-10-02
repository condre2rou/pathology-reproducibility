# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import json
import urllib.request

OUT = os.path.dirname(os.path.abspath(__file__))
API = "https://api.gdc.cancer.gov"


def gdc_post(endpoint, payload):
    req = urllib.request.Request(
        f"{API}/{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def main():

    fields = [
        "submitter_id",
        "demographic.gender",
        "demographic.race",
        "diagnoses.age_at_diagnosis",
        "diagnoses.tumor_grade",
        "diagnoses.morphology",
        "diagnoses.primary_diagnosis",
        "diagnoses.ajcc_pathologic_stage",
    ]

    extra = [
        "diagnoses.microsatellite_instability",
        "diagnoses.msi_status",
        "diagnoses.mismatch_repair_deficiency",
    ]

    payload = {
        "filters": {"op": "in", "content": {"field": "project.project_id", "value": ["TCGA-UCEC"]}},
        "fields": ",".join(fields),
        "format": "JSON",
        "size": 1000,
    }
    r = gdc_post("cases", payload)
    hits = r["data"]["hits"]
    print(f"TCGA-UCEC 病例数: {len(hits)} (total {r['data']['pagination']['total']})")

    import pandas as pd

    rows = []
    for h in hits:
        d = h.get("diagnoses", [{}])
        d = d[0] if d else {}
        demo = h.get("demographic", {}) or {}
        rows.append(
            {
                "submitter_id": h["submitter_id"],
                "case_id": h["id"],
                "gender": demo.get("gender"),
                "age_at_diagnosis": d.get("age_at_diagnosis"),
                "tumor_grade": d.get("tumor_grade"),
                "morphology": d.get("morphology"),
                "primary_diagnosis": d.get("primary_diagnosis"),
                "stage": d.get("ajcc_pathologic_stage"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "tcga_ucec_clinical.csv"), index=False, encoding="utf-8-sig")
    print("已写 tcga_ucec_clinical.csv")
    print(df.head(3).to_string())
    print("\n可用列:", df.columns.tolist())

    print("\n检查临床中是否有 MSI/MMR 信息...")
    for k in extra:
        try:
            p2 = dict(payload)
            p2["fields"] = k
            r2 = gdc_post("cases", p2)
            nonnull = sum(
                1 for h in r2["data"]["hits"] if h.get("diagnoses", [{}])[0].get(k.split(".")[-1])
            )
            print(f"  {k}: 非空 {nonnull}")
        except Exception as e:
            print(f"  {k}: 不可用 ({str(e)[:60]})")


if __name__ == "__main__":
    main()
