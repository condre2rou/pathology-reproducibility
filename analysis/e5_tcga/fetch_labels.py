# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import json
import urllib.request

import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
STUDY = "ucec_tcga_pan_can_atlas_2018"
ATTRS = ["SUBTYPE", "MSI_SENSOR_SCORE", "MSI_SCORE_MANTIS"]


def fetch(attr):
    url = f"https://www.cbioportal.org/api/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT&attributeId={attr}&projection=SUMMARY"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def main():
    data = {}
    for a in ATTRS:
        try:
            recs = fetch(a)
            for r in recs:
                data.setdefault(r["patientId"], {})[a] = r["value"]
            print(f"{a}: {len(recs)} 条")
        except Exception as e:
            print(f"{a}: ERR {str(e)[:100]}")

    df = pd.DataFrame([{"patient_id": k, **v} for k, v in data.items()])
    print(f"\n总病例 {len(df)}")
    if "SUBTYPE" in df.columns:
        print("SUBTYPE 分布:")
        print(df["SUBTYPE"].value_counts(dropna=False).to_string())

    df.to_csv(os.path.join(OUT, "tcga_ucec_labels.csv"), index=False, encoding="utf-8-sig")
    print("\n已写 tcga_ucec_labels.csv")


if __name__ == "__main__":
    main()
