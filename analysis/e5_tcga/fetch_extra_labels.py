# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import json
import urllib.request

import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
STUDY = "ucec_tcga_pan_can_atlas_2018"
PROFILE = f"{STUDY}_mutations"


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


def post(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


def main():
    res = {}

    print("拉取 TP53 突变...")
    try:
        sample_lists = get(f"https://www.cbioportal.org/api/studies/{STUDY}/sample-lists")
        all_list = [s["sampleListId"] for s in sample_lists if s["sampleListId"].endswith("_all")]
        sl = all_list[0] if all_list else f"{STUDY}_all"
        print(f"  sampleListId = {sl}")
        payload = {
            "entrezGeneIds": [7157],  # TP53
            "sampleListId": sl,
        }
        muts = post(f"https://www.cbioportal.org/api/mutations/fetch?projection=SUMMARY", payload)
        hit = {}
        for m in muts:
            pid = m.get("patientId")
            sid = m.get("sampleId")
            if pid:
                hit[pid] = 1
        print(f"  有 TP53 突变的患者 {len(hit)}")
        res["tp53"] = hit
    except Exception as e:
        print(f"  TP53 拉取失败: {str(e)[:150]}")

    print("拉取组织学亚型...")
    hist = {}
    for attr in ["ONCOTREE_CODE", "ICD_O_3_HISTOLOGY", "CANCER_TYPE_DETAILED"]:
        for dt in ["SAMPLE", "PATIENT"]:
            try:
                d = get(
                    f"https://www.cbioportal.org/api/studies/{STUDY}/clinical-data"
                    f"?clinicalDataType={dt}&attributeId={attr}&projection=SUMMARY"
                )
                if d:
                    for r in d:
                        hist.setdefault(r["patientId"], {})[attr] = r["value"]
                    print(f"  {attr}/{dt}: {len(d)} 条")
                    break
            except Exception:
                continue
    res["histology"] = hist

    pids = set(res.get("tp53", {})) | set(hist)
    rows = []
    for p in sorted(pids):
        rows.append(
            {
                "patient_id": p,
                "tp53_mut": res.get("tp53", {}).get(p, 0),
                "oncotree": hist.get(p, {}).get("ONCOTREE_CODE", ""),
                "icd_histology": hist.get(p, {}).get("ICD_O_3_HISTOLOGY", ""),
                "cancer_detailed": hist.get(p, {}).get("CANCER_TYPE_DETAILED", ""),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "tcga_extra_labels.csv"), index=False, encoding="utf-8-sig")
    print(f"\n合计 {len(df)} 例 → tcga_extra_labels.csv")
    print(f"  TP53 突变阳性: {int(df.tp53_mut.sum())}")
    if df.oncotree.astype(bool).any():
        print(f"  ONCOTREE 分布: {df[df.oncotree.astype(bool)].oncotree.value_counts().to_dict()}")
    if df.icd_histology.astype(bool).any():
        print(
            f"  ICD 组织学分布: {df[df.icd_histology.astype(bool)].icd_histology.value_counts().head(5).to_dict()}"
        )


if __name__ == "__main__":
    main()
