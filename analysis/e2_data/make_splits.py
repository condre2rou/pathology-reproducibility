# -*- coding: utf-8 -*-
"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import json
import random
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # snap-cpath
LABELS = os.path.join(ROOT, "e1_label_engine", "outputs", "labels_ec_patient.csv")
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    df = pd.read_csv(LABELS, dtype=str)

    main = df[(df["match_status"].isin(["exact", "shifted"])) & (df["mmr"] != "unknown")].copy()
    main["patient_id"] = main["year"] + "_" + main["folder"]
    main["label_bin"] = (main["mmr"] == "dMMR").astype(int)  # dMMR=1

    temporal = {
        "train": [],
        "test_2023": main[main["year"] == "2023"]["patient_id"].tolist(),
        "test_2024": main[main["year"] == "2024"]["patient_id"].tolist(),
        "train_years": ["2021", "2022"],
    }
    for year in ["2021", "2022"]:
        temporal["train"] += main[main["year"] == year]["patient_id"].tolist()

    pos = main[main["label_bin"] == 1]["patient_id"].tolist()
    neg = main[main["label_bin"] == 0]["patient_id"].tolist()
    random.Random(42).shuffle(pos)
    random.Random(42).shuffle(neg)
    folds = {f"fold_{i}": {"pos": [], "neg": []} for i in range(5)}
    for i, p in enumerate(pos):
        folds[f"fold_{i % 5}"]["pos"].append(p)
    for i, n in enumerate(neg):
        folds[f"fold_{i % 5}"]["neg"].append(n)

    cv = {}
    for i in range(5):
        test = folds[f"fold_{i}"]["pos"] + folds[f"fold_{i}"]["neg"]
        train = [
            p
            for j in range(5)
            if j != i
            for p in folds[f"fold_{j}"]["pos"] + folds[f"fold_{j}"]["neg"]
        ]
        cv[f"fold_{i}"] = {"train": train, "test": test}

    splits = {
        "temporal": temporal,
        "cv_5fold": cv,
        "n_main_analysis": len(main),
        "n_pMMR": int((main["label_bin"] == 0).sum()),
        "n_dMMR": int((main["label_bin"] == 1).sum()),
    }
    with open(os.path.join(OUT, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    cols = [
        "patient_id",
        "year",
        "folder",
        "mmr",
        "label_bin",
        "confidence",
        "noise_eps",
        "noise_flags",
        "p53",
        "er",
        "pr",
        "ki67",
        "grade",
        "image_paths",
    ]
    main[cols].to_csv(os.path.join(OUT, "manifest.csv"), index=False, encoding="utf-8-sig")

    print(f"主分析集 {len(main)} 例（pMMR {splits['n_pMMR']} / dMMR {splits['n_dMMR']}）")
    print("\n时间划分:")
    for k, v in temporal.items():
        if k != "train_years":
            print(f"  {k}: {len(v)} 例")
    print("\n5-fold 分层 CV（每折 test 例数）:")
    for k, v in cv.items():
        print(f"  {k}: train {len(v['train'])} / test {len(v['test'])}")
    print("\n已写 e2_data/manifest.csv 与 splits.json")


if __name__ == "__main__":
    main()
