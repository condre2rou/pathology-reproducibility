"""Export every main/supplementary table as editable CSV from stored estimates."""

import argparse
import csv
import json
from pathlib import Path

POLICIES = {
    "always_continue": "Always continue",
    "always_stop": "Always stop",
    "fixed_057": "Historical 0.57",
    "cohort_recalibrated": "Cohort recalibration",
    "flat_curve": "Flat curve",
    "three_way": "Three-way",
}


def load(root, path):
    return json.loads((root / path).read_text())


def csv_file(out, name, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / (name + ".csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def policy_rows(methods):
    for key, label in POLICIES.items():
        m = methods[key]
        yield dict(
            policy=label,
            accuracy=m["accuracy"],
            false_stop_percent=None if m["false_stop_rate"] is None else 100 * m["false_stop_rate"],
            missed_opportunity_percent=100 * m["missed_opportunity_rate"],
            annotation_saved_percent=100 * m["annotation_saving_fraction"],
            coverage_percent=100 * m["decision_coverage"],
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root, out = args.root, args.out
    out.mkdir(parents=True, exist_ok=False)
    ref = load(root, "e11_method_audit/outputs/corrected_reference_result.json")
    operation = load(root, "e11_method_audit/outputs/single_budget_result.json")
    meta = load(root, "e13_review/outputs/grouped_meta.json")
    sensitivity = load(root, "e13_review/outputs/sensitivity.json")
    manifest = load(root, "e11_method_audit/outputs/manifest.json")
    labels = load(root, "e1_label_engine/outputs/e1_summary.json")
    tasks = manifest["tasks"]
    csv_file(
        out,
        "Table_1",
        [
            dict(
                source="Institutional EC",
                analysis_records=labels["labels"]["main_analysis"]["n"],
                combinations=sum(t["cohort"] == "YT-EC" for t in tasks),
                role="Task characterisation and annotation evaluation",
            ),
            dict(
                source="Institutional OV",
                analysis_records=max(t["n"] for t in tasks if t["cohort"] == "YT-OV"),
                combinations=sum(t["cohort"] == "YT-OV" for t in tasks),
                role="Annotation evaluation",
            ),
            dict(
                source="TCGA (15 cohorts)",
                analysis_records="Varies by endpoint",
                combinations=sum(t["cohort"].startswith("TCGA-") for t in tasks),
                role="Annotation evaluation and WSI comparisons",
            ),
            dict(
                source="Paired H&E/IHC",
                analysis_records=load(root, "e10_expansion/ki67_172_result.json")["meta"][
                    "n_patients"
                ],
                combinations=None,
                role="Supporting Ki67 analysis",
            ),
        ],
    )
    rows = [
        dict(
            analysis="Descriptive score / historical cut",
            auroc=ref["fixed_rule"]["auroc"],
            accuracy=ref["fixed_rule"]["accuracy"],
        )
    ]
    for key in ["cohort", "source_group", "endpoint", "config"]:
        rows.append(
            dict(
                analysis=f"Held-out {key} threshold",
                auroc=None,
                accuracy=ref["grouped"][key]["accuracy"],
            )
        )
    rows.extend(
        [
            dict(
                analysis="Cohort-held-out meta-model",
                auroc=meta["reference_grouped_meta"]["auroc"],
                accuracy=meta["reference_grouped_meta"]["accuracy"],
            ),
            dict(
                analysis="Majority-class baseline",
                auroc=None,
                accuracy=ref["fixed_rule"]["majority_accuracy"],
            ),
        ]
    )
    csv_file(out, "Table_2", rows)
    csv_file(out, "Table_3", policy_rows(operation["natural"]["methods"]))
    additional = load(root, "e6_learnability/validate_scale_result.json")
    additional_rows = [
        dict(
            task=t,
            scale="tissue",
            representation=b,
            **{k: additional[f"{b}_{t}"][k] for k in ("n", "pos", "auroc")},
            fold_sd=additional[f"{b}_{t}"]["std"],
        )
        for t in ("lvsi", "myoinv")
        for b in ("phikon", "ssl", "imagenet")
    ]
    for backbone in ("phikon", "ssl", "imagenet"):
        filename = "tasks_phikon.json" if backbone == "phikon" else f"robustness_{backbone}.json"
        grade = load(root, f"e2_train/runs/{filename}")["tasks"]["grade3"]
        additional_rows.append(
            dict(
                task="grade3_control",
                scale="cell",
                representation=backbone,
                n=grade["n"],
                pos=grade["pos"],
                auroc=grade.get("auroc", grade.get("auroc_mean")),
                fold_sd=grade.get("std", grade.get("auroc_std")),
            )
        )
    csv_file(out, "Table_S1", additional_rows)
    nearest = load(root, "e6_learnability/validate_result.json")["correlations"]["knn_loo_n20"]
    random = load(root, "e6_learnability/active_select_result.json")["results"]["random_n20"]
    curve = load(root, "e6_learnability/learning_curve_result.json")["results"][
        "extrapolation (A∞)"
    ]
    csv_file(
        out,
        "Table_S2",
        [
            dict(
                analysis="Nearest neighbour",
                n=nearest["n"],
                spearman=nearest["spearman"],
                pearson=None,
                r2=None,
            ),
            dict(
                analysis="Random-subset linear probe",
                n=random["n"],
                spearman=random["spearman"],
                pearson=random["pearson"],
                r2=None,
            ),
            dict(
                analysis="Curve extrapolation",
                n=curve["n"],
                spearman=curve["spearman"],
                pearson=None,
                r2=curve["r2"],
            ),
        ],
    )
    finetune = []
    temporal_counts = load(root, "e9_revision/temporal_counts.json")["partitions"]
    for config in ("imagenet_full_PREEXISTING", "imagenet_full", "simclr_full", "imagenet_frozen"):
        result = load(root, f"e9_revision/finetune_{config}.json")
        for test in ("test_2023", "test_2024"):
            row = result[test]
            counts = temporal_counts[test]
            assert row["n"] == counts["usable_images"]
            assert row["pos"] == counts["positive_images"]
            finetune.append(
                dict(
                    configuration=config,
                    test=test,
                    n_image_rows=row["n"],
                    positive_image_rows=row["pos"],
                    records_with_usable_images=counts["records_with_usable_images"],
                    positive_records_with_usable_images=counts[
                        "positive_records_with_usable_images"
                    ],
                    evaluation_unit="image; descriptive point estimate",
                    auroc=row["auroc"],
                    auprc=row["auprc"],
                )
            )
    csv_file(out, "Table_S3", finetune)
    quality = load(root, "e9_revision/quality_strat_result.json")["tasks"]
    rows = []
    for task, measures in [
        ("dMMR", ["sharp", "bytes_per_px", "illum_uneven", "sat_mean", "contrast", "colour_cast"]),
        ("grade3", ["sharp", "bytes_per_px"]),
    ]:
        for measure in measures:
            row = dict(task=task, quality_measure=measure)
            for tertile in ["T1", "T2", "T3"]:
                for key in ["auroc", "sd", "n", "pos"]:
                    row[f"{tertile}_{key}"] = quality[task]["strata"][measure][tertile][key]
            rows.append(row)
    csv_file(out, "Table_S4", rows)
    rows = []
    for subset, values in [
        ("budget-specific", ref["budget_sensitivity"]),
        ("common", ref["common_subset"]["budgets"]),
    ]:
        for n, m in values.items():
            rows.append(
                dict(
                    subset=subset,
                    budget=int(n),
                    usable=m["n_points"],
                    saturated=m["n_saturated"],
                    auroc=m["fixed"]["auroc"],
                    recalibrated_accuracy=m["cohort_recalibrated"]["accuracy"],
                    majority_accuracy=m["fixed"]["majority_accuracy"],
                )
            )
    csv_file(out, "Table_S5", rows)
    csv_file(out, "Table_S6", load(root, "e6_learnability/pairing_table.json")["pairs"])
    csv_file(out, "Table_S7", policy_rows(operation["balanced"]["methods"]))
    phikon = load(root, "e2_train/runs/tasks_phikon.json")["tasks"]
    ssl = load(root, "e2_train/runs/robustness_ssl.json")["tasks"]
    imagenet = load(root, "e2_train/runs/robustness_imagenet.json")["tasks"]
    histology = load(root, "e6_learnability/histology_robust_result.json")["by_backbone"]
    rows = [
        dict(
            task=t,
            records=phikon[t]["n"],
            positive=phikon[t]["pos"],
            phikon=phikon[t]["auroc"],
            simclr=ssl[t]["auroc_mean"],
            imagenet=imagenet[t]["auroc_mean"],
        )
        for t in ["grade3", "Ki67_high30", "p53_abn", "PR_pos", "dMMR", "ER_pos"]
    ]
    rows.append(
        dict(
            task="serous_subtype",
            records=histology["phikon"]["n"],
            positive=histology["phikon"]["pos"],
            phikon=histology["phikon"]["auroc"],
            simclr=histology["ssl"]["auroc"],
            imagenet=histology["imagenet"]["auroc"],
        )
    )
    csv_file(out, "Table_S8", rows)
    csv_file(
        out,
        "Table_S9",
        [
            dict(
                cutoff=float(d),
                saturated=m["saturated"],
                auroc=m["fixed"]["auroc"],
                historical_accuracy=m["fixed"]["accuracy"],
                recalibrated_accuracy=m["cohort_recalibrated"]["accuracy"],
                majority_accuracy=m["fixed"]["majority_accuracy"],
            )
            for d, m in ref["delta_sensitivity"].items()
        ],
    )
    csv_file(
        out,
        "Table_S10",
        [
            dict(
                decision=d,
                counted_as_stop=int(d == "stop"),
                definitive=int(d != "insufficient"),
                correct_if="G <= 0.05" if d == "stop" else "G > 0.05",
                example_pool_size=100,
                example_label_cost=40 if d == "stop" else 100,
                example_labels_saved=60 if d == "stop" else 0,
                extra_labels="0" if d == "stop" else "N_pool - 40",
                saved_labels="N_pool - 40" if d == "stop" else "0",
            )
            for d in ["stop", "continue", "insufficient"]
        ],
    )
    rows = []
    for subset, profile in sensitivity["profiles"].items():
        for row in policy_rows(profile["methods"]):
            rows.append(
                dict(
                    subset=subset,
                    repetitions=profile["n_repetitions"],
                    tasks=profile["n_tasks"],
                    score_auroc=profile["score_auroc"],
                    **row,
                )
            )
    csv_file(out, "Table_S11", rows)
    diag = sensitivity["refit_diagnostic"]
    csv_file(
        out,
        "Table_S12",
        [
            dict(
                interval=key,
                pilots=diag["n_repetitions"],
                tasks=diag["n_tasks"],
                accuracy=diag[key]["accuracy"],
                false_stop_rate=diag[key]["false_stop_rate"],
                coverage=diag[key]["decision_coverage"],
                median_width=diag[key]["median_interval_width"],
                labels_saved=diag[key]["mean_labels_saved"],
                coverage_percent=100 * diag[key]["decision_coverage"],
                false_stop_percent=100 * diag[key]["false_stop_rate"],
                stop_percent=100 * diag[key]["stop_fraction"],
                saved_percent=100 * diag[key]["annotation_saving_fraction"],
                stop_weight=diag[key]["true_stop"] + diag[key]["false_stop"],
            )
            for key in ["pair", "refit"]
        ],
    )
    rows = []
    for key in POLICIES:
        row = dict(policy=POLICIES[key])
        for metric in [
            "accuracy",
            "false_stop_rate",
            "missed_opportunity_rate",
            "annotation_saving_fraction",
            "decision_coverage",
        ]:
            values = [
                d["methods"][key][metric]
                for d in sensitivity["delete_one_cohort"].values()
                if d["methods"][key][metric] is not None
            ]
            row[metric + "_min"] = min(values) if values else None
            row[metric + "_max"] = max(values) if values else None
        rows.append(row)
    csv_file(out, "Table_S13", rows)
    csv_file(out, "all_task_definitions", manifest["tasks"])
    csv_file(
        out,
        "degradation",
        [
            dict(level=level, task=task, **m)
            for level, v in load(root, "e10_expansion/degradation_result.json")["curves"].items()
            for task, m in v["tasks"].items()
        ],
    )
    fixed = operation["natural"]["methods"]["fixed_057"]
    key_results = dict(
        reference_auroc=ref["fixed_rule"]["auroc"],
        reference_accuracy=ref["fixed_rule"]["accuracy"],
        random_repetitions=operation["natural"]["n_repetitions"],
        random_score_auroc=operation["natural"]["single_draw_score_auroc"],
        precision=1 - fixed["false_stop_rate"],
        sensitivity=fixed["saturation_sensitivity"],
        specificity=fixed["nonsaturation_specificity"],
        false_stop_rate=fixed["false_stop_rate"],
        annotation_saving_fraction=fixed["annotation_saving_fraction"],
        grouped_meta=meta,
        refit_diagnostic=diag,
        test_uncertainty=sensitivity["test_uncertainty"],
        ki67=load(root, "e10_expansion/ki67_172_result.json"),
        extraction=labels,
        demographics=load(root, "e9_revision/demographics_result.json"),
    )
    (out / "key_results.json").write_text(
        json.dumps(key_results, indent=2, ensure_ascii=False) + "\n"
    )
    print("Exported all 16 manuscript tables, task definitions, degradation values and key results")


if __name__ == "__main__":
    main()
