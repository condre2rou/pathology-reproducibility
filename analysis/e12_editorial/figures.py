#!/usr/bin/env python3
"""Vector manuscript figures derived only from stored aggregate results."""

from pathlib import Path
from collections import Counter
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#23659A"
TEAL = "#008577"
ORANGE = "#BD652F"
GREY = "#687581"
INK = "#233342"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK,
        "text.color": INK,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
    }
)


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def framework(out):
    fig, ax = plt.subplots(figsize=(11, 8.3))
    ax.set(xlim=(0, 12), ylim=(0, 9.2))
    ax.axis("off")

    def box(x, y, w, h, title, subtitle="", colour=BLUE, fontsize=10.3):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=.08",
                facecolor="white",
                edgecolor=colour,
                linewidth=1.2,
            )
        )
        ax.text(
            x + w / 2,
            y + h * (0.65 if subtitle else 0.5),
            title,
            ha="center",
            va="center",
            fontsize=fontsize,
            weight="bold",
            color=colour,
        )
        if subtitle:
            ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=9.0)

    def arrow(a, b, colour=GREY):
        ax.add_patch(
            FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color=colour)
        )

    def panel(y, h, heading, colour):
        ax.add_patch(
            FancyBboxPatch(
                (0.08, y),
                11.8,
                h,
                boxstyle="round,pad=.03,rounding_size=.1",
                facecolor=colour,
                edgecolor="none",
            )
        )
        ax.text(0.28, y + h - 0.26, heading, fontsize=12, weight="bold", va="center")

    panel(7.05, 2.03, "A   Common feature and prediction pipeline", "#EEF4F9")
    box(0.32, 7.38, 2.30, 1.08, "Photographs / WSI", "Report / genomic labels")
    box(3.00, 7.38, 2.62, 1.08, "Frozen encoders", "Phikon · SimCLR\nResNet-50 · UNI2-h (WSI)")
    box(6.00, 7.38, 2.20, 1.08, "Unit features", "Directory record /\nTCGA participant")
    box(
        8.60,
        7.38,
        2.90,
        1.08,
        "Linear probe",
        "Training-fold standardisation\nBalanced logistic regression",
    )
    for left, right in [(2.65, 2.97), (5.65, 5.97), (8.23, 8.57)]:
        arrow((left, 7.92), (right, 7.92))
    ax.text(
        0.38, 7.14, "Task characterisation: grade, Ki67, dMMR, p53 and receptor status", fontsize=10
    )

    panel(
        4.93,
        1.90,
        "B   Balanced-pool benchmark   |   134 combinations · 17 source cohorts",
        "#F1F5F8",
    )
    box(0.32, 5.27, 2.30, 0.94, "Balanced subsets", "10–80 labels · 30 draws")
    box(3.00, 5.27, 2.62, 0.94, "Within-subset CV", "A(n) and A(full)")
    box(6.00, 5.27, 2.20, 0.94, "Finite-pool gain", "Δ = A(full) − A(40)")
    box(
        8.60,
        5.27,
        2.90,
        0.94,
        "Grouped evaluation",
        "Descriptive association\nHistorical cut / cohort holdout",
    )
    for left, right in [(2.65, 2.97), (5.65, 5.97), (8.23, 8.57)]:
        arrow((left, 5.74), (right, 5.74))

    panel(
        0.12,
        4.58,
        "C   Single random pilot   |   Split first, fit separately, evaluate together",
        "#EFF7F4",
    )
    ax.text(
        0.40,
        4.12,
        "Shared hash split: directory records (institutional) / participants (TCGA)",
        fontsize=10,
    )
    box(0.40, 2.96, 2.12, 0.90, "Training pool", "About 75% of units", TEAL)
    box(2.95, 2.96, 2.10, 0.90, "Random pilot", "40 training-pool units", TEAL)
    box(5.50, 2.96, 2.18, 0.90, "Pilot-only CV", "Score + pair interval", TEAL)
    box(8.16, 2.96, 3.10, 0.90, "Decision", "Stop / Continue / Insufficient", TEAL, 9.8)
    for left, right in [(2.55, 2.92), (5.08, 5.47), (7.71, 8.13)]:
        arrow((left, 3.41), (right, 3.41), TEAL)
    box(
        0.40,
        0.63,
        2.12,
        1.13,
        "Held-out test set",
        "About 25% of units\nExcluded from fitting",
        ORANGE,
        9.8,
    )
    box(
        2.95,
        1.35,
        2.10,
        1.15,
        "Fit two models",
        "Pilot 40 / full pool\nPool includes pilot",
        ORANGE,
        9.8,
    )
    box(
        5.50,
        1.35,
        2.18,
        1.15,
        "Paired predictions",
        "Both models on the\nsame held-out test set",
        ORANGE,
        9.8,
    )
    box(
        8.16,
        1.35,
        3.10,
        1.15,
        "Retrospective evaluation",
        "G = test AUROC difference\nErrors / Savings / Coverage",
        ORANGE,
        9.8,
    )
    arrow((4.00, 2.92), (4.00, 2.54), ORANGE)
    ax.plot([1.46, 1.46, 3.12], [2.92, 2.69, 2.69], color=ORANGE, lw=1.1)
    arrow((3.12, 2.69), (3.12, 2.54), ORANGE)
    arrow((5.08, 1.93), (5.47, 1.93), ORANGE)
    ax.plot([2.55, 6.59], [1.10, 1.10], color=ORANGE, lw=1.1)
    arrow((6.59, 1.10), (6.59, 1.31), ORANGE)
    arrow((7.71, 1.93), (8.13, 1.93), ORANGE)
    arrow((9.71, 2.92), (9.71, 2.54), ORANGE)
    ax.text(
        0.44,
        0.34,
        "Held-out outcomes evaluate decisions; they are not inputs to a target-cohort decision.",
        fontsize=9.5,
    )
    save(fig, out, "fig_e12_framework")


def spectrum(out):
    files = ["tasks_phikon.json", "robustness_ssl.json", "robustness_imagenet.json"]
    data = [json.loads((ROOT / "e2_train/runs" / f).read_text())["tasks"] for f in files]
    histo = json.loads((ROOT / "e6_learnability/histology_robust_result.json").read_text())[
        "by_backbone"
    ]
    for k, name in enumerate(("phikon", "ssl", "imagenet")):
        data[k]["subtype"] = histo[name]
    tasks = ["grade3", "Ki67_high30", "p53_abn", "PR_pos", "dMMR", "ER_pos", "subtype"]
    labels = [
        "Grade 3",
        "Ki67 ≥30%",
        "Abnormal p53",
        "PR positive",
        "dMMR",
        "ER positive",
        "Serous subtype",
    ]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for k, (name, colour, marker) in enumerate(
        [("Phikon", BLUE, "o"), ("SimCLR", TEAL, "s"), ("ImageNet", ORANGE, "^")]
    ):
        means = [data[k][t].get("auroc", data[k][t].get("auroc_mean")) for t in tasks]
        sd = [data[k][t].get("std", data[k][t].get("auroc_std")) for t in tasks]
        ax.errorbar(
            means,
            np.arange(7) + (k - 1) * 0.19,
            xerr=sd,
            fmt=marker,
            color=colour,
            markersize=5,
            capsize=2,
            elinewidth=1.0,
            label=name,
        )
    ax.set_yticks(
        range(7), [f"{name}  (n={data[0][task]['n']})" for name, task in zip(labels, tasks)]
    )
    ax.set(xlim=(0.25, 0.95), xlabel="AUROC", ylim=(6.6, -0.6))
    ax.axvline(0.5, color=GREY, ls="--", lw=0.9)
    ax.grid(axis="x", alpha=0.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_e12_task_spectrum")


def budget(root, out):
    d = json.loads((root / "corrected_reference_result.json").read_text())
    budgets = [10, 20, 30, 40, 50, 60, 80]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for name, data, colour, marker in [
        ("Eligible at each budget", d["budget_sensitivity"], BLUE, "o"),
        ("Common subset", d["common_subset"]["budgets"], TEAL, "s"),
    ]:
        axes[0].plot(
            budgets,
            [data[str(n)]["fixed"]["auroc"] for n in budgets],
            "-" + marker,
            color=colour,
            ms=4,
            label=name,
        )
        axes[1].plot(
            budgets,
            [data[str(n)]["cohort_recalibrated"]["accuracy"] for n in budgets],
            "-" + marker,
            color=colour,
            ms=4,
            label=name,
        )
        axes[1].plot(
            budgets,
            [data[str(n)]["fixed"]["majority_accuracy"] for n in budgets],
            ls=":",
            color=colour,
            lw=1.3,
        )
    for ax, title, ylabel in zip(
        axes,
        ["A   Fixed-score discrimination", "B   Cohort-recalibrated decisions"],
        ["AUROC", "Accuracy"],
    ):
        ax.set(
            title=title,
            xlabel="Balanced-subset label budget",
            ylabel=ylabel,
            ylim=(0, 1),
            xticks=budgets,
        )
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=8.5, frameon=False, loc="lower right")
    axes[1].text(
        0.04,
        0.04,
        "Dotted lines: majority baselines",
        transform=axes[1].transAxes,
        fontsize=8.5,
        color=GREY,
    )
    text = "Eligible combinations:  " + "  ·  ".join(
        f"{n}: {d['budget_sensitivity'][str(n)]['n_points']}" for n in budgets
    )
    fig.text(0.5, 0.015, text, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "fig_e12_budget")


def decisions(root, out):
    d = json.loads((root / "single_budget_result.json").read_text())
    methods = ["always_stop", "fixed_057", "cohort_recalibrated", "flat_curve", "three_way"]
    labels = ["Always stop", "Historical 0.57", "Cohort recalibration", "Flat curve", "Three-way"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), sharey=True)
    for ax, (metric, title) in zip(
        axes,
        [
            ("annotation_saving_fraction", "A   Annotation savings"),
            ("false_stop_rate", "B   False stops among stop calls"),
        ],
    ):
        for sample, colour, marker, offset, label in [
            ("natural", BLUE, "o", -0.12, "Random pilot"),
            ("balanced", ORANGE, "s", 0.12, "Balanced pilot"),
        ]:
            for i, method in enumerate(methods):
                m = d[sample]["methods"][method]
                lo, hi = m["ci"][metric]["interval"]
                y = i + offset
                ax.plot([100 * lo, 100 * hi], [y, y], color=colour, lw=1.3)
                ax.scatter(
                    [100 * m[metric]],
                    [y],
                    c=colour,
                    marker=marker,
                    s=29,
                    label=label if i == 0 else None,
                    zorder=3,
                )
        ax.set(title=title, xlim=(0, 100), xlabel="Percent", ylim=(4.6, -0.6))
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="x", alpha=0.15)
    axes[0].set_yticks(range(5), labels)
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig_e12_decisions")


def cohorts(root, out):
    d = json.loads((root / "manifest.json").read_text())
    counts = Counter(t["cohort"] for t in d["tasks"])
    ordered = sorted(counts, key=lambda x: (-counts[x], x))
    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    colours = [TEAL if name.startswith("YT") else BLUE for name in ordered]
    ax.barh(range(len(ordered)), [counts[name] for name in ordered], height=0.65, color=colours)
    for i, name in enumerate(ordered):
        ax.text(counts[name] + 0.45, i, str(counts[name]), va="center", fontsize=9)
    ax.set(
        yticks=range(len(ordered)),
        yticklabels=ordered,
        xlabel="Task–representation combinations",
        xlim=(0, 54),
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    save(fig, out, "fig_e12_cohorts")


def repeatability(root, out):
    d = pd.read_csv(root / "single_budget_decisions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.7))
    for sampling, colour, label in [
        ("natural", BLUE, "Random pilot"),
        ("balanced", ORANGE, "Balanced pilot"),
    ]:
        fixed = (
            d[(d.sampling == sampling) & (d.method == "fixed_057")].groupby("dataset").stop.mean()
        )
        tri = (
            d[(d.sampling == sampling) & (d.method == "three_way")]
            .groupby("dataset")
            .decisive.mean()
        )
        axes[0].hist(
            fixed, bins=np.linspace(0, 1, 11), histtype="step", lw=1.6, color=colour, label=label
        )
        axes[1].hist(
            1 - tri, bins=np.linspace(0, 1, 11), histtype="step", lw=1.6, color=colour, label=label
        )
    for ax, title, xlabel in zip(
        axes,
        ["A   Historical-rule repeatability", "B   Three-way uncertainty"],
        ["Fraction of pilots called stop", "Fraction with insufficient evidence"],
    ):
        ax.set(title=title, xlabel=xlabel, ylabel="Number of combinations", xlim=(0, 1))
        ax.grid(axis="y", alpha=0.15)
    axes[0].legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_e12_repeatability")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "e11_method_audit/outputs")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    framework(args.out)
    spectrum(args.out)
    budget(args.root, args.out)
    decisions(args.root, args.out)
    cohorts(args.root, args.out)
    repeatability(args.root, args.out)
    print("Generated six vector figure sets (PDF, SVG, PNG).")


if __name__ == "__main__":
    main()
