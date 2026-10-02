"""Preserved computation; see docs/experiments.md for its protocol."""

import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
S1, S2, S3, S4 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
CUT = "#6b4fa0"
INK, INK2, MUTED = ("#0b0b0b", "#52514e", "#898781")
GRID, BASE, SURF = ("#e1e0d9", "#c3c2b7", "#fcfcfb")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": BASE,
        "axes.linewidth": 0.8,
        "axes.facecolor": SURF,
        "figure.facecolor": "white",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    }
)


def style_axes(ax, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def chance_line(ax, y=0.5):
    ax.axhline(y, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)


def save(fig, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  OK {name}.png / .pdf / .svg")


def fig2():
    res = ["native", "512", "384", "256", "192", "128", "96"]
    with open(
        os.path.join(os.path.dirname(OUT), "e2_train/runs/resolution_ladder_ssl.json")
    ) as stream:
        data = json.load(stream)["tasks"]
    grade, ki67, dmmr, p53 = [
        [data[t]["curve"][r]["auroc"] for r in res]
        for t in ("grade3", "Ki67_high30", "dMMR", "p53_abn")
    ]
    x = np.arange(len(res))
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.plot(x, grade, "-o", color=S1, lw=1.8, ms=4.5, label="grade (morphology)", zorder=3)
    ax.plot(x, ki67, "-s", color=S2, lw=1.8, ms=4.0, label="Ki67 (proliferation)", zorder=3)
    ax.plot(x, p53, "-^", color=S4, lw=1.8, ms=4.5, label="p53 (molecular)", zorder=3)
    ax.plot(x, dmmr, "-D", color=S3, lw=1.8, ms=4.0, label="dMMR (molecular)", zorder=3)
    chance_line(ax)
    ax.text(6.35, 0.503, "chance", fontsize=6.5, color=MUTED, va="bottom", ha="right")
    ax.text(6.3, grade[-1] + 0.02, "grade", fontsize=6.5, color=S1, ha="right")
    ax.text(6.3, dmmr[-1] - 0.032, "dMMR", fontsize=6.5, color=S3, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(res, fontsize=6.3)
    ax.set_xlim(-0.4, 6.6)
    ax.set_ylim(0.42, 0.82)
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_xlabel("Input short-side resolution (px)")
    style_axes(ax)
    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=6.3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.2,
        handletextpad=0.5,
    )
    save(fig, "fig2_resolution_ladder")


def fig3():
    labels = ["Microphoto\n(morphology)", "Microphoto\n(molecular)", "WSI\n(molecular)"]
    sub = ["grade", "dMMR", "dMMR"]
    with open(os.path.join(os.path.dirname(OUT), "e2_train/runs/tasks_phikon.json")) as stream:
        photo = json.load(stream)["tasks"]
    with open(os.path.join(os.path.dirname(OUT), "e5_tcga/ucec_wsi_tasks.json")) as stream:
        wsi = json.load(stream)
    estimates = [photo["grade3"], photo["dMMR"], wsi["dMMR"]]
    vals = [m["auroc"] for m in estimates]
    sds = [m["std"] for m in estimates]
    colors = [S1, S3, S1]
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    x = np.arange(3)
    bars = ax.bar(
        x,
        vals,
        0.46,
        yerr=sds,
        error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=2.4, capthick=0.8),
        color=colors,
        zorder=3,
    )
    bars[2].set_hatch("///")
    bars[2].set_edgecolor(SURF)
    chance_line(ax)
    ax.text(3.25, 0.503, "chance", fontsize=6.5, color=MUTED, va="bottom", ha="right")
    for b, v, sd, s in zip(bars, vals, sds, sub):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + sd + 0.01,
            f"{v:.3f}\n{s}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            color=INK,
            linespacing=1.15,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_xlim(-0.5, 3.3)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("AUROC")
    style_axes(ax)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=S1, label="Microphotography"),
            Patch(facecolor=S1, hatch="///", edgecolor=SURF, label="WSI"),
        ],
        frameon=False,
        fontsize=6.5,
        ncol=2,
        loc="upper center",
        handlelength=1.4,
        columnspacing=1.6,
        bbox_to_anchor=(0.5, -0.17),
    )
    save(fig, "fig3_core_comparison")


def fig4():
    labels = [
        "GRADE-only\nbaseline",
        "G1 only\n(n=39)",
        "G2 only\n(n=55)",
        "G3 only\n(n=102)",
        "All\n(n=200)",
    ]
    with open(os.path.join(os.path.dirname(OUT), "e5_tcga/confound_result.json")) as stream:
        data = json.load(stream)
    vals = [
        data["grade_only_auroc"],
        *[data["by_grade"][g]["auroc"] for g in ("G1", "G2", "G3")],
        data["auroc_all"],
    ]
    colors = [S3, S4, S2, S1, INK2]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, 0.58, color=colors, zorder=3)
    chance_line(ax)
    ax.text(4.42, 0.503, "chance", fontsize=6.5, color=MUTED, va="bottom", ha="left")
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.013,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.1)
    ax.set_xlim(-0.6, 5.0)
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("AUROC")
    style_axes(ax)
    save(fig, "fig4_confound_grade")


def fig6():
    tasks = ["grade\n(morphology)", "dMMR / MSI\n(molecular)", "ER\n(IHC)", "PR\n(IHC)"]
    with open(os.path.join(os.path.dirname(OUT), "e2_train/runs/tasks_phikon.json")) as stream:
        local = json.load(stream)["tasks"]
    with open(os.path.join(os.path.dirname(OUT), "e5_tcga/ucec_wsi_tasks.json")) as stream:
        ucec = json.load(stream)
    with open(os.path.join(os.path.dirname(OUT), "e5_tcga/multi_result.json")) as stream:
        multi = json.load(stream)
    photo = [local[t]["auroc"] for t in ("grade3", "dMMR", "ER_pos", "PR_pos")]
    wsi = [
        [ucec["grade"]["auroc"], multi["STAD"]["morphology"]["auroc"]],
        [
            ucec["dMMR"]["auroc"],
            multi["STAD"]["molecular"]["auroc"],
            multi["COAD"]["molecular"]["auroc"],
        ],
        [multi["BRCA_IDC"]["molecular"]["auroc"]],
        [multi["BRCA_IDC_PR"]["molecular"]["auroc"]],
    ]
    wsi_lbl = [["UCEC", "STAD"], ["UCEC", "STAD", "COAD"], ["BRCA"], ["BRCA"]]
    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    for i, (p, ws, lbs) in enumerate(zip(photo, wsi, wsi_lbl)):
        ax.scatter(
            ws, [i] * len(ws), s=44, marker="D", color=S1, zorder=4, edgecolor=SURF, linewidth=0.8
        )
        ax.scatter([p], [i], s=58, marker="o", color=S2, zorder=4, edgecolor=SURF, linewidth=0.8)
        order = np.argsort(ws)
        for rank, j in enumerate(order):
            dy = -0.3 if rank % 2 == 0 else -0.56
            ax.plot([ws[j], ws[j]], [i - 0.13, i + dy + 0.13], color=BASE, lw=0.7, zorder=1)
            ax.text(ws[j], i + dy, lbs[j], fontsize=5.3, color=MUTED, ha="center", va="top")
    ax.axvline(0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(0.505, 1.52, "chance", fontsize=6.2, color=MUTED, va="center", ha="left")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=6.8)
    ax.set_ylim(-0.95, 4.15)
    ax.set_xlim(0.44, 0.99)
    ax.set_xlabel("AUROC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=S2, ms=6, label="Microphotography (ours)"),
            Line2D([], [], marker="D", ls="", color=S1, ms=5, label="WSI (TCGA)"),
        ],
        frameon=False,
        fontsize=6.4,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        handletextpad=0.4,
        borderpad=0.0,
    )
    save(fig, "fig6_cross_cancer")


def fig8():
    """Preserved computation; see docs/experiments.md for its protocol."""
    import json

    p = os.path.join(os.path.dirname(OUT), "e10_expansion", "degradation_result.json")
    if not os.path.exists(p):
        print("  跳过 fig8（degradation_result.json 不存在）")
        return
    d = json.load(open(p, encoding="utf-8"))
    curves = d["curves"]
    order = [
        "native",
        "down2",
        "down4",
        "down8",
        "blur1",
        "blur2",
        "blur4",
        "jpeg50",
        "jpeg20",
        "jpeg5",
        "contrast0.5",
        "bright0.6",
    ]
    order = [k for k in order if k in curves]
    labels = ["native", "÷2", "÷4", "÷8", "σ1", "σ2", "σ4", "q50", "q20", "q5", "×0.5", "×0.6"]
    lab = dict(zip(order, labels))
    CELL = {"grade3": ("grade 3", S1, "-"), "Ki67_high30": ("Ki67 high", S1, "--")}
    MOL = {"dMMR": ("dMMR", S2, "-"), "ER_pos": ("ER", S2, "--"), "PR_pos": ("PR", S2, ":")}
    ALL = {**CELL, **MOL}
    END_DY = {"dMMR": 9, "ER_pos": 0, "PR_pos": -9, "grade3": 0, "Ki67_high30": 0}
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(order))
    for key, (name, col, ls) in ALL.items():
        y = [
            curves[k]["tasks"][key]["auroc"] if key in curves[k]["tasks"] else np.nan for k in order
        ]
        mk = "o" if col == S1 else "D"
        ax.plot(
            x,
            y,
            color=col,
            ls=ls,
            lw=1.4,
            marker=mk,
            ms=3.2,
            mec="white",
            mew=0.6,
            label=name,
            zorder=3,
        )
        if not np.isnan(y[-1]):
            ax.annotate(
                name,
                (x[-1], y[-1]),
                xytext=(4, END_DY.get(key, 0)),
                textcoords="offset points",
                fontsize=6.6,
                color=col,
                va="center",
                ha="left",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkA=0, shrinkB=1)
                if END_DY.get(key, 0)
                else None,
            )
    chance_line(ax, 0.5)
    ax.text(0.05, 0.512, "chance", fontsize=6.2, color=MUTED, va="bottom")
    for xi in [3.5, 6.5, 9.5]:
        ax.axvline(xi, color=GRID, lw=0.8, zorder=1)
    for xc, t in [(1.5, "downsample"), (5.0, "blur"), (8.0, "JPEG"), (10.5, "illum.")]:
        ax.text(xc, 0.905, t, fontsize=6.4, color=INK2, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([lab[k] for k in order], fontsize=6.8)
    ax.set_xlim(-0.5, len(order) + 1.6)
    ax.set_ylim(0.4, 0.93)
    ax.set_xlabel("Degradation level applied to the same fixed images")
    ax.set_ylabel("AUROC")
    style_axes(ax)
    save(fig, "fig8_degradation")


if __name__ == "__main__":
    fig2()
    fig3()
    fig4()
    fig6()
    fig8()
    import subprocess
    import sys
    from pathlib import Path

    root = Path(OUT).parent
    subprocess.run([sys.executable, str(root / "e11_method_audit/figures.py")], check=True)
    subprocess.run([sys.executable, str(root / "e12_editorial/figures.py")], check=True)
    subprocess.run([sys.executable, str(root / "e13_review/figures.py")], check=True)
