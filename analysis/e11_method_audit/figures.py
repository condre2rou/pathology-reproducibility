"""Publication figures from aggregate E11 outputs, no patient-level data."""

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)
BLUE = "#2765a5"
ORANGE = "#c66a22"
GREEN = "#398269"
GREY = "#6b7280"


def save(fig, out, name):
    out.mkdir(exist_ok=True, parents=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def saturation(root, out):
    df = pd.read_csv(root / "corrected_points.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    origins = [
        ("Institutional EC", df.cohort == "YT-EC", BLUE, "o"),
        ("Institutional OV", df.cohort == "YT-OV", ORANGE, "s"),
        ("TCGA (15 cohorts)", df.cohort.str.startswith("TCGA-"), GREY, "^"),
    ]
    for name, mask, color, marker in origins:
        group = df[mask]
        gain = group.full - group.a40
        ax.errorbar(group.a40, gain, xerr=group.sd40, fmt="none", ecolor=color, alpha=0.15, lw=0.6)
        ax.scatter(
            group.a40,
            gain,
            c=color,
            s=26,
            marker=marker,
            edgecolors="white",
            linewidths=0.3,
            label=name,
            zorder=3,
        )
    ax.axvline(0.57, c="#8655a0", ls="--", lw=1.1)
    ax.axhline(0.05, c=GREY, ls="--", lw=1.1)
    ax.set(
        xlabel="Mean A(40): balanced label-pool draws",
        ylabel="Finite-pool headroom: A(full) − A(40)",
    )
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.grid(alpha=0.15)
    fig.tight_layout()
    save(fig, out, "fig_e11_saturation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE / "outputs")
    ap.add_argument("--out", type=Path, default=HERE.parent / "figures")
    args = ap.parse_args()

    saturation(args.root, args.out)


if __name__ == "__main__":
    main()
