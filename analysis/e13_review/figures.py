#!/usr/bin/env python3
"""Replace the supporting noise plot with documented seed variability."""

from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "e12_editorial"))
from figures import plt, save, BLUE, TEAL


def main():
    data = json.loads((ROOT / "e13_review/outputs/synthetic_noise.json").read_text())["results"]
    rates = [0, 10, 20, 30, 40]
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for mean, sd, label, colour in [
        ("ce_mean", "ce_sd", "Unweighted logistic loss", BLUE),
        ("weighted_mean", "weighted_sd", "Simulated noise weighting", TEAL),
    ]:
        ax.errorbar(
            rates,
            [data[f"{r / 100:.2f}"][mean] for r in rates],
            yerr=[data[f"{r / 100:.2f}"][sd] for r in rates],
            fmt="-o",
            capsize=3,
            label=label,
            color=colour,
        )
    ax.set(
        xlabel="Injected training-label flip probability (%)",
        ylabel="Grade 3 AUROC",
        xticks=rates,
        ylim=(0.45, 0.85),
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    save(fig, ROOT / "figures", "fig5_snap_noise")


if __name__ == "__main__":
    main()
