#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


IN_DIR = Path("evaluations/diagnosis_accuracy")
OUT_DIR = Path("evaluations/diagnosis_accuracy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_TIMING = IN_DIR / "accuracy_by_timing_sequential.csv"
FIG_TIMING = OUT_DIR / "Figure7.pdf"


TIMING_COLORS = {
    "full": "#613859",
    "three_char": "#53BBB1",
    "high_class": "#E7E294",
}

FONT_SIZE_LABEL = 14
FONT_SIZE_TICK = 13
FONT_SIZE_LEGEND = 13


def style_axes(ax):
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_accuracy_by_timing():
    df = pd.read_csv(CSV_TIMING)

    # Figure uses the Sequential Information - Abstention Aware setting.
    df = df[df["strategy"] == "sequential_info_abstention_aware"].copy()

    # Compare premature vs on-time diagnoses only.
    df = df[df["timing_category"].isin(["premature", "on_time"])].copy()

    timing_order = ["premature", "on_time"]
    timing_labels = {
        "premature": "Premature",
        "on_time": "On-time",
    }

    rows = []

    for timing in timing_order:
        sub = df[df["timing_category"] == timing]
        total = sub["total_diagnosed_cases"].sum()

        if total == 0:
            continue

        rows.append({
            "timing_category": timing,
            "label": timing_labels[timing],
            "accuracy_diagnosis_level_pct": 100 * sub["correct_diagnosis_level"].sum() / total,
            "accuracy_3char_pct": 100 * sub["correct_3char"].sum() / total,
            "accuracy_high_class_pct": 100 * sub["correct_high_class"].sum() / total,
            "total_diagnosed_cases": total,
        })

    plot_df = pd.DataFrame(rows)

    if plot_df.empty:
        raise ValueError(
            "No premature or on-time diagnosis rows found for "
            "sequential_info_abstention_aware."
        )

    x = np.arange(len(plot_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(4, 4.2))

    ax.bar(
        x - width,
        plot_df["accuracy_diagnosis_level_pct"],
        width,
        label="Full ICD code",
        color=TIMING_COLORS["full"],
    )

    ax.bar(
        x,
        plot_df["accuracy_3char_pct"],
        width,
        label="3-character ICD",
        color=TIMING_COLORS["three_char"],
    )

    ax.bar(
        x + width,
        plot_df["accuracy_high_class_pct"],
        width,
        label="High-level class",
        color=TIMING_COLORS["high_class"],
    )

    ax.set_ylabel("Diagnosis accuracy (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], fontsize=FONT_SIZE_TICK)

    style_axes(ax)

    ax.legend(
        frameon=True,
        fontsize=FONT_SIZE_LEGEND,
    )

    fig.tight_layout()
    fig.savefig(FIG_TIMING, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {FIG_TIMING}")


def main():
    plot_accuracy_by_timing()


if __name__ == "__main__":
    main()