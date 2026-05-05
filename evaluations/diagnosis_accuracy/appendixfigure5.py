#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

IN_DIR = Path("evaluations/diagnosis_accuracy")
OUT_DIR = Path("evaluations/diagnosis_accuracy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_DETAIL = IN_DIR / "diagnosis_accuracy_per_sample_all_strategies.csv"
FIG_OUT = OUT_DIR / "AppendixFigure5.pdf"

MODEL_NAME_MAP = {
    "gemma-3-27b": "Gemma 3 27B",
    "gemma-3-4b": "Gemma 3 4B",
    "claude-opus-4.6": "Claude 4.6",
    "gpt-5.4": "GPT-5.4",
    "gemini-2.5-flash-no-thinking": "Gemini #no-think",
    "gemini-2.5-flash-thinking": "Gemini #think",
    "med42-8b": "Med42-8B",
    "medgemma-27b": "MedGemma 27B",
    "mistral-small-3.1-24B": "Mistral 24B",
    "qwen_3-32b": "Qwen 32B",
}

STRATEGY = "full_info_no_abstention"
MIN_CASES = 5

FONT_SIZE_LABEL = 14
FONT_SIZE_TICK = 12


def style_axes(ax):
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def validate_columns(df, required_cols):
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def plot_accuracy_per_diagnosis():
    df = pd.read_csv(CSV_DETAIL)
    df.columns = df.columns.str.strip()

    required_cols = {
        "model",
        "strategy",
        "sample_id",
        "gt_3char",
        "match_3char",
    }
    validate_columns(df, required_cols)

    df = df[df["model"].isin(MODEL_NAME_MAP.keys())].copy()
    df = df[df["strategy"] == STRATEGY].copy()
    df = df[df["gt_3char"].notna()].copy()

    summary = (
        df.groupby("gt_3char", as_index=False)
        .agg(
            total_cases=("sample_id", "count"),
            correct_3char=("match_3char", "sum"),
        )
    )

    summary["accuracy_3char_pct"] = (
        100 * summary["correct_3char"] / summary["total_cases"]
    )

    summary = summary[summary["total_cases"] >= MIN_CASES].copy()
    summary = summary.sort_values("accuracy_3char_pct", ascending=False)

    top_6 = summary.head(6).copy()
    bottom_6 = summary.tail(6).copy()

    top_6["section"] = "Top 6"
    bottom_6["section"] = "Bottom 6"

    plot_df = pd.concat([top_6, bottom_6], ignore_index=True)
    plot_df = plot_df.drop_duplicates(subset=["gt_3char"])

    fig, ax = plt.subplots(figsize=(6.5, 5))

    y_pos = range(len(plot_df))

    ax.barh(
        y_pos,
        plot_df["accuracy_3char_pct"],
        height=0.7,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["gt_3char"], fontsize=FONT_SIZE_TICK)

    ax.invert_yaxis()

    for i, row in plot_df.iterrows():
        ax.text(
            row["accuracy_3char_pct"] + 1,
            i,
            f"n={row['total_cases']}",
            va="center",
            fontsize=9,
        )

    if len(plot_df) > 6:
        separator_y = 5.5
        ax.axhline(separator_y, color="black", linewidth=0.8)

        ax.text(
            102,
            2.5,
            "Top 6",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

        ax.text(
            102,
            min(8.5, len(plot_df) - 1),
            "Bottom 6",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("3-character ICD accuracy (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Ground-truth ICD category", fontsize=FONT_SIZE_LABEL)
    ax.set_xlim(0, 115)

    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK)

    style_axes(ax)

    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {FIG_OUT}")


def main():
    plot_accuracy_per_diagnosis()


if __name__ == "__main__":
    main()