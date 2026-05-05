#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

IN_DIR = Path("evaluations/diagnosis_accuracy")
OUT_DIR = Path("evaluations/diagnosis_accuracy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_DETAIL = IN_DIR / "diagnosis_accuracy_per_sample_all_strategies.csv"

FIG_GROUPED = OUT_DIR / "Figure8.pdf"
FIG_DELTA = OUT_DIR / "Figure_3char_accuracy_drop_matched_sequential_vs_fullinfo.pdf"

MODEL_NAME_MAP = {
    "gemma-3-27b": "Gemma 3 27B",
    "gemma-3-4b": "Gemma 3 4B",
    "claude-opus-4.6": "Claude 4.6",
    "gpt-5.4": "GPT-5.4",
    "gemini-2.5-flash-no-thinking": "Gemini Flash 2.5",
    "med42-8b": "Med42-8B",
    "medgemma-27b": "MedGemma 27B",
    "mistral-small-3.1-24B": "Mistral 24B",
    "qwen_3-32b": "Qwen 32B",
}

STRATEGY_LABELS = {
    "full_info_no_abstention": "Full info - no abstention",
    "full_info_abstention_aware": "Full info - abstention",
    "sequential_info_neutral_prompting": "Sequential - no abstention",
    "sequential_info_abstention_aware": "Sequential - abstention",
}

STRATEGIES_TO_COMPARE = list(STRATEGY_LABELS.keys())

COMPARISONS = {
    "No-abstention setting": (
        "full_info_no_abstention",
        "sequential_info_neutral_prompting",
    ),
    "Abstention-aware setting": (
        "full_info_abstention_aware",
        "sequential_info_abstention_aware",
    ),
}

COMPARISON_COLORS = {
    "No-abstention setting": "#4C78A8",
    "Abstention-aware setting": "#D64B45",
}

STRATEGY_COLORS = {
    "full_info_no_abstention": "#4C9A3F",
    "full_info_abstention_aware": "#A8D49E",
    "sequential_info_neutral_prompting": "#4C78A8",
    "sequential_info_abstention_aware": "#96C1EE",
}

FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 11
FONT_SIZE_LEGEND = 12


def pretty_model_name(model):
    return MODEL_NAME_MAP.get(model, model)


def style_axes(ax):
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def compute_accuracy(df):
    return (
        df.groupby(["model", "strategy"], as_index=False)
        .agg(
            total_cases=("sample_id", "count"),
            correct_3char=("match_3char", "sum"),
        )
        .assign(
            accuracy_3char_pct=lambda x: 100 * x["correct_3char"] / x["total_cases"]
        )
    )


def plot_grouped_strategy_accuracy():
    df = pd.read_csv(CSV_DETAIL)

    df = df[df["model"].isin(MODEL_NAME_MAP.keys())].copy()
    df = df[df["strategy"].isin(STRATEGIES_TO_COMPARE)].copy()

    acc = compute_accuracy(df)

    pivot = acc.pivot(
        index="model",
        columns="strategy",
        values="accuracy_3char_pct",
    )

    pivot = pivot.dropna(subset=["full_info_no_abstention"])
    pivot = pivot.sort_values("full_info_no_abstention", ascending=False)

    models = [pretty_model_name(m) for m in pivot.index]
    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, strategy in enumerate(STRATEGIES_TO_COMPARE):
        if strategy not in pivot.columns:
            continue

        offset = (i - 1.5) * width

        ax.bar(
            x + offset,
            pivot[strategy],
            width,
            label=STRATEGY_LABELS[strategy],
            color=STRATEGY_COLORS[strategy],
        )

    ax.set_ylabel("3-char ICD accuracy (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=FONT_SIZE_TICK)

    style_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.48),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.45)
    fig.savefig(FIG_GROUPED, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {FIG_GROUPED}")



def main():
    plot_grouped_strategy_accuracy()


if __name__ == "__main__":
    main()