#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

EXCLUDE_MODELS = {
    "gemini-2.5-flash-thinking",
    "gemini-2.5-flash-thinking2",
}

MODEL_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]


def pretty_model_name(model: str) -> str:
    return MODEL_NAME_MAP.get(model, model)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Stage 1 clarification rates with model names in the legend"
    )
    parser.add_argument(
        "--input_csv",
        default="evaluations/clarification_seeking/fig4.csv",
        help="Input CSV file",
    )
    parser.add_argument(
        "--output_file",
        default="evaluations/clarification_seeking/Figure4.pdf",
        help="Output figure file",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of raw model names to include",
    )
    parser.add_argument(
        "--sort_by",
        choices=["stage1", "model"],
        default="stage1",
        help="Sorting method",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df.columns = df.columns.str.strip()

    required_columns = {"model", "pct_stage1_clarify"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    df = df[~df["model"].isin(EXCLUDE_MODELS)].copy()

    if args.models:
        requested = set(args.models)
        available = set(df["model"].tolist())
        not_found = sorted(requested - available)
        if not_found:
            raise ValueError(f"These models were not found in the CSV: {not_found}")
        df = df[df["model"].isin(args.models)].copy()

    if df.empty:
        raise ValueError("No data after filtering models.")

    df["pretty_model"] = df["model"].apply(pretty_model_name)

    if args.sort_by == "stage1":
        df = df.sort_values(by="pct_stage1_clarify", ascending=False)
    else:
        df = df.sort_values(by="pretty_model", ascending=True)

    models = df["pretty_model"].tolist()
    values = df["pct_stage1_clarify"].tolist()
    n_models = len(models)

    if len(MODEL_COLORS) < n_models:
        raise ValueError(
            f"Not enough colors in MODEL_COLORS: got {len(MODEL_COLORS)}, need {n_models}."
        )

    x_center = 0.0
    total_group_width = 2.2
    bar_width = total_group_width / n_models * 0.9
    offsets = np.linspace(
        -total_group_width / 2 + bar_width / 2,
        total_group_width / 2 - bar_width / 2,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(5, 4.8))

    for i, (model, value) in enumerate(zip(models, values)):
        ax.bar(
            x_center + offsets[i],
            value,
            width=bar_width,
            color=MODEL_COLORS[i],
            label=model,
        )

    ymax = max(values) if values else 0
    ymax = max(5, np.ceil((ymax + 3) / 5) * 5)

    left_edge = x_center + offsets[0] - bar_width / 2
    right_edge = x_center + offsets[-1] + bar_width / 2

    ax.set_xlim(left_edge - 0.02, right_edge + 0.5)
    ax.set_ylim(0, 100)

    ax.set_xticks([])
    ax.set_ylabel("% of samples", fontsize=13)
    ax.set_xlabel("# Clarify at Stage 1", fontsize=13)
    ax.tick_params(axis="y", labelsize=10)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=12,
        borderpad=0.4,
        labelspacing=0.2,
        handletextpad=0.4,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(args.output_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()