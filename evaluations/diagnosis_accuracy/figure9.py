#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


IN_DIR = Path("evaluations/diagnosis_accuracy")
OUT_DIR = Path("evaluations/diagnosis_accuracy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_DETAIL = IN_DIR / "diagnosis_accuracy_per_sample_all_strategies.csv"
FIG_OUT = OUT_DIR / "Figure9.pdf"

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

GRANULARITY_LABELS = {
    "match_diagnosis_level": "Diagnosis-level ICD",
    "match_3char": "3-character ICD",
    "match_high_class": "High-level class",
}

COLORS = {
    "agreement": "#58AC49",
    "disagreement": "#D64641",
}

FONT_SIZE_LABEL = 14
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 12


def style_axes(ax):
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def validate_columns(df, required_cols):
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def plot_accuracy_by_doctor_agreement_all_granularities():
    df = pd.read_csv(CSV_DETAIL)

    df = df[df["model"].isin(MODEL_NAME_MAP.keys())].copy()
    df = df[df["strategy"] == "full_info_no_abstention"].copy()

    required_cols = {
        "doctor_agreement",
        *GRANULARITY_LABELS.keys(),
    }
    validate_columns(df, required_cols)

    rows = []

    for agreement_status in ["agreement", "disagreement"]:
        sub = df[df["doctor_agreement"] == agreement_status]

        for metric, label in GRANULARITY_LABELS.items():
            accuracy = 100 * sub[metric].sum() / len(sub) if len(sub) else 0

            rows.append({
                "doctor_agreement": agreement_status,
                "granularity": label,
                "accuracy_pct": accuracy,
                "total_cases": len(sub),
            })

    plot_df = pd.DataFrame(rows)

    granularities = list(GRANULARITY_LABELS.values())
    x = np.arange(len(granularities))
    width = 0.35

    agreement_values = (
        plot_df[plot_df["doctor_agreement"] == "agreement"]
        .set_index("granularity")
        .loc[granularities, "accuracy_pct"]
        .values
    )

    disagreement_values = (
        plot_df[plot_df["doctor_agreement"] == "disagreement"]
        .set_index("granularity")
        .loc[granularities, "accuracy_pct"]
        .values
    )

    fig, ax = plt.subplots(figsize=(5, 4.5))

    ax.bar(
        x - width / 2,
        agreement_values,
        width,
        label="Doctor agreement",
        color=COLORS["agreement"],
    )

    ax.bar(
        x + width / 2,
        disagreement_values,
        width,
        label="Doctor disagreement",
        color=COLORS["disagreement"],
    )

    ax.set_ylabel("Diagnosis accuracy (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(
        granularities,
        rotation=20,
        ha="right",
        fontsize=FONT_SIZE_TICK,
    )

    style_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {FIG_OUT}")


def main():
    plot_accuracy_by_doctor_agreement_all_granularities()


if __name__ == "__main__":
    main()