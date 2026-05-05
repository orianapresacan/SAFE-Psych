#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


GT_PATH = "evaluations/data_gt.json"
OUT_DIR = Path("evaluations/dataset_analysis")
TOP_N_DIAGNOSES = 15

STEP_ORDER = ["S1", "S2", "S3", "S4", "Final S"]


# -------------------------
# Plot settings
# -------------------------
AGE_COLOR = "#AD94BB"
DIAGNOSIS_COLOR = "#577292"
ACTION_COLORS = ["#8EA1AD", "#9CC75B", "#DD615D"]

AGE_BAR_WIDTH = 0.75
DIAGNOSIS_BAR_WIDTH = 0.75
ACTION_BAR_WIDTH = 0.75

AGE_FONT_SIZE = 14
DIAGNOSIS_FONT_SIZE = 14
ACTION_FONT_SIZE = 14
ACTION_LEGEND_FONT_SIZE = 12


def clean(x):
    if x is None:
        return ""
    return str(x).strip()


def apply_font_sizes(ax, font_size):
    ax.yaxis.label.set_size(font_size)
    ax.xaxis.label.set_size(font_size)

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)


def get_plot_step(step, total_steps):
    if step == total_steps:
        return "Final S"
    return f"S{step}"


def is_sufficient(item):
    return clean(item.get("diagnosis_sufficiency")) == "1"


def get_num_sections(item):
    return int(item.get("num_sections_in_input", 0))


def get_tau_exp(item):
    T_i = get_num_sections(item)

    if is_sufficient(item):
        value = clean(item.get("earliest_sufficient_section"))
        if value == "":
            return None
        return int(float(value))

    return T_i


def derive_action_labels(item):
    T_i = get_num_sections(item)
    tau = get_tau_exp(item)

    if T_i == 0 or tau is None:
        return []

    labels = []

    for step in range(1, T_i + 1):
        if not is_sufficient(item):
            label = "Abstain" if step == T_i else "Clarify"
        else:
            label = "Clarify" if step < tau else "Diagnose"

        labels.append({
            "id": clean(item.get("id")),
            "step_label": get_plot_step(step, T_i),
            "expert_action": label,
        })

    return labels


def plot_age_distribution(df):
    age_counts = (
        df["age_group"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
        .value_counts()
        .sort_index()
    )

    print("\nAge group distribution:")
    print(age_counts.to_string())

    fig, ax = plt.subplots(figsize=(4.5, 4))
    age_counts.plot(kind="bar", ax=ax, color=AGE_COLOR, width=AGE_BAR_WIDTH)

    ax.set_ylabel("Number of patients", fontsize=AGE_FONT_SIZE)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    ax.set_xlabel("")
    ax.set_xticklabels(age_counts.index, rotation=45, ha="right")

    apply_font_sizes(ax, AGE_FONT_SIZE)

    plt.tight_layout()

    out_path = OUT_DIR / "AppendixFigure1a.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")


def plot_top_diagnoses(df):
    diagnoses = (
        df["diagnosis"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    diagnoses = diagnoses[diagnoses != ""]

    diagnosis_counts = (
        diagnoses
        .str.upper()
        .value_counts()
        .head(TOP_N_DIAGNOSES)
    )

    print(f"\nTop {TOP_N_DIAGNOSES} diagnoses:")
    print(diagnosis_counts.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    diagnosis_counts.plot(
        kind="bar",
        ax=ax,
        color=DIAGNOSIS_COLOR,
        width=DIAGNOSIS_BAR_WIDTH,
    )

    ax.set_ylabel("Number of patients", fontsize=DIAGNOSIS_FONT_SIZE)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    ax.set_xlabel("")
    ax.set_xticklabels(diagnosis_counts.index, rotation=45, ha="right")

    apply_font_sizes(ax, DIAGNOSIS_FONT_SIZE)

    plt.tight_layout()

    out_path = OUT_DIR / "AppendixFigure1b.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")


def plot_action_labels(gt):
    action_rows = []

    for item in gt:
        action_rows.extend(derive_action_labels(item))

    action_df = pd.DataFrame(action_rows)

    action_counts = (
        action_df
        .groupby(["step_label", "expert_action"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=STEP_ORDER, columns=["Clarify", "Diagnose", "Abstain"], fill_value=0)
    )

    action_percent = action_counts.div(action_counts.sum(axis=1), axis=0) * 100
    action_percent = action_percent.reindex(STEP_ORDER)

    print("\nAction label counts by step:")
    print(action_counts.to_string())

    print("\nAction label percentages by step:")
    print(action_percent.round(2).to_string())

    fig, ax = plt.subplots(figsize=(5, 4))

    action_percent.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=ACTION_BAR_WIDTH,
        color=ACTION_COLORS,
    )

    ax.set_xticks(range(len(action_percent.index)))
    ax.set_xticklabels(action_percent.index, rotation=0)

    ax.set_xlabel("")
    ax.xaxis.label.set_visible(False)

    ax.set_ylabel("Percentage of labels", fontsize=ACTION_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100)

    apply_font_sizes(ax, ACTION_FONT_SIZE)

    ax.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        fontsize=ACTION_LEGEND_FONT_SIZE,
    )

    plt.tight_layout()

    out_path = OUT_DIR / "AppendixFigure2b.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = json.loads(Path(GT_PATH).read_text(encoding="utf-8"))
    df = pd.DataFrame(gt)

    plot_age_distribution(df)
    plot_top_diagnoses(df)
    plot_action_labels(gt)


if __name__ == "__main__":
    main()