#!/usr/bin/env python3
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("evaluations/premature_diagnoses/per_sample_diagnosis_timing_by_strategy.csv")
GT_PATH = Path("evaluations/data_gt.json")
OUT_DIR = Path("evaluations/premature_diagnoses")

STRATEGIES = ["neutral", "abstention_aware"]

LINE_STYLES = {
    "neutral": "-",
    "abstention_aware": "-",
    "expert": "--",
}

MARKERS = {
    "neutral": "o",
    "abstention_aware": "o",
    "expert": "s",
}

LINE_WIDTHS = {
    "neutral": 2.2,
    "abstention_aware": 2.2,
    "expert": 2,
}

STRATEGY_LABELS = {
    "neutral": "Neutral",
    "abstention_aware": "Abstention",
    "expert": "Expert sufficient",
}

STAGE_ORDER = [1, 2, 3, 4, 5]
STAGE_LABELS = ["S1", "S2", "S3", "S4", "Final"]

COLORS = {
    "neutral": "#B8560C",
    "abstention_aware": "#615CA5",
    "expert": "#686868",
}

TIMING_ORDER = ["premature", "on_time", "late", "never"]
TIMING_LABELS = {
    "premature": "Premature",
    "on_time": "On-time",
    "late": "Late",
    "never": "Never",
}

TIMING_COLORS = {
    "premature": "#E94C4C",
    "on_time": "#369734",
    "late": "#2781E9",
    "never": "#ACACAC",
}


def clean(x):
    if x is None:
        return ""
    return str(x).strip()


def load_data():
    df = pd.read_csv(CSV_PATH)

    df = df[df["strategy"].isin(STRATEGIES)].copy()

    df["model_first_diagnosis_step"] = pd.to_numeric(
        df["model_first_diagnosis_step"],
        errors="coerce",
    )

    return df


def load_expert_steps():
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))

    rows = []

    for item in gt:
        sample_id = clean(item.get("id"))
        sufficiency = clean(item.get("diagnosis_sufficiency"))
        earliest = clean(item.get("earliest_sufficient_section"))
        total_steps = clean(item.get("num_sections_in_input"))

        if not sample_id or not total_steps:
            continue

        total_steps = int(float(total_steps))

        if sufficiency == "1":
            if earliest == "":
                continue
            expert_step = int(float(earliest))
        elif sufficiency == "0":
            expert_step = total_steps
        else:
            continue

        if expert_step == total_steps:
            expert_step = 5

        rows.append({
            "sample_id": sample_id,
            "expert_step": expert_step,
        })

    return pd.DataFrame(rows)


def plot_cumulative_diagnosis_curve(df):
    rows = []

    for strategy in STRATEGIES:
        sdf = df[df["strategy"] == strategy].copy()
        total = len(sdf)

        for stage in STAGE_ORDER:
            diagnosed_by_stage = (
                sdf["model_first_diagnosis_step"].notna()
                & (sdf["model_first_diagnosis_step"] <= stage)
            ).sum()

            rows.append({
                "strategy": strategy,
                "stage": stage,
                "pct_diagnosed": diagnosed_by_stage / total * 100 if total else 0,
            })

    expert_df = load_expert_steps()
    total_expert = len(expert_df)

    for stage in STAGE_ORDER:
        expert_sufficient_by_stage = (expert_df["expert_step"] <= stage).sum()

        rows.append({
            "strategy": "expert",
            "stage": stage,
            "pct_diagnosed": expert_sufficient_by_stage / total_expert * 100 if total_expert else 0,
        })

    plot_df = pd.DataFrame(rows)

    print("\nCumulative diagnosis/sufficiency percentages:")
    print(
        plot_df
        .pivot(index="stage", columns="strategy", values="pct_diagnosed")
        .round(2)
        .to_string()
    )

    fig, ax = plt.subplots(figsize=(4, 4.2))

    for strategy in ["neutral", "abstention_aware", "expert"]:
        sdf = plot_df[plot_df["strategy"] == strategy]

        ax.plot(
            sdf["stage"],
            sdf["pct_diagnosed"],
            marker=MARKERS[strategy],
            linestyle=LINE_STYLES[strategy],
            linewidth=LINE_WIDTHS[strategy],
            color=COLORS[strategy],
            label=STRATEGY_LABELS[strategy],
        )

    ax.set_xticks(STAGE_ORDER)
    ax.set_xticklabels(STAGE_LABELS)

    ax.set_ylim(0, 105)
    ax.set_ylabel("Cumulative cases (%)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.03, 0.65),
        fontsize=12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out_path = OUT_DIR / "Figure6.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")


def plot_timing_breakdown(df):
    summary = (
        df.groupby(["strategy", "timing_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=STRATEGIES, columns=TIMING_ORDER, fill_value=0)
    )

    summary_pct = summary.div(summary.sum(axis=1), axis=0) * 100

    print("\nTiming breakdown counts:")
    print(summary.to_string())

    print("\nTiming breakdown percentages:")
    print(summary_pct.round(2).to_string())

    fig, ax = plt.subplots(figsize=(4, 4.2))

    summary_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=0.6,
        color=[TIMING_COLORS[c] for c in TIMING_ORDER],
    )

    ax.set_xticklabels(
        [STRATEGY_LABELS[s] for s in summary_pct.index],
        rotation=0,
        ha="center",
        fontsize=12,
    )

    ax.set_ylabel("Cumulative cases reaching decision (%)", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=10)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    labels = [TIMING_LABELS[l] for l in labels]

    ax.legend(
        handles,
        labels,
        title="",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        fontsize=12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out_path = OUT_DIR / "Figure5.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    plot_cumulative_diagnosis_curve(df)
    plot_timing_breakdown(df)


if __name__ == "__main__":
    main()