import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


DIAG_CSV = Path("evaluations/diagnosis_accuracy/diagnosis_accuracy_per_sample_all_strategies.csv")
ABSTENTION_CSV = Path("evaluations/abstention_rate/abstention_summary.csv")

OUT_DIR = Path("evaluations/reasoning_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = OUT_DIR / "gemini_thinking_summary_single_plot.csv"
OUT_FIG = OUT_DIR / "AppendixFigure6.pdf"


GEMINI_MODELS = {
    "gemini-2.5-flash-no-thinking": {
        "label": "0",
        "budget": 0,
    },
    "gemini-2.5-flash-thinking": {
        "label": "128",
        "budget": 128,
    },
    "gemini-2.5-flash-thinking2": {
        "label": "2048",
        "budget": 2048,
    },
}

THINKING_ORDER = [
    "gemini-2.5-flash-no-thinking",
    "gemini-2.5-flash-thinking",
    "gemini-2.5-flash-thinking2",
]

STRATEGY = "sequential_info_abstention_aware"

# Use either "match_3char" or "match_diagnosis_level".
ACCURACY_COLUMN = "match_3char"
ACCURACY_LABEL = "3-character ICD accuracy"


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def pct_mean(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return np.nan
    return 100 * series.mean()


def load_accuracy():
    df = pd.read_csv(DIAG_CSV)
    df.columns = df.columns.str.strip()

    required_cols = {"model", "strategy", ACCURACY_COLUMN}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Diagnosis CSV missing columns: {sorted(missing)}")

    print("\nDiagnosis CSV models:")
    print(sorted(df["model"].unique()))

    print("\nDiagnosis CSV strategies:")
    print(sorted(df["strategy"].unique()))

    df = df[
        (df["model"].isin(GEMINI_MODELS.keys()))
        & (df["strategy"] == STRATEGY)
    ].copy()

    rows = []

    for model in THINKING_ORDER:
        sub = df[df["model"] == model]

        rows.append({
            "model": model,
            "thinking_label": GEMINI_MODELS[model]["label"],
            "thinking_budget": GEMINI_MODELS[model]["budget"],
            "n_accuracy_cases": len(sub),
            "accuracy_pct": pct_mean(sub[ACCURACY_COLUMN]),
        })

    return pd.DataFrame(rows)


def load_abstention_errors():
    df = pd.read_csv(ABSTENTION_CSV)
    df.columns = df.columns.str.strip()

    required_cols = {
        "model",
        "strategy",
        "final_human_sufficiency",
        "abstention_rate",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Abstention CSV missing columns: {sorted(missing)}")

    print("\nAbstention CSV models:")
    print(sorted(df["model"].unique()))

    print("\nAbstention CSV strategies:")
    print(sorted(df["strategy"].unique()))

    df = df[
        (df["model"].isin(GEMINI_MODELS.keys()))
        & (df["strategy"] == STRATEGY)
    ].copy()

    rows = []

    for model in THINKING_ORDER:
        sub = df[df["model"] == model]

        insufficient = sub[sub["final_human_sufficiency"] == "insufficient"]
        sufficient = sub[sub["final_human_sufficiency"] == "sufficient"]

        abstain_insufficient = (
            100 * insufficient["abstention_rate"].mean()
            if len(insufficient)
            else np.nan
        )

        abstain_sufficient = (
            100 * sufficient["abstention_rate"].mean()
            if len(sufficient)
            else np.nan
        )

        under_abstention = (
            100 - abstain_insufficient
            if not np.isnan(abstain_insufficient)
            else np.nan
        )

        over_abstention = abstain_sufficient

        rows.append({
            "model": model,
            "thinking_label": GEMINI_MODELS[model]["label"],
            "thinking_budget": GEMINI_MODELS[model]["budget"],
            "n_abstention_rows": len(sub),
            "abstention_rate_insufficient_pct": abstain_insufficient,
            "abstention_rate_sufficient_pct": abstain_sufficient,
            "under_abstention_pct": under_abstention,
            "over_abstention_pct": over_abstention,
        })

    return pd.DataFrame(rows)


def plot_single_line(summary):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    x = summary["thinking_budget"].to_numpy()

    ax.plot(
        x,
        summary["accuracy_pct"],
        marker="o",
        linewidth=1.8,
        markersize=6,
        label=ACCURACY_LABEL,
        zorder=3,
    )

    ax.plot(
        x,
        summary["under_abstention_pct"],
        marker="o",
        linewidth=1.8,
        markersize=6,
        label="Under-abstention",
        zorder=3,
    )

    ax.plot(
        x,
        summary["over_abstention_pct"],
        marker="o",
        linewidth=1.8,
        markersize=6,
        label="Over-abstention",
        zorder=3,
    )

    ax.set_xscale("symlog", linthresh=128)
    ax.set_xticks(summary["thinking_budget"])
    ax.set_xticklabels(summary["thinking_label"])

    ax.set_xlabel("Gemini 2.5 Flash thinking budget")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))

    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )

    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_FIG}")


def main():
    acc_df = load_accuracy()
    abst_df = load_abstention_errors()

    summary = acc_df.merge(
        abst_df,
        on=["model", "thinking_label", "thinking_budget"],
        how="outer",
    )

    summary = summary.sort_values("thinking_budget")

    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"\nWrote {OUT_SUMMARY}")

    print("\nGemini thinking summary:")
    print(summary)

    plot_single_line(summary)


if __name__ == "__main__":
    main()