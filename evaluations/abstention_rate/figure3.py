import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = "evaluations/abstention_rate/abstention_summary.csv"

STRATEGIES = {
    "full_info": "full_info_abstention_aware",
    "sequential_info": "sequential_info_abstention_aware",
}

OUT_SCATTER_FILES = {
    "full_info": "evaluations/abstention_rate/figure3_full_info.pdf",
    "sequential_info": "evaluations/abstention_rate/figure3_sequential_info.pdf",
}

MODEL_LABELS = {
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

EXCLUDE_MODELS_FROM_SCATTER = {
    "gemini-2.5-flash-thinking",
    "gemini-2.5-flash-thinking2",
}

SCATTER_POINT_COLOR = "#576F91"
SCATTER_POINT_CI_COLOR = "#23223F"
SCATTER_POINT_SIZE = 92
SCATTER_LABEL_SIZE = 14

LABEL_OFFSETS_FULL_INFO  = {
    "mistral-small-3.1-24B": (-4, 12),
    "gemini-2.5-flash-no-thinking": (-5, 25),
    "gemini-2.5-flash-thinking": (-15, 18),
    "gemini-2.5-flash-thinking2": (18, 18),
    "qwen_3-32b": (-36, -5),
    "medgemma-27b": (10, 40),
    "gpt-5.4": (-25, -10),
    "claude-opus-4.6": (18, 20),
    "gemma-3-4b": (18, 16),
    "med42-8b": (18, -4),
    "gemma-3-27b": (-15, -15),
}

LABEL_OFFSETS_SEQUENTIAL_INFO = {
    # crowded upper-left cluster
    "gemini-2.5-flash-no-thinking": (1, 20),
    "mistral-small-3.1-24B": (-5, -15),

    # middle
    "gpt-5.4": (-28, -18),

    # lower-right cluster
    "qwen_3-32b": (-28, -8),
    "gemma-3-27b": (-18, -28),
    "medgemma-27b": (0, 38),
    "gemma-3-4b": (20, 10),
    "med42-8b": (22, 2),
    "claude-opus-4.6": (32, -12),
}

CI_CSV_PATH = "evaluations/multiple_seed_runs/scatter_axis_ci_by_model.csv"

CI_MODELS = {
    "gpt-5.4",
    "med42-8b",
    "mistral-small-3.1-24B",
    "qwen_3-32b",
}

def build_scatter_ci_lookup(ci_df):
    ci_df.columns = ci_df.columns.str.strip()

    required_cols = [
        "model",
        "mean_under_abstention_pct",
        "under_t_ci95_lower_pct",
        "under_t_ci95_upper_pct",
        "mean_over_abstention_pct",
        "over_t_ci95_lower_pct",
        "over_t_ci95_upper_pct",
    ]

    missing = [c for c in required_cols if c not in ci_df.columns]
    if missing:
        raise ValueError(f"CI file is missing required columns: {missing}")

    lookup = {}

    for _, row in ci_df.iterrows():
        model = row["model"]

        if model not in CI_MODELS:
            continue

        x = row["mean_under_abstention_pct"]
        y = row["mean_over_abstention_pct"]

        lookup[model] = {
            "x_lower": max(0.0, x - row["under_t_ci95_lower_pct"]),
            "x_upper": max(0.0, row["under_t_ci95_upper_pct"] - x),
            "y_lower": max(0.0, y - row["over_t_ci95_lower_pct"]),
            "y_upper": max(0.0, row["over_t_ci95_upper_pct"] - y),
        }

    return lookup

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def make_plot_df(df, scenario):
    strategy = STRATEGIES[scenario]

    sdf = df[
        (df["scenario"] == scenario)
        & (df["strategy"] == strategy)
    ].copy()

    if sdf.empty:
        print(f"Warning: no rows found for scenario={scenario}, strategy={strategy}")

    insuff = sdf[sdf["final_human_sufficiency"] == "insufficient"][
        ["model", "abstention_rate"]
    ].rename(columns={"abstention_rate": "abstention_insufficient"})

    suff = sdf[sdf["final_human_sufficiency"] == "sufficient"][
        ["model", "abstention_rate"]
    ].rename(columns={"abstention_rate": "abstention_sufficient"})

    plot_df = insuff.merge(suff, on="model", how="inner")

    plot_df["under_abstention"] = (1 - plot_df["abstention_insufficient"]) * 100
    plot_df["over_abstention"] = plot_df["abstention_sufficient"] * 100

    return plot_df


def plot_tradeoff_scatter(df, scenario, scatter_ci=None):
    plot_df = make_plot_df(df, scenario)
    plot_df = plot_df[~plot_df["model"].isin(EXCLUDE_MODELS_FROM_SCATTER)].copy()

    if plot_df.empty:
        print(f"Skipping {scenario}: no paired sufficient/insufficient rows found.")
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for _, row in plot_df.iterrows():
        model = row["model"]
        x = row["under_abstention"]
        y = row["over_abstention"]

        if scenario == "full_info" and scatter_ci is not None and model in scatter_ci:
            ci = scatter_ci[model]

            ax.errorbar(
                x,
                y,
                xerr=[[ci["x_lower"]], [ci["x_upper"]]],
                yerr=[[ci["y_lower"]], [ci["y_upper"]]],
                fmt="o",
                markersize=8,
                color=SCATTER_POINT_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.8,
                ecolor=SCATTER_POINT_CI_COLOR,
                elinewidth=1.2,
                capsize=5,
                capthick=1.2,
                zorder=3,
            )
        else:
            ax.scatter(
                x,
                y,
                s=SCATTER_POINT_SIZE,
                color=SCATTER_POINT_COLOR,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

    for _, row in plot_df.iterrows():
        model = row["model"]
        label = MODEL_LABELS.get(model, model)
        x = row["under_abstention"]
        y = row["over_abstention"]

        label_offsets = (
            LABEL_OFFSETS_FULL_INFO
            if scenario == "full_info"
            else LABEL_OFFSETS_SEQUENTIAL_INFO
        )

        dx, dy = label_offsets.get(model, (8, 8))

        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=SCATTER_LABEL_SIZE,
            ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                color="#9A9A9A",
                linewidth=0.55,
                shrinkA=3,
                shrinkB=4,
            ),
        )

    ax.set_xlim(0, 112)
    ax.set_ylim(-6, 105)
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))

    ax.set_xlabel("Under-abstention (%)")
    ax.set_ylabel("Over-abstention (%)")

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)

    plt.tight_layout()

    out_path = OUT_SCATTER_FILES[scenario]
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}")


def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    scatter_ci = None

    try:
        ci_df = pd.read_csv(CI_CSV_PATH)
        scatter_ci = build_scatter_ci_lookup(ci_df)
        print(f"Loaded CI data from {CI_CSV_PATH}")
        print("CI models:", sorted(scatter_ci.keys()))
    except FileNotFoundError:
        print(f"No CI file found at {CI_CSV_PATH}; plotting without CI bars.")

    print("Available scenarios:", sorted(df["scenario"].unique()))
    print("Available strategies:", sorted(df["strategy"].unique()))

    for scenario in ["full_info", "sequential_info"]:
        plot_tradeoff_scatter(df, scenario, scatter_ci=scatter_ci)


if __name__ == "__main__":
    main()