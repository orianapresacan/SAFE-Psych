import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = "evaluations/abstention_rate/abstention_summary.csv"
CI_CSV_PATH = "evaluations/multiple_seed_runs/scatter_axis_ci_by_model.csv"

MODEL_LABELS = {
    "gemma-3-27b": "Gemma 3 27B",
    "gemma-3-4b": "Gemma 3 4B",
    "claude-opus-4.6": "Claude 4.6",
    "gpt-5.4": "GPT 5.4",
    "gemini-2.5-flash-no-thinking": "Gemini Flash 2.5",
    "med42-8b": "Med42 8B",
    "medgemma-27b": "MedGemma 27B",
    "mistral-small-3.1-24B": "Mistral 3.1 24B",
    "qwen_3-32b": "Qwen 3 32B",
}

CI_MODELS = {
    "gpt-5.4",
    "med42-8b",
    "mistral-small-3.1-24B",
    "qwen_3-32b",
}

EXCLUDE_MODELS_FROM_PLOTS = {
    "gemini-2.5-flash-thinking",
    "gemini-2.5-flash-thinking2",
}

SUFFICIENCY_ORDER = ["insufficient", "sufficient"]

SUFFICIENCY_LABELS = {
    "insufficient": "Insufficient information (↑ better)",
    "sufficient": "Sufficient information (↓ better)",
}

COLORS_TWO_BAR = {
    "insufficient": "#007a35",
    "sufficient": "#A70101",
}

FIGURES = [
    {
        "scenario": "full_info",
        "out": "evaluations/abstention_rate/fig3_full_info_abstention_aware",
        "strategies": ["full_info_abstention_aware"],
    },
    {
        "scenario": "sequential_info",
        "out": "evaluations/abstention_rate/fig3_sequential_info_abstention_aware",
        "strategies": ["sequential_info_abstention_aware"],
    },
]


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def get_model_order(sdf, sort_strategy):
    model_order_df = (
        sdf[
            (sdf["strategy"] == sort_strategy)
            & (sdf["final_human_sufficiency"] == "insufficient")
        ]
        .sort_values("abstention_rate", ascending=False)
    )
    return model_order_df["model"].tolist()


def build_abstention_ci_lookup(ci_df):
    lookup = {}

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

    for _, row in ci_df.iterrows():
        model = row["model"]

        if model not in CI_MODELS:
            continue

        lookup[(model, "insufficient")] = {
            # For insufficient cases, the bar shows abstention rate,
            # while the CI file stores under-abstention = 100 - abstention rate.
            # Convert under-abstention CI back to abstention-rate CI.
            "mean": 100 - row["mean_under_abstention_pct"],
            "lower": 100 - row["under_t_ci95_upper_pct"],
            "upper": 100 - row["under_t_ci95_lower_pct"],
        }

        lookup[(model, "sufficient")] = {
            # For sufficient cases, the bar shows over-abstention,
            # which is the same as abstention rate.
            "mean": row["mean_over_abstention_pct"],
            "lower": row["over_t_ci95_lower_pct"],
            "upper": row["over_t_ci95_upper_pct"],
        }

    return lookup

def should_use_ci_for_config(scenario, strategies):
    """
    Add CI bars only to the main full-info abstention-aware figure.
    """
    return (
        scenario == "full_info"
        and strategies == ["full_info_abstention_aware"]
    )


def plot_figure(df, config, ci_lookup=None):
    scenario = config["scenario"]
    strategies = config["strategies"]

    sdf = df[df["scenario"] == scenario].copy()
    models = get_model_order(sdf, strategies[0])
    models = [m for m in models if m not in EXCLUDE_MODELS_FROM_PLOTS]

    if not models:
        print(f"Warning: no models found for scenario={scenario}, strategy={strategies[0]}")
        return

    fig, ax = plt.subplots(figsize=(8.5, 2.7))

    x = list(range(len(models)))
    total_bars = len(strategies) * len(SUFFICIENCY_ORDER)
    width = 0.76 / total_bars

    use_ci_for_this_figure = should_use_ci_for_config(
        scenario=scenario,
        strategies=strategies,
    )

    for s_idx, strategy in enumerate(strategies):
        for i, sufficiency in enumerate(SUFFICIENCY_ORDER):
            values = []

            ci_offsets = []
            ci_values = []
            ci_lower_errors = []
            ci_upper_errors = []

            use_ci_for_this_bar_group = (
                ci_lookup is not None
                and use_ci_for_this_figure
                and strategy == "full_info_abstention_aware"
            )

            bar_index = s_idx * len(SUFFICIENCY_ORDER) + i
            offsets = np.array([
                pos + (bar_index - total_bars / 2) * width + width / 2
                for pos in x
            ])

            for model_idx, model in enumerate(models):
                row = sdf[
                    (sdf["model"] == model)
                    & (sdf["strategy"] == strategy)
                    & (sdf["final_human_sufficiency"] == sufficiency)
                ]

                if row.empty:
                    values.append(np.nan)
                    continue

                value = row["abstention_rate"].iloc[0] * 100

                if (
                    use_ci_for_this_bar_group
                    and model in CI_MODELS
                    and (model, sufficiency) in ci_lookup
                ):
                    ci = ci_lookup[(model, sufficiency)]

                    value = ci["mean"]
                    lower = ci["lower"]
                    upper = ci["upper"]

                    ci_offsets.append(offsets[model_idx])
                    ci_values.append(value)
                    ci_lower_errors.append(max(0.0, value - lower))
                    ci_upper_errors.append(max(0.0, upper - value))

                values.append(value)

            values = np.array(values, dtype=float)

            ax.bar(
                offsets,
                values,
                width=width,
                label=SUFFICIENCY_LABELS[sufficiency],
                color=COLORS_TWO_BAR[sufficiency],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )

            if len(ci_offsets) > 0:
                ax.errorbar(
                    np.array(ci_offsets),
                    np.array(ci_values),
                    yerr=np.vstack([
                        np.array(ci_lower_errors),
                        np.array(ci_upper_errors),
                    ]),
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.9,
                    capsize=4.5,
                    capthick=0.9,
                    alpha=0.9,
                    zorder=10,
                    clip_on=False,
                )

    nice_labels = [MODEL_LABELS.get(m, m) for m in models]

    ax.set_xticks(x)
    ax.set_xticklabels(nice_labels, rotation=28, ha="right")
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Abstention rate (%)")

    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=2,
        handlelength=1.5,
        columnspacing=1.4,
        labelspacing=0.6,
        fontsize=10.5,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1.28])

    pdf_path = f"{config['out']}.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {pdf_path}")


def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    df = df[~df["model"].isin(EXCLUDE_MODELS_FROM_PLOTS)].copy()

    ci_lookup = None

    try:
        ci_df = pd.read_csv(CI_CSV_PATH)
        ci_lookup = build_abstention_ci_lookup(ci_df)
        print(f"Loaded CI data from {CI_CSV_PATH}")

        loaded_ci_models = sorted({model for model, _ in ci_lookup.keys()})
        print(f"CI bars will be added only for: {loaded_ci_models}")

    except FileNotFoundError:
        print(f"No CI file found at {CI_CSV_PATH}; plotting without CI bars.")

    for config in FIGURES:
        plot_figure(df, config, ci_lookup=ci_lookup)


if __name__ == "__main__":
    main()