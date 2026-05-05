#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

import pandas as pd


STRATEGIES = [
    "full_info_no_abstention",
    "full_info_abstention_aware",
    "sequential_info_neutral_prompting",
    "sequential_info_abstention_aware",
    "sequential_info_clarification_only",
    "structured_actions",
]


def parse_model_and_strategy(folder_name: str):
    """
    Extract model and strategy from folder names like:
      gpt-5.4_full_info_abstention_aware
      med42-8b_sequential_info_neutral_prompting

    Returns:
      model, strategy
    """
    for strategy in sorted(STRATEGIES, key=len, reverse=True):
        marker = f"_{strategy}"

        if folder_name.endswith(marker):
            model = folder_name[: -len(marker)]
            return model, strategy

        if marker in folder_name:
            model = folder_name.split(marker)[0]
            return model, strategy

    return folder_name, "unknown"


def seconds_to_hms(seconds):
    seconds = float(seconds)

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize runtime from timing.json files."
    )
    parser.add_argument(
        "--runs_dir",
        default="runs",
        help="Directory containing model/strategy run folders.",
    )
    parser.add_argument(
        "--out_csv",
        default="evaluations/computation_time/runtime_summary.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_csv = Path(args.out_csv)

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir.resolve()}")

    timing_files = sorted(runs_dir.glob("*/timing.json"))

    if not timing_files:
        raise FileNotFoundError(
            f"No timing.json files found under {runs_dir.resolve()}"
        )

    rows = []

    for timing_path in timing_files:
        folder = timing_path.parent.name
        model, strategy = parse_model_and_strategy(folder)

        with timing_path.open("r", encoding="utf-8") as f:
            timing = json.load(f)

        total_time_seconds = timing.get("total_time_seconds")
        num_samples = timing.get("num_samples")

        if total_time_seconds is None:
            print(f"Warning: missing total_time_seconds in {timing_path}")
            continue

        total_time_seconds = float(total_time_seconds)

        if num_samples is not None:
            num_samples = int(num_samples)
            seconds_per_sample = (
                total_time_seconds / num_samples if num_samples > 0 else None
            )
        else:
            seconds_per_sample = None

        rows.append({
            "model": model,
            "strategy": strategy,
            "run_folder": folder,
            "num_samples": num_samples,
            "total_time_seconds": total_time_seconds,
            "total_time_minutes": total_time_seconds / 60,
            "total_time_hours": total_time_seconds / 3600,
            "total_time_hms": seconds_to_hms(total_time_seconds),
            "seconds_per_sample": seconds_per_sample,
            "minutes_per_sample": (
                seconds_per_sample / 60
                if seconds_per_sample is not None
                else None
            ),
            "timing_file": str(timing_path),
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(["strategy", "model"]).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Wrote {out_csv}")
    print("\nRuntime summary:")
    print(
        df[
            [
                "model",
                "strategy",
                "num_samples",
                "total_time_hms",
                "total_time_hours",
                "seconds_per_sample",
            ]
        ].to_string(index=False)
    )

    print("\nRuntime by strategy:")
    strategy_summary = (
        df.groupby("strategy", as_index=False)
        .agg(
            n_runs=("model", "count"),
            total_time_hours=("total_time_hours", "sum"),
            mean_time_hours=("total_time_hours", "mean"),
            mean_seconds_per_sample=("seconds_per_sample", "mean"),
        )
        .sort_values("strategy")
    )
    print(strategy_summary.to_string(index=False))


if __name__ == "__main__":
    main()