import json
import glob
from pathlib import Path

import pandas as pd


GT_PATH = "evaluations/data_gt.json"
RUNS_DIR = "runs"

OUT_CSV = "evaluations/abstention_rate/abstention_summary.csv"
OUT_PER_CASE_CSV = "evaluations/abstention_rate/abstention_per_case.csv"
OUT_EMPTY_QC_CSV = "evaluations/abstention_rate/empty_output_qc.csv"


STRATEGIES = {
    "full_info_no_abstention": {
        "scenario": "full_info",
        "metric_type": "full_note_response",
    },
    "full_info_abstention_aware": {
        "scenario": "full_info",
        "metric_type": "full_note_response",
    },
    "sequential_info_neutral_prompting": {
        "scenario": "sequential_info",
        "metric_type": "trajectory_level",
    },
    "sequential_info_abstention_aware": {
        "scenario": "sequential_info",
        "metric_type": "trajectory_level",
    },
}


def load_json_or_jsonl(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    obj = json.loads(text)

    if isinstance(obj, list):
        return obj

    return [obj]


def is_empty_text(value):
    return not isinstance(value, str) or not value.strip()


def get_final_sufficiency(gt_item):
    row = gt_item.get("original_row", gt_item)
    value = str(row.get("diagnosis_sufficiency", "")).strip()

    if value == "1":
        return "sufficient"

    if value == "0":
        return "insufficient"

    return "unknown"


def infer_model_from_path(file, strategy):
    """
    Infers model name from folder name.

    Example:
      gemini-2.5-flash-thinking2_sequential_info_abstention_aware
      -> gemini-2.5-flash-thinking2
    """
    folder = Path(file).parent.name
    marker = f"_{strategy}"

    if marker in folder:
        return folder.split(marker)[0]

    return folder


def trajectory_ever_diagnosed(run_item):
    if run_item.get("stopping_reason") == "Diagnose":
        return True

    for step in run_item.get("trajectory", []):
        judge_action = step.get("judge_action")

        structured = step.get("structured_action_raw")
        if not isinstance(structured, dict):
            structured = {}

        structured_action = structured.get("action")

        if judge_action == "Diagnose" or structured_action == "Diagnose":
            return True

    return False


def get_empty_output_stats(run_item):
    empty_model_outputs = 0
    empty_judge_outputs = 0
    empty_steps = []

    trajectory = run_item.get("trajectory", [])
    if not isinstance(trajectory, list):
        trajectory = []

    for step_idx, step in enumerate(trajectory, start=1):
        model_empty = is_empty_text(step.get("model_output_raw"))
        judge_empty = is_empty_text(step.get("judge_output_raw"))

        if model_empty:
            empty_model_outputs += 1

        if judge_empty:
            empty_judge_outputs += 1

        if model_empty or judge_empty:
            empty_steps.append({
                "step": step.get("step_index", step_idx),
                "model_empty": model_empty,
                "judge_empty": judge_empty,
            })

    return {
        "n_steps_checked": len(trajectory),
        "empty_model_outputs": empty_model_outputs,
        "empty_judge_outputs": empty_judge_outputs,
        "has_empty_model_output": int(empty_model_outputs > 0),
        "has_empty_judge_output": int(empty_judge_outputs > 0),
        "empty_steps": json.dumps(empty_steps, ensure_ascii=False),
    }


def main():
    gt = json.loads(Path(GT_PATH).read_text(encoding="utf-8"))

    sufficiency_by_id = {
        str(item["id"]): get_final_sufficiency(item)
        for item in gt
        if get_final_sufficiency(item) != "unknown"
    }

    print("GT cases with known final sufficiency:", len(sufficiency_by_id))
    print(pd.Series(sufficiency_by_id).value_counts())

    files = glob.glob(f"{RUNS_DIR}/**/trajectories*.json*", recursive=True)

    if not files:
        raise FileNotFoundError(
            f"No trajectories*.json/jsonl files found under {RUNS_DIR}"
        )

    print(f"\nFound {len(files)} trajectory files under {RUNS_DIR}")

    rows = []

    seen_models_from_record = set()
    seen_models_from_path = set()
    seen_strategies = set()
    skipped_strategies = set()

    thinking2_records_seen = 0
    thinking2_records_kept = 0

    for file in files:
        records = load_json_or_jsonl(file)

        for rec in records:
            record_model = rec.get("model_evaluated")
            strategy = rec.get("strategy")

            seen_models_from_record.add(record_model)
            seen_strategies.add(strategy)

            if strategy not in STRATEGIES:
                skipped_strategies.add(strategy)
                continue

            model = infer_model_from_path(file, strategy)
            seen_models_from_path.add(model)

            if "thinking2" in str(record_model).lower() or "thinking2" in model.lower():
                thinking2_records_seen += 1

            sample_id = str(rec.get("sample_id"))
            if sample_id not in sufficiency_by_id:
                continue

            final_sufficiency = sufficiency_by_id[sample_id]

            ever_diagnosed = trajectory_ever_diagnosed(rec)
            abstained = not ever_diagnosed

            if final_sufficiency == "insufficient":
                behavior_label = (
                    "appropriate_abstention"
                    if abstained
                    else "failure_to_abstain"
                )
            elif final_sufficiency == "sufficient":
                behavior_label = (
                    "over_abstention"
                    if abstained
                    else "appropriate_diagnosis"
                )
            else:
                behavior_label = "unknown"

            empty_stats = get_empty_output_stats(rec)

            rows.append({
                "model": model,
                "model_evaluated_raw": record_model,
                "strategy": strategy,
                "scenario": STRATEGIES[strategy]["scenario"],
                "metric_type": STRATEGIES[strategy]["metric_type"],
                "sample_id": sample_id,
                "final_human_sufficiency": final_sufficiency,
                "abstained": int(abstained),
                "ever_diagnosed": int(ever_diagnosed),
                "behavior_label": behavior_label,
                "num_steps_executed": rec.get("num_steps_executed"),
                "stopping_reason": rec.get("stopping_reason"),
                "source_file": file,
                **empty_stats,
            })

            if "thinking2" in str(record_model).lower() or "thinking2" in model.lower():
                thinking2_records_kept += 1

    per_case = pd.DataFrame(rows)

    if per_case.empty:
        print("\nModels seen from record field:")
        print(sorted(map(str, seen_models_from_record)))

        print("\nModels inferred from path:")
        print(sorted(map(str, seen_models_from_path)))

        print("\nStrategies seen:")
        print(sorted(map(str, seen_strategies)))

        print("\nSkipped strategies:")
        print(sorted(map(str, skipped_strategies)))

        raise ValueError("No matching model outputs found for selected strategies.")

    per_case["valid_for_metrics"] = (
        (per_case["has_empty_model_output"] == 0)
        & (per_case["has_empty_judge_output"] == 0)
    )

    per_case_valid = per_case[per_case["valid_for_metrics"]].copy()

    summary = (
        per_case_valid
        .groupby(
            [
                "scenario",
                "metric_type",
                "strategy",
                "model",
                "final_human_sufficiency",
            ],
            as_index=False,
        )
        .agg(
            n_cases=("sample_id", "nunique"),
            n_abstained=("abstained", "sum"),
            n_diagnosed=("ever_diagnosed", "sum"),
            abstention_rate=("abstained", "mean"),
            diagnosis_rate=("ever_diagnosed", "mean"),
            cases_with_empty_model_output=("has_empty_model_output", "sum"),
            cases_with_empty_judge_output=("has_empty_judge_output", "sum"),
            total_empty_model_outputs=("empty_model_outputs", "sum"),
            total_empty_judge_outputs=("empty_judge_outputs", "sum"),
        )
        .sort_values(
            [
                "scenario",
                "strategy",
                "model",
                "final_human_sufficiency",
            ]
        )
    )

    empty_qc = (
        per_case
        .groupby(["model", "strategy"], as_index=False)
        .agg(
            n_cases=("sample_id", "nunique"),
            valid_cases=("valid_for_metrics", "sum"),
            total_steps_checked=("n_steps_checked", "sum"),
            cases_with_empty_model_output=("has_empty_model_output", "sum"),
            cases_with_empty_judge_output=("has_empty_judge_output", "sum"),
            total_empty_model_outputs=("empty_model_outputs", "sum"),
            total_empty_judge_outputs=("empty_judge_outputs", "sum"),
        )
        .sort_values(["model", "strategy"])
    )

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(OUT_CSV, index=False)
    per_case.to_csv(OUT_PER_CASE_CSV, index=False)
    empty_qc.to_csv(OUT_EMPTY_QC_CSV, index=False)

    print("\nModels seen from record field:")
    print(sorted(map(str, seen_models_from_record)))

    print("\nModels inferred from path:")
    print(sorted(map(str, seen_models_from_path)))

    print("\nStrategies seen:")
    print(sorted(map(str, seen_strategies)))

    if skipped_strategies:
        print("\nSkipped strategies:")
        print(sorted(map(str, skipped_strategies)))

    print("\nThinking2 records seen:", thinking2_records_seen)
    print("Thinking2 records kept:", thinking2_records_kept)

    print("\nSummary rows for thinking2:")
    thinking2_summary = summary[
        summary["model"].str.contains("thinking2", case=False, na=False)
    ]
    print(thinking2_summary)

    print(f"\nWrote summary CSV: {OUT_CSV}")
    print(f"Wrote per-case CSV: {OUT_PER_CASE_CSV}")
    print(f"Wrote empty-output QC CSV: {OUT_EMPTY_QC_CSV}")


if __name__ == "__main__":
    main()