#!/usr/bin/env python3
# clarifying_without_instruction.py

import csv
import json
from pathlib import Path
from collections import Counter


RUNS_DIR = Path("runs")
GT_PATH = Path("evaluations/data_gt.json")
OUT_CSV = Path("evaluations/clarification_seeking/fig4.csv")

TARGET_STRATEGY = "sequential_info_neutral_prompting"


def find_trajectory_files(runs_dir: Path):
    trajectory_files = []

    for folder in runs_dir.iterdir():
        if not folder.is_dir():
            continue

        if TARGET_STRATEGY not in folder.name:
            continue

        traj_path = folder / "trajectories.jsonl"

        if traj_path.exists():
            trajectory_files.append(traj_path)
        else:
            print(f"Warning: missing trajectories.jsonl in {folder}")

    return sorted(trajectory_files)


def extract_model_name(jsonl_path: Path) -> str:
    name = jsonl_path.parent.name

    if TARGET_STRATEGY in name:
        return name.split(f"_{TARGET_STRATEGY}")[0]

    return name


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON in {path} line {line_num}: {e}")


def load_gt_stage1_clarify_ids(gt_path: Path):
    with gt_path.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)

    clarify_ids = set()

    for item in gt_data:
        sample_id = str(item["id"]).strip()
        row = item.get("original_row", item)

        label = str(row.get("presenting_symptoms_label", "")).strip()

        if label in {"", "Clarify"}:
            clarify_ids.add(sample_id)

    return clarify_ids


def get_stage1_action(record: dict):
    for step in record.get("trajectory", []):
        if step.get("step_index") == 1:
            if step.get("new_judge_action"):
                return step.get("new_judge_action")

            if step.get("new_judge_output", {}).get("action"):
                return step["new_judge_output"]["action"]

            return step.get("judge_action")

    return None


def main():
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"Could not find runs directory: {RUNS_DIR.resolve()}")

    if not GT_PATH.exists():
        raise FileNotFoundError(f"Could not find GT file: {GT_PATH.resolve()}")

    jsonl_paths = find_trajectory_files(RUNS_DIR)

    if not jsonl_paths:
        raise FileNotFoundError(
            f"No folders containing '{TARGET_STRATEGY}' with trajectories.jsonl "
            f"found in {RUNS_DIR.resolve()}"
        )

    print(f"Found {len(jsonl_paths)} trajectory files:")
    for path in jsonl_paths:
        print(f"  {path}")

    gt_stage1_clarify_ids = load_gt_stage1_clarify_ids(GT_PATH)
    print(f"\nLoaded {len(gt_stage1_clarify_ids)} GT cases with Stage 1 = Clarify")

    model_total = Counter()
    model_stage1_clarify = Counter()
    model_not_stage1_clarify = Counter()

    for jsonl_path in jsonl_paths:
        model_name = extract_model_name(jsonl_path)

        for record in load_jsonl(jsonl_path):
            sample_id = str(record.get("sample_id", record.get("id", ""))).strip()

            if sample_id not in gt_stage1_clarify_ids:
                continue

            model_total[model_name] += 1

            stage1_action = str(get_stage1_action(record)).strip()

            if stage1_action == "Clarify":
                model_stage1_clarify[model_name] += 1
            else:
                model_not_stage1_clarify[model_name] += 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "total_gt_stage1_clarify_cases",
            "count_stage1_clarify",
            "count_not_stage1_clarify",
            "pct_stage1_clarify",
            "pct_not_stage1_clarify",
        ])

        for model in sorted(model_total):
            total = model_total[model]
            count_clarify = model_stage1_clarify[model]
            count_not = model_not_stage1_clarify[model]

            pct_clarify = round(100 * count_clarify / total, 2) if total else 0.0
            pct_not = round(100 * count_not / total, 2) if total else 0.0

            writer.writerow([
                model,
                total,
                count_clarify,
                count_not,
                pct_clarify,
                pct_not,
            ])

    print(f"\nWrote {OUT_CSV}")

    print("\n=== Stage 1 Clarification on GT-Stage1-Clarify cases ===")
    for model in sorted(model_total):
        total = model_total[model]
        count_clarify = model_stage1_clarify[model]
        count_not = model_not_stage1_clarify[model]
        pct_clarify = round(100 * count_clarify / total, 2) if total else 0.0

        print(f"\nModel: {model}")
        print(f"  Total GT Stage-1-Clarify cases: {total}")
        print(f"  Clarified at Stage 1: {count_clarify} ({pct_clarify}%)")
        print(f"  Did not clarify at Stage 1: {count_not}")


if __name__ == "__main__":
    main()