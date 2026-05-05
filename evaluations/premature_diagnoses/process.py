#!/usr/bin/env python3
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

RUNS_DIR = Path("runs")
GT_PATH = Path("evaluations/data_gt.json")
OUT_DIR = Path("evaluations/premature_diagnoses")


STRATEGY_MARKERS = {
    "_sequential_info_neutral_prompting": "neutral",
    "_sequential_info_abstention_aware": "abstention_aware",
}


def extract_model_and_strategy(folder_name: str):
    for marker, strategy in STRATEGY_MARKERS.items():
        if marker in folder_name:
            model = folder_name.split(marker)[0]
            return model, strategy
    return folder_name, "unknown"


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


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def parse_int(value):
    value = normalize_text(value)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def load_gt_earliest_diagnosis_info(gt_path: Path):
    with gt_path.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_map = {}

    for item in gt_data:
        sample_id = normalize_text(item.get("id"))
        if not sample_id:
            continue

        row = item.get("original_row", item)

        diagnosis_agreement = normalize_text(row.get("diagnosis_agreement", "")).upper()
        earliest = parse_int(row.get("earliest_sufficient_section"))
        num_sections = parse_int(item.get("num_sections_in_input"))

        if diagnosis_agreement not in {"1", "TRUE"}:
            continue
        if earliest is None:
            continue
        if num_sections is None:
            continue

        gt_map[sample_id] = {
            "earliest_sufficient_section": earliest,
            "num_sections_in_input": num_sections,
        }

    return gt_map


def get_action(step: dict):
    if step.get("new_judge_action"):
        return normalize_text(step.get("new_judge_action")).lower()

    if step.get("new_judge_output", {}).get("action"):
        return normalize_text(step["new_judge_output"]["action"]).lower()

    return normalize_text(step.get("judge_action")).lower()


def get_first_diagnose_step(record: dict):
    trajectory = record.get("trajectory", [])

    for step in sorted(trajectory, key=lambda x: x.get("step_index", 999)):
        action = get_action(step)
        step_index = step.get("step_index")

        if action == "diagnose" and isinstance(step_index, int):
            return step_index

    return None


def get_diagnose_steps(record: dict):
    trajectory = record.get("trajectory", [])
    steps = set()

    for step in trajectory:
        action = get_action(step)
        step_index = step.get("step_index")

        if action == "diagnose" and isinstance(step_index, int):
            steps.add(step_index)

    return steps


def classify_timing(model_first_diagnosis_step, gold_earliest_stage):
    if model_first_diagnosis_step is None:
        return "never"
    if model_first_diagnosis_step < gold_earliest_stage:
        return "premature"
    if model_first_diagnosis_step == gold_earliest_stage:
        return "on_time"
    return "late"


def pct(count, total):
    return round(100 * count / total, 2) if total else 0.0


def main():
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"Could not find folder: {RUNS_DIR.resolve()}")

    if not GT_PATH.exists():
        raise FileNotFoundError(f"Could not find GT file: {GT_PATH.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    matched_files = [
        p for p in RUNS_DIR.rglob("*trajectories*.jsonl")
        if p.is_file()
        and (
            "_sequential_info_neutral_prompting" in str(p.parent)
            or "_sequential_info_abstention_aware" in str(p.parent)
        )
    ]

    if not matched_files:
        print("No sequential neutral or abstention-aware trajectories found.")
        return

    print(f"Found {len(matched_files)} trajectory files:")
    for p in sorted(matched_files):
        model, strategy = extract_model_and_strategy(p.parent.name)
        print(f"  model={model}, strategy={strategy}, file={p}")

    gt_map = load_gt_earliest_diagnosis_info(GT_PATH)
    print(f"\nLoaded {len(gt_map)} GT cases with diagnosis=TRUE and valid earliest_sufficient_section")

    # Keyed by (model, strategy)
    model_total = Counter()
    model_premature = Counter()
    model_on_time = Counter()
    model_late = Counter()
    model_never = Counter()

    model_premature_stage_distance_sum = Counter()
    model_premature_stage_distance_n = Counter()

    stage_total = defaultdict(Counter)
    stage_diagnose = defaultdict(Counter)

    per_sample_rows = []

    for traj_file in sorted(matched_files):
        folder_name = traj_file.parent.name
        model_name, strategy = extract_model_and_strategy(folder_name)
        key = (model_name, strategy)

        for record in load_jsonl(traj_file):
            sample_id = normalize_text(record.get("sample_id", record.get("id", "")))
            if not sample_id or sample_id not in gt_map:
                continue

            gt_info = gt_map[sample_id]
            gold_earliest_stage = gt_info["earliest_sufficient_section"]
            num_sections_in_input = gt_info["num_sections_in_input"]

            model_first_diagnosis_step = get_first_diagnose_step(record)
            diagnose_steps = get_diagnose_steps(record)
            timing_category = classify_timing(model_first_diagnosis_step, gold_earliest_stage)

            model_total[key] += 1

            if timing_category == "premature":
                model_premature[key] += 1
            elif timing_category == "on_time":
                model_on_time[key] += 1
            elif timing_category == "late":
                model_late[key] += 1
            elif timing_category == "never":
                model_never[key] += 1

            stages_too_early = ""
            if (
                model_first_diagnosis_step is not None
                and model_first_diagnosis_step < gold_earliest_stage
            ):
                stages_too_early = gold_earliest_stage - model_first_diagnosis_step
                model_premature_stage_distance_sum[key] += stages_too_early
                model_premature_stage_distance_n[key] += 1

            per_sample_rows.append([
                model_name,
                strategy,
                sample_id,
                num_sections_in_input,
                gold_earliest_stage,
                model_first_diagnosis_step if model_first_diagnosis_step is not None else "",
                timing_category,
                stages_too_early,
            ])

            for stage in range(1, num_sections_in_input + 1):
                stage_total[key][stage] += 1
                if stage in diagnose_steps:
                    stage_diagnose[key][stage] += 1

    detail_csv = OUT_DIR / "per_sample_diagnosis_timing_by_strategy.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "strategy",
            "sample_id",
            "num_sections_in_input",
            "gold_earliest_sufficient_section",
            "model_first_diagnosis_step",
            "timing_category",
            "stages_too_early",
        ])
        writer.writerows(per_sample_rows)

    summary_csv = OUT_DIR / "fig5a_diagnosis_timing_by_strategy.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "strategy",
            "total_cases",
            "count_premature",
            "count_on_time",
            "count_late",
            "count_never",
            "pct_premature",
            "pct_on_time",
            "pct_late",
            "pct_never",
            "mean_stages_too_early_among_premature",
        ])

        for key in sorted(model_total):
            model, strategy = key
            total = model_total[key]

            count_premature = model_premature[key]
            count_on_time = model_on_time[key]
            count_late = model_late[key]
            count_never = model_never[key]

            n_early = model_premature_stage_distance_n[key]
            mean_early = (
                round(model_premature_stage_distance_sum[key] / n_early, 2)
                if n_early else ""
            )

            writer.writerow([
                model,
                strategy,
                total,
                count_premature,
                count_on_time,
                count_late,
                count_never,
                pct(count_premature, total),
                pct(count_on_time, total),
                pct(count_late, total),
                pct(count_never, total),
                mean_early,
            ])

    per_stage_csv = OUT_DIR / "fig5b_diagnosis_by_stage_by_strategy.csv"
    with per_stage_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "strategy",
            "stage",
            "total_cases",
            "count_diagnose",
            "pct_diagnose",
        ])

        for key in sorted(stage_total):
            model, strategy = key
            for stage in sorted(stage_total[key]):
                total_cases = stage_total[key][stage]
                count_diagnose = stage_diagnose[key][stage]

                writer.writerow([
                    model,
                    strategy,
                    stage,
                    total_cases,
                    count_diagnose,
                    pct(count_diagnose, total_cases),
                ])

    print(f"\nWrote {detail_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {per_stage_csv}")

    print("\n=== Diagnosis timing relative to earliest sufficient section ===")
    for key in sorted(model_total):
        model, strategy = key
        total = model_total[key]

        count_premature = model_premature[key]
        count_on_time = model_on_time[key]
        count_late = model_late[key]
        count_never = model_never[key]

        n_early = model_premature_stage_distance_n[key]
        mean_early = (
            round(model_premature_stage_distance_sum[key] / n_early, 2)
            if n_early else None
        )

        print(f"\nModel: {model}")
        print(f"  Strategy: {strategy}")
        print(f"  Total cases: {total}")
        print(f"  Premature: {count_premature} ({pct(count_premature, total)}%)")
        print(f"  On-time: {count_on_time} ({pct(count_on_time, total)}%)")
        print(f"  Late: {count_late} ({pct(count_late, total)}%)")
        print(f"  Never: {count_never} ({pct(count_never, total)}%)")
        if mean_early is not None:
            print(f"  Mean stages too early among premature: {mean_early}")


if __name__ == "__main__":
    main()