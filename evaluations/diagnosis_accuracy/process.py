import json
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter


RUNS_DIR = Path("runs")
GT_PATH = Path("evaluations/data_gt.json")
OUT_DIR = Path("evaluations/diagnosis_accuracy")


STRATEGY_MARKERS = [
    "_full_info_no_abstention",
    "_full_info_abstention_aware",
    "_sequential_info_neutral_prompting",
    "_sequential_info_abstention_aware",
    "_sequential_info_clarification_only",
    "_structured_actions",
]


ICD10_MAIN_GROUPS = [
    ("F00-F09", "Organic, including symptomatic, mental disorders"),
    ("F10-F19", "Mental and behavioural disorders due to psychoactive substance use"),
    ("F20-F29", "Schizophrenia, schizotypal, and delusional disorders"),
    ("F30-F39", "Mood [affective] disorders"),
    ("F40-F49", "Neurotic, stress-related, and somatoform disorders"),
    ("F50-F59", "Behavioural syndromes associated with physiological disturbances and physical factors"),
    ("F60-F69", "Disorders of personality and behaviour in adult persons"),
    ("F70-F79", "Mental retardation/Intellectual disabilities"),
    ("F80-F89", "Disorders of psychological development"),
    ("F90-F98", "Behavioural and emotional disorders with onset usually occurring in childhood and adolescence"),
    ("F99", "Unspecified mental disorder"),
]

def has_decimal(code):
    code = normalize_icd10(code)
    return bool(code and "." in code)


def normalize_text(value):
    return "" if value is None else str(value).strip()

def diagnosis_level_match(gt_code, pred_code):
    """
    Flexible benchmark diagnosis-level matching.

    If the ground-truth code includes a decimal, compare both codes at the
    benchmark diagnosis level: base ICD code + first decimal digit.

      GT F31.1, model F31.11 -> correct
      GT F31.1, model F31.2  -> incorrect

    If the ground-truth code has no decimal, compare only the 3-character base.

      GT F29, model F29.1 -> correct
      GT F29, model F29   -> correct
      GT F29, model F30.1 -> incorrect
    """
    gt_raw = normalize_icd10(gt_code)
    pred_raw = normalize_icd10(pred_code)

    if not gt_raw or not pred_raw:
        return 0

    if has_decimal(gt_raw):
        gt_level = icd10_diagnosis_level(gt_raw)
        pred_level = icd10_diagnosis_level(pred_raw)
        return int(bool(gt_level and pred_level and gt_level == pred_level))

    gt_base = icd10_3char(gt_raw)
    pred_base = icd10_3char(pred_raw)
    return int(bool(gt_base and pred_base and gt_base == pred_base))


def normalize_icd10(code):
    code = normalize_text(code).upper().replace(" ", "")
    return code or None


def icd10_diagnosis_level(code):
    """
    Canonical ICD-10 code at the benchmark diagnosis level.

    The benchmark ground truth uses codes such as F32.1.
    If the model outputs a more specific extension such as F32.10 or F32.11,
    this function collapses it to F32.1.

    Examples:
      F32.1  -> F32.1
      F32.10 -> F32.1
      F32.11 -> F32.1
      F70.01 -> F70.0
      F20    -> F20
    """
    code = normalize_icd10(code)

    if not code:
        return None

    match = re.match(r"^([A-Z]\d{2})(?:\.([0-9A-Za-z]))?", code)

    if not match:
        return None

    base = match.group(1)
    first_decimal = match.group(2)

    if first_decimal:
        return f"{base}.{first_decimal}"

    return base


def icd10_3char(code):
    """
    Extracts the 3-character ICD-10 category.

    Examples:
      F32.1  -> F32
      F32.10 -> F32
      F20    -> F20
    """
    code = normalize_icd10(code)

    if not code:
        return None

    match = re.match(r"^([A-Z]\d{2})", code)

    return match.group(1) if match else None


def icd10_main_group(code):
    """
    Maps an ICD-10 code to a high-level ICD-10 psychiatric group.
    """
    cls = icd10_3char(code)

    if not cls:
        return None

    letter = cls[0]

    try:
        number = int(cls[1:])
    except ValueError:
        return None

    for group_range, _ in ICD10_MAIN_GROUPS:
        if "-" in group_range:
            start, end = group_range.split("-")

            if (
                letter == start[0] == end[0]
                and int(start[1:]) <= number <= int(end[1:])
            ):
                return group_range

        else:
            if cls == group_range:
                return group_range

    return None


def parse_int(value):
    value = normalize_text(value)

    if not value:
        return None

    try:
        return int(float(value))
    except ValueError:
        return None


def is_yes(value):
    return normalize_text(value).lower() in {"1", "true", "yes"}


def pct(num, den):
    return round(100 * num / den, 2) if den else 0.0


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON in {path} line {line_num}: {e}")


def extract_model_and_strategy(folder_name):
    for marker in STRATEGY_MARKERS:
        if marker in folder_name:
            return folder_name.split(marker)[0], marker.lstrip("_")

    return folder_name, "unknown"


def load_gt(gt_path):
    """
    Loads ground-truth diagnosis labels.

    Keeps only cases that are marked sufficient for diagnosis and have a
    non-empty ground-truth diagnosis.
    """
    with gt_path.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt = {}

    for item in gt_data:
        sample_id = normalize_text(item.get("id"))

        if not sample_id:
            continue

        row = item.get("original_row", item)

        diagnosis = normalize_icd10(row.get("diagnosis", ""))

        diagnosis_sufficiency = row.get("diagnosis_sufficiency", "")
        diagnosis_sufficiency_agreement = row.get("diagnosis_sufficiency_agreement", "")
        diagnosis_agreement = row.get("diagnosis_agreement", "")

        earliest = parse_int(row.get("earliest_sufficient_section"))
        num_sections = parse_int(item.get("num_sections_in_input"))

        # Keep only cases judged sufficient for diagnosis,
        # because this script evaluates diagnosis accuracy.
        if not is_yes(diagnosis_sufficiency):
            continue

        if not diagnosis:
            continue

        sufficiency_agreement_status = (
            "agreement"
            if is_yes(diagnosis_sufficiency_agreement)
            else "disagreement"
        )

        diagnosis_agreement_status = (
            "agreement"
            if is_yes(diagnosis_agreement)
            else "disagreement"
        )

        combined_doctor_agreement = (
            "agreement"
            if (
                is_yes(diagnosis_sufficiency_agreement)
                and is_yes(diagnosis_agreement)
            )
            else "disagreement"
        )

        gt[sample_id] = {
            "diagnosis": diagnosis,
            "diagnosis_sufficiency": "sufficient",
            "diagnosis_sufficiency_agreement": sufficiency_agreement_status,
            "diagnosis_agreement": diagnosis_agreement_status,
            "doctor_agreement": combined_doctor_agreement,
            "earliest_sufficient_section": earliest,
            "num_sections_in_input": num_sections,
        }

    return gt


def extract_predicted_icd10_code(text):
    """
    Extracts the first ICD-10-like code from the model/judge output.

    If the output contains a field named predicted_icd10_codes, extraction is
    restricted to the text following that field.
    """
    if not text:
        return None

    match = re.search(r"predicted_icd10_codes:\s*(.*)", text, flags=re.IGNORECASE)

    if match:
        text = match.group(1)

    codes = re.findall(r"\b([A-Za-z]\d{2}(?:\.[0-9A-Za-z]+)?)\b", text)

    if not codes:
        return None

    return normalize_icd10(codes[0])


def get_action(step):
    if step.get("new_judge_action"):
        return normalize_text(step.get("new_judge_action")).lower()

    if step.get("new_judge_output", {}).get("action"):
        return normalize_text(step["new_judge_output"]["action"]).lower()

    return normalize_text(step.get("judge_action")).lower()


def get_first_diagnosis(record):
    """
    Returns the first trajectory step where the judge action is Diagnose,
    together with the ICD-10 code extracted from that step.
    """
    trajectory = record.get("trajectory", [])

    for step in sorted(trajectory, key=lambda x: x.get("step_index", 999)):
        action = get_action(step)
        step_index = step.get("step_index")
        raw = step.get("judge_output_raw", "")

        code = extract_predicted_icd10_code(raw)

        if action == "diagnose" and isinstance(step_index, int):
            return step_index, code

    return None, None


def get_predicted_code(record):
    """
    Returns the first predicted ICD-10 code associated with a Diagnose action.

    If no Diagnose action contains a code, falls back to the last ICD-10 code
    found anywhere in the trajectory.
    """
    trajectory = record.get("trajectory", [])
    fallback_code = None

    for step in sorted(trajectory, key=lambda x: x.get("step_index", 999)):
        action = get_action(step)
        raw = step.get("judge_output_raw", "")

        code = extract_predicted_icd10_code(raw)

        if code:
            fallback_code = code

        if action == "diagnose" and code:
            return code

    return fallback_code


def classify_timing(first_step, gold_earliest):
    if first_step is None:
        return "never"

    if gold_earliest is None:
        return "unknown"

    if first_step < gold_earliest:
        return "premature"

    if first_step == gold_earliest:
        return "on_time"

    return "late"


def matches(gt_code, pred_code):
    """
    Computes diagnosis matches at three levels:

    1. diagnosis_level:
       Flexible benchmark diagnosis-level match.
       If GT has a decimal, compare base + first decimal digit.
       If GT has no decimal, compare the 3-character base.

    2. 3char:
       ICD-10 3-character category.

    3. high_class:
       Broad ICD-10 psychiatric group.
    """
    gt_raw = normalize_icd10(gt_code)
    pred_raw = normalize_icd10(pred_code)

    gt_diagnosis_level = icd10_diagnosis_level(gt_raw)
    pred_diagnosis_level = icd10_diagnosis_level(pred_raw)

    gt_3char = icd10_3char(gt_raw)
    pred_3char = icd10_3char(pred_raw)

    gt_high_class = icd10_main_group(gt_raw)
    pred_high_class = icd10_main_group(pred_raw)

    return {
        "gt_raw": gt_raw,
        "pred_raw": pred_raw,

        "gt_diagnosis_level": gt_diagnosis_level,
        "pred_diagnosis_level": pred_diagnosis_level,
        "gt_3char": gt_3char,
        "pred_3char": pred_3char,
        "gt_high_class": gt_high_class,
        "pred_high_class": pred_high_class,

        "match_diagnosis_level": diagnosis_level_match(gt_raw, pred_raw),
        "match_3char": int(
            bool(gt_3char and pred_3char and gt_3char == pred_3char)
        ),
        "match_high_class": int(
            bool(gt_high_class and pred_high_class and gt_high_class == pred_high_class)
        ),
    }


def find_trajectory_files():
    return sorted(
        p for p in RUNS_DIR.rglob("*trajectories*.jsonl")
        if p.is_file()
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_gt(GT_PATH)
    files = find_trajectory_files()

    if not files:
        raise FileNotFoundError(f"No trajectories files found under {RUNS_DIR}")

    print(f"Loaded {len(gt)} GT diagnosis cases")
    print(f"Found {len(files)} trajectory files")

    per_sample_rows = []

    for path in files:
        model, strategy = extract_model_and_strategy(path.parent.name)

        for record in load_jsonl(path):
            sample_id = normalize_text(record.get("sample_id", record.get("id", "")))

            if sample_id not in gt:
                continue

            gt_code = gt[sample_id]["diagnosis"]
            earliest = gt[sample_id]["earliest_sufficient_section"]

            first_step, first_diag_code = get_first_diagnosis(record)
            pred_code = first_diag_code or get_predicted_code(record)
            timing = classify_timing(first_step, earliest)

            if not pred_code:
                continue

            match_info = matches(gt_code, pred_code)

            per_sample_rows.append({
                "model": model,
                "strategy": strategy,
                "sample_id": sample_id,
                "doctor_agreement": gt[sample_id]["doctor_agreement"],
                "diagnosis_sufficiency_agreement": gt[sample_id]["diagnosis_sufficiency_agreement"],
                "diagnosis_agreement": gt[sample_id]["diagnosis_agreement"],
                "gold_earliest_sufficient_section": earliest,
                "model_first_diagnosis_step": first_step if first_step is not None else "",
                "timing_category": timing,
                **match_info,
            })

    # --------------------------------------------------
    # 1. Accuracy by granularity: full_info_no_abstention
    # --------------------------------------------------
    granularity_stats = defaultdict(Counter)

    for row in per_sample_rows:
        if row["strategy"] != "full_info_no_abstention":
            continue

        key = row["model"]
        granularity_stats[key]["total"] += 1
        granularity_stats[key]["correct_diagnosis_level"] += row["match_diagnosis_level"]
        granularity_stats[key]["correct_3char"] += row["match_3char"]
        granularity_stats[key]["correct_high_class"] += row["match_high_class"]

    out1 = OUT_DIR / "accuracy_by_granularity_full_info_no_abstention.csv"

    with out1.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "model",
            "strategy",
            "total_diagnosed_cases",
            "correct_diagnosis_level",
            "accuracy_diagnosis_level_pct",
            "correct_3char",
            "accuracy_3char_pct",
            "correct_high_class",
            "accuracy_high_class_pct",
        ])

        for model in sorted(granularity_stats):
            stats = granularity_stats[model]
            total = stats["total"]

            writer.writerow([
                model,
                "full_info_no_abstention",
                total,
                stats["correct_diagnosis_level"],
                pct(stats["correct_diagnosis_level"], total),
                stats["correct_3char"],
                pct(stats["correct_3char"], total),
                stats["correct_high_class"],
                pct(stats["correct_high_class"], total),
            ])

    # --------------------------------------------------
    # 2. Accuracy vs timing: sequential strategies
    # --------------------------------------------------
    timing_stats = defaultdict(Counter)

    for row in per_sample_rows:
        if row["strategy"] not in {
            "sequential_info_abstention_aware",
            "sequential_info_neutral_prompting",
        }:
            continue

        if row["timing_category"] == "unknown":
            continue

        key = (row["model"], row["strategy"], row["timing_category"])

        timing_stats[key]["total"] += 1
        timing_stats[key]["correct_diagnosis_level"] += row["match_diagnosis_level"]
        timing_stats[key]["correct_3char"] += row["match_3char"]
        timing_stats[key]["correct_high_class"] += row["match_high_class"]

    out2 = OUT_DIR / "accuracy_by_timing_sequential.csv"

    with out2.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "model",
            "strategy",
            "timing_category",
            "total_diagnosed_cases",
            "correct_diagnosis_level",
            "accuracy_diagnosis_level_pct",
            "correct_3char",
            "accuracy_3char_pct",
            "correct_high_class",
            "accuracy_high_class_pct",
        ])

        for (model, strategy, timing), stats in sorted(timing_stats.items()):
            total = stats["total"]

            writer.writerow([
                model,
                strategy,
                timing,
                total,
                stats["correct_diagnosis_level"],
                pct(stats["correct_diagnosis_level"], total),
                stats["correct_3char"],
                pct(stats["correct_3char"], total),
                stats["correct_high_class"],
                pct(stats["correct_high_class"], total),
            ])

    # --------------------------------------------------
    # 3. Scatter: safety vs accuracy
    # x = premature diagnosis rate
    # y = diagnosis-level accuracy
    # strategy = sequential_info_abstention_aware
    # --------------------------------------------------
    scatter_total = Counter()
    scatter_premature = Counter()
    scatter_diag_total = Counter()
    scatter_correct_diagnosis_level = Counter()

    for row in per_sample_rows:
        if row["strategy"] != "sequential_info_abstention_aware":
            continue

        if row["timing_category"] == "unknown":
            continue

        model = row["model"]
        scatter_total[model] += 1

        if row["timing_category"] == "premature":
            scatter_premature[model] += 1

        scatter_diag_total[model] += 1
        scatter_correct_diagnosis_level[model] += row["match_diagnosis_level"]

    out3 = OUT_DIR / "scatter_premature_vs_diagnosis_level_accuracy_sequential_abstention_aware.csv"

    with out3.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "model",
            "strategy",
            "total_diagnosed_cases",
            "premature_diagnosis_rate_pct",
            "accuracy_diagnosis_level_pct",
            "correct_diagnosis_level",
        ])

        for model in sorted(scatter_total):
            writer.writerow([
                model,
                "sequential_info_abstention_aware",
                scatter_total[model],
                pct(scatter_premature[model], scatter_total[model]),
                pct(
                    scatter_correct_diagnosis_level[model],
                    scatter_diag_total[model],
                ),
                scatter_correct_diagnosis_level[model],
            ])

    # --------------------------------------------------
    # 4. Per-sample detail for auditing
    # --------------------------------------------------
    out_detail = OUT_DIR / "diagnosis_accuracy_per_sample_all_strategies.csv"

    with out_detail.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "strategy",
            "sample_id",
            "doctor_agreement",
            "diagnosis_sufficiency_agreement",
            "diagnosis_agreement",
            "gold_earliest_sufficient_section",
            "model_first_diagnosis_step",
            "timing_category",

            "gt_raw",
            "pred_raw",
            "gt_diagnosis_level",
            "pred_diagnosis_level",
            "gt_3char",
            "pred_3char",
            "gt_high_class",
            "pred_high_class",

            "match_diagnosis_level",
            "match_3char",
            "match_high_class",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    print(f"Wrote {out1}")
    print(f"Wrote {out2}")
    print(f"Wrote {out3}")
    print(f"Wrote {out_detail}")


if __name__ == "__main__":
    main()