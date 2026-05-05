#!/usr/bin/env python3
import json
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


RUNS_DIR = Path("evaluations/multiple_seed_runs/runs_seeds")
STRATEGY = "full_info_abstention_aware"
DATA_GT_PATH = Path("evaluations/data_gt.json")

OUT_DIR = Path("multiple_seed_runs")
OUT_PER_SEED = OUT_DIR / "seed_variability_per_seed.csv"
OUT_SUMMARY = OUT_DIR / "seed_variability_summary.csv"
OUT_SCATTER_CI = OUT_DIR / "scatter_axis_ci_by_model.csv"

VALID_SUFFICIENCY = {"sufficient", "insufficient"}


ICD10_MAIN_GROUPS = [
    ("F00-F09", "Organic, including symptomatic, mental disorders"),
    ("F10-F19", "Mental and behavioural disorders due to psychoactive substance use"),
    ("F20-F29", "Schizophrenia, schizotypal, and delusional disorders"),
    ("F30-F39", "Mood [affective] disorders"),
    ("F40-F48", "Neurotic, stress-related, and somatoform disorders"),
    ("F50-F59", "Behavioural syndromes associated with physiological disturbances and physical factors"),
    ("F60-F69", "Disorders of personality and behaviour in adult persons"),
    ("F70-F79", "Mental retardation / intellectual disabilities"),
    ("F80-F89", "Disorders of psychological development"),
    ("F90-F98", "Behavioural and emotional disorders with onset usually occurring in childhood and adolescence"),
    ("F99", "Unspecified mental disorder"),
]


def normalize_text(value):
    return "" if value is None else str(value).strip()


def normalize_icd10(code):
    code = normalize_text(code).upper().replace(" ", "")
    return code or None


def icd10_diagnosis_level(code):
    """
    Benchmark diagnosis level.

    Keeps base ICD code + first decimal digit:
      F31.1  -> F31.1
      F31.11 -> F31.1
      F31.10 -> F31.1
      F29    -> F29
    """
    code = normalize_icd10(code)

    if not code:
        return None

    match = re.match(r"^([A-Z]\d{2})(?:\.([0-9A-Za-z]))?", code)

    if not match:
        return None

    base = match.group(1)
    first_decimal = match.group(2)

    return f"{base}.{first_decimal}" if first_decimal else base


def icd10_3char(code):
    code = normalize_icd10(code)

    if not code:
        return None

    match = re.match(r"^([A-Z]\d{2})", code)
    return match.group(1) if match else None


def icd10_main_group(code):
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


def has_decimal(code):
    code = normalize_icd10(code)
    return bool(code and "." in code)


def diagnosis_level_match(gt_code, pred_code):
    """
    Flexible benchmark diagnosis-level match.

    If GT has a decimal:
      GT F31.1, model F31.11 -> correct
      GT F31.1, model F31.2  -> incorrect

    If GT has no decimal:
      GT F29, model F29.1 -> correct
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


def matches(gt_code, pred_code):
    gt_raw = normalize_icd10(gt_code)
    pred_raw = normalize_icd10(pred_code)

    gt_3char = icd10_3char(gt_raw)
    pred_3char = icd10_3char(pred_raw)

    gt_high = icd10_main_group(gt_raw)
    pred_high = icd10_main_group(pred_raw)

    return {
        "match_diagnosis_level": diagnosis_level_match(gt_raw, pred_raw),
        "match_3char": int(bool(gt_3char and pred_3char and gt_3char == pred_3char)),
        "match_high_class": int(bool(gt_high and pred_high and gt_high == pred_high)),
    }


def extract_predicted_icd10_code(text):
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

    structured = step.get("structured_action_raw")
    if isinstance(structured, dict) and structured.get("action"):
        return normalize_text(structured.get("action")).lower()

    return normalize_text(step.get("judge_action")).lower()


def get_first_diagnosis_code(record):
    trajectory = record.get("trajectory", [])

    for step in sorted(trajectory, key=lambda x: x.get("step_index", 999)):
        action = get_action(step)
        raw = step.get("judge_output_raw", "")

        code = extract_predicted_icd10_code(raw)

        if action == "diagnose":
            return code

    return None


def get_any_predicted_code(record):
    """
    Fallback: returns last ICD-like code found anywhere in the trajectory.
    """
    fallback = None

    for step in sorted(record.get("trajectory", []), key=lambda x: x.get("step_index", 999)):
        raw = step.get("judge_output_raw", "")
        code = extract_predicted_icd10_code(raw)

        if code:
            fallback = code

    return fallback


def is_abstained(record):
    if normalize_text(record.get("stopping_reason")).lower() == "abstain":
        return True

    for step in record.get("trajectory", []):
        if get_action(step) == "abstain":
            return True

    return False


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def parse_run_folder(folder_name):
    """
    Expected:
      <model>_<strategy>_seed-<seed>
    """
    seed_match = re.search(r"_seed-(\d+)(?:_|$)", folder_name)
    seed = int(seed_match.group(1)) if seed_match else None

    marker = f"_{STRATEGY}_"

    if marker not in folder_name:
        return None, seed

    model = folder_name.split(marker)[0]

    return model, seed


def load_ground_truth(path):
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    gt = {}

    for rec in records:
        sample_id = str(rec["id"])
        suff_value = normalize_text(rec.get("diagnosis_sufficiency"))
        diagnosis = normalize_icd10(rec.get("diagnosis"))

        if suff_value == "1":
            sufficiency = "sufficient"
        elif suff_value == "0":
            sufficiency = "insufficient"
        else:
            sufficiency = None

        gt[sample_id] = {
            "sufficiency": sufficiency,
            "diagnosis": diagnosis,
        }

    return gt


def mean_sd_t_ci(values, ci=95):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    n = len(values)

    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }

    mean = values.mean()

    if n == 1:
        return {
            "n": n,
            "mean": mean,
            "sd": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }

    sd = values.std(ddof=1)
    sem = sd / np.sqrt(n)

    alpha = 1 - ci / 100
    critical = t.ppf(1 - alpha / 2, df=n - 1)

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_lower": mean - critical * sem,
        "ci_upper": mean + critical * sem,
    }


def add_pct_columns(df):
    pct_source_cols = [
        "mean",
        "sd",
        "ci95_lower",
        "ci95_upper",
        "abstention_rate",
        "under_abstention",
        "over_abstention",
        "accuracy_diagnosis_level",
        "accuracy_3char",
        "accuracy_high_class",
        "mean_under_abstention",
        "std_under_abstention",
        "under_t_ci95_lower",
        "under_t_ci95_upper",
        "mean_over_abstention",
        "std_over_abstention",
        "over_t_ci95_lower",
        "over_t_ci95_upper",
    ]

    for col in pct_source_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f"{col}_pct"] = df[col] * 100

    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(DATA_GT_PATH)
    files = sorted(RUNS_DIR.glob("**/trajectories.jsonl"))

    if not files:
        raise FileNotFoundError(f"No trajectories.jsonl files found under {RUNS_DIR}")

    rows = []

    for file in files:
        model_from_folder, seed = parse_run_folder(file.parent.name)

        if model_from_folder is None or seed is None:
            continue

        for rec in load_jsonl(file):
            if rec.get("strategy") != STRATEGY:
                continue

            sample_id = str(rec.get("sample_id"))
            gt_item = gt.get(sample_id)

            if gt_item is None:
                continue

            sufficiency = gt_item["sufficiency"]
            gt_code = gt_item["diagnosis"]

            if sufficiency not in VALID_SUFFICIENCY:
                continue

            abstained = int(is_abstained(rec))

            pred_code = get_first_diagnosis_code(rec) or get_any_predicted_code(rec)

            match_info = {
                "match_diagnosis_level": np.nan,
                "match_3char": np.nan,
                "match_high_class": np.nan,
            }

            has_diagnosis_eval = bool(
                sufficiency == "sufficient"
                and gt_code
                and pred_code
                and not abstained
            )

            if has_diagnosis_eval:
                match_info = matches(gt_code, pred_code)

            rows.append({
                "model": rec.get("model_evaluated") or model_from_folder,
                "seed": seed,
                "sample_id": sample_id,
                "final_human_sufficiency": sufficiency,
                "abstained": abstained,
                "pred_code": pred_code,
                "gt_code": gt_code,
                "has_diagnosis_eval": int(has_diagnosis_eval),
                "source_file": str(file),
                **match_info,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No matching records found.")

    dupes = df.duplicated(["model", "seed", "sample_id"], keep=False)
    if dupes.any():
        print("\nWarning: duplicate model/seed/sample_id rows found.")
        print(df.loc[dupes, ["model", "seed", "sample_id", "source_file"]])

    # ------------------------------------------------------------------
    # Per-seed abstention metrics
    # ------------------------------------------------------------------
    abst_per_seed = (
        df.groupby(["model", "seed", "final_human_sufficiency"], as_index=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            n_abstained=("abstained", "sum"),
            abstention_rate=("abstained", "mean"),
        )
    )

    abst_per_seed["under_abstention"] = np.where(
        abst_per_seed["final_human_sufficiency"] == "insufficient",
        1 - abst_per_seed["abstention_rate"],
        np.nan,
    )

    abst_per_seed["over_abstention"] = np.where(
        abst_per_seed["final_human_sufficiency"] == "sufficient",
        abst_per_seed["abstention_rate"],
        np.nan,
    )

    # ------------------------------------------------------------------
    # Per-seed diagnosis accuracy
    # ------------------------------------------------------------------
    diag_df = df[df["has_diagnosis_eval"] == 1].copy()

    diag_per_seed = (
        diag_df.groupby(["model", "seed"], as_index=False)
        .agg(
            n_diagnosis_eval=("sample_id", "nunique"),
            accuracy_diagnosis_level=("match_diagnosis_level", "mean"),
            accuracy_3char=("match_3char", "mean"),
            accuracy_high_class=("match_high_class", "mean"),
        )
    )

    per_seed = abst_per_seed.merge(
        diag_per_seed,
        on=["model", "seed"],
        how="left",
    )

    # ------------------------------------------------------------------
    # Summary across seeds
    # ------------------------------------------------------------------
    summary_rows = []

    for model, model_df in per_seed.groupby("model"):
        # under-abstention
        under_values = model_df.loc[
            model_df["final_human_sufficiency"] == "insufficient",
            "under_abstention",
        ].to_numpy()

        under_stats = mean_sd_t_ci(under_values)

        summary_rows.append({
            "model": model,
            "metric": "under_abstention",
            "n_seeds": under_stats["n"],
            "mean": under_stats["mean"],
            "sd": under_stats["sd"],
            "ci95_lower": under_stats["ci_lower"],
            "ci95_upper": under_stats["ci_upper"],
        })

        # over-abstention
        over_values = model_df.loc[
            model_df["final_human_sufficiency"] == "sufficient",
            "over_abstention",
        ].to_numpy()

        over_stats = mean_sd_t_ci(over_values)

        summary_rows.append({
            "model": model,
            "metric": "over_abstention",
            "n_seeds": over_stats["n"],
            "mean": over_stats["mean"],
            "sd": over_stats["sd"],
            "ci95_lower": over_stats["ci_lower"],
            "ci95_upper": over_stats["ci_upper"],
        })

        # diagnosis accuracy metrics
        model_diag = diag_per_seed[diag_per_seed["model"] == model]

        for metric in [
            "accuracy_diagnosis_level",
            "accuracy_3char",
            "accuracy_high_class",
        ]:
            stats = mean_sd_t_ci(model_diag[metric].to_numpy())

            summary_rows.append({
                "model": model,
                "metric": metric,
                "n_seeds": stats["n"],
                "mean": stats["mean"],
                "sd": stats["sd"],
                "ci95_lower": stats["ci_lower"],
                "ci95_upper": stats["ci_upper"],
            })

    summary = pd.DataFrame(summary_rows).sort_values(["metric", "model"])
    summary = add_pct_columns(summary)

    # ------------------------------------------------------------------
    # Scatter-axis CI table
    # ------------------------------------------------------------------
    scatter_rows = []

    for model, model_summary in summary.groupby("model"):
        row = {"model": model}

        for metric, prefix in [
            ("under_abstention", "under"),
            ("over_abstention", "over"),
        ]:
            m = model_summary[model_summary["metric"] == metric]

            if len(m):
                m = m.iloc[0]
                row[f"mean_{prefix}_abstention"] = m["mean"]
                row[f"std_{prefix}_abstention"] = m["sd"]
                row[f"{prefix}_t_ci95_lower"] = m["ci95_lower"]
                row[f"{prefix}_t_ci95_upper"] = m["ci95_upper"]
                row[f"n_seeds_{prefix}"] = m["n_seeds"]

        scatter_rows.append(row)

    scatter_ci = pd.DataFrame(scatter_rows).sort_values("model")
    scatter_ci = add_pct_columns(scatter_ci)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    per_seed = add_pct_columns(per_seed)

    per_seed.to_csv(OUT_PER_SEED, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    scatter_ci.to_csv(OUT_SCATTER_CI, index=False)

    print(f"\nWrote: {OUT_PER_SEED}")
    print(f"Wrote: {OUT_SUMMARY}")
    print(f"Wrote: {OUT_SCATTER_CI}")

    print("\nSummary:")
    print(
        summary[
            [
                "model",
                "metric",
                "n_seeds",
                "mean_pct",
                "sd_pct",
                "ci95_lower_pct",
                "ci95_upper_pct",
            ]
        ].to_string(index=False)
    )

    print("\nScatter CI:")
    print(scatter_ci.to_string(index=False))


if __name__ == "__main__":
    main()