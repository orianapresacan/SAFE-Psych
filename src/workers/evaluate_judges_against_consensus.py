import argparse
import json
import math
from collections import Counter

from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


TASKS = {
    "primary_answer": {
        "human_col": "PRIMARY_ANSWER__ann1",
        "judge_col": "primary_answer",
    },
    "info_request": {
        "human_col": "INFO_REQUEST__ann1",
        "judge_col": "info_request",
    },
    "action": {
        "human_col": "ACTION_CALC__consensus",
        "judge_col": "judge_action",
    },
}

FLEISS_TASKS = {
    "primary_answer": {
        "human_cols": [
            "PRIMARY_ANSWER__ann1",
            "PRIMARY_ANSWER__ann2",
            "PRIMARY_ANSWER__ann3",
        ],
        "judge_col": "primary_answer",
    },
    "info_request": {
        "human_cols": [
            "INFO_REQUEST__ann1",
            "INFO_REQUEST__ann2",
            "INFO_REQUEST__ann3",
        ],
        "judge_col": "info_request",
    },
    "action": {
        "human_cols": [
            "ACTION_CALC__ann1",
            "ACTION_CALC__ann2",
            "ACTION_CALC__ann3",
        ],
        "judge_col": "judge_action",
    },
}

def fleiss_kappa_for_rater_columns(df, rating_cols):
    """
    Computes Fleiss' kappa across multiple raters.
    Each row is one item; rating_cols are the raters.
    """
    clean = df[rating_cols].copy()

    for col in rating_cols:
        clean[col] = clean[col].map(normalize_label)

    clean = clean.dropna()

    if clean.empty:
        return math.nan, 0

    labels = sorted(
        set(
            label
            for col in rating_cols
            for label in clean[col].tolist()
            if label is not None
        )
    )

    matrix = []

    for _, row in clean.iterrows():
        counts = Counter(row[col] for col in rating_cols)
        matrix.append([counts.get(label, 0) for label in labels])

    matrix = np.array(matrix)

    try:
        return float(fleiss_kappa(matrix, method="fleiss")), len(clean)
    except Exception:
        return math.nan, len(clean)
    

def normalize_label(x):
    if pd.isna(x):
        return None

    x = str(x).strip().lower()

    if x in {"", "nan", "none/null", "null"}:
        return None

    aliases = {
        "diagnose": "diagnose",
        "clarify": "clarify",
        "abstain": "abstain",
        "no": "none",
        "n/a": "none",
        "na": "none",
        "not_applicable": "none",
        "not applicable": "none",
    }

    return aliases.get(x, x)


def map_to_action(primary, info_request):
    primary = normalize_label(primary)
    info_request = normalize_label(info_request)

    if primary == "committed":
        return "CLARIFY" if info_request == "specific" else "DIAGNOSE"

    if primary == "differential":
        return "CLARIFY" if info_request == "specific" else "ABSTAIN"

    if primary == "none":
        return "CLARIFY" if info_request == "specific" else "ABSTAIN"

    return "ABSTAIN"


def read_judge_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            judge_output = obj.get("new_judge_output", {}) or {}

            rows.append({
                "annotation_id": str(obj.get("annotation_id")),
                "primary_answer": normalize_label(judge_output.get("primary_answer")),
                "info_request": normalize_label(judge_output.get("info_request")),
                "judge_action": normalize_label(judge_output.get("action")),
            })

    return pd.DataFrame(rows)


def fleiss_kappa_from_two_raters(labels_a, labels_b):
    pairs = [
        (a, b)
        for a, b in zip(labels_a, labels_b)
        if a is not None and b is not None and not pd.isna(a) and not pd.isna(b)
    ]

    if not pairs:
        return math.nan

    categories = sorted(set(x for pair in pairs for x in pair))
    n_items = len(pairs)
    n_raters = 2

    category_counts = []

    for a, b in pairs:
        counts = Counter([a, b])
        category_counts.append([counts.get(cat, 0) for cat in categories])

    p_i = []
    for counts in category_counts:
        agreement = sum(c * c for c in counts) - n_raters
        agreement /= n_raters * (n_raters - 1)
        p_i.append(agreement)

    p_bar = sum(p_i) / n_items

    category_totals = [
        sum(row[j] for row in category_counts) for j in range(len(categories))
    ]
    p_j = [total / (n_items * n_raters) for total in category_totals]

    p_e = sum(p * p for p in p_j)

    if p_e == 1:
        return math.nan

    return (p_bar - p_e) / (1 - p_e)


def compute_task_metrics(df, task_name, human_col, judge_col):
    sub = df[["annotation_id", human_col, judge_col]].copy()
    sub = sub.dropna(subset=[human_col, judge_col])

    y_human = sub[human_col].tolist()
    y_judge = sub[judge_col].tolist()

    if len(sub) == 0:
        return {
            "task": task_name,
            "n": 0,
            "accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "macro_f1": math.nan,
            "weighted_f1": math.nan,
            "cohen_kappa": math.nan,
            "fleiss_kappa": math.nan,
        }

    return {
        "task": task_name,
        "n": len(sub),
        "accuracy": accuracy_score(y_human, y_judge),
        "balanced_accuracy": balanced_accuracy_score(y_human, y_judge),
        "macro_f1": f1_score(y_human, y_judge, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_human, y_judge, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_human, y_judge),
        "fleiss_kappa_consensus_judge": fleiss_kappa_from_two_raters(y_human, y_judge),
    }


def print_diagnostics(df):
    for task, cols in TASKS.items():
        sub = df[["annotation_id", cols["human_col"], cols["judge_col"]]].dropna()

        if sub.empty:
            continue

        y_human = sub[cols["human_col"]].tolist()
        y_judge = sub[cols["judge_col"]].tolist()
        labels = sorted(set(y_human) | set(y_judge))

        print(f"\n=== {task.upper()} ===")

        print("\nHuman label distribution:")
        print(pd.Series(y_human).value_counts().to_string())

        print("\nJudge label distribution:")
        print(pd.Series(y_judge).value_counts().to_string())

        print("\nClassification report:")
        print(
            classification_report(
                y_human,
                y_judge,
                labels=labels,
                zero_division=0,
            )
        )

        print("\nConfusion matrix:")
        cm = pd.DataFrame(
            confusion_matrix(y_human, y_judge, labels=labels),
            index=[f"human:{x}" for x in labels],
            columns=[f"judge:{x}" for x in labels],
        )
        print(cm.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus", required=True, help="Human labels CSV")
    parser.add_argument("--judge", required=True, help="Judge JSONL")
    parser.add_argument("--out", default="agreement_report.csv")
    parser.add_argument("--merged-out", default="merged_human_judge_labels.csv")
    args = parser.parse_args()

    human = pd.read_csv(args.consensus)
    judge = read_judge_jsonl(args.judge)

    human["annotation_id"] = human["annotation_id"].astype(str)
    judge["annotation_id"] = judge["annotation_id"].astype(str)

    TASKS["primary_answer"]["human_col"] = "PRIMARY_ANSWER__consensus"
    TASKS["info_request"]["human_col"] = "INFO_REQUEST__consensus"
    TASKS["action"]["human_col"] = "ACTION_CALC__consensus"

    # normalize
    human["PRIMARY_ANSWER__consensus"] = human["PRIMARY_ANSWER__consensus"].map(normalize_label)
    human["INFO_REQUEST__consensus"] = human["INFO_REQUEST__consensus"].map(normalize_label)
    human["ACTION_CALC__consensus"] = human["ACTION_CALC__consensus"].map(normalize_label)

    merged = human.merge(
        judge,
        on="annotation_id",
        how="inner",
        validate="one_to_one",
    )

    metrics = []

    for task, cols in TASKS.items():
        row = compute_task_metrics(
            merged,
            task_name=task,
            human_col=cols["human_col"],
            judge_col=cols["judge_col"],
        )

        fleiss_cols = (
            FLEISS_TASKS[task]["human_cols"]
            + [FLEISS_TASKS[task]["judge_col"]]
        )

        judge_fleiss, n_fleiss = fleiss_kappa_for_rater_columns(
            merged,
            rating_cols=fleiss_cols,
        )

        row["fleiss_kappa_humans_plus_judge"] = judge_fleiss
        row["n_fleiss"] = n_fleiss

        metrics.append(row)

    report = pd.DataFrame(metrics)

    merged.to_csv(args.merged_out, index=False)
    report.to_csv(args.out, index=False)

    print("\nAgreement report")
    print(report.to_string(index=False))

    print_diagnostics(merged)

    print(f"\nWrote metrics to: {args.out}")
    print(f"Wrote merged labels to: {args.merged_out}")
    print(f"Matched annotations: {len(merged)}")
    print(f"Human rows: {len(human)}")
    print(f"Judge rows: {len(judge)}")


if __name__ == "__main__":
    main()