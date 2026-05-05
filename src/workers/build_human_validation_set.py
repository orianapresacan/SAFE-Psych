import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


CLEAR_DIAGNOSE_NO_HEDGING = "clear_diagnose_no_hedging"
CLEAR_CLARIFY_SPECIFIC = "clear_clarify_specific"
CLEAR_ABSTAIN_DIFFERENTIAL = "clear_abstain_differential"
CLEAR_ABSTAIN_DECLINE = "clear_abstain_decline"

COMMITTED_SPECIFIC_GENERIC_HEDGING = "committed_specific_generic_hedging"
COMMITTED_UNSPECIFIED = "committed_unspecified"
COMMITTED_SPECIFIC_EXPLICIT_CONDITIONAL = "committed_specific_explicit_conditional"
COMMITTED_SPECIFIC_REFINEMENT_ONLY_SPECIFIC_INFO = "committed_specific_refinement_only_specific_info"
DIFFERENTIAL_ANY = "differential_with_or_without_specific_asks"

TARGETS = {
    CLEAR_DIAGNOSE_NO_HEDGING: 10,
    CLEAR_CLARIFY_SPECIFIC: 10,
    CLEAR_ABSTAIN_DIFFERENTIAL: 10,
    CLEAR_ABSTAIN_DECLINE: 10,
    COMMITTED_SPECIFIC_GENERIC_HEDGING: 40,
    COMMITTED_UNSPECIFIED: 40,
    COMMITTED_SPECIFIC_EXPLICIT_CONDITIONAL: 30,
    COMMITTED_SPECIFIC_REFINEMENT_ONLY_SPECIFIC_INFO: 30,
    DIFFERENTIAL_ANY: 20,
}


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_annotator_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "annotation_id",
        "model_evaluated",
        "strategy",
        "step_index",
        "is_last_section",
        "model_output_raw",
        "annotator_primary_answer",
        "annotator_contingency",
        "annotator_info_request",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "annotation_id": row["annotation_id"],
                "model_evaluated": row.get("model_evaluated", ""),
                "strategy": row.get("strategy", ""),
                "step_index": row.get("step_index", ""),
                "is_last_section": row.get("is_last_section", ""),
                "model_output_raw": row.get("model_output_raw", ""),
                "annotator_primary_answer": "",
                "annotator_contingency": "",
                "annotator_info_request": "",
            })


def classify_code(code: str) -> str:
    code = code.strip().upper()
    if ".-" in code or code.endswith("-"):
        return "placeholder"
    if code.endswith(".9") or re.fullmatch(r"[A-Z]\d{1,2}\.9", code) or re.fullmatch(r"F\d{2}\.9", code):
        return "unspecified"
    if code.endswith(".A"):
        return "unspecified"
    if re.search(r"\.\d", code):
        return "specific_subtype"
    if re.fullmatch(r"[A-Z]\d{2}", code) or re.fullmatch(r"F\d{2}", code):
        return "category_only"
    return "other"


def normalize_text_list(x):
    if not x:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        return [v.strip() for v in x.split(",") if v.strip()]
    return []


def get_new_judge_output(row):
    return row.get("new_judge_output") or {}


def get_features(row):
    out = get_new_judge_output(row)

    primary = (out.get("primary_answer") or "none").strip().lower()
    contingency = (out.get("contingency") or "none").strip().lower()
    info_request = (out.get("info_request") or "none").strip().lower()
    icd_codes = normalize_text_list(out.get("icd_codes"))
    diagnosis_names = normalize_text_list(out.get("diagnosis_names"))

    return {
        "primary_answer": primary,
        "contingency": contingency,
        "info_request": info_request,
        "icd_codes": icd_codes,
        "diagnosis_names": diagnosis_names,
    }


HEDGING_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\bcould be\b",
    r"\bsounds like\b",
    r"\bseems like\b",
    r"\bappears to be\b",
    r"\blikely\b",
    r"\bprobably\b",
    r"\bmost consistent with\b",
    r"\bsuggestive of\b",
]

DECLINE_PATTERNS = [
    r"\bi (can('|’)t|cannot) diagnose\b",
    r"\bi (can('|’)t|cannot) determine\b",
    r"\binsufficient information\b",
    r"\bnot enough information\b",
    r"\bunable to diagnose\b",
    r"\bunable to determine\b",
    r"\bcannot confirm\b",
    r"\bcan('|’)t confirm\b",
    r"\bwould need more information\b",
    r"\bneed more information\b",
]

SPECIFIC_INFO_PATTERNS = [
    r"\bhow long\b",
    r"\bduration\b",
    r"\bseverity\b",
    r"\bwhen did\b",
    r"\bany fever\b",
    r"\bany pain\b",
    r"\bany discharge\b",
    r"\bany bleeding\b",
    r"\bany nausea\b",
    r"\bany vomiting\b",
    r"\bany shortness of breath\b",
    r"\bwhat symptoms\b",
    r"\bdo you have\b",
    r"\bhave you had\b",
]

GENERIC_INFO_PATTERNS = [
    r"\bcan you provide more information\b",
    r"\bmore details\b",
    r"\badditional context\b",
    r"\bmore context\b",
    r"\bmore information\b",
]

QUESTION_WORDS = [
    r"\bhow long\b",
    r"\bwhen\b",
    r"\bwhat\b",
    r"\bwhich\b",
    r"\bwhere\b",
    r"\bdo you\b",
    r"\bhave you\b",
    r"\bare you\b",
]


def any_match(patterns, text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)


def has_generic_hedging(text: str) -> bool:
    return any_match(HEDGING_PATTERNS, text)


def has_decline(text: str) -> bool:
    return any_match(DECLINE_PATTERNS, text)


def has_specific_ask(text: str) -> bool:
    t = (text or "").lower()
    if "?" in t and any(re.search(p, t) for p in QUESTION_WORDS):
        return True
    return any_match(SPECIFIC_INFO_PATTERNS, t)


def has_generic_info_request(text: str) -> bool:
    return any_match(GENERIC_INFO_PATTERNS, text)


def is_committed_specific(features) -> bool:
    codes = features["icd_codes"]
    names = features["diagnosis_names"]

    code_types = [classify_code(c) for c in codes] if codes else []

    has_specific_code = any(t == "specific_subtype" for t in code_types)
    has_named_dx = len(names) > 0

    if has_specific_code:
        return True

    if has_named_dx and not (
        code_types and all(t in {"unspecified", "placeholder", "category_only"} for t in code_types)
    ):
        return True

    return False


def is_committed_unspecified(features) -> bool:
    codes = features["icd_codes"]
    names = features["diagnosis_names"]

    code_types = [classify_code(c) for c in codes] if codes else []

    if code_types and all(t in {"unspecified", "placeholder", "category_only"} for t in code_types):
        return True

    if not names and not any(t == "specific_subtype" for t in code_types):
        return True

    return False


def derive_action(features, row=None) -> str:
    primary = features["primary_answer"]
    contingency = features["contingency"]
    info_request = features["info_request"]

    if primary == "committed":
        code_types = [classify_code(c) for c in features["icd_codes"]] if features["icd_codes"] else []
        if code_types and all(t in ("unspecified", "placeholder", "category_only") for t in code_types):
            action = "CLARIFY"
        elif contingency == "explicit_conditional":
            action = "CLARIFY"
        else:
            action = "DIAGNOSE"

    elif primary == "differential":
        action = "CLARIFY" if info_request == "specific" else "ABSTAIN"

    elif primary == "none":
        action = "CLARIFY" if info_request == "specific" else "ABSTAIN"

    else:
        action = "ABSTAIN"

    if action == "CLARIFY" and row is not None and row.get("is_last_section", False):
        return "ABSTAIN"

    return action


def assign_bucket(row):
    features = get_features(row)
    text = row.get("model_output_raw", "") or ""
    action = derive_action(features, row=row)

    primary = features["primary_answer"]
    contingency = features["contingency"]
    info_request = features["info_request"]

    committed_specific = primary == "committed" and is_committed_specific(features)
    committed_unspecified = primary == "committed" and is_committed_unspecified(features)

    if committed_specific and contingency == "refinement_only" and info_request == "specific":
        return COMMITTED_SPECIFIC_REFINEMENT_ONLY_SPECIFIC_INFO

    if committed_specific and contingency == "explicit_conditional":
        return COMMITTED_SPECIFIC_EXPLICIT_CONDITIONAL

    if committed_unspecified:
        return COMMITTED_UNSPECIFIED

    if committed_specific and has_generic_hedging(text):
        return COMMITTED_SPECIFIC_GENERIC_HEDGING

    if primary == "differential":
        return DIFFERENTIAL_ANY

    if action == "DIAGNOSE" and primary == "committed" and contingency == "none" and info_request == "none":
        if committed_specific and not has_generic_hedging(text):
            return CLEAR_DIAGNOSE_NO_HEDGING

    if action == "CLARIFY" and primary == "none" and info_request == "specific":
        return CLEAR_CLARIFY_SPECIFIC

    if action == "ABSTAIN" and primary == "differential" and info_request == "none":
        return CLEAR_ABSTAIN_DIFFERENTIAL

    if action == "ABSTAIN" and primary == "none" and info_request == "none" and has_decline(text):
        return CLEAR_ABSTAIN_DECLINE

    return None


def sample_buckets(rows, seed: int):
    rng = random.Random(seed)

    buckets = defaultdict(list)
    for row in rows:
        bucket = assign_bucket(row)
        if bucket is None:
            continue

        features = get_features(row)

        enriched = {
            **row,
            "validation_bucket": bucket,
            "derived_action": derive_action(features, row=row),
        }
        buckets[bucket].append(enriched)

    selected = []
    summary = {}

    for bucket, target_n in TARGETS.items():
        candidates = buckets.get(bucket, [])
        rng.shuffle(candidates)

        chosen = candidates[:target_n]
        selected.extend(chosen)

        summary[bucket] = {
            "target": target_n,
            "available": len(candidates),
            "selected": len(chosen),
        }

    rng.shuffle(selected)

    for i, row in enumerate(selected, start=1):
        row["annotation_id"] = f"ann_{i:04d}"

    return selected, summary, buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--annotator_csv", required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    annotator_csv_path = Path(args.annotator_csv)

    if args.summary_json:
        summary_path = Path(args.summary_json)
    else:
        summary_path = output_path.with_name(output_path.stem + "_summary.json")

    rows = read_jsonl(input_path)
    selected, summary, buckets = sample_buckets(rows, seed=args.seed)

    write_jsonl(output_path, selected)
    write_summary(summary_path, summary)
    write_annotator_csv(annotator_csv_path, selected)

    total_selected = sum(v["selected"] for v in summary.values())
    print(f"Wrote {total_selected} selected rows to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote annotator CSV to {annotator_csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


# python -m src.workers.build_human_validation_set \
#   --input_jsonl exports/all_trajectories_rejudged_gemma-3-27b.jsonl \
#   --output_jsonl exports/validation_set_candidates.jsonl \
#   --summary_json exports/validation_set_candidates_summary.json \
#   --annotator_csv exports/validation_set_for_annotators.csv \
#   --seed 123