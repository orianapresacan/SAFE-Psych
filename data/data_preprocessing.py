import argparse
import csv
import json
from pathlib import Path

SECTION_COLUMNS = [
    "presenting_symptoms",
    "psychiatric_history",
    "psychiatric_exam",
    "psychological_exam",
    "secondary_diagnoses",
]


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def parse_csv(input_csv: Path):
    data = []
    data_gt = []

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {
            "id",
            "age_group",
            "sex",
            "presenting_symptoms",
            "psychiatric_history",
            "psychiatric_exam",
            "psychological_exam",
            "secondary_diagnoses",
            "diagnosis_sufficiency",
            "diagnosis_sufficiency_agreement",
            "diagnosis",
            "diagnosis_agreement",
            "earliest_sufficient_section",
        }

        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            row_id = clean(row["id"])

            sections = []
            sections_metadata = []
            compact_order = 1

            for original_order, content_col in enumerate(SECTION_COLUMNS, start=1):
                content = clean(row.get(content_col, ""))
                present = bool(content)
                assigned_order = compact_order if present else None

                if present:
                    sections.append({
                        "order": compact_order,
                        "content": content,
                    })
                    compact_order += 1

                sections_metadata.append({
                    "section_name": content_col,
                    "original_order": original_order,
                    "present": present,
                    "assigned_order": assigned_order,
                    "content": content,
                })

            data.append({
                "id": row_id,
                "sections": sections,
            })

            data_gt.append({
                "id": row_id,
                "age_group": clean(row["age_group"]),
                "sex": clean(row["sex"]),
                "diagnosis_sufficiency": clean(row["diagnosis_sufficiency"]),
                "diagnosis_sufficiency_agreement": clean(row["diagnosis_sufficiency_agreement"]),
                "diagnosis": clean(row["diagnosis"]),
                "diagnosis_agreement": clean(row["diagnosis_agreement"]),
                "earliest_sufficient_section": clean(row["earliest_sufficient_section"]),
                "num_sections_in_input": len(sections),
                "last_section_order": len(sections),
                "sections_metadata": sections_metadata,
            })

    return data, data_gt


def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark CSV to samples.json and data_gt.json"
    )
    parser.add_argument(
        "--input_csv",
        default=None,
        help="Optional input CSV path. Defaults to data/data.csv relative to this script.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # This assumes the script lives inside the data folder.
    data_dir = script_dir

    # This assumes evaluations/ is next to data/.
    evaluations_dir = data_dir.parent / "evaluations"

    input_csv = Path(args.input_csv) if args.input_csv else data_dir / "data.csv"
    output_json = data_dir / "samples.json"
    output_gt_json = evaluations_dir / "data_gt.json"

    evaluations_dir.mkdir(parents=True, exist_ok=True)

    data, data_gt = parse_csv(input_csv)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with output_gt_json.open("w", encoding="utf-8") as f:
        json.dump(data_gt, f, ensure_ascii=False, indent=2)

    print(f"Wrote {output_json}")
    print(f"Wrote {output_gt_json}")


if __name__ == "__main__":
    main()