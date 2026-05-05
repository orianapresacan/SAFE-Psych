import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


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


def stratified_sample(rows, max_samples, seed=123):
    rng = random.Random(seed)

    buckets = defaultdict(list)
    for row in rows:
        key = (row.get("model_evaluated"), row.get("strategy"))
        buckets[key].append(row)

    bucket_keys = sorted(buckets.keys())
    for key in bucket_keys:
        rng.shuffle(buckets[key])

    num_buckets = len(bucket_keys)
    if num_buckets == 0 or len(rows) <= max_samples:
        rng.shuffle(rows)
        return rows[:max_samples]

    base_quota = max_samples // num_buckets
    remainder = max_samples % num_buckets

    selected = []
    selected_counts = {}

    # First pass: assign base quota
    for key in bucket_keys:
        take = min(base_quota, len(buckets[key]))
        selected.extend(buckets[key][:take])
        selected_counts[key] = take

    # Second pass: distribute leftovers
    remaining_capacity_keys = [
        key for key in bucket_keys
        if selected_counts[key] < len(buckets[key])
    ]

    idx = 0
    while len(selected) < max_samples and remaining_capacity_keys:
        key = remaining_capacity_keys[idx % len(remaining_capacity_keys)]
        already_taken = selected_counts[key]

        if already_taken < len(buckets[key]):
            selected.append(buckets[key][already_taken])
            selected_counts[key] += 1

        remaining_capacity_keys = [
            k for k in bucket_keys
            if selected_counts[k] < len(buckets[k])
        ]
        idx += 1

    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--blind_output_jsonl", required=True)
    parser.add_argument("--key_output_jsonl", required=True)
    parser.add_argument("--id_prefix", default="ann")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    blind_output_path = Path(args.blind_output_jsonl)
    key_output_path = Path(args.key_output_jsonl)

    rows = read_jsonl(input_path)
    rows = [row for row in rows if row.get("strategy") != "structured_actions"]

    sampled_rows = stratified_sample(
        rows=rows,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    blind_rows = []
    key_rows = []

    for i, row in enumerate(sampled_rows, start=1):
        annotation_id = f"{args.id_prefix}_{i:06d}"

        blind_rows.append({
            "annotation_id": annotation_id,
            "model_output_raw": row.get("model_output_raw", ""),
        })

        key_rows.append({
            "annotation_id": annotation_id,
            **row,
        })

    write_jsonl(blind_output_path, blind_rows)
    write_jsonl(key_output_path, key_rows)

    print(f"Input rows after filtering: {len(rows)}")
    print(f"Sampled rows: {len(sampled_rows)}")
    print(f"Wrote {len(blind_rows)} blind annotation rows to {blind_output_path}")
    print(f"Wrote {len(key_rows)} key rows to {key_output_path}")

    # Optional summary
    summary = defaultdict(int)
    for row in sampled_rows:
        summary[(row.get("model_evaluated"), row.get("strategy"))] += 1

    print("\nSample distribution by model and strategy:")
    for (model, strategy), count in sorted(summary.items()):
        print(f"  model={model} | strategy={strategy} -> {count}")


if __name__ == "__main__":
    main()
