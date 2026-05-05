import argparse
import json
from pathlib import Path

from src.judge import run_judge
from src.models.factory import create_model


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--judge_key", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context_window", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--google_api_key", default=None)
    parser.add_argument("--anthropic_api_key", default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)

    judge = create_model(
        model_key=args.judge_key,
        device=args.device,
        max_model_len=args.context_window,
        gpu_memory_utilization=args.gpu_memory_utilization,
        openai_api_key=args.openai_api_key,
        google_api_key=args.google_api_key,
        anthropic_api_key=args.anthropic_api_key,
        deterministic=True,
    )

    outputs = []

    for row in rows:
        model_output = (row.get("model_output_raw") or "").strip()
        is_last_section = bool(row.get("is_last_section", False))

        if not model_output:
            outputs.append({
                **row,
                "action": "Abstain",
                "judge_model": args.judge_key,
                "judge_output_raw": "",
                "structured_action_raw": None,
            })
            continue

        judge_result = run_judge(
            judge_model=judge,
            model_output=model_output,
            max_tokens=args.max_tokens,
            is_last_section=is_last_section,
        )

        outputs.append({
            **row,
            "action": judge_result.get("action"),
            "judge_model": args.judge_key,
            "judge_output_raw": judge_result.get("raw_output", ""),
            "structured_action_raw": judge_result,
        })

    write_jsonl(args.output_jsonl, outputs)
    print(f"Wrote {len(outputs)} judged rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()