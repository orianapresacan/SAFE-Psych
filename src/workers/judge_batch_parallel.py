import argparse
import json
import asyncio
import random
from pathlib import Path

from src.judge import run_judge
from src.models.factory import create_model


def row_key(row):
    return (str(row.get("sample_id")), int(row.get("step_index", 0)))


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, row):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


async def safe_run_judge(row, judge, args, sem):
    model_output = (row.get("model_output_raw") or "").strip()
    is_last_section = bool(row.get("is_last_section", False))

    if not model_output:
        return {
            **row,
            "action": "",
            "judge_model": args.judge_key,
            "judge_output_raw": "",
            "structured_action_raw": None,
        }

    max_retries = args.max_retries

    async with sem:
        for attempt in range(max_retries):
            try:
                judge_result = await asyncio.to_thread(
                    run_judge,
                    judge_model=judge,
                    model_output=model_output,
                    max_tokens=args.max_tokens,
                    is_last_section=is_last_section,
                )

                return {
                    **row,
                    "action": judge_result.get("action"),
                    "judge_model": args.judge_key,
                    "judge_output_raw": judge_result.get("raw_output", ""),
                    "structured_action_raw": {
                        "primary_answer": judge_result.get("primary_answer"),
                        "info_request": judge_result.get("info_request"),
                        "icd_codes": judge_result.get("icd_codes"),
                        "diagnosis_names": judge_result.get("diagnosis_names"),
                        "action": judge_result.get("action"),
                    },
                }

            except Exception as e:
                msg = str(e).lower()
                is_rate_limit = "429" in msg or "rate limit" in msg or "rate_limit" in msg

                if not is_rate_limit or attempt == max_retries - 1:
                    return {
                        **row,
                        "action": None,
                        "judge_model": args.judge_key,
                        "judge_output_raw": "",
                        "structured_action_raw": None,
                        "error": str(e),
                    }

                wait = min(args.max_backoff, (2 ** attempt) + random.uniform(0, 1))
                print(f"Rate limit for {row_key(row)}. Retrying in {wait:.2f}s")
                await asyncio.sleep(wait)


async def run_all(rows, judge, args):
    existing_outputs = read_jsonl(args.output_jsonl)
    done_keys = {row_key(r) for r in existing_outputs if "error" not in r}

    rows_to_process = [r for r in rows if row_key(r) not in done_keys]

    print(f"Input rows: {len(rows)}")
    print(f"Already completed: {len(done_keys)}")
    print(f"Remaining: {len(rows_to_process)}")

    sem = asyncio.Semaphore(args.concurrency)

    tasks = [
        safe_run_judge(row, judge, args, sem)
        for row in rows_to_process
    ]

    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        append_jsonl(args.output_jsonl, result)

        completed += 1
        if completed % 25 == 0:
            print(f"Saved {completed}/{len(rows_to_process)} new rows")

    print(f"Finished. Saved {completed} new rows.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--judge_key", required=True)

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context_window", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_tokens", type=int, default=64)

    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--google_api_key", default=None)
    parser.add_argument("--anthropic_api_key", default=None)

    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=12)
    parser.add_argument("--max_backoff", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    random.seed(args.seed)

    rows = read_jsonl(args.input_jsonl)

    judge = create_model(
        model_key=args.judge_key,
        device=args.device,
        max_model_len=args.context_window,
        gpu_memory_utilization=args.gpu_memory_utilization,
        openai_api_key=args.openai_api_key,
        google_api_key=args.google_api_key,
        anthropic_api_key=args.anthropic_api_key,
        seed=args.seed,
        deterministic=True,
    )

    asyncio.run(run_all(rows, judge, args))

    print(f"Wrote/resumed outputs at: {args.output_jsonl}")


if __name__ == "__main__":
    main()