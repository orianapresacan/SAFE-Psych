import argparse
import json
import asyncio
from pathlib import Path
import random 

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

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tmp_path.replace(output_path)


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def is_api_model(model_key: str) -> bool:
    return model_key.startswith(("claude", "gemini", "gpt"))


def generate_one(model, conversation, do_sample, temperature, top_p, max_tokens):
    generate_kwargs = {}
    if getattr(model, "model_name", "").startswith("Qwen/Qwen3"):
        generate_kwargs["enable_thinking"] = False

    return model.generate(
        messages=conversation,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        **generate_kwargs,
    )


async def generate_one_async(row, model, args, do_sample, sem):
    async with sem:
        for attempt in range(12):
            try:
                text = await asyncio.to_thread(
                    generate_one,
                    model,
                    row["conversation"],
                    do_sample,
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                )

                if text and text.strip():
                    return {
                        "sample_id": row["sample_id"],
                        "model_output": text.strip(),
                    }

                raise RuntimeError("Empty model output")

            except Exception as e:
                msg = str(e).lower()

                is_retryable = (
                    "429" in msg
                    or "rate limit" in msg
                    or "rate_limit" in msg
                    or "resource_exhausted" in msg
                    or "too many requests" in msg
                    or "empty model output" in msg
                    or "timeout" in msg
                    or "temporarily unavailable" in msg
                    or "503" in msg
                    or "unavailable" in msg
                    or "500" in msg
                )

                if not is_retryable or attempt == 11:
                    return {
                        "sample_id": row["sample_id"],
                        "model_output": "",
                        "error": str(e),
                    }

                wait = min(60, (2 ** attempt) + random.uniform(0, 1))
                print(
                    f"Retryable generation error for {row['sample_id']}: {e}. "
                    f"Retrying in {wait:.2f}s"
                )
                await asyncio.sleep(wait)


async def generate_api_rows(rows, model, args, do_sample):
    sem = asyncio.Semaphore(args.api_concurrency)

    tasks = [
        generate_one_async(row, model, args, do_sample, sem)
        for row in rows
    ]

    outputs = []
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        outputs.append(result)
        completed += 1

        if completed % args.save_every == 0:
            write_jsonl(args.output_jsonl, outputs)
            print(f"Saved {completed}/{len(rows)} generations")

    return outputs


def generate_local_rows(rows, model, args, do_sample):
    outputs = []

    for i, row in enumerate(rows, start=1):
        text = generate_one(
            model=model,
            conversation=row["conversation"],
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        outputs.append({
            "sample_id": row["sample_id"],
            "model_output": text,
        })

        if i % args.save_every == 0:
            write_jsonl(args.output_jsonl, outputs)
            print(f"Saved {i}/{len(rows)} generations")

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context_window", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--do_sample", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--google_api_key", default=None)
    parser.add_argument("--anthropic_api_key", default=None)
    parser.add_argument("--api_concurrency", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)

    if args.seed is not None:
        random.seed(args.seed)

    model = create_model(
        model_key=args.model_key,
        device=args.device,
        max_model_len=args.context_window,
        gpu_memory_utilization=args.gpu_memory_utilization,
        openai_api_key=args.openai_api_key,
        google_api_key=args.google_api_key,
        anthropic_api_key=args.anthropic_api_key,
        seed=args.seed,
    )

    do_sample = str_to_bool(args.do_sample)

    if is_api_model(args.model_key):
        outputs = asyncio.run(generate_api_rows(rows, model, args, do_sample))
    else:
        outputs = generate_local_rows(rows, model, args, do_sample)

    write_jsonl(args.output_jsonl, outputs)


if __name__ == "__main__":
    main()