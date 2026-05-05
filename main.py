import argparse
import random
from datetime import datetime

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from src.inference import run_inference


DEFAULT_SEED = 123
JUDGE_SEED = 123

MODEL_TEMPERATURE = 0.7
MODEL_TOP_P = 0.95
MODEL_MAX_TOKENS = 512

JUDGE_MAX_TOKENS = 128
MODEL_CONTEXT_WINDOW = 8192
JUDGE_CONTEXT_WINDOW = 4096
GPU_MEMORY_UTILIZATION = 0.9

SAVE_EVERY = 25
API_CONCURRENCY = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def build_run_name(model_name: str, strategy: str, judge_name: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{strategy}_judge-{judge_name}_seed-{seed}_{ts}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "llama3.1-8b",
            "mistral-small-3.1-24B",
            "gemma-3-4b",
            "gemma-3-12b",
            "gemma-3-27b",
            "gemma-4-31b",
            "medgemma-27b",
            "med42-8b",
            "qwen_3-32b",
            "gpt-5.4",
            "claude-opus-4.6",
            "gemini-2.5-pro",
            "gpt-4o-mini",
            "gemini-2.5-flash-no-thinking",
            "gemini-2.5-flash-thinking"
        ],
    )
    parser.add_argument(
        "--judge",
        required=True,
        choices=[
            "llama3.1-8b",
            "qwen_3-32b",
            "gemma-3-27b",
            "qwen_2.5-14b",
            "mistral-small-3.1-24B",
            "gemma-3-12b",
            "gpt-4o-mini",
            "gpt-5.4"
        ],
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=[
            "full_info_no_abstention",
            "full_info_abstention_aware",
            "sequential_info_neutral_prompting",
            "sequential_info_abstention_aware",
            "sequential_info_clarification_only",
            "structured_actions",
        ],
    )
    parser.add_argument("--data_path", default="data/samples.json")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed to use (default: {DEFAULT_SEED})",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    run_name = build_run_name(args.model, args.strategy, args.judge, args.seed)
        
    run_inference(
        data_path=args.data_path,
        model_key=args.model,
        judge_key=args.judge,
        run_dir=f"runs/{run_name}",
        strategy=args.strategy,
        num_samples=args.num_samples,
        seed=args.seed,
        judge_seed=JUDGE_SEED,
        device=args.device,
        model_context_window=MODEL_CONTEXT_WINDOW,
        judge_context_window=JUDGE_CONTEXT_WINDOW,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        model_do_sample=True,
        model_temperature=MODEL_TEMPERATURE,
        model_top_p=MODEL_TOP_P,
        model_max_tokens=MODEL_MAX_TOKENS,
        judge_max_tokens=JUDGE_MAX_TOKENS,
        save_every=SAVE_EVERY,
        api_concurrency=API_CONCURRENCY,
    )

if __name__ == "__main__":
    main()