from pathlib import Path
import json
import random
import subprocess
import sys
import time
from copy import deepcopy
import os

from .dataset import load_dataset
from .prompt_loader import load_prompt
from .rollout import build_user_prompt, parse_structured_action


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_subprocess(module_name: str, args: list[str], extra_env: dict | None = None) -> None:
    cmd = [sys.executable, "-m", module_name] + args
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, check=True, env=env)


def run_inference(
    data_path,
    model_key,
    judge_key,
    run_dir,
    strategy,
    num_samples=None,
    seed=None,
    judge_seed=123,
    device="cuda",
    model_context_window=8192,
    judge_context_window=4096,
    gpu_memory_utilization=0.90,
    model_do_sample=True,
    model_temperature=0.7,
    model_top_p=0.95,
    model_max_tokens=512,
    judge_max_tokens=128,
    save_every=25,
    api_concurrency=8,
):
    total_start = time.perf_counter()

    samples = load_dataset(data_path)

    subprocess_env = {}

    if seed is not None:
        random.seed(seed)
        random.shuffle(samples)

    if num_samples is not None:
        samples = samples[:num_samples]

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    tmp_path = run_path / "_tmp"
    tmp_path.mkdir(parents=True, exist_ok=True)

    traj_path = run_path / "trajectories.jsonl"
    timing_path = run_path / "timing.json"

    system_prompt = load_prompt("system_instruction")

    trajectories = {}
    sample_lookup = {sample.sample_id: sample for sample in samples}
    sample_start_times = {}
    sample_timings = []

    for sample in samples:
        trajectories[sample.sample_id] = {
            "sample_id": sample.sample_id,
            "num_sections_total": len(sample.sections),
            "num_steps_executed": 0,
            "stopping_reason": None,
            "trajectory": [],
            "_conversation": [{"role": "system", "content": system_prompt}],
            "_revealed_sections": [],
            "_stopped": False,
        }
        sample_start_times[sample.sample_id] = time.perf_counter()

    max_steps = max((len(s.sections) for s in samples), default=0)

    for step_idx in range(max_steps):
        active_rows = []

        for sample in samples:
            state = trajectories[sample.sample_id]
            if state["_stopped"]:
                continue
            if strategy.startswith("full_info_"):
                if step_idx > 0:
                    continue

                is_first = True
                is_last = True
                section = sample.sections[0]
                section_text = section.content.strip()

                state["_revealed_sections"] = [
                    {"order": sec.order, "content": sec.content.strip()}
                    for sec in sample.sections
                ]
            else:
                section = sample.sections[step_idx]
                is_first = step_idx == 0
                is_last = step_idx == len(sample.sections) - 1
                section_text = section.content.strip()

                state["_revealed_sections"].append({
                    "order": section.order,
                    "content": section_text,
                })

            user_prompt = build_user_prompt(
                strategy=strategy,
                revealed_sections=state["_revealed_sections"],
                new_section_text=section_text,
                is_first=is_first,
                is_last=is_last,
            )

            state["_conversation"].append({"role": "user", "content": user_prompt})

            active_rows.append({
                "sample_id": sample.sample_id,
                "step_index": step_idx + 1,
                "section_order": section.order,
                "is_last": is_last,
                "user_prompt": user_prompt,
                "conversation": deepcopy(state["_conversation"]),
                "revealed_sections": deepcopy(state["_revealed_sections"]),
            })

        if not active_rows:
            break

        step_input_path = tmp_path / f"step_{step_idx+1:02d}_generate_input.jsonl"
        step_output_path = tmp_path / f"step_{step_idx+1:02d}_generate_output.jsonl"

        _write_jsonl(step_input_path, active_rows)

        _run_subprocess(
            "src.workers.generate_batch",
            [
                "--input_jsonl", str(step_input_path),
                "--output_jsonl", str(step_output_path),
                "--model_key", model_key,
                "--device", device,
                "--context_window", str(model_context_window),
                "--gpu_memory_utilization", str(gpu_memory_utilization),
                "--max_tokens", str(model_max_tokens),
                "--do_sample", "false" if strategy == "structured_actions" else ("true" if model_do_sample else "false"),
                "--temperature", str(model_temperature),
                "--top_p", str(model_top_p),
                "--api_concurrency", str(api_concurrency),
                "--save_every", str(save_every),
                "--seed", str(seed if seed is not None else 0),
            ],
            extra_env=subprocess_env,
        )

        generation_rows = _read_jsonl(step_output_path)
        generation_by_sample = {row["sample_id"]: row for row in generation_rows}

        if strategy == "structured_actions":
            for row in active_rows:
                sample_id = row["sample_id"]
                state = trajectories[sample_id]
                gen_row = generation_by_sample[sample_id]
                model_output = gen_row["model_output"]

                state["_conversation"].append({"role": "assistant", "content": model_output})

                action = parse_structured_action(model_output)

                step_record = {
                    "step_index": row["step_index"],
                    "section_order": row["section_order"],
                    "revealed_section_orders": [s["order"] for s in state["_revealed_sections"]],
                    "user_prompt": row["user_prompt"],
                    "conversation_state": deepcopy(state["_conversation"]),
                    "model_output_raw": model_output,
                    "judge_output_raw": None,
                    "judge_action": action,
                    "is_last_section": row["is_last"],
                    "structured_action_raw": model_output,
                }

                state["trajectory"].append(step_record)
                state["num_steps_executed"] += 1

                if action == "Diagnose":
                    state["stopping_reason"] = "Diagnose"
                    state["_stopped"] = True
                elif action == "Abstain":
                    state["stopping_reason"] = "Abstain"
                    state["_stopped"] = True
                elif action == "Unparsed":
                    state["stopping_reason"] = "UnparsedAction"
                    state["_stopped"] = True

        else:
            judge_input_rows = []

            for row in active_rows:
                sample_id = row["sample_id"]
                state = trajectories[sample_id]
                gen_row = generation_by_sample[sample_id]
                model_output = gen_row["model_output"]

                state["_conversation"].append({"role": "assistant", "content": model_output})

                judge_input_rows.append({
                    "sample_id": sample_id,
                    "model_output_raw": model_output,
                    "is_last_section": row["is_last"],
                })

            judge_input_path = tmp_path / f"step_{step_idx+1:02d}_judge_input.jsonl"
            judge_output_path = tmp_path / f"step_{step_idx+1:02d}_judge_output.jsonl"

            _write_jsonl(judge_input_path, judge_input_rows)

            _run_subprocess(
                "src.workers.judge_batch_parallel",
                [
                    "--input_jsonl", str(judge_input_path),
                    "--output_jsonl", str(judge_output_path),
                    "--judge_key", judge_key,
                    "--device", device,
                    "--context_window", str(judge_context_window),
                    "--gpu_memory_utilization", str(gpu_memory_utilization),
                    "--max_tokens", str(judge_max_tokens),
                    "--seed", str(judge_seed),
                ],
                extra_env=subprocess_env,
            )

            judge_rows = _read_jsonl(judge_output_path)
            judge_by_sample = {row["sample_id"]: row for row in judge_rows}

            for row in active_rows:
                sample_id = row["sample_id"]
                state = trajectories[sample_id]
                judge_row = judge_by_sample[sample_id]

                action = judge_row["action"]
                judge_output = judge_row.get("judge_output_raw", "")
                structured_action = judge_row.get("structured_action_raw")

                if isinstance(structured_action, dict):
                    structured_action = {
                        k: v
                        for k, v in structured_action.items()
                        if k != "raw_output"
                    }

                model_output = generation_by_sample[sample_id]["model_output"]

                if strategy.startswith("full_info_") and action == "Clarify":
                    action = "Abstain"

                step_record = {
                    "step_index": row["step_index"],
                    "section_order": row["section_order"],
                    "revealed_section_orders": [s["order"] for s in state["_revealed_sections"]],
                    "user_prompt": row["user_prompt"],
                    "conversation_state": deepcopy(state["_conversation"]),
                    "model_output_raw": model_output,
                    "judge_output_raw": judge_output,
                    "judge_action": action,
                    "is_last_section": row["is_last"],
                    "structured_action_raw": structured_action,
                }

                state["trajectory"].append(step_record)
                state["num_steps_executed"] += 1

                if strategy.startswith("full_info_"):
                    state["stopping_reason"] = action
                    state["_stopped"] = True
                elif action == "Diagnose":
                    state["stopping_reason"] = "Diagnose"
                    state["_stopped"] = True
                elif action == "Abstain":
                    state["stopping_reason"] = "Abstain"
                    state["_stopped"] = True

        for row in active_rows:
            sample_id = row["sample_id"]
            state = trajectories[sample_id]
            if state["_stopped"] and "_elapsed_recorded" not in state:
                sample_elapsed = time.perf_counter() - sample_start_times[sample_id]
                sample_timings.append({
                    "sample_id": sample_id,
                    "seconds": sample_elapsed,
                })
                state["_elapsed_recorded"] = True

    for sample in samples:
        state = trajectories[sample.sample_id]
        if state["stopping_reason"] is None:
            state["stopping_reason"] = "Finished"

        if "_elapsed_recorded" not in state:
            sample_elapsed = time.perf_counter() - sample_start_times[sample.sample_id]
            sample_timings.append({
                "sample_id": sample.sample_id,
                "seconds": sample_elapsed,
            })
            state["_elapsed_recorded"] = True

    with open(traj_path, "w", encoding="utf-8") as traj_file:
        for sample in samples:
            state = trajectories[sample.sample_id]

            out = {
                "sample_id": state["sample_id"],
                "num_sections_total": state["num_sections_total"],
                "num_steps_executed": state["num_steps_executed"],
                "stopping_reason": state["stopping_reason"],
                "trajectory": state["trajectory"],
                "model_evaluated": model_key,
                "judge_model": None if strategy == "structured_actions" else judge_key,
                "judge_seed": judge_seed if strategy != "structured_actions" else None,
                "strategy": strategy,
                "seed": seed,
                "model_do_sample": False if strategy == "structured_actions" else model_do_sample,
                "model_temperature": None if (strategy == "structured_actions" or not model_do_sample) else model_temperature,
                "model_top_p": None if (strategy == "structured_actions" or not model_do_sample) else model_top_p,
                "model_max_tokens": model_max_tokens,
                "judge_do_sample": False if strategy != "structured_actions" else None,
                "judge_max_tokens": judge_max_tokens if strategy != "structured_actions" else None,
            }
            traj_file.write(json.dumps(out, ensure_ascii=False) + "\n")

    total_elapsed = time.perf_counter() - total_start

    timing_summary = {
        "total_time_seconds": total_elapsed,
        "num_samples": len(samples),
        "avg_time_per_sample_seconds": total_elapsed / len(samples) if samples else 0.0,
        "sample_times": sample_timings,
    }

    timing_path.write_text(
        json.dumps(timing_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Total time: {total_elapsed:.2f}s")
    if samples:
        print(f"Average per sample: {total_elapsed / len(samples):.2f}s")