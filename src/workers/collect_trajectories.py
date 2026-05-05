import argparse
import json
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


def collect_trajectories(runs_dir: Path):
    collected = []

    for traj_path in sorted(runs_dir.rglob("trajectories.jsonl")):
        run_dir = traj_path.parent
        run_name = run_dir.name

        sample_rows = read_jsonl(traj_path)

        for sample_row in sample_rows:
            sample_id = sample_row.get("sample_id")
            strategy = sample_row.get("strategy")
            seed = sample_row.get("seed")
            model_evaluated = sample_row.get("model_evaluated")
            judge_model = sample_row.get("judge_model")
            stopping_reason = sample_row.get("stopping_reason")
            num_sections_total = sample_row.get("num_sections_total")
            num_steps_executed = sample_row.get("num_steps_executed")

            for step in sample_row.get("trajectory", []):
                collected.append({
                    "source_run_name": run_name,
                    "source_run_dir": str(run_dir),
                    "source_file": str(traj_path),
                    "sample_id": sample_id,
                    "strategy": strategy,
                    "seed": seed,
                    "model_evaluated": model_evaluated,
                    "judge_model": judge_model,
                    "stopping_reason": stopping_reason,
                    "num_sections_total": num_sections_total,
                    "num_steps_executed": num_steps_executed,
                    "step_index": step.get("step_index"),
                    "section_order": step.get("section_order"),
                    "revealed_section_orders": step.get("revealed_section_orders"),
                    "is_last_section": step.get("is_last_section"),
                    "user_prompt": step.get("user_prompt"),
                    "conversation_state": step.get("conversation_state"),
                    "model_output_raw": step.get("model_output_raw"),
                    "judge_action": step.get("judge_action"),
                    "judge_output_raw": step.get("judge_output_raw"),
                    "structured_action_raw": step.get("structured_action_raw"),
                })

    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--output_jsonl", required=True)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_path = Path(args.output_jsonl)

    rows = collect_trajectories(runs_dir)
    write_jsonl(output_path, rows)

    print(f"Collected {len(rows)} trajectory steps into {output_path}")


if __name__ == "__main__":
    main()
