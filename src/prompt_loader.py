from pathlib import Path


PROMPTS_DIR = Path("prompts")


def _read_text(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_prompt(name: str) -> str:
    return _read_text(PROMPTS_DIR / f"{name}.txt")


def load_strategy_prompt(strategy: str, name: str) -> str:
    return _read_text(PROMPTS_DIR / strategy / f"{name}.txt")