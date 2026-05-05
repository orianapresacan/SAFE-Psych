from .prompt_loader import load_prompt


VALID_PRIMARY = {"committed", "differential", "none"}
VALID_INFO_REQUEST = {"specific", "generic_only", "none"}

VALID_ACTIONS = {"Diagnose", "Clarify", "Abstain"}


def build_judge_messages(model_output):
    template = load_prompt("judge")
    prompt = template.replace("{model_response}", model_output)
    return [{"role": "user", "content": prompt}]


def parse_judge_output(raw_text: str) -> dict:
    parsed = {
        "primary_answer": None,
        "info_request": None,
        "icd_codes": [],
        "diagnosis_names": [],
    }

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip().upper()
        value = value.strip()

        if key == "PRIMARY_ANSWER":
            value_lower = value.lower()
            if value_lower in VALID_PRIMARY:
                parsed["primary_answer"] = value_lower

        elif key == "INFO_REQUEST":
            value_lower = value.lower()
            if value_lower in VALID_INFO_REQUEST:
                parsed["info_request"] = value_lower

        elif key == "ICD_CODES":
            if value and value.lower() != "empty":
                parsed["icd_codes"] = [
                    x.strip() for x in value.split(",") if x.strip()
                ]

        elif key == "DIAGNOSIS_NAMES":
            if value and value.lower() != "empty":
                parsed["diagnosis_names"] = [
                    x.strip() for x in value.split(",") if x.strip()
                ]

    return parsed


def derive_action(parsed: dict, is_last_section: bool = False) -> str:
    primary = (parsed.get("primary_answer") or "none").strip().lower()
    info_request = (parsed.get("info_request") or "none").strip().lower()

    if primary == "committed":
        action = "Clarify" if info_request == "specific" else "Diagnose"

    elif primary == "differential":
        action = "Clarify" if info_request == "specific" else "Abstain"

    elif primary == "none":
        action = "Clarify" if info_request == "specific" else "Abstain"

    else:
        action = "Abstain"

    if action == "Clarify" and is_last_section:
        action = "Abstain"

    return action


def run_judge(judge_model, model_output, max_tokens=128, is_last_section: bool = False):
    messages = build_judge_messages(model_output)

    raw_output = judge_model.generate(
        messages=messages,
        do_sample=False,
        max_tokens=max_tokens,
    )

    parsed = parse_judge_output(raw_output)
    parsed["raw_output"] = raw_output
    parsed["action"] = derive_action(parsed, is_last_section=is_last_section)

    return parsed