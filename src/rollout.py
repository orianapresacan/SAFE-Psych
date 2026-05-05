from .prompt_loader import load_prompt, load_strategy_prompt


def build_sections_text(revealed_sections) -> str:
    return "\n\n".join(sec["content"] for sec in revealed_sections)


def parse_structured_action(raw_text: str) -> str:
    text = raw_text.strip().lower()

    if not text:
        return "Unparsed"

    first_line = text.splitlines()[0].strip()

    if first_line == "clarify":
        return "Clarify"
    if first_line == "abstain":
        return "Abstain"
    if first_line == "diagnose" or first_line.startswith("diagnose:"):
        return "Diagnose"

    return "Unparsed"


def _get_final_notice_name(strategy: str) -> str | None:
    if strategy == "sequential_info_abstention_aware":
        return "final_step_abstention_notice"
    if strategy in {
        "sequential_info_clarification_only",
        "sequential_info_neutral_prompting",
    }:
        return "final_step_notice"
    return None


def build_user_prompt(strategy, revealed_sections, new_section_text, is_first, is_last):
    sections_text = build_sections_text(revealed_sections)

    if strategy.startswith("full_info_"):
        template = load_prompt(strategy)
        if "{sections}" not in template:
            raise ValueError(f"Prompt template must contain {{sections}}: {strategy}")
        return template.format(sections=sections_text).strip()

    if strategy == "structured_actions":
        prompt_name = "structured_action_final" if is_last else "structured_action_intermediate"
        template = load_strategy_prompt("structured_action", prompt_name)
        if "{sections}" not in template:
            raise ValueError(
                f"Prompt template must contain {{sections}}: structured_action/{prompt_name}"
            )
        return template.format(sections=sections_text).strip()

    if is_first:
        template = load_strategy_prompt(strategy, "initial_instruction")
        if "{sections}" not in template:
            raise ValueError(
                f"Prompt template must contain {{sections}}: {strategy}/initial_instruction"
            )
        user_prompt = template.format(sections=sections_text).strip()
    else:
        template = load_strategy_prompt(strategy, "additional_info_prefix")
        if "{sections}" not in template:
            raise ValueError(
                f"Prompt template must contain {{sections}}: {strategy}/additional_info_prefix"
            )
        user_prompt = template.format(sections=new_section_text).strip()

    if is_last:
        final_notice_name = _get_final_notice_name(strategy)
        if final_notice_name is not None:
            final_notice = load_strategy_prompt(strategy, final_notice_name)
            user_prompt = f"{user_prompt}\n\n{final_notice}"

    return user_prompt