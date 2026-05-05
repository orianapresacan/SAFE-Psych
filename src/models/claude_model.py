import anthropic
from .base import BaseModel
from .keys import resolve_api_key


class ClaudeModel(BaseModel):
    def __init__(self, model_name: str, api_key: str | None = None, **kwargs):
        self.model_name = model_name

        api_key = resolve_api_key(
            explicit_key=api_key,
            env_var="ANTHROPIC_API_KEY",
            label="Anthropic API key",
        )

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, messages, max_tokens, **kwargs):
        system_blocks = []
        user_assistant_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_blocks.append(msg["content"])
            else:
                user_assistant_messages.append(msg)

        request_kwargs = {
            "model": self.model_name,
            "messages": user_assistant_messages,
            "max_tokens": max_tokens,
        }

        if system_blocks:
            request_kwargs["system"] = "\n\n".join(system_blocks)

        response = self.client.messages.create(**request_kwargs)

        text_parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ]

        full_text = "".join(text_parts).strip()
        if full_text:
            return full_text

        stop_reason = getattr(response, "stop_reason", None)
        raise ValueError(f"Claude returned no text content. stop_reason={stop_reason}")