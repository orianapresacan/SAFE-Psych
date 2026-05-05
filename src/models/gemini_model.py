from google import genai
from google.genai import types
from .base import BaseModel
from .keys import resolve_api_key


class GeminiModel(BaseModel):
    def __init__(self, model_name: str, api_key: str | None = None, thinking_budget=None, **kwargs):
        self.model_name = model_name
        self.thinking_budget = thinking_budget

        api_key = resolve_api_key(
            explicit_key=api_key,
            env_var="GOOGLE_API_KEY",
            label="Google API key",
        )

        self.client = genai.Client(api_key=api_key)

    def generate(self, messages, max_tokens, **kwargs):
        system_parts = []
        contents = []

        for msg in messages:
            role = msg["role"]
            text = msg["content"]

            if role == "system":
                system_parts.append(text)
            elif role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": text}],
                })
            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": text}],
                })
        
        config_kwargs = {
            "system_instruction": "\n\n".join(system_parts) if system_parts else None,
            "max_output_tokens": max_tokens,
        }

        if self.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )

        config = types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        text = getattr(response, "text", None)

        if text and text.strip():
            return text.strip()
        
        candidate = response.candidates[0] if getattr(response, "candidates", None) else None
        finish_reason = getattr(candidate, "finish_reason", None)
        
        raise RuntimeError(
            f"Gemini returned empty text. "
            f"candidate_finish_reason={finish_reason}, "
            f"prompt_feedback={getattr(response, 'prompt_feedback', None)}, "
            f"candidates={getattr(response, 'candidates', None)}"
        )

