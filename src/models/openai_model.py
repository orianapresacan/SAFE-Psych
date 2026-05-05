from openai import OpenAI

from .base import BaseModel
from .keys import resolve_api_key


class OpenAIModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        seed: int | None = None,
        deterministic: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.seed = seed
        self.deterministic = deterministic

        api_key = resolve_api_key(
            explicit_key=api_key,
            env_var="OPENAI_API_KEY",
            label="OpenAI API key",
        )

        self.client = OpenAI(api_key=api_key)

    def generate(self, messages, max_tokens, **kwargs):
        seed = kwargs.pop("seed", self.seed)

        request = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }

        if seed is not None:
            request["seed"] = seed

        if self.deterministic:
            request["temperature"] = 0
            request["top_p"] = 1

        response = self.client.chat.completions.create(**request)
        return (response.choices[0].message.content or "").strip()