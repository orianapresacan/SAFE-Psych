from .registry import MODEL_REGISTRY


def create_model(
    model_key: str,
    device: str = "cuda",
    openai_api_key: str | None = None,
    google_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    **model_kwargs,
):
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")

    config = MODEL_REGISTRY[model_key]
    provider = config["provider"]
    model_name = config["model_name"]

    if provider == "vllm":
        from .vllm_model import VLLMModel
        return VLLMModel(model_name, device=device, **model_kwargs)

    if provider == "hf":
        from .hf_model import HFModel
        return HFModel(model_name, **model_kwargs)

    if provider == "openai":
        from .openai_model import OpenAIModel
        return OpenAIModel(model_name, api_key=openai_api_key, **model_kwargs)

    if provider == "anthropic":
        from .claude_model import ClaudeModel
        return ClaudeModel(model_name, api_key=anthropic_api_key, **model_kwargs)

    if provider == "gemini":
        from .gemini_model import GeminiModel
        return GeminiModel(model_name, api_key=google_api_key, **model_kwargs)

    raise ValueError(f"Unsupported provider: {provider}")