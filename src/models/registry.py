MODEL_REGISTRY = {
    "gpt-5.4": {
        "provider": "openai",
        "model_name": "gpt-5.4",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model_name": "gpt-4o-mini",
    },
    "claude-opus-4.6": {
        "provider": "anthropic",
        "model_name": "claude-opus-4-6",
    },
    "claude-sonnet-4.6": {
        "provider": "anthropic",
        "model_name": "claude-sonnet-4-6",
    },
    "gemini-2.5-pro": {
        "provider": "gemini",
        "model_name": "gemini-2.5-pro",
    },
    "gemini-2.5-flash-thinking": {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "thinking_budget": 2048,
    },

    "gemini-2.5-flash-no-thinking": {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "thinking_budget": 0,
    },
    "llama3.1-8b": {
        "provider": "vllm",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "qwen_3-32b": {
        "provider": "vllm",
        "model_name": "Qwen/Qwen3-32B",
    },
    "qwen_2.5-14b": {
        "provider": "vllm",
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
    },
    "mistral-small-3.1-24B": {
        "provider": "vllm",
        "model_name": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    },
    "gemma-3-4b": {
        "provider": "vllm",
        "model_name": "google/gemma-3-4B-it",
    },
    "gemma-3-12b": {
        "provider": "vllm",
        "model_name": "google/gemma-3-12B-it",
    },
    "gemma-3-27b": {
        "provider": "vllm",
        "model_name": "google/gemma-3-27B-it",
    },
    "gemma-4-31b": {
        "provider": "hf",
        "model_name": "google/gemma-4-31B-it",
    },
    "medgemma-27b": {
        "provider": "vllm",
        "model_name": "google/medgemma-27b-text-it",
    },
    "med42-8b": {
        "provider": "vllm",
        "model_name": "m42-health/Llama3-Med42-8B",
    }
}