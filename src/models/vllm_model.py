from pathlib import Path
import json

from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams

from .base import BaseModel


class VLLMModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.95,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        self.model_name = model_name
        self.deterministic = deterministic

        llm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "disable_log_stats": True,
        }
        
        if seed is not None:
            llm_kwargs["seed"] = seed

        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs

        print("LLM kwargs =", llm_kwargs)
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

        if not getattr(self.tokenizer, "chat_template", None):
            try:
                template_path = hf_hub_download(
                    repo_id=model_name,
                    filename="chat_template.json",
                )
                template_data = json.loads(
                    Path(template_path).read_text(encoding="utf-8")
                )
                chat_template = template_data.get("chat_template")
                if chat_template:
                    self.tokenizer.chat_template = chat_template
            except Exception:
                pass

    def generate(
        self,
        messages,
        max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        **kwargs,
    ):
        chat_template_kwargs = {}

        if (
            self.model_name.startswith("Qwen/Qwen3")
            and kwargs.get("enable_thinking") is False
        ):
            chat_template_kwargs["enable_thinking"] = False

        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                f"Tokenizer for {self.model_name} has no chat template, "
                "and chat_template.json could not be loaded."
            )

        if do_sample:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        else:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
            )

        if "mistral" in self.model_name.lower():
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            outputs = self.llm.generate([prompt_token_ids], sampling_params)
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            outputs = self.llm.generate([prompt], sampling_params)

        return outputs[0].outputs[0].text.strip()