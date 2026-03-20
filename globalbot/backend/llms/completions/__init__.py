from __future__ import annotations

from typing import Any

from globalbot.backend.llms.completions.base import BaseCompletionLLM
from globalbot.backend.llms.completions.openai import OpenAICompletion

ONLINE_PROVIDERS = {"openai"}
OFFLINE_PROVIDERS = {"vllm"}


def init_completion_model(provider: str, **kwargs: Any) -> BaseCompletionLLM:
    provider = provider.lower()
    if provider == "openai":
        return OpenAICompletion(**kwargs)
    if provider == "vllm":
        from globalbot.backend.llms.completions.vllm import VLLMCompletion
        return VLLMCompletion(**kwargs)
    raise ValueError(f"Unknown completion provider: {provider!r}. Available: {ONLINE_PROVIDERS | OFFLINE_PROVIDERS}")


__all__ = [
    "BaseCompletionLLM",
    "OpenAICompletion",
    "init_completion_model",
    "ONLINE_PROVIDERS",
    "OFFLINE_PROVIDERS",
]