from __future__ import annotations

from typing import Any, Optional

from globalbot.backend.rerankers.base import BaseReranker

RERANKER_PROVIDERS = {"cross-encoder", "cohere", "llm"}


def init_reranker(method: str = "cross-encoder", **kwargs: Any) -> BaseReranker:
    method = method.lower()

    if method == "cross-encoder":
        from globalbot.backend.rerankers.rerankers import CrossEncoderReranker
        return CrossEncoderReranker(**kwargs)

    if method == "cohere":
        from globalbot.backend.rerankers.rerankers import CohereReranker
        return CohereReranker(**kwargs)

    if method == "llm":
        from globalbot.backend.rerankers.rerankers import LLMReranker
        return LLMReranker(**kwargs)

    raise ValueError(f"Unknown reranker method: {method!r}. Available: {RERANKER_PROVIDERS}")


__all__ = [
    "BaseReranker",
    "init_reranker",
    "RERANKER_PROVIDERS",
]