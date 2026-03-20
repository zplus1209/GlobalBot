from __future__ import annotations

from typing import Any, Optional

from globalbot.backend.chunkings.base import BaseChunker
from globalbot.backend.chunkings.fixed import (
    RecursiveChunker,
    CharacterChunker,
    TokenChunker,
    SentenceChunker,
    MarkdownChunker,
)

CHUNKER_PROVIDERS = {"recursive", "character", "token", "sentence", "markdown", "semantic", "llm"}


def init_chunker(method: str = "recursive", **kwargs: Any) -> BaseChunker:
    method = method.lower()

    if method == "recursive":
        return RecursiveChunker(**kwargs)

    if method == "character":
        return CharacterChunker(**kwargs)

    if method == "token":
        return TokenChunker(**kwargs)

    if method == "sentence":
        return SentenceChunker(**kwargs)

    if method == "markdown":
        return MarkdownChunker(**kwargs)

    if method == "semantic":
        from globalbot.backend.chunkings.semantic import ClusterSemanticChunker
        return ClusterSemanticChunker(**kwargs)

    if method == "llm":
        from globalbot.backend.chunkings.llm_chunker import LLMChunker
        return LLMChunker(**kwargs)

    raise ValueError(f"Unknown chunking method: {method!r}. Available: {CHUNKER_PROVIDERS}")


__all__ = [
    "BaseChunker",
    "RecursiveChunker",
    "CharacterChunker",
    "TokenChunker",
    "SentenceChunker",
    "MarkdownChunker",
    "init_chunker",
    "CHUNKER_PROVIDERS",
]