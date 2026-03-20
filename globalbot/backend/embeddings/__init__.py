from __future__ import annotations

from typing import Any

from globalbot.backend.embeddings.base import BaseEmbeddings
from globalbot.backend.embeddings.openai import OpenAIEmbeddings, AzureOpenAIEmbeddings, OpenAICompatibleEmbeddings
from globalbot.backend.embeddings.ollama import OllamaEmbeddings
from globalbot.backend.embeddings.langchain_based import (
    LangChainEmbeddings,
    LCOpenAIEmbeddings,
    LCAzureOpenAIEmbeddings,
    LCOllamaEmbeddings,
    LCHuggingFaceEmbeddings,
    LCFastEmbedEmbeddings,
    LCCohereEmbeddings,
    LCGoogleEmbeddings,
)

ONLINE_PROVIDERS = {"openai", "azure", "cohere", "google", "openai_compat"}
OFFLINE_PROVIDERS = {"ollama", "hf", "fastembed"}


def init_embedding_model(
    provider: str,
    use_langchain: bool = False,
    **kwargs: Any,
) -> BaseEmbeddings:
    provider = provider.lower()

    if provider in ONLINE_PROVIDERS:
        if use_langchain:
            return _init_online_lc(provider, **kwargs)
        return _init_online(provider, **kwargs)

    if provider in OFFLINE_PROVIDERS:
        if use_langchain:
            return _init_offline_lc(provider, **kwargs)
        return _init_offline(provider, **kwargs)

    raise ValueError(f"Unknown provider: {provider!r}. Online: {ONLINE_PROVIDERS}, Offline: {OFFLINE_PROVIDERS}")


def _init_online(provider: str, **kwargs: Any) -> BaseEmbeddings:
    if provider == "openai":
        return OpenAIEmbeddings(**kwargs)
    if provider == "azure":
        return AzureOpenAIEmbeddings(**kwargs)
    if provider == "openai_compat":
        return OpenAICompatibleEmbeddings(**kwargs)
    return _init_online_lc(provider, **kwargs)


def _init_online_lc(provider: str, **kwargs: Any) -> BaseEmbeddings:
    if provider == "openai":
        return LCOpenAIEmbeddings(**kwargs)
    if provider == "azure":
        return LCAzureOpenAIEmbeddings(**kwargs)
    if provider == "cohere":
        return LCCohereEmbeddings(**kwargs)
    if provider == "google":
        return LCGoogleEmbeddings(**kwargs)
    raise ValueError(f"No LangChain wrapper for online provider: {provider!r}")


def _init_offline(provider: str, **kwargs: Any) -> BaseEmbeddings:
    if provider == "ollama":
        return OllamaEmbeddings(**kwargs)
    if provider == "hf":
        from globalbot.backend.embeddings.hf import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(**kwargs)
    if provider == "fastembed":
        from globalbot.backend.embeddings.fastEmbed import FastEmbedEmbeddings
        return FastEmbedEmbeddings(**kwargs)
    raise ValueError(f"Unknown offline provider: {provider!r}")


def _init_offline_lc(provider: str, **kwargs: Any) -> BaseEmbeddings:
    if provider == "ollama":
        return LCOllamaEmbeddings(**kwargs)
    if provider == "hf":
        return LCHuggingFaceEmbeddings(**kwargs)
    if provider == "fastembed":
        return LCFastEmbedEmbeddings(**kwargs)
    raise ValueError(f"No LangChain wrapper for offline provider: {provider!r}")


__all__ = [
    "BaseEmbeddings",
    "OpenAIEmbeddings",
    "AzureOpenAIEmbeddings",
    "OpenAICompatibleEmbeddings",
    "OllamaEmbeddings",
    "LangChainEmbeddings",
    "LCOpenAIEmbeddings",
    "LCAzureOpenAIEmbeddings",
    "LCOllamaEmbeddings",
    "LCHuggingFaceEmbeddings",
    "LCFastEmbedEmbeddings",
    "LCCohereEmbeddings",
    "LCGoogleEmbeddings",
    "init_embedding_model",
    "ONLINE_PROVIDERS",
    "OFFLINE_PROVIDERS",
]