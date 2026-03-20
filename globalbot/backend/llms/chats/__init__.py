from __future__ import annotations

from typing import Any, Optional

from globalbot.backend.llms.chats.base import BaseChatLLM
from globalbot.backend.llms.chats.openai_impl import ChatOpenAI, AzureChatOpenAI
from globalbot.backend.llms.chats.ollama_impl import ChatOllama, ChatOpenAICompatible
from globalbot.backend.llms.chats.langchain_based import (
    LCChatOllama,
    LCChatOpenAI,
    LCChatOpenAICompatible,
    LCAzureChatOpenAI,
    LCChatAnthropic,
    LCChatGoogleGenAI,
    LCChatCohere,
    LCChatBedrock,
    LCChatGroq,
    LCChatMistralAI,
    LCChatHuggingFace,
)

ONLINE_PROVIDERS = {"openai", "azure", "anthropic", "google", "cohere", "groq", "mistral", "bedrock"}
OFFLINE_PROVIDERS = {"ollama", "vllm", "hf", "onnx", "openai_compat"}


def init_chat_model(
    provider: str,
    use_langchain: bool = False,
    **kwargs: Any,
) -> BaseChatLLM:
    provider = provider.lower()

    if provider in ONLINE_PROVIDERS:
        if use_langchain:
            return _init_online_lc(provider, **kwargs)
        return _init_online(provider, **kwargs)

    if provider in OFFLINE_PROVIDERS:
        return _init_offline(provider, **kwargs)

    raise ValueError(f"Unknown provider: {provider!r}. Online: {ONLINE_PROVIDERS}, Offline: {OFFLINE_PROVIDERS}")


def _init_online(provider: str, **kwargs: Any) -> BaseChatLLM:
    if provider == "openai":
        return ChatOpenAI(**kwargs)
    if provider == "azure":
        return AzureChatOpenAI(**kwargs)
    if provider in ("anthropic", "google", "cohere", "groq", "mistral", "bedrock"):
        return _init_online_lc(provider, **kwargs)
    raise ValueError(f"No native client for online provider: {provider!r}")


def _init_online_lc(provider: str, **kwargs: Any) -> BaseChatLLM:
    if provider == "openai":
        return LCChatOpenAI(**kwargs)
    if provider == "azure":
        return LCAzureChatOpenAI(**kwargs)
    if provider == "anthropic":
        return LCChatAnthropic(**kwargs)
    if provider == "google":
        return LCChatGoogleGenAI(**kwargs)
    if provider == "cohere":
        return LCChatCohere(**kwargs)
    if provider == "groq":
        return LCChatGroq(**kwargs)
    if provider == "mistral":
        return LCChatMistralAI(**kwargs)
    if provider == "bedrock":
        return LCChatBedrock(**kwargs)
    raise ValueError(f"No LangChain wrapper for provider: {provider!r}")


def _init_offline(provider: str, **kwargs: Any) -> BaseChatLLM:
    if provider == "ollama":
        return ChatOllama(**kwargs)
    if provider == "openai_compat":
        return ChatOpenAICompatible(**kwargs)
    if provider == "vllm":
        from globalbot.backend.llms.chats.vllm import ChatVLLM
        return ChatVLLM(**kwargs)
    if provider == "hf":
        from globalbot.backend.llms.chats.hf import ChatHuggingFace
        return ChatHuggingFace(**kwargs)
    if provider == "onnx":
        from globalbot.backend.llms.chats.onnx import ChatONNX
        return ChatONNX(**kwargs)
    raise ValueError(f"Unknown offline provider: {provider!r}")


__all__ = [
    "BaseChatLLM",
    "ChatOpenAI",
    "AzureChatOpenAI",
    "ChatOllama",
    "ChatOpenAICompatible",
    "LCChatOllama",
    "LCChatOpenAI",
    "LCChatOpenAICompatible",
    "LCAzureChatOpenAI",
    "LCChatAnthropic",
    "LCChatGoogleGenAI",
    "LCChatCohere",
    "LCChatBedrock",
    "LCChatGroq",
    "LCChatMistralAI",
    "LCChatHuggingFace",
    "init_chat_model",
    "ONLINE_PROVIDERS",
    "OFFLINE_PROVIDERS",
]