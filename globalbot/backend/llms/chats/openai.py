from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional

from pydantic import model_validator

from globalbot.backend.base import AIMessage, AnyMessage
from globalbot.backend.llms.chats.base import BaseChatLLM


def _to_openai_messages(messages: List[AnyMessage]) -> List[dict]:
    role_map = {"HumanMessage": "user", "AIMessage": "assistant", "SystemMessage": "system"}
    result = []
    for m in messages:
        role = role_map.get(type(m).__name__, "user")
        content = m.content if isinstance(m.content, str) else str(m.content)
        result.append({"role": role, "content": content})
    return result


class ChatOpenAI(BaseChatLLM):
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: int = 60
    context_length: Optional[int] = None

    _client: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "ChatOpenAI":
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        if not self.name:
            self.name = f"openai/{self.model}"
        return self

    def _call(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        self.log.debug("llm.openai.call", model=self.model, n=len(messages))
        params: dict = dict(
            model=self.model,
            messages=_to_openai_messages(messages),
            temperature=self.temperature,
        )
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        params.update(kwargs)

        response = self._client.chat.completions.create(**params)
        choice = response.choices[0]
        msg = AIMessage(content=choice.message.content or "")

        if response.usage:
            msg.total_tokens = response.usage.total_tokens
            msg.prompt_tokens = response.usage.prompt_tokens
            msg.completion_tokens = response.usage.completion_tokens
            self.log.debug(
                "llm.tokens",
                total=msg.total_tokens,
                prompt=msg.prompt_tokens,
                completion=msg.completion_tokens,
            )
        return msg

    def _stream(self, messages: List[AnyMessage], **kwargs: Any) -> Iterator[AIMessage]:
        self.log.debug("llm.openai.stream", model=self.model)
        params: dict = dict(
            model=self.model,
            messages=_to_openai_messages(messages),
            temperature=self.temperature,
            stream=True,
        )
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        params.update(kwargs)

        accumulated = ""
        for chunk in self._client.chat.completions.create(**params):
            delta = chunk.choices[0].delta
            if delta.content:
                accumulated += delta.content
            yield AIMessage(content=accumulated)

    async def _acall(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        params: dict = dict(
            model=self.model,
            messages=_to_openai_messages(messages),
            temperature=self.temperature,
        )
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        params.update(kwargs)

        response = await client.chat.completions.create(**params)
        choice = response.choices[0]
        msg = AIMessage(content=choice.message.content or "")
        if response.usage:
            msg.total_tokens = response.usage.total_tokens
            msg.prompt_tokens = response.usage.prompt_tokens
            msg.completion_tokens = response.usage.completion_tokens
        return msg

    async def _astream(self, messages: List[AnyMessage], **kwargs: Any) -> AsyncIterator[AIMessage]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        params: dict = dict(
            model=self.model,
            messages=_to_openai_messages(messages),
            temperature=self.temperature,
            stream=True,
        )
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens

        accumulated = ""
        async for chunk in await client.chat.completions.create(**params):
            delta = chunk.choices[0].delta
            if delta.content:
                accumulated += delta.content
            yield AIMessage(content=accumulated)


class AzureChatOpenAI(BaseChatLLM):
    model: str = "gpt-4o"
    azure_endpoint: str = ""
    api_key: Optional[str] = None
    api_version: str = "2024-02-15-preview"
    azure_deployment: str = ""
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: int = 60

    _client: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "AzureChatOpenAI":
        from openai import AzureOpenAI
        self._client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            timeout=self.timeout,
        )
        if not self.name:
            self.name = f"azure/{self.azure_deployment or self.model}"
        return self

    def _call(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        deployment = self.azure_deployment or self.model
        self.log.debug("llm.azure.call", deployment=deployment, n=len(messages))
        params: dict = dict(
            model=deployment,
            messages=_to_openai_messages(messages),
            temperature=self.temperature,
        )
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        params.update(kwargs)

        response = self._client.chat.completions.create(**params)
        choice = response.choices[0]
        msg = AIMessage(content=choice.message.content or "")
        if response.usage:
            msg.total_tokens = response.usage.total_tokens
            msg.prompt_tokens = response.usage.prompt_tokens
            msg.completion_tokens = response.usage.completion_tokens
        return msg

    def _stream(self, messages: List[AnyMessage], **kwargs: Any) -> Iterator[AIMessage]:
        deployment = self.azure_deployment or self.model
        params: dict = dict(
            model=deployment,
            messages=_to_openai_messages(messages),
            temperature=self.temperature,
            stream=True,
        )
        accumulated = ""
        for chunk in self._client.chat.completions.create(**params):
            delta = chunk.choices[0].delta
            if delta.content:
                accumulated += delta.content
            yield AIMessage(content=accumulated)
