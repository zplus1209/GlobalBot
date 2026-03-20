from __future__ import annotations

from typing import Any, Iterator, List, Optional

from openai import OpenAI
from pydantic import model_validator

from globalbot.backend.base import AIMessage, AnyMessage
from globalbot.backend.llms.chats.base import BaseChatLLM, _to_openai_messages


class ChatVLLM(BaseChatLLM):
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "dummy"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: int = 120

    _client: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "ChatVLLM":
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        if not self.name:
            self.name = f"vllm/{self.model}"
        return self

    def _call(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        self.log.debug("llm.vllm.call", model=self.model, n=len(messages))
        params: dict = dict(
            model=self.model,
            messages=_to_openai_messages(messages),
            temperature=self.temperature,
        )
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        params.update(kwargs)
        response = self._client.chat.completions.create(**params)
        msg = AIMessage(content=response.choices[0].message.content or "")
        if response.usage:
            msg.total_tokens = response.usage.total_tokens
            msg.prompt_tokens = response.usage.prompt_tokens
            msg.completion_tokens = response.usage.completion_tokens
        return msg

    def _stream(self, messages: List[AnyMessage], **kwargs: Any) -> Iterator[AIMessage]:
        self.log.debug("llm.vllm.stream", model=self.model)
        params: dict = dict(
            model=self.model,
            messages=_to_openai_messages(messages),
            temperature=self.temperature,
            stream=True,
        )
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        accumulated = ""
        for chunk in self._client.chat.completions.create(**params):
            delta = chunk.choices[0].delta
            if delta.content:
                accumulated += delta.content
            yield AIMessage(content=accumulated)