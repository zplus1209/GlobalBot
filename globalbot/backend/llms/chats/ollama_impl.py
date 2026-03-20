from __future__ import annotations

from typing import Any, Iterator, List, Optional

from ollama import Client
from pydantic import model_validator

from globalbot.backend.base import AIMessage, AnyMessage
from globalbot.backend.llms.chats.base import BaseChatLLM, _to_openai_messages


def _to_ollama_messages(messages: List[AnyMessage]) -> List[dict]:
    role_map = {"HumanMessage": "user", "AIMessage": "assistant", "SystemMessage": "system"}
    return [
        {
            "role": role_map.get(type(m).__name__, "user"),
            "content": m.content if isinstance(m.content, str) else str(m.content),
        }
        for m in messages
    ]


class ChatOllama(BaseChatLLM):
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout: int = 120
    context_window: Optional[int] = None

    @model_validator(mode="after")
    def _set_name(self) -> "ChatOllama":
        if not self.name:
            self.name = f"ollama/{self.model}"
        return self

    def _get_options(self) -> dict:
        opts: dict = {"temperature": self.temperature}
        if self.context_window:
            opts["num_ctx"] = self.context_window
        return opts

    def _call(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        client = Client(host=self.base_url)
        self.log.debug("llm.ollama.call", model=self.model, n=len(messages))
        response = client.chat(model=self.model, messages=_to_ollama_messages(messages), options=self._get_options())
        content = response.message.content or ""
        msg = AIMessage(content=content)
        if hasattr(response, "prompt_eval_count"):
            msg.prompt_tokens = response.prompt_eval_count
        if hasattr(response, "eval_count"):
            msg.completion_tokens = response.eval_count
            msg.total_tokens = (msg.prompt_tokens or 0) + msg.completion_tokens
        return msg

    def _stream(self, messages: List[AnyMessage], **kwargs: Any) -> Iterator[AIMessage]:
        client = Client(host=self.base_url)
        self.log.debug("llm.ollama.stream", model=self.model)
        accumulated = ""
        for chunk in client.chat(model=self.model, messages=_to_ollama_messages(messages), options=self._get_options(), stream=True):
            if chunk.message and chunk.message.content:
                accumulated += chunk.message.content
            yield AIMessage(content=accumulated)

    async def _acall(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._call(messages, **kwargs))


class ChatOpenAICompatible(BaseChatLLM):
    model: str = ""
    base_url: str = ""
    api_key: str = "dummy"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: int = 120

    _client: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "ChatOpenAICompatible":
        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        if not self.name:
            self.name = f"openai-compat/{self.model}"
        return self

    def _call(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        self.log.debug("llm.compat.call", model=self.model, base_url=self.base_url)
        params: dict = dict(model=self.model, messages=_to_openai_messages(messages), temperature=self.temperature)
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
        params: dict = dict(model=self.model, messages=_to_openai_messages(messages), temperature=self.temperature, stream=True)
        accumulated = ""
        for chunk in self._client.chat.completions.create(**params):
            delta = chunk.choices[0].delta
            if delta.content:
                accumulated += delta.content
            yield AIMessage(content=accumulated)


if __name__ == "__main__":
    llm = ChatOllama(model="qwen3.5:35b")

    q = "Bạn là chuyên gia AI, hãy cho tôi biết DiT là gì? Và flow matching giúp gì?"

    ans = llm.run(q)

    print(ans.text)