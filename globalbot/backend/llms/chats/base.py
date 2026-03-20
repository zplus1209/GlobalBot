from __future__ import annotations

import time
import asyncio
from abc import abstractmethod
from typing import Any, AsyncIterator, Iterator, List, Optional, Union

from globalbot.backend.base import (
    BaseComponent,
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from globalbot.backend.base.schema import get_text

MessageInput = Union[str, dict, AnyMessage]


def _normalise(messages: List[MessageInput]) -> List[AnyMessage]:
    out: List[AnyMessage] = []
    for m in messages:
        if isinstance(m, (HumanMessage, AIMessage, SystemMessage)):
            out.append(m)
        elif isinstance(m, str):
            out.append(HumanMessage(content=m))
        elif isinstance(m, dict):
            role = m.get("role", "user").lower()
            content = m.get("content", "")
            if role in ("assistant", "ai"):
                out.append(AIMessage(content=content))
            elif role == "system":
                out.append(SystemMessage(content=content))
            else:
                out.append(HumanMessage(content=content))
        else:
            out.append(HumanMessage(content=get_text(m)))
    return out


def _preview(messages: List[AnyMessage]) -> str:
    if not messages:
        return "(empty)"
    last = messages[-1]
    return f"{type(last).__name__}: {last.text[:60].replace(chr(10), ' ')!r}"


def _to_openai_messages(messages: List[AnyMessage]) -> List[dict]:
    role_map = {"HumanMessage": "user", "AIMessage": "assistant", "SystemMessage": "system"}
    return [
        {
            "role": role_map.get(type(m).__name__, "user"),
            "content": m.content if isinstance(m.content, str) else str(m.content),
        }
        for m in messages
    ]


class BaseChatLLM(BaseComponent):
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: int = 60

    def run(self, messages: Union[MessageInput, List[MessageInput]], **kwargs: Any) -> AIMessage:
        if isinstance(messages, (str, dict)) or not isinstance(messages, list):
            messages = [messages]
        normed = _normalise(messages)
        label = self.name or type(self).__name__
        with self.timed(
            "llm.call.start",
            "llm.call.end",
            model=label,
            n_messages=len(normed),
            last=_preview(normed),
        ) as ctx:
            result = self._call(normed, **kwargs)
            ctx["tokens_total"] = result.total_tokens
            ctx["tokens_prompt"] = result.prompt_tokens
            ctx["tokens_completion"] = result.completion_tokens
            ctx["reply_chars"] = len(result.text)
        return result

    def stream(self, messages: Union[MessageInput, List[MessageInput]], **kwargs: Any) -> Iterator[AIMessage]:
        if isinstance(messages, (str, dict)) or not isinstance(messages, list):
            messages = [messages]
        normed = _normalise(messages)
        label = self.name or type(self).__name__
        self.log.info("llm.stream.start", model=label, n_messages=len(normed))
        t0 = time.perf_counter()
        n = 0
        last: Optional[AIMessage] = None
        try:
            for chunk in self._stream(normed, **kwargs):
                n += 1
                last = chunk
                yield chunk
        except Exception as exc:
            ms = int((time.perf_counter() - t0) * 1000)
            self.log.error("llm.stream.error", model=label, duration_ms=ms, error=str(exc)[:120])
            raise
        ms = int((time.perf_counter() - t0) * 1000)
        self.log.info("llm.stream.end", model=label, duration_ms=ms, n_chunks=n, chars=len(last.text) if last else 0)

    async def arun(self, messages: Union[MessageInput, List[MessageInput]], **kwargs: Any) -> AIMessage:
        if isinstance(messages, (str, dict)) or not isinstance(messages, list):
            messages = [messages]
        normed = _normalise(messages)
        label = self.name or type(self).__name__
        with self.timed("llm.acall.start", "llm.acall.end", model=label, n_messages=len(normed)) as ctx:
            result = await self._acall(normed, **kwargs)
            ctx["reply_chars"] = len(result.text)
        return result

    async def astream(self, messages: Union[MessageInput, List[MessageInput]], **kwargs: Any) -> AsyncIterator[AIMessage]:
        if isinstance(messages, (str, dict)) or not isinstance(messages, list):
            messages = [messages]
        normed = _normalise(messages)
        async for chunk in self._astream(normed, **kwargs):
            yield chunk

    @abstractmethod
    def _call(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        raise NotImplementedError

    def _stream(self, messages: List[AnyMessage], **kwargs: Any) -> Iterator[AIMessage]:
        yield self._call(messages, **kwargs)

    async def _acall(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._call(messages, **kwargs))

    async def _astream(self, messages: List[AnyMessage], **kwargs: Any) -> AsyncIterator[AIMessage]:
        result = await self._acall(messages, **kwargs)
        yield result

    def chat(self, user: str, system: Optional[str] = None, **kwargs: Any) -> str:
        msgs: List[MessageInput] = []
        if system:
            msgs.append(SystemMessage(content=system))
        msgs.append(HumanMessage(content=user))
        return self.run(msgs, **kwargs).text