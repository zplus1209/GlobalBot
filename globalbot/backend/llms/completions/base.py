from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, Iterator, List, Union

from globalbot.backend.base import BaseComponent, AIMessage


class BaseCompletionLLM(BaseComponent):
    temperature: float = 0.0
    max_tokens: int = 256
    timeout: int = 60

    def run(self, prompt: Union[str, List[str]], **kwargs: Any) -> AIMessage:
        prompts = [prompt] if isinstance(prompt, str) else prompt
        label = self.name or type(self).__name__
        with self.timed(
            "completion.call.start",
            "completion.call.end",
            model=label,
            preview=prompts[0][:60] if prompts else "",
        ) as ctx:
            result = self._call(prompts, **kwargs)
            ctx["reply_chars"] = len(result.text)
        return result

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[AIMessage]:
        label = self.name or type(self).__name__
        self.log.info("completion.stream.start", model=label)
        yield from self._stream(prompt, **kwargs)

    async def arun(self, prompt: Union[str, List[str]], **kwargs: Any) -> AIMessage:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(prompt, **kwargs))

    @abstractmethod
    def _call(self, prompts: List[str], **kwargs: Any) -> AIMessage:
        raise NotImplementedError

    def _stream(self, prompt: str, **kwargs: Any) -> Iterator[AIMessage]:
        yield self._call([prompt], **kwargs)
