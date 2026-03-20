from __future__ import annotations

from typing import Any, Iterator, List, Optional

from pydantic import model_validator

from globalbot.backend.base import AIMessage
from globalbot.backend.llms.completions.base import BaseCompletionLLM


class OpenAICompletion(BaseCompletionLLM):
    model: str = "gpt-3.5-turbo-instruct"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    max_tokens: int = 256
    timeout: int = 60

    _client: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "OpenAICompletion":
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        if not self.name:
            self.name = f"openai-completion/{self.model}"
        return self

    def _call(self, prompts: List[str], **kwargs: Any) -> AIMessage:
        self.log.debug("completion.openai.call", model=self.model)
        response = self._client.completions.create(
            model=self.model,
            prompt=prompts[0],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        msg = AIMessage(content=response.choices[0].text or "")
        if response.usage:
            msg.total_tokens = response.usage.total_tokens
            msg.prompt_tokens = response.usage.prompt_tokens
            msg.completion_tokens = response.usage.completion_tokens
        return msg

    def _stream(self, prompt: str, **kwargs: Any) -> Iterator[AIMessage]:
        self.log.debug("completion.openai.stream", model=self.model)
        accumulated = ""
        for chunk in self._client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        ):
            text = chunk.choices[0].text or ""
            accumulated += text
            yield AIMessage(content=accumulated)
