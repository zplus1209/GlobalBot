from __future__ import annotations

from typing import Any, Iterator, List, Optional

from pydantic import model_validator

from globalbot.backend.base import AIMessage, AnyMessage
from globalbot.backend.llms.chats.base import BaseChatLLM


def _to_prompt(messages: List[AnyMessage], chat_template: Optional[str] = None) -> str:
    parts = []
    for m in messages:
        role = type(m).__name__.replace("Message", "").lower()
        content = m.content if isinstance(m.content, str) else str(m.content)
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>")
    return "\n".join(parts)


class ChatONNX(BaseChatLLM):
    model_path: str = ""
    tokenizer_path: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.0
    device: str = "cpu"

    _pipeline: Any = None

    @model_validator(mode="after")
    def _init_pipeline(self) -> "ChatONNX":
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer, TextGenerationPipeline

        tokenizer_path = self.tokenizer_path or self.model_path
        model = ORTModelForCausalLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=self.device if self.device != "cpu" else -1,
        )
        if not self.name:
            self.name = f"onnx/{self.model_path.split('/')[-1]}"
        return self

    def _call(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        self.log.debug("llm.onnx.call", model=self.name)
        prompt = _to_prompt(messages)
        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
            return_full_text=False,
        )
        content = outputs[0]["generated_text"] if outputs else ""
        return AIMessage(content=content)

    def _stream(self, messages: List[AnyMessage], **kwargs: Any) -> Iterator[AIMessage]:
        yield self._call(messages, **kwargs)