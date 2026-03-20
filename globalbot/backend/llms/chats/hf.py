from __future__ import annotations

from typing import Any, Iterator, List, Optional

from pydantic import model_validator

from globalbot.backend.base import AIMessage, AnyMessage
from globalbot.backend.llms.chats.base import BaseChatLLM


class ChatHuggingFace(BaseChatLLM):
    model_id: str = "HuggingFaceH4/zephyr-7b-beta"
    device: str = "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.0
    torch_dtype: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    _pipeline: Any = None
    _tokenizer: Any = None

    @model_validator(mode="after")
    def _init_pipeline(self) -> "ChatHuggingFace":
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        dtype_map = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map.get(self.torch_dtype, "auto")

        load_kwargs: dict = {"torch_dtype": dtype, "device_map": self.device}
        if self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            load_kwargs["load_in_4bit"] = True

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self._tokenizer = tokenizer
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        if not self.name:
            self.name = f"hf/{self.model_id}"
        return self

    def _messages_to_prompt(self, messages: List[AnyMessage]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            role_map = {"HumanMessage": "user", "AIMessage": "assistant", "SystemMessage": "system"}
            lc_msgs = [
                {"role": role_map.get(type(m).__name__, "user"),
                 "content": m.content if isinstance(m.content, str) else str(m.content)}
                for m in messages
            ]
            return self._tokenizer.apply_chat_template(lc_msgs, tokenize=False, add_generation_prompt=True)
        parts = []
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            parts.append(content)
        return "\n".join(parts)

    def _call(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        self.log.debug("llm.hf.call", model=self.model_id)
        prompt = self._messages_to_prompt(messages)
        gen_kwargs: dict = dict(
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["do_sample"] = True
        outputs = self._pipeline(prompt, **gen_kwargs)
        content = outputs[0]["generated_text"] if outputs else ""
        return AIMessage(content=content)

    def _stream(self, messages: List[AnyMessage], **kwargs: Any) -> Iterator[AIMessage]:
        from transformers import TextIteratorStreamer
        import threading

        prompt = self._messages_to_prompt(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["do_sample"] = True

        thread = threading.Thread(target=self._pipeline.model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated = ""
        for token in streamer:
            accumulated += token
            yield AIMessage(content=accumulated)
        thread.join()