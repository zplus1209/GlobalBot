from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional

from pydantic import Field, model_validator, ConfigDict

from globalbot.backend.llms.chats.base import BaseChatLLM
from globalbot.backend.base import AIMessage, AnyMessage, HumanMessage, SystemMessage


class LangChainChatLLM(BaseChatLLM):
    lc_model: Any = Field(default=None, exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed = True
    )
    
    def _get_model(self) -> Any:
        if self.lc_model is None:
            raise ValueError(f"{type(self).__name__}: lc_model is not set.")
        return self.lc_model

    def _to_lc_messages(self, messages: List[AnyMessage]) -> List[Any]:
        from langchain_core.messages import (
            AIMessage as LCAIMessage,
            HumanMessage as LCHumanMessage,
            SystemMessage as LCSystemMessage,
        )
        _map = {
            HumanMessage: LCHumanMessage,
            AIMessage: LCAIMessage,
            SystemMessage: LCSystemMessage,
        }
        return [_map.get(type(m), LCHumanMessage)(content=m.content) for m in messages]

    def _parse_response(self, response: Any) -> AIMessage:
        if not hasattr(response, "content"):
            return AIMessage(content=str(response))
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        msg = AIMessage(content=content)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            u = response.usage_metadata
            msg.total_tokens = u.get("total_tokens")
            msg.prompt_tokens = u.get("input_tokens")
            msg.completion_tokens = u.get("output_tokens")
        elif hasattr(response, "response_metadata"):
            rm = response.response_metadata or {}
            usage = rm.get("token_usage") or rm.get("usage", {})
            if usage:
                msg.total_tokens = usage.get("total_tokens")
                msg.prompt_tokens = usage.get("prompt_tokens")
                msg.completion_tokens = usage.get("completion_tokens")
        if msg.total_tokens:
            self.log.debug(
                "llm.tokens",
                total=msg.total_tokens,
                prompt=msg.prompt_tokens,
                completion=msg.completion_tokens,
            )
        return msg

    def _call(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        lc = self._get_model()
        lc_msgs = self._to_lc_messages(messages)
        self.log.debug("llm.lc.invoke", lc_type=type(lc).__name__, n=len(lc_msgs))
        response = lc.invoke(lc_msgs, config={"run_name": self.name or type(self).__name__})
        return self._parse_response(response)

    def _stream(self, messages: List[AnyMessage], **kwargs: Any) -> Iterator[AIMessage]:
        lc = self._get_model()
        lc_msgs = self._to_lc_messages(messages)
        if not hasattr(lc, "stream"):
            yield self._call(messages, **kwargs)
            return
        accumulated = ""
        for chunk in lc.stream(lc_msgs):
            if hasattr(chunk, "content"):
                c = chunk.content
                if isinstance(c, str):
                    accumulated += c
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict):
                            accumulated += part.get("text", "")
            yield AIMessage(content=accumulated)

    async def _acall(self, messages: List[AnyMessage], **kwargs: Any) -> AIMessage:
        lc = self._get_model()
        if hasattr(lc, "ainvoke"):
            response = await lc.ainvoke(self._to_lc_messages(messages))
            return self._parse_response(response)
        return await super()._acall(messages, **kwargs)

    async def _astream(self, messages: List[AnyMessage], **kwargs: Any) -> AsyncIterator[AIMessage]:
        lc = self._get_model()
        if not hasattr(lc, "astream"):
            result = await self._acall(messages, **kwargs)
            yield result
            return
        accumulated = ""
        async for chunk in lc.astream(self._to_lc_messages(messages)):
            if hasattr(chunk, "content") and isinstance(chunk.content, str):
                accumulated += chunk.content
            yield AIMessage(content=accumulated)


class LCChatOllama(LangChainChatLLM):
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_ctx: Optional[int] = None

    @model_validator(mode="after")
    def _build(self) -> "LCChatOllama":
        from langchain_ollama import ChatOllama
        kwargs: dict = dict(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
        )
        if self.num_ctx:
            kwargs["num_ctx"] = self.num_ctx
        self.lc_model = ChatOllama(**kwargs)
        if not self.name:
            self.name = f"lc/ollama/{self.model}"
        return self


class LCChatOpenAI(LangChainChatLLM):
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    @model_validator(mode="after")
    def _build(self) -> "LCChatOpenAI":
        from langchain_openai import ChatOpenAI
        self.lc_model = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not self.name:
            self.name = f"lc/openai/{self.model}"
        return self


class LCChatOpenAICompatible(LangChainChatLLM):
    model: str = ""
    base_url: str = ""
    api_key: str = "dummy"
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    @model_validator(mode="after")
    def _build(self) -> "LCChatOpenAICompatible":
        from langchain_openai import ChatOpenAI
        self.lc_model = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not self.name:
            self.name = f"lc/openai-compat/{self.model}"
        return self


class LCAzureChatOpenAI(LangChainChatLLM):
    model: str = "gpt-4o"
    azure_endpoint: str = ""
    api_key: Optional[str] = None
    api_version: str = "2024-02-15-preview"
    azure_deployment: str = ""
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    @model_validator(mode="after")
    def _build(self) -> "LCAzureChatOpenAI":
        from langchain_openai import AzureChatOpenAI
        self.lc_model = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            azure_deployment=self.azure_deployment,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not self.name:
            self.name = f"lc/azure/{self.azure_deployment or self.model}"
        return self


class LCChatAnthropic(LangChainChatLLM):
    model: str = "claude-3-5-haiku-20241022"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024

    @model_validator(mode="after")
    def _build(self) -> "LCChatAnthropic":
        from langchain_anthropic import ChatAnthropic
        self.lc_model = ChatAnthropic(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not self.name:
            self.name = f"lc/anthropic/{self.model}"
        return self


class LCChatGoogleGenAI(LangChainChatLLM):
    model: str = "gemini-1.5-flash"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    @model_validator(mode="after")
    def _build(self) -> "LCChatGoogleGenAI":
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.lc_model = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        if not self.name:
            self.name = f"lc/google/{self.model}"
        return self


class LCChatCohere(LangChainChatLLM):
    model: str = "command-r-plus"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    @model_validator(mode="after")
    def _build(self) -> "LCChatCohere":
        from langchain_cohere import ChatCohere
        self.lc_model = ChatCohere(
            model=self.model,
            cohere_api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not self.name:
            self.name = f"lc/cohere/{self.model}"
        return self


class LCChatBedrock(LangChainChatLLM):
    model_id: str = "anthropic.claude-3-5-haiku-20241022-v1:0"
    region_name: str = "us-east-1"
    temperature: float = 0.0
    max_tokens: int = 1024

    @model_validator(mode="after")
    def _build(self) -> "LCChatBedrock":
        from langchain_aws import ChatBedrock
        self.lc_model = ChatBedrock(
            model_id=self.model_id,
            region_name=self.region_name,
            model_kwargs={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        )
        if not self.name:
            self.name = f"lc/bedrock/{self.model_id}"
        return self


class LCChatGroq(LangChainChatLLM):
    model: str = "llama-3.1-8b-instant"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    @model_validator(mode="after")
    def _build(self) -> "LCChatGroq":
        from langchain_groq import ChatGroq
        self.lc_model = ChatGroq(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not self.name:
            self.name = f"lc/groq/{self.model}"
        return self


class LCChatMistralAI(LangChainChatLLM):
    model: str = "mistral-large-latest"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    @model_validator(mode="after")
    def _build(self) -> "LCChatMistralAI":
        from langchain_mistralai import ChatMistralAI
        self.lc_model = ChatMistralAI(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not self.name:
            self.name = f"lc/mistral/{self.model}"
        return self


class LCChatHuggingFace(LangChainChatLLM):
    model_id: str = "HuggingFaceH4/zephyr-7b-beta"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_new_tokens: int = 512
    task: str = "text-generation"

    @model_validator(mode="after")
    def _build(self) -> "LCChatHuggingFace":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        endpoint = HuggingFaceEndpoint(
            repo_id=self.model_id,
            huggingfacehub_api_token=self.api_key,
            task=self.task,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        self.lc_model = ChatHuggingFace(llm=endpoint)
        if not self.name:
            self.name = f"lc/hf/{self.model_id}"
        return self
