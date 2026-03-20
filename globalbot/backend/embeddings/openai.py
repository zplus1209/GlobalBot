from __future__ import annotations

from typing import Any, List, Optional

from openai import OpenAI, AsyncOpenAI
from pydantic import model_validator

from globalbot.backend.embeddings.base import BaseEmbeddings


class OpenAIEmbeddings(BaseEmbeddings):
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 60
    context_length: Optional[int] = None
    dimensions: Optional[int] = None

    _client: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "OpenAIEmbeddings":
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        if not self.name:
            self.name = f"openai/{self.model}"
        return self

    def _embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        self.log.debug("embeddings.openai.call", model=self.model, n=len(texts))
        params: dict = dict(model=self.model, input=texts)
        if self.dimensions:
            params["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**params)
        return [item.embedding for item in response.data]

    def _embed_query(self, text: str, **kwargs: Any) -> List[float]:
        self.log.debug("embeddings.openai.query", model=self.model)
        params: dict = dict(model=self.model, input=text)
        if self.dimensions:
            params["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**params)
        return response.data[0].embedding

    async def _aembed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        params: dict = dict(model=self.model, input=texts)
        if self.dimensions:
            params["dimensions"] = self.dimensions
        response = await client.embeddings.create(**params)
        return [item.embedding for item in response.data]

    async def _aembed_query(self, text: str, **kwargs: Any) -> List[float]:
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        params: dict = dict(model=self.model, input=text)
        if self.dimensions:
            params["dimensions"] = self.dimensions
        response = await client.embeddings.create(**params)
        return response.data[0].embedding


class AzureOpenAIEmbeddings(BaseEmbeddings):
    model: str = "text-embedding-ada-002"
    azure_endpoint: str = ""
    api_key: Optional[str] = None
    api_version: str = "2024-02-15-preview"
    azure_deployment: str = ""
    timeout: int = 60

    _client: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "AzureOpenAIEmbeddings":
        from openai import AzureOpenAI
        self._client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint, api_key=self.api_key,
            api_version=self.api_version, timeout=self.timeout,
        )
        if not self.name:
            self.name = f"azure/{self.azure_deployment or self.model}"
        return self

    def _embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        deployment = self.azure_deployment or self.model
        self.log.debug("embeddings.azure.call", deployment=deployment, n=len(texts))
        response = self._client.embeddings.create(model=deployment, input=texts)
        return [item.embedding for item in response.data]

    def _embed_query(self, text: str, **kwargs: Any) -> List[float]:
        deployment = self.azure_deployment or self.model
        response = self._client.embeddings.create(model=deployment, input=text)
        return response.data[0].embedding


class OpenAICompatibleEmbeddings(BaseEmbeddings):
    model: str = ""
    base_url: str = ""
    api_key: str = "dummy"
    timeout: int = 60
    context_length: Optional[int] = None

    _client: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "OpenAICompatibleEmbeddings":
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        if not self.name:
            self.name = f"openai-compat/{self.model}"
        return self

    def _embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        self.log.debug("embeddings.compat.call", model=self.model, base_url=self.base_url, n=len(texts))
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def _embed_query(self, text: str, **kwargs: Any) -> List[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding