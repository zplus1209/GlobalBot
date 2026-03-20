from __future__ import annotations

from typing import Any, List, Optional

from pydantic import Field, model_validator, ConfigDict

from globalbot.backend.embeddings.base import BaseEmbeddings


class LangChainEmbeddings(BaseEmbeddings):
    lc_model: Any = Field(default=None, exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed = True
    )
    
    def _get_model(self) -> Any:
        if self.lc_model is None:
            raise ValueError(f"{type(self).__name__}: lc_model is not set.")
        return self.lc_model

    def _embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        return self._get_model().embed_documents(texts)

    def _embed_query(self, text: str, **kwargs: Any) -> List[float]:
        return self._get_model().embed_query(text)

    async def _aembed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        lc = self._get_model()
        if hasattr(lc, "aembed_documents"):
            return await lc.aembed_documents(texts)
        return await super()._aembed_documents(texts)

    async def _aembed_query(self, text: str, **kwargs: Any) -> List[float]:
        lc = self._get_model()
        if hasattr(lc, "aembed_query"):
            return await lc.aembed_query(text)
        return await super()._aembed_query(text)


class LCOpenAIEmbeddings(LangChainEmbeddings):
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    dimensions: Optional[int] = None

    @model_validator(mode="after")
    def _build(self) -> "LCOpenAIEmbeddings":
        from langchain_openai import OpenAIEmbeddings
        kwargs: dict = dict(model=self.model, openai_api_key=self.api_key, openai_api_base=self.base_url)
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        self.lc_model = OpenAIEmbeddings(**kwargs)
        if not self.name:
            self.name = f"lc/openai/{self.model}"
        return self


class LCAzureOpenAIEmbeddings(LangChainEmbeddings):
    model: str = "text-embedding-ada-002"
    azure_endpoint: str = ""
    api_key: Optional[str] = None
    api_version: str = "2024-02-15-preview"
    azure_deployment: str = ""

    @model_validator(mode="after")
    def _build(self) -> "LCAzureOpenAIEmbeddings":
        from langchain_openai import AzureOpenAIEmbeddings
        self.lc_model = AzureOpenAIEmbeddings(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            azure_deployment=self.azure_deployment or self.model,
        )
        if not self.name:
            self.name = f"lc/azure/{self.azure_deployment or self.model}"
        return self


class LCOllamaEmbeddings(LangChainEmbeddings):
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"

    @model_validator(mode="after")
    def _build(self) -> "LCOllamaEmbeddings":
        from langchain_ollama import OllamaEmbeddings
        self.lc_model = OllamaEmbeddings(model=self.model, base_url=self.base_url)
        if not self.name:
            self.name = f"lc/ollama/{self.model}"
        return self


class LCHuggingFaceEmbeddings(LangChainEmbeddings):
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"

    @model_validator(mode="after")
    def _build(self) -> "LCHuggingFaceEmbeddings":
        from langchain_huggingface import HuggingFaceEmbeddings
        self.lc_model = HuggingFaceEmbeddings(
            model_name=self.model_id,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        if not self.name:
            self.name = f"lc/hf/{self.model_id}"
        return self


class LCFastEmbedEmbeddings(LangChainEmbeddings):
    model: str = "BAAI/bge-small-en-v1.5"
    max_length: int = 512

    @model_validator(mode="after")
    def _build(self) -> "LCFastEmbedEmbeddings":
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        self.lc_model = FastEmbedEmbeddings(model_name=self.model, max_length=self.max_length)
        if not self.name:
            self.name = f"lc/fastembed/{self.model}"
        return self


class LCCohereEmbeddings(LangChainEmbeddings):
    model: str = "embed-english-v3.0"
    api_key: Optional[str] = None

    @model_validator(mode="after")
    def _build(self) -> "LCCohereEmbeddings":
        from langchain_cohere import CohereEmbeddings
        self.lc_model = CohereEmbeddings(model=self.model, cohere_api_key=self.api_key)
        if not self.name:
            self.name = f"lc/cohere/{self.model}"
        return self


class LCGoogleEmbeddings(LangChainEmbeddings):
    model: str = "models/text-embedding-004"
    api_key: Optional[str] = None

    @model_validator(mode="after")
    def _build(self) -> "LCGoogleEmbeddings":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        self.lc_model = GoogleGenerativeAIEmbeddings(model=self.model, google_api_key=self.api_key)
        if not self.name:
            self.name = f"lc/google/{self.model}"
        return self