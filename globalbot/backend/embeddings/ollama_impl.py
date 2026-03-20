from __future__ import annotations

from typing import Any, List, Optional

from pydantic import model_validator

from globalbot.backend.embeddings.base import BaseEmbeddings


class OllamaEmbeddings(BaseEmbeddings):
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    context_length: Optional[int] = None

    @model_validator(mode="after")
    def _set_name(self) -> "OllamaEmbeddings":
        if not self.name:
            self.name = f"ollama/{self.model}"
        return self

    def _embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        from ollama import Client
        client = Client(host=self.base_url)
        self.log.debug("embeddings.ollama.call", model=self.model, n=len(texts))
        vectors = []
        for text in texts:
            response = client.embed(model=self.model, input=text)
            vectors.append(response.embeddings[0])
        return vectors

    def _embed_query(self, text: str, **kwargs: Any) -> List[float]:
        from ollama import Client
        client = Client(host=self.base_url)
        self.log.debug("embeddings.ollama.query", model=self.model)
        response = client.embed(model=self.model, input=text)
        return response.embeddings[0]

    async def _aembed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._embed_documents(texts, **kwargs))

    async def _aembed_query(self, text: str, **kwargs: Any) -> List[float]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._embed_query(text, **kwargs))