from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, List, Optional

from globalbot.backend.base import BaseComponent


class BaseEmbeddings(BaseComponent):
    context_length: Optional[int] = None

    def run(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        label = self.name or type(self).__name__
        with self.timed("embeddings.call.start", "embeddings.call.end", model=label, n=len(texts)) as ctx:
            result = self._embed_documents(texts, **kwargs)
            ctx["dim"] = len(result[0]) if result else 0
        return result

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        return self._embed_query(text, **kwargs)

    async def arun(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        return await self._aembed_documents(texts, **kwargs)

    async def aembed_query(self, text: str, **kwargs: Any) -> List[float]:
        return await self._aembed_query(text, **kwargs)

    @abstractmethod
    def _embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def _embed_query(self, text: str, **kwargs: Any) -> List[float]:
        raise NotImplementedError

    async def _aembed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._embed_documents(texts, **kwargs))

    async def _aembed_query(self, text: str, **kwargs: Any) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._embed_query(text, **kwargs))

    def as_langchain(self):
        from langchain_core.embeddings import Embeddings as LCEmbeddings

        outer = self

        class _Adapter(LCEmbeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return outer._embed_documents(texts)

            def embed_query(self, text: str) -> List[float]:
                return outer._embed_query(text)

        return _Adapter()