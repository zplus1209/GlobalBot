from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from globalbot.backend.base import BaseComponent, Document, RetrievedDocument


class BaseVectorStore(BaseComponent):
    collection_name: str = "globalbot"

    @abstractmethod
    def add(
        self,
        docs: List[Document],
        embeddings: List[List[float]],
        **kwargs: Any,
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedDocument]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: List[str], **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def drop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    def run(self, docs: List[Document], embeddings: List[List[float]], **kwargs: Any) -> List[str]:
        return self.add(docs, embeddings, **kwargs)