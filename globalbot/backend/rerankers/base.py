from __future__ import annotations

from abc import abstractmethod
from typing import Any, List

from globalbot.backend.base import BaseComponent, RetrievedDocument


class BaseReranker(BaseComponent):
    top_k: int = 5

    @abstractmethod
    def rerank(self, query: str, docs: List[RetrievedDocument], **kwargs: Any) -> List[RetrievedDocument]:
        raise NotImplementedError

    def run(self, query: str, docs: List[RetrievedDocument], **kwargs: Any) -> List[RetrievedDocument]:
        if not docs:
            return []
        self.log.debug("reranker.run", n_input=len(docs), top_k=self.top_k)
        results = self.rerank(query, docs, **kwargs)
        self.log.debug("reranker.done", n_output=len(results))
        return results