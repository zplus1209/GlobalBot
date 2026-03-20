from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from globalbot.backend.base import BaseComponent, Document, RetrievedDocument


class BM25Retriever(BaseComponent):
    top_k: int = 5
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25

    _corpus: List[str] = []
    _docs: List[Document] = []
    _bm25: Any = None

    def model_post_init(self, __context: Any) -> None:
        self._corpus = []
        self._docs = []
        self._bm25 = None
        if not self.name:
            self.name = "bm25"

    def add(self, docs: List[Document]) -> None:
        self._docs.extend(docs)
        self._corpus.extend([doc.page_content for doc in docs])
        self._build_index()

    def _build_index(self) -> None:
        from rank_bm25 import BM25Okapi
        tokenized = [text.lower().split() for text in self._corpus]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b, epsilon=self.epsilon)
        self.log.debug("bm25.index_built", n_docs=len(self._corpus))

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedDocument]:
        if not self._bm25:
            self.log.warning("bm25.empty_index")
            return []

        k = top_k or self.top_k
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        scored_docs = sorted(
            zip(scores, self._docs),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, doc in scored_docs[:k]:
            if filters:
                if not all(doc.metadata.get(fk) == fv for fk, fv in filters.items()):
                    continue
            results.append(RetrievedDocument(
                page_content=doc.page_content,
                metadata=doc.metadata,
                doc_id=doc.doc_id,
                score=float(score),
            ))
        return results

    def run(self, query: str, **kwargs: Any) -> List[RetrievedDocument]:
        return self.query(query, **kwargs)

    def __len__(self) -> int:
        return len(self._docs)