from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import model_validator

from globalbot.backend.base import BaseComponent, RetrievedDocument
from globalbot.backend.embeddings.base import BaseEmbeddings
from globalbot.backend.storages.vectorstores.base import BaseVectorStore


class RAGRetriever(BaseComponent):
    vectorstore: BaseVectorStore
    embeddings: BaseEmbeddings
    top_k: int = 5
    score_threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None
    reranker: Optional[Any] = None

    @model_validator(mode="after")
    def _set_name(self) -> "RAGRetriever":
        if not self.name:
            self.name = f"retriever/{self.vectorstore.name}"
        return self

    def run(self, query: str, **kwargs: Any) -> List[RetrievedDocument]:
        self.log.debug("retriever.query", query=query[:80], top_k=self.top_k)

        fetch_k = self.top_k * 3 if self.reranker else self.top_k

        with self.timed("retriever.embed.start", "retriever.embed.end"):
            embedding = self.embeddings.embed_query(query)

        with self.timed("retriever.search.start", "retriever.search.end"):
            results = self.vectorstore.query(
                embedding=embedding,
                top_k=fetch_k,
                filters=self.filters or kwargs.get("filters"),
            )

        if self.score_threshold is not None:
            results = [r for r in results if r.score >= self.score_threshold]

        if self.reranker and results:
            with self.timed("retriever.rerank.start", "retriever.rerank.end"):
                results = self.reranker.run(query=query, docs=results)

        self.log.debug("retriever.results", n=len(results))
        return results[: self.top_k]


class KeywordRetriever(BaseComponent):
    top_k: int = 5
    score_threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None
    reranker: Optional[Any] = None

    _bm25: Any = None

    @model_validator(mode="after")
    def _set_name(self) -> "KeywordRetriever":
        if not self.name:
            self.name = "keyword_retriever/bm25"
        return self

    def build_index(self, docs: List[Any]) -> None:
        from globalbot.backend.storages.keyword_search import BM25Retriever
        self._bm25 = BM25Retriever(top_k=self.top_k * 3 if self.reranker else self.top_k)
        self._bm25.add(docs)
        self.log.info("keyword_retriever.index_built", n_docs=len(docs))

    def run(self, query: str, **kwargs: Any) -> List[RetrievedDocument]:
        if not self._bm25:
            self.log.warning("keyword_retriever.no_index")
            return []

        self.log.debug("keyword_retriever.query", query=query[:80])

        with self.timed("keyword_retriever.search.start", "keyword_retriever.search.end"):
            results = self._bm25.query(
                query=query,
                filters=self.filters or kwargs.get("filters"),
            )

        if self.score_threshold is not None:
            results = [r for r in results if r.score >= self.score_threshold]

        if self.reranker and results:
            with self.timed("keyword_retriever.rerank.start", "keyword_retriever.rerank.end"):
                results = self.reranker.run(query=query, docs=results)

        self.log.debug("keyword_retriever.results", n=len(results))
        return results[: self.top_k]


class HybridRetriever(BaseComponent):
    vector_retriever: RAGRetriever
    keyword_retriever: KeywordRetriever
    top_k: int = 5
    vector_weight: float = 0.5
    reranker: Optional[Any] = None

    @model_validator(mode="after")
    def _set_name(self) -> "HybridRetriever":
        if not self.name:
            self.name = "hybrid_retriever"
        return self

    def run(self, query: str, **kwargs: Any) -> List[RetrievedDocument]:
        self.log.debug("hybrid_retriever.query", query=query[:80])

        vector_results = self.vector_retriever.run(query, **kwargs)
        keyword_results = self.keyword_retriever.run(query, **kwargs)

        merged = self._reciprocal_rank_fusion(vector_results, keyword_results)

        if self.reranker and merged:
            with self.timed("hybrid_retriever.rerank.start", "hybrid_retriever.rerank.end"):
                merged = self.reranker.run(query=query, docs=merged)

        self.log.debug("hybrid_retriever.results", n=len(merged))
        return merged[: self.top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievedDocument],
        keyword_results: List[RetrievedDocument],
        k: int = 60,
    ) -> List[RetrievedDocument]:
        scores: Dict[str, float] = {}
        docs_by_id: Dict[str, RetrievedDocument] = {}

        for rank, doc in enumerate(vector_results):
            rrf = self.vector_weight / (k + rank + 1)
            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + rrf
            docs_by_id[doc.doc_id] = doc

        kw_weight = 1.0 - self.vector_weight
        for rank, doc in enumerate(keyword_results):
            rrf = kw_weight / (k + rank + 1)
            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + rrf
            if doc.doc_id not in docs_by_id:
                docs_by_id[doc.doc_id] = doc

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        results = []
        for doc_id in sorted_ids:
            doc = docs_by_id[doc_id].model_copy()
            doc.score = scores[doc_id]
            results.append(doc)
        return results