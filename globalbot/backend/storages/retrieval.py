from __future__ import annotations

from typing import Any, Dict, List, Optional

from globalbot.backend.base import BaseComponent, RetrievedDocument
from globalbot.backend.embeddings.base import BaseEmbeddings
from globalbot.backend.storages.vectorstores.base import BaseVectorStore


class RAGRetriever(BaseComponent):
    vectorstore: BaseVectorStore
    embeddings: BaseEmbeddings
    top_k: int = 5
    score_threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        if not self.name:
            self.name = f"retriever/{self.vectorstore.name}"

    def run(self, query: str, **kwargs: Any) -> List[RetrievedDocument]:
        self.log.debug("retriever.query", query=query[:80], top_k=self.top_k)

        with self.timed("retriever.embed.start", "retriever.embed.end"):
            embedding = self.embeddings.embed_query(query)

        with self.timed("retriever.search.start", "retriever.search.end"):
            results = self.vectorstore.query(
                embedding=embedding,
                top_k=self.top_k,
                filters=self.filters or kwargs.get("filters"),
            )

        if self.score_threshold is not None:
            results = [r for r in results if r.score >= self.score_threshold]

        self.log.debug("retriever.results", n=len(results))
        return results