from __future__ import annotations

from typing import Any, List, Optional

from pydantic import model_validator

from globalbot.backend.base import RetrievedDocument
from globalbot.backend.rerankers.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    top_k: int = 5
    batch_size: int = 32

    _model: Any = None

    @model_validator(mode="after")
    def _init_model(self) -> "CrossEncoderReranker":
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_id, device=self.device)
        if not self.name:
            self.name = f"cross-encoder/{self.model_id}"
        return self

    def rerank(self, query: str, docs: List[RetrievedDocument], **kwargs: Any) -> List[RetrievedDocument]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[: self.top_k]:
            new_doc = doc.model_copy()
            new_doc.score = float(score)
            results.append(new_doc)
        return results


class CohereReranker(BaseReranker):
    model: str = "rerank-english-v3.0"
    api_key: Optional[str] = None
    top_k: int = 5

    _client: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "CohereReranker":
        import cohere
        self._client = cohere.Client(api_key=self.api_key)
        if not self.name:
            self.name = f"cohere/{self.model}"
        return self

    def rerank(self, query: str, docs: List[RetrievedDocument], **kwargs: Any) -> List[RetrievedDocument]:
        self.log.debug("reranker.cohere.rerank", n=len(docs))
        response = self._client.rerank(
            model=self.model,
            query=query,
            documents=[doc.page_content for doc in docs],
            top_n=self.top_k,
        )
        results = []
        for item in response.results:
            doc = docs[item.index].model_copy()
            doc.score = item.relevance_score
            results.append(doc)
        return results


class LLMReranker(BaseReranker):
    top_k: int = 5
    batch_size: int = 10

    _llm: Any = None

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        if not self.name:
            self.name = f"llm-reranker/{type(llm).__name__}"

    def rerank(self, query: str, docs: List[RetrievedDocument], **kwargs: Any) -> List[RetrievedDocument]:
        import re

        scored = []
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i: i + self.batch_size]
            prompt = self._build_prompt(query, batch)
            response = self._llm.chat(user=prompt)
            scores = self._parse_scores(response, len(batch))
            for doc, score in zip(batch, scores):
                new_doc = doc.model_copy()
                new_doc.score = score
                scored.append(new_doc)

        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[: self.top_k]

    def _build_prompt(self, query: str, docs: List[RetrievedDocument]) -> str:
        lines = [f"Query: {query}\n"]
        for i, doc in enumerate(docs):
            lines.append(f"[{i}] {doc.page_content[:300]}")
        lines.append(
            "\nScore each document from 0.0 to 1.0 based on relevance to the query."
            "\nRespond ONLY with a comma-separated list of scores in order, e.g.: 0.9, 0.2, 0.7"
        )
        return "\n".join(lines)

    def _parse_scores(self, response: str, n: int) -> List[float]:
        import re
        nums = re.findall(r"\d+(?:\.\d+)?", response)
        scores = [float(x) for x in nums[:n]]
        while len(scores) < n:
            scores.append(0.0)
        return scores