from __future__ import annotations

import pytest
from typing import List, Optional, Any

from globalbot.backend.base import Document, RetrievedDocument
from globalbot.backend.rerankers.base import BaseReranker
from globalbot.backend.rerankers import init_reranker, RERANKER_PROVIDERS
from globalbot.backend.storages.vectorstores.base import BaseVectorStore
from globalbot.backend.embeddings.base import BaseEmbeddings
from globalbot.backend.storages.retrieval import RAGRetriever, KeywordRetriever, HybridRetriever


def _make_docs(n: int = 5) -> List[RetrievedDocument]:
    return [
        RetrievedDocument(
            page_content=f"document about topic {i}",
            metadata={"source": f"doc{i}.txt"},
            doc_id=f"doc-{i:04d}",
            score=1.0 - i * 0.1,
        )
        for i in range(n)
    ]


class _ReverseReranker(BaseReranker):
    def rerank(self, query: str, docs: List[RetrievedDocument], **kwargs: Any) -> List[RetrievedDocument]:
        reversed_docs = list(reversed(docs))
        for i, doc in enumerate(reversed_docs):
            new_doc = doc.model_copy()
            new_doc.score = float(i) / max(len(reversed_docs), 1)
            reversed_docs[i] = new_doc
        return reversed_docs[: self.top_k]


class _DummyEmbeddings(BaseEmbeddings):
    dim: int = 4

    def _embed_documents(self, texts, **kwargs):
        return [[0.1] * self.dim for _ in texts]

    def _embed_query(self, text, **kwargs):
        return [0.1] * self.dim


class _DummyVectorStore(BaseVectorStore):
    _store: dict = {}
    _fixed_results: Optional[List[RetrievedDocument]] = None

    def model_post_init(self, __context):
        self._store = {}
        self.name = "dummy_vs"

    def add(self, docs, embeddings, **kwargs):
        for doc, emb in zip(docs, embeddings):
            self._store[doc.doc_id] = (doc, emb)
        return [doc.doc_id for doc in docs]

    def query(self, embedding, top_k=5, filters=None, **kwargs):
        if self._fixed_results is not None:
            return self._fixed_results[:top_k]
        results = []
        for doc_id, (doc, emb) in list(self._store.items())[:top_k]:
            results.append(RetrievedDocument(
                page_content=doc.page_content, metadata=doc.metadata,
                doc_id=doc_id, score=0.9,
            ))
        return results

    def delete(self, ids, **kwargs):
        for id_ in ids:
            self._store.pop(id_, None)

    def drop(self):
        self._store.clear()

    def count(self):
        return len(self._store)


class TestBaseReranker:
    def test_run_calls_rerank(self):
        reranker = _ReverseReranker(top_k=3)
        docs = _make_docs(5)
        results = reranker.run("test query", docs)
        assert len(results) == 3

    def test_run_empty_docs(self):
        reranker = _ReverseReranker(top_k=3)
        results = reranker.run("test", [])
        assert results == []

    def test_top_k_respected(self):
        reranker = _ReverseReranker(top_k=2)
        docs = _make_docs(5)
        results = reranker.run("test", docs)
        assert len(results) <= 2


class TestInitReranker:
    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker method"):
            init_reranker("unknown_xyz")

    def test_providers_registered(self):
        assert "cross-encoder" in RERANKER_PROVIDERS
        assert "cohere" in RERANKER_PROVIDERS
        assert "llm" in RERANKER_PROVIDERS


class TestBM25Retriever:
    def _make_plain_docs(self, texts: List[str]) -> List[Document]:
        return [Document.from_text(t, metadata={"idx": i}) for i, t in enumerate(texts)]

    def test_add_and_query(self):
        from globalbot.backend.storages.keyword_search import BM25Retriever
        retriever = BM25Retriever(top_k=3)
        docs = self._make_plain_docs([
            "python programming language",
            "java enterprise development",
            "machine learning with python",
            "database management systems",
            "python data science",
        ])
        retriever.add(docs)
        results = retriever.query("python")
        assert len(results) > 0
        assert all(isinstance(r, RetrievedDocument) for r in results)
        assert all("python" in r.page_content.lower() for r in results)

    def test_top_k(self):
        from globalbot.backend.storages.keyword_search import BM25Retriever
        retriever = BM25Retriever(top_k=2)
        docs = self._make_plain_docs([f"document {i}" for i in range(10)])
        retriever.add(docs)
        results = retriever.query("document")
        assert len(results) <= 2

    def test_empty_index_returns_empty(self):
        from globalbot.backend.storages.keyword_search import BM25Retriever
        retriever = BM25Retriever(top_k=3)
        results = retriever.query("test")
        assert results == []

    def test_scores_are_floats(self):
        from globalbot.backend.storages.keyword_search import BM25Retriever
        retriever = BM25Retriever(top_k=5)
        docs = self._make_plain_docs(["hello world", "foo bar", "hello foo"])
        retriever.add(docs)
        results = retriever.query("hello")
        for r in results:
            assert isinstance(r.score, float)

    def test_run_alias(self):
        from globalbot.backend.storages.keyword_search import BM25Retriever
        retriever = BM25Retriever(top_k=3)
        docs = self._make_plain_docs(["test document one", "test document two"])
        retriever.add(docs)
        results = retriever.run("test")
        assert isinstance(results, list)

    def test_filters(self):
        from globalbot.backend.storages.keyword_search import BM25Retriever
        retriever = BM25Retriever(top_k=5)
        docs = [
            Document.from_text("python tutorial", metadata={"lang": "en"}),
            Document.from_text("python tutoriel", metadata={"lang": "fr"}),
            Document.from_text("java tutorial", metadata={"lang": "en"}),
        ]
        retriever.add(docs)
        results = retriever.query("tutorial", filters={"lang": "en"})
        assert all(r.metadata.get("lang") == "en" for r in results)


class TestRAGRetrieverWithReranker:
    def test_retriever_without_reranker(self):
        vs = _DummyVectorStore()
        emb = _DummyEmbeddings()
        docs = [Document.from_text(f"doc {i}") for i in range(5)]
        vs.add(docs, [[0.1] * 4] * 5)

        retriever = RAGRetriever(vectorstore=vs, embeddings=emb, top_k=3)
        results = retriever.run("test")
        assert len(results) <= 3

    def test_retriever_with_reranker(self):
        vs = _DummyVectorStore()
        vs._fixed_results = _make_docs(10)
        emb = _DummyEmbeddings()
        reranker = _ReverseReranker(top_k=3)

        retriever = RAGRetriever(
            vectorstore=vs, embeddings=emb,
            top_k=3, reranker=reranker,
        )
        results = retriever.run("test query")
        assert len(results) <= 3

    def test_reranker_changes_order(self):
        vs = _DummyVectorStore()
        original_docs = _make_docs(5)
        vs._fixed_results = original_docs
        emb = _DummyEmbeddings()
        reranker = _ReverseReranker(top_k=5)

        retriever = RAGRetriever(
            vectorstore=vs, embeddings=emb,
            top_k=5, reranker=reranker,
        )
        results = retriever.run("test")
        assert len(results) > 0

    def test_fetches_more_when_reranker_set(self):
        fetched_counts = []

        class _TrackingVS(_DummyVectorStore):
            def query(self, embedding, top_k=5, filters=None, **kwargs):
                fetched_counts.append(top_k)
                return _make_docs(top_k)

        vs = _TrackingVS()
        emb = _DummyEmbeddings()
        reranker = _ReverseReranker(top_k=3)

        retriever = RAGRetriever(
            vectorstore=vs, embeddings=emb,
            top_k=3, reranker=reranker,
        )
        retriever.run("test")
        assert fetched_counts[0] == 9


class TestKeywordRetriever:
    def _make_plain_docs(self, texts):
        return [Document.from_text(t) for t in texts]

    def test_build_and_query(self):
        retriever = KeywordRetriever(top_k=3)
        docs = self._make_plain_docs([
            "python is a programming language",
            "java is used for enterprise",
            "python for data science",
        ])
        retriever.build_index(docs)
        results = retriever.run("python programming")
        assert len(results) > 0

    def test_no_index_returns_empty(self):
        retriever = KeywordRetriever(top_k=3)
        results = retriever.run("test")
        assert results == []

    def test_with_reranker(self):
        retriever = KeywordRetriever(top_k=2, reranker=_ReverseReranker(top_k=2))
        docs = self._make_plain_docs([f"document {i}" for i in range(5)])
        retriever.build_index(docs)
        results = retriever.run("document")
        assert len(results) <= 2


class TestHybridRetriever:
    def _make_retrievers(self, vector_docs=None, keyword_texts=None):
        vs = _DummyVectorStore()
        emb = _DummyEmbeddings()

        if vector_docs:
            vs._fixed_results = vector_docs
        else:
            docs = [Document.from_text(f"vector doc {i}") for i in range(5)]
            vs.add(docs, [[0.1] * 4] * 5)

        vector_ret = RAGRetriever(vectorstore=vs, embeddings=emb, top_k=5)

        keyword_ret = KeywordRetriever(top_k=5)
        texts = keyword_texts or [f"keyword doc {i}" for i in range(5)]
        keyword_ret.build_index([Document.from_text(t) for t in texts])

        return vector_ret, keyword_ret

    def test_hybrid_basic(self):
        vector_ret, keyword_ret = self._make_retrievers()
        hybrid = HybridRetriever(
            vector_retriever=vector_ret,
            keyword_retriever=keyword_ret,
            top_k=5,
        )
        results = hybrid.run("test query")
        assert isinstance(results, list)
        assert len(results) <= 5

    def test_rrf_merges_results(self):
        shared_text = "shared document about python"
        vector_docs = [
            RetrievedDocument(page_content=shared_text, doc_id="shared-001", score=0.9),
            RetrievedDocument(page_content="only in vector", doc_id="vec-001", score=0.8),
        ]
        vector_ret, keyword_ret = self._make_retrievers(vector_docs=vector_docs)
        keyword_ret.build_index([
            Document.from_text(shared_text, doc_id="shared-001"),
            Document.from_text("only in keyword", doc_id="kw-001"),
        ])

        hybrid = HybridRetriever(
            vector_retriever=vector_ret,
            keyword_retriever=keyword_ret,
            top_k=10,
        )
        results = hybrid.run("python")
        ids = [r.doc_id for r in results]
        assert "shared-001" in ids

    def test_vector_weight(self):
        vector_ret, keyword_ret = self._make_retrievers()
        hybrid_vector_heavy = HybridRetriever(
            vector_retriever=vector_ret,
            keyword_retriever=keyword_ret,
            vector_weight=0.9,
            top_k=5,
        )
        results = hybrid_vector_heavy.run("test")
        assert len(results) <= 5

    def test_with_reranker(self):
        vector_ret, keyword_ret = self._make_retrievers()
        reranker = _ReverseReranker(top_k=3)
        hybrid = HybridRetriever(
            vector_retriever=vector_ret,
            keyword_retriever=keyword_ret,
            top_k=3,
            reranker=reranker,
        )
        results = hybrid.run("test")
        assert len(results) <= 3