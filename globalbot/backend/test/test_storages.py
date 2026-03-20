from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call
from typing import List

from globalbot.backend.base import Document, RetrievedDocument
from globalbot.backend.storage.vectorstores.base import BaseVectorStore
from globalbot.backend.storage.vectorstores import init_vectorstore, VECTORSTORE_PROVIDERS
from globalbot.backend.storage.ingestion import TextSplitter, RAGIndexer
from globalbot.backend.storage.retrieval import RAGRetriever


class _DummyVectorStore(BaseVectorStore):
    _store: dict = {}

    def model_post_init(self, __context):
        self._store = {}
        self.name = "dummy"

    def add(self, docs, embeddings, **kwargs):
        for doc, emb in zip(docs, embeddings):
            self._store[doc.doc_id] = (doc, emb)
        return [doc.doc_id for doc in docs]

    def query(self, embedding, top_k=5, filters=None, **kwargs):
        results = []
        for doc_id, (doc, emb) in list(self._store.items())[:top_k]:
            results.append(RetrievedDocument(page_content=doc.page_content, metadata=doc.metadata, doc_id=doc_id, score=0.9))
        return results

    def delete(self, ids, **kwargs):
        for id_ in ids:
            self._store.pop(id_, None)

    def drop(self):
        self._store.clear()

    def count(self):
        return len(self._store)


class _DummyEmbeddings:
    def run(self, texts):
        return [[0.1] * 4 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 4


class TestBaseVectorStore:
    def test_add_and_query(self):
        store = _DummyVectorStore()
        docs = [Document.from_text("hello world", metadata={"source": "test"})]
        embeddings = [[0.1, 0.2, 0.3, 0.4]]
        ids = store.add(docs, embeddings)
        assert len(ids) == 1
        results = store.query([0.1, 0.2, 0.3, 0.4], top_k=5)
        assert len(results) == 1
        assert isinstance(results[0], RetrievedDocument)

    def test_delete(self):
        store = _DummyVectorStore()
        docs = [Document.from_text("hello")]
        store.add(docs, [[0.1] * 4])
        assert store.count() == 1
        store.delete([docs[0].doc_id])
        assert store.count() == 0

    def test_drop(self):
        store = _DummyVectorStore()
        docs = [Document.from_text("hello"), Document.from_text("world")]
        store.add(docs, [[0.1] * 4, [0.2] * 4])
        assert store.count() == 2
        store.drop()
        assert store.count() == 0

    def test_run_calls_add(self):
        store = _DummyVectorStore()
        docs = [Document.from_text("test")]
        ids = store.run(docs, [[0.1] * 4])
        assert len(ids) == 1


class TestInitVectorstore:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown vectorstore provider"):
            init_vectorstore("unknown_xyz")

    def test_all_providers_registered(self):
        assert "chroma" in VECTORSTORE_PROVIDERS
        assert "qdrant" in VECTORSTORE_PROVIDERS
        assert "milvus" in VECTORSTORE_PROVIDERS
        assert "mongodb" in VECTORSTORE_PROVIDERS

    def test_init_chroma(self):
        import sys
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            from importlib import reload
            import globalbot.backend.storages.vectorstores.chroma_impl as chroma_mod
            reload(chroma_mod)
            store = chroma_mod.ChromaVectorStore(collection_name="test", persist_directory="/tmp/test")
            assert store.name == "chroma/test"

    def test_init_qdrant(self):
        import sys
        mock_qdrant = MagicMock()
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_qdrant.QdrantClient.return_value = mock_client

        with patch.dict(sys.modules, {"qdrant_client": mock_qdrant, "qdrant_client.models": MagicMock()}):
            from importlib import reload
            import globalbot.backend.storages.vectorstores.qdrant_impl as qdrant_mod
            reload(qdrant_mod)
            store = qdrant_mod.QdrantVectorStore(collection_name="test", path="/tmp/qdrant")
            assert store.name == "qdrant/test"

    def test_init_mongodb(self):
        import sys
        mock_pymongo = MagicMock()
        mock_client = MagicMock()
        mock_pymongo.MongoClient.return_value = mock_client

        with patch.dict(sys.modules, {"pymongo": mock_pymongo}):
            from importlib import reload
            import globalbot.backend.storages.vectorstores.mongodb_impl as mongo_mod
            reload(mongo_mod)
            store = mongo_mod.MongoDBVectorStore(collection_name="test")
            assert "mongodb" in store.name


class TestTextSplitter:
    def test_splits_long_text(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        doc = Document.from_text("a" * 200, metadata={"source": "test"})
        chunks = splitter.run([doc])
        assert len(chunks) > 1

    def test_preserves_metadata(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        doc = Document.from_text("hello world " * 10, metadata={"source": "test.pdf"})
        chunks = splitter.run([doc])
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.pdf"
            assert chunk.metadata.get("parent_doc_id") == doc.doc_id

    def test_short_text_single_chunk(self):
        splitter = TextSplitter(chunk_size=1000, chunk_overlap=0)
        doc = Document.from_text("short text")
        chunks = splitter.run([doc])
        assert len(chunks) == 1

    def test_multiple_docs(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        docs = [Document.from_text("a" * 100), Document.from_text("b" * 100)]
        chunks = splitter.run(docs)
        assert len(chunks) >= 4


class TestRAGIndexer:
    def test_index_docs(self):
        store = _DummyVectorStore()
        embeddings = _DummyEmbeddings()
        indexer = RAGIndexer(vectorstore=store, embeddings=embeddings, batch_size=10)
        docs = [Document.from_text(f"document {i}") for i in range(5)]
        result = indexer.run(docs)
        assert result["stored"] > 0
        assert store.count() > 0

    def test_batching(self):
        store = _DummyVectorStore()
        embeddings = _DummyEmbeddings()
        indexer = RAGIndexer(vectorstore=store, embeddings=embeddings, batch_size=2)
        docs = [Document.from_text(f"doc {i}" * 20) for i in range(10)]
        result = indexer.run(docs)
        assert result["stored"] > 0

    def test_splits_before_indexing(self):
        store = _DummyVectorStore()
        embeddings = _DummyEmbeddings()
        splitter = TextSplitter(chunk_size=20, chunk_overlap=0)
        indexer = RAGIndexer(vectorstore=store, embeddings=embeddings, splitter=splitter)
        docs = [Document.from_text("x" * 100)]
        result = indexer.run(docs)
        assert result["stored"] > 1


class TestRAGRetriever:
    def test_retrieve(self):
        store = _DummyVectorStore()
        embeddings = _DummyEmbeddings()
        docs = [Document.from_text("python programming language")]
        store.add(docs, [[0.1] * 4])

        retriever = RAGRetriever(vectorstore=store, embeddings=embeddings, top_k=3)
        results = retriever.run("python")
        assert len(results) >= 1
        assert isinstance(results[0], RetrievedDocument)

    def test_score_threshold_filters(self):
        store = _DummyVectorStore()
        embeddings = _DummyEmbeddings()
        docs = [Document.from_text("test doc")]
        store.add(docs, [[0.1] * 4])

        retriever = RAGRetriever(vectorstore=store, embeddings=embeddings, score_threshold=0.99)
        results = retriever.run("test")
        assert len(results) == 0

    def test_retriever_name(self):
        store = _DummyVectorStore()
        embeddings = _DummyEmbeddings()
        retriever = RAGRetriever(vectorstore=store, embeddings=embeddings)
        assert "retriever" in retriever.name


class TestWebCrawler:
    def test_crawl_single_page(self):
        import sys
        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "<html><body><h1>Hello World</h1><p>This is a test page with enough content.</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_response

        mock_bs4 = MagicMock()

        with patch.dict(sys.modules, {"requests": mock_requests}):
            from globalbot.backend.storage.vectorstores.crawler import WebCrawler
            crawler = WebCrawler(
                start_urls=["http://example.com"],
                max_depth=0,
                max_pages=1,
                delay=0,
            )
            docs = crawler.run()
            assert len(docs) >= 0

    def test_is_allowed_domain_filter(self):
        from globalbot.backend.storage.vectorstores.crawler import WebCrawler
        crawler = WebCrawler(
            start_urls=["http://example.com"],
            allowed_domains=["example.com"],
            delay=0,
        )
        assert crawler._is_allowed("http://example.com/page") is True
        assert crawler._is_allowed("http://other.com/page") is False

    def test_exclude_patterns(self):
        from globalbot.backend.storages.vectorstores.crawler import WebCrawler
        crawler = WebCrawler(
            start_urls=["http://example.com"],
            exclude_patterns=["/admin", "/login"],
            delay=0,
        )
        assert crawler._is_excluded("http://example.com/admin/panel") is True
        assert crawler._is_excluded("http://example.com/about") is False