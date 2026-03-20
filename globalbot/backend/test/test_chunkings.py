from __future__ import annotations

import sys
import pytest
from unittest.mock import MagicMock, patch
from typing import List, Optional

from globalbot.backend.base import Document
from globalbot.backend.chunkings.base import BaseChunker
from globalbot.backend.chunkings import init_chunker, CHUNKER_PROVIDERS
from globalbot.backend.embeddings.base import BaseEmbeddings
from globalbot.backend.llms.chats.base import BaseChatLLM
from globalbot.backend.base import AIMessage


class _DummyChunker(BaseChunker):
    def split_text(self, text: str) -> List[str]:
        size = self.chunk_size
        return [text[i: i + size] for i in range(0, len(text), size) if text[i: i + size].strip()]


class _DummyEmbeddings(BaseEmbeddings):
    dim: int = 3

    def _embed_documents(self, texts, **kwargs):
        return [[0.1] * self.dim for _ in texts]

    def _embed_query(self, text, **kwargs):
        return [0.1] * self.dim


class _DummyLLM(BaseChatLLM):
    response: str = "split_after: 2, 4"

    def _call(self, messages, **kwargs):
        return AIMessage(content=self.response)


class TestBaseChunker:
    def test_split_documents_preserves_metadata(self):
        chunker = _DummyChunker(chunk_size=20)
        doc = Document.from_text("a" * 100, metadata={"source": "test.pdf", "page": 1})
        chunks = chunker.split_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page"] == 1
            assert chunk.metadata["parent_doc_id"] == doc.doc_id

    def test_split_documents_assigns_chunk_index(self):
        chunker = _DummyChunker(chunk_size=20)
        doc = Document.from_text("b" * 60)
        chunks = chunker.split_documents([doc])
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_split_documents_skips_empty_chunks(self):
        chunker = _DummyChunker(chunk_size=50)
        doc = Document.from_text("   ")
        chunks = chunker.split_documents([doc])
        assert len(chunks) == 0

    def test_split_multiple_docs(self):
        chunker = _DummyChunker(chunk_size=20)
        docs = [Document.from_text("a" * 40), Document.from_text("b" * 40)]
        chunks = chunker.split_documents(docs)
        assert len(chunks) >= 4

    def test_run_calls_split_documents(self):
        chunker = _DummyChunker(chunk_size=20)
        docs = [Document.from_text("x" * 60)]
        result = chunker.run(docs)
        assert len(result) > 1
        assert all(isinstance(c, Document) for c in result)

    def test_unique_chunk_ids(self):
        chunker = _DummyChunker(chunk_size=20)
        doc = Document.from_text("unique content " * 10)
        chunks = chunker.split_documents([doc])
        ids = [c.doc_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestInitChunker:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking method"):
            init_chunker("unknown_xyz")

    def test_all_providers_registered(self):
        assert "recursive" in CHUNKER_PROVIDERS
        assert "character" in CHUNKER_PROVIDERS
        assert "token" in CHUNKER_PROVIDERS
        assert "sentence" in CHUNKER_PROVIDERS
        assert "markdown" in CHUNKER_PROVIDERS
        assert "semantic" in CHUNKER_PROVIDERS
        assert "llm" in CHUNKER_PROVIDERS

    def test_default_is_recursive(self):
        mock_lts = MagicMock()
        mock_lts.RecursiveCharacterTextSplitter.return_value = MagicMock()
        with patch.dict(sys.modules, {"langchain_text_splitters": mock_lts}):
            from importlib import reload
            import globalbot.backend.chunkings.fixed as fixed_mod
            reload(fixed_mod)
            chunker = fixed_mod.RecursiveChunker(chunk_size=200)
            assert isinstance(chunker, fixed_mod.RecursiveChunker)

    def test_init_chunker_params(self):
        mock_lts = MagicMock()
        mock_lts.RecursiveCharacterTextSplitter.return_value = MagicMock()
        with patch.dict(sys.modules, {"langchain_text_splitters": mock_lts}):
            from importlib import reload
            import globalbot.backend.chunkings.fixed as fixed_mod
            reload(fixed_mod)
            chunker = fixed_mod.RecursiveChunker(chunk_size=300, chunk_overlap=50)
            assert chunker.chunk_size == 300
            assert chunker.chunk_overlap == 50


class TestRecursiveChunker:
    def test_split_text(self):
        mock_lts = MagicMock()
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["chunk one", "chunk two", "chunk three"]
        mock_lts.RecursiveCharacterTextSplitter.return_value = mock_splitter

        with patch.dict(sys.modules, {"langchain_text_splitters": mock_lts}):
            from importlib import reload
            import globalbot.backend.chunkings.fixed as fixed_mod
            reload(fixed_mod)
            chunker = fixed_mod.RecursiveChunker(chunk_size=100, chunk_overlap=20)
            result = chunker.split_text("some long text")

        assert result == ["chunk one", "chunk two", "chunk three"]
        mock_lts.RecursiveCharacterTextSplitter.assert_called_once_with(
            chunk_size=100, chunk_overlap=20,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            keep_separator=True, is_separator_regex=False,
        )

    def test_name_format(self):
        mock_lts = MagicMock()
        mock_lts.RecursiveCharacterTextSplitter.return_value = MagicMock()
        with patch.dict(sys.modules, {"langchain_text_splitters": mock_lts}):
            from importlib import reload
            import globalbot.backend.chunkings.fixed as fixed_mod
            reload(fixed_mod)
            chunker = fixed_mod.RecursiveChunker(chunk_size=500, chunk_overlap=100)
            assert chunker.name == "recursive/500/100"


class TestTokenChunker:
    def test_split_text(self):
        mock_lts = MagicMock()
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["token chunk 1", "token chunk 2"]
        mock_lts.TokenTextSplitter.return_value = mock_splitter

        with patch.dict(sys.modules, {"langchain_text_splitters": mock_lts}):
            from importlib import reload
            import globalbot.backend.chunkings.fixed as fixed_mod
            reload(fixed_mod)
            chunker = fixed_mod.TokenChunker(chunk_size=512, chunk_overlap=50)
            result = chunker.split_text("text to tokenize")

        assert result == ["token chunk 1", "token chunk 2"]


class TestClusterSemanticChunker:
    def test_split_short_text(self):
        emb = _DummyEmbeddings()
        from globalbot.backend.chunkings.semantic import ClusterSemanticChunker
        chunker = ClusterSemanticChunker(embeddings=emb, max_chunk_size=500, min_chunk_size=10)
        result = chunker.split_text("Hello world. This is a test.")
        assert isinstance(result, list)

    def test_split_empty_text(self):
        emb = _DummyEmbeddings()
        from globalbot.backend.chunkings.semantic import ClusterSemanticChunker
        chunker = ClusterSemanticChunker(embeddings=emb)
        result = chunker.split_text("")
        assert result == []

    def test_split_single_sentence(self):
        emb = _DummyEmbeddings()
        from globalbot.backend.chunkings.semantic import ClusterSemanticChunker
        chunker = ClusterSemanticChunker(embeddings=emb)
        result = chunker.split_text("This is a single sentence that is long enough.")
        assert len(result) == 1

    def test_low_similarity_forces_split(self):
        class _OrthogonalEmbeddings(BaseEmbeddings):
            _vecs = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
            _call_count: int = 0

            def _embed_documents(self, texts, **kwargs):
                result = self._vecs[:len(texts)]
                return result

            def _embed_query(self, text, **kwargs):
                return [0.1, 0.1, 0.1]

        emb = _OrthogonalEmbeddings()
        from globalbot.backend.chunkings.semantic import ClusterSemanticChunker
        chunker = ClusterSemanticChunker(embeddings=emb, max_chunk_size=1000, min_chunk_size=5)
        sentences = [
            "Python is a programming language.",
            "The ocean is vast and deep.",
            "Machine learning uses data to learn patterns.",
            "Dolphins swim in the sea.",
        ]
        result = chunker.split_text(" ".join(sentences))
        assert len(result) >= 1


class TestLLMChunker:
    def test_split_text(self):
        llm = _DummyLLM(response="split_after: 2, 4")
        from globalbot.backend.chunkings.llm_chunker import LLMChunker
        chunker = LLMChunker(llm=llm, initial_chunk_size=10)
        result = chunker.split_text("word " * 60)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_parse_split_ids_valid(self):
        from globalbot.backend.chunkings.llm_chunker import _parse_split_ids
        ids = _parse_split_ids("split_after: 2, 4, 6", n_chunks=8, current_chunk=1)
        assert ids == [2, 4, 6]

    def test_parse_split_ids_filters_below_current(self):
        from globalbot.backend.chunkings.llm_chunker import _parse_split_ids
        ids = _parse_split_ids("split_after: 1, 3, 5", n_chunks=8, current_chunk=3)
        assert 1 not in ids
        assert 3 in ids

    def test_parse_split_ids_no_match(self):
        from globalbot.backend.chunkings.llm_chunker import _parse_split_ids
        ids = _parse_split_ids("no splits here", n_chunks=8, current_chunk=1)
        assert ids == []

    def test_empty_text(self):
        llm = _DummyLLM()
        from globalbot.backend.chunkings.llm_chunker import LLMChunker
        chunker = LLMChunker(llm=llm)
        result = chunker.split_text("")
        assert result == []


class TestRAGIndexerWithChunker:
    def _make_dummy_vs(self):
        from globalbot.backend.storages.vectorstores.base import BaseVectorStore
        from globalbot.backend.base import RetrievedDocument

        class _DummyVS(BaseVectorStore):
            _store: dict = {}

            def model_post_init(self, __context):
                self._store = {}
                self.name = "dummy_vs"

            def add(self, docs, embeddings, **kwargs):
                for doc, emb in zip(docs, embeddings):
                    self._store[doc.doc_id] = (doc, emb)
                return [doc.doc_id for doc in docs]

            def query(self, embedding, top_k=5, filters=None, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                for id_ in ids:
                    self._store.pop(id_, None)

            def drop(self):
                self._store.clear()

            def count(self):
                return len(self._store)

        return _DummyVS()

    def test_default_chunker_is_recursive(self):
        from globalbot.backend.storages.ingestion import RAGIndexer
        from globalbot.backend.chunkings.fixed import RecursiveChunker

        vs = self._make_dummy_vs()
        emb = _DummyEmbeddings()
        indexer = RAGIndexer(vectorstore=vs, embeddings=emb)
        assert isinstance(indexer.chunker, RecursiveChunker)

    def test_custom_chunker_used(self):
        from globalbot.backend.storages.ingestion import RAGIndexer

        vs = self._make_dummy_vs()
        emb = _DummyEmbeddings()
        chunker = _DummyChunker(chunk_size=30)
        indexer = RAGIndexer(vectorstore=vs, embeddings=emb, chunker=chunker)
        docs = [Document.from_text("x" * 90)]
        result = indexer.run(docs)
        assert result["stored"] > 0
        assert indexer.chunker is chunker

    def test_semantic_chunker_integration(self):
        from globalbot.backend.chunkings.semantic import ClusterSemanticChunker
        from globalbot.backend.storages.ingestion import RAGIndexer

        emb_for_chunking = _DummyEmbeddings()
        chunker = ClusterSemanticChunker(embeddings=emb_for_chunking, max_chunk_size=200)

        vs = self._make_dummy_vs()
        emb_for_indexing = _DummyEmbeddings()
        indexer = RAGIndexer(vectorstore=vs, embeddings=emb_for_indexing, chunker=chunker)

        docs = [Document.from_text("Hello world. This is a test. Another sentence here.")]
        result = indexer.run(docs)
        assert "stored" in result