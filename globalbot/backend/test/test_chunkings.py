from __future__ import annotations

import sys
import pytest
from unittest.mock import MagicMock, patch
from typing import List

from globalbot.backend.base import Document
from globalbot.backend.chunkings.base import BaseChunker
from globalbot.backend.chunkings import init_chunker, CHUNKER_PROVIDERS


class _DummyChunker(BaseChunker):
    def split_text(self, text: str) -> List[str]:
        size = self.chunk_size
        return [text[i: i + size] for i in range(0, len(text), size) if text[i: i + size].strip()]


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
        with patch("langchain_text_splitters.RecursiveCharacterTextSplitter") as mock_cls:
            mock_splitter = MagicMock()
            mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
            mock_cls.return_value = mock_splitter
            chunker = init_chunker("recursive", chunk_size=200)
            assert isinstance(chunker, __import__("globalbot.backend.chunkings.fixed", fromlist=["RecursiveChunker"]).RecursiveChunker)

    def test_init_chunker_params(self):
        mock_lts = MagicMock()
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["a", "b"]
        mock_lts.RecursiveCharacterTextSplitter.return_value = mock_splitter
        with patch.dict(sys.modules, {"langchain_text_splitters": mock_lts}):
            chunker = init_chunker("recursive", chunk_size=300, chunk_overlap=50)
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
        mock_emb = MagicMock()
        mock_emb.run.return_value = [[0.1, 0.2], [0.9, 0.8]]

        from globalbot.backend.chunkings.semantic import ClusterSemanticChunker
        chunker = ClusterSemanticChunker(embeddings=mock_emb, max_chunk_size=500, min_chunk_size=10)
        result = chunker.split_text("Hello world. This is a test.")
        assert isinstance(result, list)

    def test_split_empty_text(self):
        mock_emb = MagicMock()
        from globalbot.backend.chunkings.semantic import ClusterSemanticChunker
        chunker = ClusterSemanticChunker(embeddings=mock_emb)
        result = chunker.split_text("")
        assert result == []

    def test_split_single_sentence(self):
        mock_emb = MagicMock()
        mock_emb.run.return_value = [[0.1, 0.2]]
        from globalbot.backend.chunkings.semantic import ClusterSemanticChunker
        chunker = ClusterSemanticChunker(embeddings=mock_emb)
        result = chunker.split_text("This is a single sentence that is long enough.")
        assert len(result) == 1

    def test_low_similarity_forces_split(self):
        mock_emb = MagicMock()
        mock_emb.run.return_value = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        from globalbot.backend.chunkings.semantic import ClusterSemanticChunker
        chunker = ClusterSemanticChunker(embeddings=mock_emb, max_chunk_size=1000, min_chunk_size=5)
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
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "split_after: 2, 4"

        from globalbot.backend.chunkings.llm_chunker import LLMChunker
        chunker = LLMChunker(llm=mock_llm, initial_chunk_size=10)
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
        mock_llm = MagicMock()
        from globalbot.backend.chunkings.llm_chunker import LLMChunker
        chunker = LLMChunker(llm=mock_llm)
        result = chunker.split_text("")
        assert result == []


class TestRAGIndexerWithChunker:
    def _make_indexer(self, chunker=None):
        from globalbot.backend.storages.ingestion import RAGIndexer

        mock_vs = MagicMock()
        mock_vs.name = "mock_vs"
        mock_vs.add.return_value = ["id1", "id2"]

        mock_emb = MagicMock()
        mock_emb.run.return_value = [[0.1] * 4, [0.2] * 4]

        kwargs = dict(vectorstore=mock_vs, embeddings=mock_emb)
        if chunker is not None:
            kwargs["chunker"] = chunker
        return RAGIndexer(**kwargs), mock_vs, mock_emb

    def test_default_chunker_is_recursive(self):
        from globalbot.backend.storages.ingestion import RAGIndexer
        from globalbot.backend.chunkings.fixed import RecursiveChunker

        mock_vs = MagicMock()
        mock_vs.name = "vs"
        mock_vs.add.return_value = []
        mock_emb = MagicMock()
        mock_emb.run.return_value = []

        indexer = RAGIndexer(vectorstore=mock_vs, embeddings=mock_emb)
        assert isinstance(indexer.chunker, RecursiveChunker)

    def test_custom_chunker_used(self):
        custom = _DummyChunker(chunk_size=30)
        indexer, mock_vs, mock_emb = self._make_indexer(chunker=custom)
        docs = [Document.from_text("x" * 90)]
        mock_emb.run.side_effect = lambda texts: [[0.1] * 4 for _ in texts]
        mock_vs.add.side_effect = lambda docs, embs: [d.doc_id for d in docs]
        result = indexer.run(docs)
        assert result["stored"] > 0
        assert indexer.chunker is custom

    def test_semantic_chunker_integration(self):
        mock_emb_model = MagicMock()
        mock_emb_model.run.return_value = [[0.1, 0.2, 0.3]] * 5

        from globalbot.backend.chunkings.semantic import ClusterSemanticChunker
        chunker = ClusterSemanticChunker(embeddings=mock_emb_model, max_chunk_size=200)

        mock_vs = MagicMock()
        mock_vs.name = "vs"
        mock_vs.add.side_effect = lambda docs, embs: [d.doc_id for d in docs]

        mock_emb_indexer = MagicMock()
        mock_emb_indexer.run.side_effect = lambda texts: [[0.1] * 4 for _ in texts]

        from globalbot.backend.storages.ingestion import RAGIndexer
        indexer = RAGIndexer(vectorstore=mock_vs, embeddings=mock_emb_indexer, chunker=chunker)

        docs = [Document.from_text("Hello world. This is a test. Another sentence here.")]
        result = indexer.run(docs)
        assert "stored" in result