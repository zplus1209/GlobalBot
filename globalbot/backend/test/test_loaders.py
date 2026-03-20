from __future__ import annotations

import json
import os
import tempfile
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from globalbot.backend.base import Document, RetrievedDocument, AIMessage
from globalbot.backend.embeddings.base import BaseEmbeddings
from globalbot.backend.storages.vectorstores.base import BaseVectorStore
from globalbot.backend.llms.chats.base import BaseChatLLM
from globalbot.backend.loaders.mineru_loader import MinerUParser, ParsedPDF, ParsedElement
from globalbot.backend.loaders.ingestion_pipeline import PDFIngestionPipeline
from globalbot.backend.loaders.chat_pipeline import RAGChatPipeline, ChatMessage


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
        if self._fixed_results:
            return self._fixed_results[:top_k]
        return [
            RetrievedDocument(
                page_content=doc.page_content,
                metadata=doc.metadata,
                doc_id=doc_id,
                score=0.9,
            )
            for doc_id, (doc, _) in list(self._store.items())[:top_k]
        ]

    def delete(self, ids, **kwargs):
        for id_ in ids:
            self._store.pop(id_, None)

    def drop(self):
        self._store.clear()

    def count(self):
        return len(self._store)


class _DummyLLM(BaseChatLLM):
    response: str = "This is an answer based on context."

    def _call(self, messages, **kwargs):
        return AIMessage(content=self.response)


def _make_content_list():
    return [
        {"type": "text", "text": "Introduction to machine learning.", "page_no": 0, "text_level": "paragraph"},
        {"type": "text", "text": "Deep learning is a subset of ML.", "page_no": 0, "text_level": "paragraph"},
        {"type": "table", "table_caption": "Performance Table", "table_body": "| Model | Acc |\n|---|---|\n| A | 90% |", "table_footnote": "", "page_no": 1},
        {"type": "image", "img_caption": "Figure 1: Model architecture", "img_footnote": "", "page_no": 2, "img_path": ""},
        {"type": "text", "text": "   ", "page_no": 3},
    ]


class TestParsedElement:
    def test_auto_id_generated(self):
        el = ParsedElement(
            element_type="text",
            content="some text",
            metadata={"source": "test.pdf"},
            page_no=1,
        )
        assert el.element_id != ""
        assert el.element_id.startswith("doc-")

    def test_custom_id(self):
        el = ParsedElement(
            element_type="text",
            content="text",
            element_id="custom-id-001",
        )
        assert el.element_id == "custom-id-001"


class TestParsedPDF:
    def test_all_elements(self):
        parsed = ParsedPDF(source="test.pdf")
        parsed.texts = [ParsedElement("text", "t", page_no=0)]
        parsed.tables = [ParsedElement("table", "tbl", page_no=1)]
        parsed.images = [ParsedElement("image", "img", page_no=2)]
        assert len(parsed.all_elements) == 3


class TestMinerUParser:
    def _make_parser_with_mock(self, content_list):
        parser = MinerUParser.__new__(MinerUParser)
        parser.parse_method = "auto"
        parser.output_dir = None
        parser.lang = None
        parser.backend = "pipeline"
        parser.extract_images = True
        parser.extract_tables = True
        parser.name = "mineru/auto"
        object.__setattr__(parser, "_rag_logger", None)
        parser._content_list = content_list
        return parser

    def test_parse_content_list_texts(self):
        parser = MinerUParser(parse_method="auto")
        content_list = _make_content_list()
        result = parser._parse_content_list(content_list, "test.pdf", "/tmp", "test")
        assert len(result.texts) == 2
        assert all(el.element_type == "text" for el in result.texts)

    def test_parse_content_list_tables(self):
        parser = MinerUParser(parse_method="auto")
        content_list = _make_content_list()
        result = parser._parse_content_list(content_list, "test.pdf", "/tmp", "test")
        assert len(result.tables) == 1
        assert "Performance Table" in result.tables[0].content

    def test_parse_content_list_images(self):
        parser = MinerUParser(parse_method="auto")
        content_list = _make_content_list()
        result = parser._parse_content_list(content_list, "test.pdf", "/tmp", "test")
        assert len(result.images) == 1
        assert result.images[0].element_type == "image"

    def test_skip_empty_text(self):
        parser = MinerUParser(parse_method="auto")
        content_list = [{"type": "text", "text": "   ", "page_no": 0}]
        result = parser._parse_content_list(content_list, "test.pdf", "/tmp", "test")
        assert len(result.texts) == 0

    def test_extract_tables_false(self):
        parser = MinerUParser(parse_method="auto", extract_tables=False)
        content_list = _make_content_list()
        result = parser._parse_content_list(content_list, "test.pdf", "/tmp", "test")
        assert len(result.tables) == 0

    def test_extract_images_false(self):
        parser = MinerUParser(parse_method="auto", extract_images=False)
        content_list = _make_content_list()
        result = parser._parse_content_list(content_list, "test.pdf", "/tmp", "test")
        assert len(result.images) == 0

    def test_to_documents(self):
        parser = MinerUParser(parse_method="auto")
        content_list = _make_content_list()
        parsed = parser._parse_content_list(content_list, "test.pdf", "/tmp", "test")
        docs_map = parser.to_documents(parsed)

        assert "texts" in docs_map
        assert "tables" in docs_map
        assert "images" in docs_map
        assert all(isinstance(d, Document) for d in docs_map["texts"])
        assert all(isinstance(d, Document) for d in docs_map["tables"])

    def test_metadata_preserved(self):
        parser = MinerUParser(parse_method="auto")
        content_list = [{"type": "text", "text": "Hello", "page_no": 5, "text_level": "heading"}]
        result = parser._parse_content_list(content_list, "doc.pdf", "/tmp", "doc")
        assert result.texts[0].metadata["page_no"] == 5
        assert result.texts[0].metadata["source"] == "doc.pdf"

    def test_encode_image_missing_file(self):
        parser = MinerUParser()
        encoded = parser._encode_image("/nonexistent/path.png")
        assert encoded == ""

    def test_encode_image_real_file(self):
        parser = MinerUParser()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)
            tmp = f.name
        try:
            encoded = parser._encode_image(tmp)
            assert len(encoded) > 0
        finally:
            os.unlink(tmp)


class TestPDFIngestionPipeline:
    def _make_pipeline(self, chunker=None):
        vs = _DummyVectorStore()
        emb = _DummyEmbeddings()
        pipeline = PDFIngestionPipeline(
            vectorstore=vs,
            embeddings=emb,
            chunker=chunker,
        )
        return pipeline, vs, emb

    def test_default_chunker_is_recursive(self):
        from globalbot.backend.chunkings.fixed import RecursiveChunker
        pipeline, _, _ = self._make_pipeline()
        assert isinstance(pipeline.chunker, RecursiveChunker)

    def test_ingest_with_mock_parser(self):
        pipeline, vs, _ = self._make_pipeline()

        mock_parsed = ParsedPDF(source="test.pdf")
        mock_parsed.texts = [
            ParsedElement("text", "Machine learning intro " * 5, metadata={"source": "test.pdf"}, page_no=0),
            ParsedElement("text", "Deep learning concepts " * 5, metadata={"source": "test.pdf"}, page_no=1),
        ]
        mock_parsed.tables = [
            ParsedElement("table", "| A | B |\n|---|---|\n| 1 | 2 |", metadata={"source": "test.pdf"}, page_no=2),
        ]
        mock_parsed.images = []

        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_parsed
        mock_parser.to_documents.return_value = {
            "texts": [Document.from_text(el.content, el.metadata, doc_id=el.element_id) for el in mock_parsed.texts],
            "tables": [Document.from_text(el.content, el.metadata, doc_id=el.element_id) for el in mock_parsed.tables],
            "images": [],
        }
        pipeline.parser = mock_parser

        stats = pipeline.ingest("test.pdf")
        assert stats["parsed"]["texts"] == 2
        assert stats["parsed"]["tables"] == 1
        assert vs.count() > 0

    def test_ingest_texts_chunked(self):
        from globalbot.backend.chunkings.base import BaseChunker

        class _SplitChunker(BaseChunker):
            def split_text(self, text):
                return [text[i: i + 30] for i in range(0, len(text), 30) if text[i: i + 30].strip()]

        pipeline, vs, _ = self._make_pipeline(chunker=_SplitChunker(chunk_size=30))

        mock_parsed = ParsedPDF(source="test.pdf")
        mock_parsed.texts = [ParsedElement("text", "x" * 120, metadata={"source": "test.pdf"}, page_no=0)]
        mock_parsed.tables = []
        mock_parsed.images = []

        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_parsed
        mock_parser.to_documents.return_value = {
            "texts": [Document.from_text(el.content, el.metadata) for el in mock_parsed.texts],
            "tables": [],
            "images": [],
        }
        pipeline.parser = mock_parser

        stats = pipeline.ingest("test.pdf")
        assert stats["indexed"]["text_chunks"] > 1

    def test_embed_and_store(self):
        pipeline, vs, _ = self._make_pipeline()
        docs = [Document.from_text(f"doc {i}") for i in range(5)]
        count = pipeline._embed_and_store(docs, vs)
        assert count == 5
        assert vs.count() == 5


class TestRAGChatPipeline:
    def _make_pipeline(self, fixed_results=None):
        vs = _DummyVectorStore()
        if fixed_results:
            vs._fixed_results = fixed_results
        emb = _DummyEmbeddings()

        from globalbot.backend.storages.retrieval import RAGRetriever
        retriever = RAGRetriever(vectorstore=vs, embeddings=emb, top_k=3)
        llm = _DummyLLM()
        pipeline = RAGChatPipeline(retriever=retriever, llm=llm)
        return pipeline, vs

    def test_chat_returns_message(self):
        fixed_docs = [
            RetrievedDocument(page_content="ML is machine learning.", doc_id="d1", score=0.9),
        ]
        pipeline, _ = self._make_pipeline(fixed_results=fixed_docs)
        msg = pipeline.chat("What is ML?")
        assert isinstance(msg, ChatMessage)
        assert msg.role == "assistant"
        assert len(msg.content) > 0

    def test_chat_returns_sources(self):
        fixed_docs = [
            RetrievedDocument(page_content="context text", doc_id="d1", score=0.9),
        ]
        pipeline, _ = self._make_pipeline(fixed_results=fixed_docs)
        pipeline.return_sources = True
        msg = pipeline.chat("test question")
        assert len(msg.sources) > 0

    def test_history_accumulates(self):
        pipeline, _ = self._make_pipeline()
        pipeline.chat("First question")
        pipeline.chat("Second question")
        assert len(pipeline.history) == 4

    def test_reset_history(self):
        pipeline, _ = self._make_pipeline()
        pipeline.chat("Question")
        pipeline.reset_history()
        assert len(pipeline.history) == 0

    def test_max_history_truncated(self):
        pipeline, _ = self._make_pipeline()
        pipeline.max_history = 2
        for i in range(10):
            pipeline.chat(f"Question {i}")
        assert len(pipeline.history) == 20

    def test_build_context_with_docs(self):
        pipeline, _ = self._make_pipeline()
        docs = [
            RetrievedDocument(page_content="doc 1 content", doc_id="d1", score=0.9, metadata={"source": "a.pdf", "page_no": 1}),
            RetrievedDocument(page_content="doc 2 content", doc_id="d2", score=0.8, metadata={"source": "b.pdf"}),
        ]
        context = pipeline._build_context(docs)
        assert "doc 1 content" in context
        assert "doc 2 content" in context
        assert "a.pdf" in context
        assert "page: 1" in context

    def test_build_context_empty(self):
        pipeline, _ = self._make_pipeline()
        context = pipeline._build_context([])
        assert "No relevant context" in context

    def test_stream(self):
        fixed_docs = [RetrievedDocument(page_content="context", doc_id="d1", score=0.9)]
        pipeline, _ = self._make_pipeline(fixed_results=fixed_docs)
        chunks = list(pipeline.stream("test question"))
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_run_alias(self):
        pipeline, _ = self._make_pipeline()
        result = pipeline.run("test")
        assert isinstance(result, ChatMessage)

    def test_custom_system_prompt(self):
        vs = _DummyVectorStore()
        emb = _DummyEmbeddings()
        from globalbot.backend.storages.retrieval import RAGRetriever
        retriever = RAGRetriever(vectorstore=vs, embeddings=emb, top_k=3)
        llm = _DummyLLM()

        custom_system = "You are an expert in AI."
        pipeline = RAGChatPipeline(
            retriever=retriever,
            llm=llm,
            system_prompt=custom_system,
        )
        assert pipeline.system_prompt == custom_system
        msg = pipeline.chat("test")
        assert isinstance(msg, ChatMessage)