"""
Unit tests — no running server needed.
    pytest tests/test_units.py -v
"""
import json
import uuid
import pytest


# ─── DocumentRecord / store ───────────────────────────────────────────────────

class TestDocumentStore:

    def test_save_and_load(self, tmp_path, monkeypatch):
        import api.store as store_mod
        monkeypatch.setattr(store_mod, "UPLOAD_DIR", tmp_path)
        monkeypatch.setattr(store_mod, "META_DIR", tmp_path / ".meta")
        (tmp_path / ".meta").mkdir()

        from api.store import DocumentRecord
        rec = DocumentRecord(
            doc_id="test-123",
            filename="sample.pdf",
            file_path="/tmp/test-123.pdf",
            mime_type="application/pdf",
            pages=5, blocks_count=12, chunks_count=24,
            status="ready",
        )
        rec.save()

        loaded = DocumentRecord.load("test-123")
        assert loaded is not None
        assert loaded.doc_id == "test-123"
        assert loaded.status == "ready"
        assert loaded.pages == 5

    def test_load_nonexistent(self, tmp_path, monkeypatch):
        import api.store as store_mod
        monkeypatch.setattr(store_mod, "META_DIR", tmp_path / ".meta")
        (tmp_path / ".meta").mkdir()
        from api.store import DocumentRecord
        assert DocumentRecord.load("nonexistent") is None

    def test_to_dict_has_required_fields(self):
        from api.store import DocumentRecord
        rec = DocumentRecord(
            doc_id="abc", filename="f.pdf", file_path="/f.pdf", mime_type="application/pdf"
        )
        d = rec.to_dict()
        for field in ["doc_id", "filename", "file_path", "mime_type", "status", "pages", "created_at"]:
            assert field in d


# ─── Chunker ─────────────────────────────────────────────────────────────────

class TestChunker:

    def _make_block(self, label, **kwargs):
        base = {
            "bbox": [10, 20, 200, 60], "page": 1,
            "label": label, "origin_label": label,
            "doc_id": "doc-001",
        }
        base.update(kwargs)
        return base

    def test_text_block_produces_document(self):
        from rag.chunker import chunk_blocks
        blocks = [self._make_block("text", content="Aviation fuel JIG standard Q3.")]
        docs = chunk_blocks(blocks, doc_id="doc-001")
        assert len(docs) == 1
        assert "Aviation fuel" in docs[0].page_content

    def test_empty_text_skipped(self):
        from rag.chunker import chunk_blocks
        blocks = [self._make_block("text", content="   ")]
        docs = chunk_blocks(blocks, doc_id="doc-001")
        assert len(docs) == 0

    def test_table_block(self):
        from rag.chunker import chunk_blocks
        blocks = [self._make_block(
            "table",
            content="raw ocr",
            table_title="Fuel specs",
            column_headers=["Parameter", "Value"],
            rows=[["Flash point", ">38°C"], ["Density", "775–840 kg/m³"]],
            notes="Per JIG standard",
        )]
        docs = chunk_blocks(blocks, doc_id="doc-001")
        assert len(docs) == 1
        text = docs[0].page_content
        assert "Flash point" in text
        assert "Fuel specs" in text

    def test_image_block(self):
        from rag.chunker import chunk_blocks
        blocks = [self._make_block(
            "image",
            content="",
            image_caption="Fueling manifold diagram",
            figure_type="Technical Diagram",
            key_elements=["3-inch flange", "ball valve"],
            purpose="Shows manifold layout",
        )]
        docs = chunk_blocks(blocks, doc_id="doc-001")
        assert len(docs) == 1
        assert "manifold" in docs[0].page_content.lower()

    def test_chart_block(self):
        from rag.chunker import chunk_blocks
        blocks = [self._make_block(
            "chart",
            content="",
            image_caption="Flow rate vs pressure",
            chart_data={"trend": "linear", "x_axis": {"label": "Pressure"}, "y_axis": {"label": "Flow rate"}},
        )]
        docs = chunk_blocks(blocks, doc_id="doc-001")
        assert len(docs) == 1

    def test_metadata_bbox_stored(self):
        from rag.chunker import chunk_blocks
        bbox = [50, 100, 400, 200]
        blocks = [self._make_block("text", content="Some text.", bbox=bbox)]
        docs = chunk_blocks(blocks, doc_id="doc-001")
        stored = json.loads(docs[0].metadata["bbox"])
        assert stored == bbox

    def test_metadata_page_stored(self):
        from rag.chunker import chunk_blocks
        blocks = [self._make_block("text", content="Page 3 content.", page=3)]
        docs = chunk_blocks(blocks, doc_id="doc-001")
        assert docs[0].metadata["page"] == 3

    def test_doc_id_in_metadata(self):
        from rag.chunker import chunk_blocks
        blocks = [self._make_block("text", content="text", doc_id="my-doc")]
        docs = chunk_blocks(blocks, doc_id="my-doc")
        assert docs[0].metadata["doc_id"] == "my-doc"

    def test_chunk_id_unique(self):
        from rag.chunker import chunk_blocks
        blocks = [
            self._make_block("text", content="First sentence."),
            self._make_block("text", content="Second sentence."),
        ]
        docs = chunk_blocks(blocks, doc_id="doc-001")
        ids = [d.metadata["chunk_id"] for d in docs]
        assert len(set(ids)) == 2

    def test_mixed_blocks(self):
        from rag.chunker import chunk_blocks
        blocks = [
            self._make_block("text", content="Intro text."),
            self._make_block("table", content="", table_title="T", column_headers=["A"], rows=[["1"]]),
            self._make_block("image", content="", image_caption="Figure 1"),
            self._make_block("text", content="   "),
        ]
        docs = chunk_blocks(blocks, doc_id="doc-001")
        assert len(docs) == 3


# ─── RetrievedChunk ───────────────────────────────────────────────────────────

class TestRetrievedChunk:

    def test_bbox_parsing(self):
        from langchain_core.documents import Document
        from rag.core import RetrievedChunk
        doc = Document(page_content="test", metadata={"bbox": "[10, 20, 200, 60]", "page": "2", "label": "table"})
        chunk = RetrievedChunk(doc, 0.85)
        assert chunk.bbox == [10, 20, 200, 60]
        assert chunk.page == 2
        assert chunk.label == "table"

    def test_to_dict_structure(self):
        from langchain_core.documents import Document
        from rag.core import RetrievedChunk
        doc = Document(
            page_content="fuel info",
            metadata={
                "bbox": "[0,0,100,50]", "page": "1", "label": "text",
                "origin_label": "text", "doc_id": "d1",
                "image_path": "", "chunk_id": "c1",
            }
        )
        chunk = RetrievedChunk(doc, 0.9)
        d = chunk.to_dict()
        for key in ["content", "score", "page", "bbox", "label", "origin_label", "doc_id", "image_path", "chunk_id"]:
            assert key in d
        assert d["score"] == 0.9
        assert d["content"] == "fuel info"

    def test_malformed_bbox_returns_empty(self):
        from langchain_core.documents import Document
        from rag.core import RetrievedChunk
        doc = Document(page_content="x", metadata={"bbox": "not-json", "page": "1", "label": "text"})
        chunk = RetrievedChunk(doc, 0.5)
        assert chunk.bbox == []
