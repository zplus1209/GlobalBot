"""
Run:
    pytest tests/test_api.py -v

Requires server running:
    python serve.py --mode online --model_name gemini --model_version gemini-1.5-flash
"""
import io
import json
import time
import pytest
import requests

BASE = "http://localhost:5002"
TIMEOUT = 10


# ─── helpers ─────────────────────────────────────────────────────────────────

def _minimal_pdf_bytes() -> bytes:
    """1-page PDF with text 'Aviation fuel standard JIG'. No external deps."""
    return b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 54>>
stream
BT /F1 12 Tf 72 720 Td (Aviation fuel standard JIG) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000378 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
441
%%EOF"""


def _wait_for_ready(doc_id: str, timeout: int = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE}/api/documents/{doc_id}", timeout=TIMEOUT)
        data = r.json()
        if data["status"] in ("ready", "error"):
            return data
        time.sleep(3)
    raise TimeoutError(f"Document {doc_id} not ready after {timeout}s")


# ─── document CRUD ────────────────────────────────────────────────────────────

class TestDocumentsCRUD:

    def test_list_empty_or_existing(self):
        r = requests.get(f"{BASE}/api/documents", timeout=TIMEOUT)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_upload_pdf(self):
        pdf = _minimal_pdf_bytes()
        r = requests.post(
            f"{BASE}/api/documents",
            files={"file": ("test_doc.pdf", io.BytesIO(pdf), "application/pdf")},
            timeout=TIMEOUT,
        )
        assert r.status_code == 202
        data = r.json()
        assert "doc_id" in data
        assert data["status"] == "processing"
        assert data["filename"] == "test_doc.pdf"
        return data["doc_id"]

    def test_upload_unsupported_type(self):
        r = requests.post(
            f"{BASE}/api/documents",
            files={"file": ("data.csv", io.BytesIO(b"a,b\n1,2"), "text/csv")},
            timeout=TIMEOUT,
        )
        assert r.status_code == 400

    def test_get_document(self):
        doc_id = self.test_upload_pdf()
        r = requests.get(f"{BASE}/api/documents/{doc_id}", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert data["doc_id"] == doc_id
        assert data["filename"] == "test_doc.pdf"
        assert data["mime_type"] == "application/pdf"

    def test_get_nonexistent(self):
        r = requests.get(f"{BASE}/api/documents/nonexistent-id", timeout=TIMEOUT)
        assert r.status_code == 404

    def test_serve_file(self):
        doc_id = self.test_upload_pdf()
        r = requests.get(f"{BASE}/api/documents/{doc_id}/file", timeout=TIMEOUT)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/pdf")

    def test_delete_document(self):
        doc_id = self.test_upload_pdf()
        r = requests.delete(f"{BASE}/api/documents/{doc_id}", timeout=TIMEOUT)
        assert r.status_code == 204
        r2 = requests.get(f"{BASE}/api/documents/{doc_id}", timeout=TIMEOUT)
        assert r2.status_code == 404

    def test_delete_nonexistent(self):
        r = requests.delete(f"{BASE}/api/documents/nonexistent", timeout=TIMEOUT)
        assert r.status_code == 404

    def test_full_pipeline_ready(self):
        doc_id = self.test_upload_pdf()
        rec = _wait_for_ready(doc_id, timeout=120)
        assert rec["status"] == "ready", f"Pipeline error: {rec.get('error')}"
        assert rec["blocks_count"] >= 0
        assert rec["chunks_count"] >= 0

    def test_get_blocks_after_processing(self):
        doc_id = self.test_upload_pdf()
        _wait_for_ready(doc_id, timeout=120)
        r = requests.get(f"{BASE}/api/documents/{doc_id}/blocks", timeout=TIMEOUT)
        assert r.status_code == 200
        blocks = r.json()
        assert isinstance(blocks, list)
        for b in blocks:
            assert "bbox" in b
            assert "label" in b
            assert "page" in b


# ─── chat / document Q&A ─────────────────────────────────────────────────────

class TestDocumentChat:

    @pytest.fixture(scope="class")
    def ready_doc_id(self):
        pdf = _minimal_pdf_bytes()
        r = requests.post(
            f"{BASE}/api/documents",
            files={"file": ("aviation.pdf", io.BytesIO(pdf), "application/pdf")},
            timeout=10,
        )
        doc_id = r.json()["doc_id"]
        rec = _wait_for_ready(doc_id, timeout=120)
        assert rec["status"] == "ready"
        return doc_id

    def test_ask_document_basic(self, ready_doc_id):
        r = requests.post(
            f"{BASE}/api/chat/document",
            json={
                "doc_id": ready_doc_id,
                "messages": [{"role": "user", "content": "What is this document about?"}],
                "k": 3,
            },
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert "content" in data
        assert isinstance(data["content"], str)
        assert len(data["content"]) > 0
        assert "retrieved_docs" in data
        assert isinstance(data["retrieved_docs"], list)

    def test_retrieved_docs_have_bbox(self, ready_doc_id):
        r = requests.post(
            f"{BASE}/api/chat/document",
            json={
                "doc_id": ready_doc_id,
                "messages": [{"role": "user", "content": "What standards are mentioned?"}],
                "k": 5,
            },
            timeout=30,
        )
        data = r.json()
        for doc in data["retrieved_docs"]:
            assert "bbox" in doc
            assert "page" in doc
            assert "label" in doc
            assert "score" in doc
            assert "doc_id" in doc
            assert isinstance(doc["bbox"], list)
            assert 0.0 <= doc["score"] <= 1.0

    def test_ask_nonexistent_doc(self):
        r = requests.post(
            f"{BASE}/api/chat/document",
            json={"doc_id": "nonexistent", "messages": [{"role": "user", "content": "test"}]},
            timeout=TIMEOUT,
        )
        assert r.status_code == 404

    def test_ask_not_ready_doc(self):
        pdf = _minimal_pdf_bytes()
        r = requests.post(
            f"{BASE}/api/documents",
            files={"file": ("temp.pdf", io.BytesIO(pdf), "application/pdf")},
            timeout=TIMEOUT,
        )
        doc_id = r.json()["doc_id"]
        r2 = requests.post(
            f"{BASE}/api/chat/document",
            json={"doc_id": doc_id, "messages": [{"role": "user", "content": "test"}]},
            timeout=TIMEOUT,
        )
        assert r2.status_code == 400

    def test_ask_no_user_message(self, ready_doc_id):
        r = requests.post(
            f"{BASE}/api/chat/document",
            json={"doc_id": ready_doc_id, "messages": [{"role": "assistant", "content": "hi"}]},
            timeout=TIMEOUT,
        )
        assert r.status_code == 400


# ─── chat / knowledge base ───────────────────────────────────────────────────

class TestKnowledgeChat:

    def test_ask_knowledge_basic(self):
        r = requests.post(
            f"{BASE}/api/chat/knowledge",
            json={"messages": [{"role": "user", "content": "What information is available?"}], "k": 3},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert "content" in data
        assert "retrieved_docs" in data

    def test_multi_turn_knowledge(self):
        history = [
            {"role": "user", "content": "Tell me about fuel standards."},
            {"role": "assistant", "content": "I found information about fuel standards."},
            {"role": "user", "content": "Can you elaborate?"},
        ]
        r = requests.post(
            f"{BASE}/api/chat/knowledge",
            json={"messages": history, "k": 3},
            timeout=30,
        )
        assert r.status_code == 200
        assert len(r.json()["content"]) > 0

    def test_no_user_message(self):
        r = requests.post(
            f"{BASE}/api/chat/knowledge",
            json={"messages": [{"role": "assistant", "content": "hi"}]},
            timeout=TIMEOUT,
        )
        assert r.status_code == 400

    def test_retrieved_docs_structure(self):
        r = requests.post(
            f"{BASE}/api/chat/knowledge",
            json={"messages": [{"role": "user", "content": "any information"}], "k": 5},
            timeout=30,
        )
        data = r.json()
        for doc in data.get("retrieved_docs", []):
            assert set(doc.keys()) >= {"content", "score", "page", "bbox", "label", "doc_id"}
