"""
Run:
    pytest tests/test_api_v2.py -v

Requires server running on localhost:5002.
"""
from __future__ import annotations

import io
import time

import requests

BASE = "http://localhost:5002"
TIMEOUT = 10


def _minimal_pdf_bytes() -> bytes:
    return b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 56>>
stream
BT /F1 12 Tf 72 720 Td (GlobalBot CRUD API sample text) Tj ET
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
0000000380 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
443
%%EOF"""


def _wait_ready(doc_id: str, timeout: int = 120) -> dict:
    until = time.time() + timeout
    while time.time() < until:
        r = requests.get(f"{BASE}/api/pipeline/files/{doc_id}", timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if data["status"] in ("ready", "error"):
            return data
        time.sleep(2)
    raise TimeoutError(doc_id)


def test_pipeline_file_crud_and_ask():
    upload = requests.post(
        f"{BASE}/api/pipeline/files",
        files={"file": ("crud.pdf", io.BytesIO(_minimal_pdf_bytes()), "application/pdf")},
        timeout=TIMEOUT,
    )
    assert upload.status_code == 202
    doc_id = upload.json()["doc_id"]

    listing = requests.get(f"{BASE}/api/pipeline/files", timeout=TIMEOUT)
    assert listing.status_code == 200
    assert any(d["doc_id"] == doc_id for d in listing.json())

    ready = _wait_ready(doc_id)
    assert ready["status"] == "ready", ready.get("error")

    ask = requests.post(
        f"{BASE}/api/pipeline/files/{doc_id}/ask",
        json={"query": "Tài liệu nói gì?", "k": 3},
        timeout=30,
    )
    assert ask.status_code == 200
    data = ask.json()
    assert "answer" in data
    assert isinstance(data["retrieved_docs"], list)

    delete = requests.delete(f"{BASE}/api/pipeline/files/{doc_id}", timeout=TIMEOUT)
    assert delete.status_code == 204


def test_knowledge_chat_crud():
    create = requests.post(
        f"{BASE}/api/knowledge/chats",
        json={"query": "Cho biết dữ liệu hiện có", "k": 3},
        timeout=30,
    )
    assert create.status_code == 200
    chat_id = create.json()["chat_id"]

    get_one = requests.get(f"{BASE}/api/knowledge/chats/{chat_id}", timeout=TIMEOUT)
    assert get_one.status_code == 200

    add_msg = requests.post(
        f"{BASE}/api/knowledge/chats/{chat_id}/messages",
        json={"query": "Mở rộng câu trả lời", "k": 3},
        timeout=30,
    )
    assert add_msg.status_code == 200

    delete = requests.delete(f"{BASE}/api/knowledge/chats/{chat_id}", timeout=TIMEOUT)
    assert delete.status_code == 204
