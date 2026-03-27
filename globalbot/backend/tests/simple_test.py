"""
Simple test script to demonstrate the two flows:
1. Upload file -> Process -> Chunk -> RAG -> retrieval -> LLM -> Answer
2. Chat: Knowledge DB -> retrieval -> LLM -> Answer

Run:
    python simple_test.py

Requires server running:
    python serve.py --mode online --model_name gemini --model_version gemini-1.5-flash
"""
import io
import json
import time
import requests

BASE = "http://localhost:5002"
TIMEOUT = 10

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

def wait_for_ready(doc_id: str, timeout: int = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE}/api/documents/{doc_id}", timeout=TIMEOUT)
        data = r.json()
        if data["status"] in ("ready", "error"):
            return data
        time.sleep(3)
    raise TimeoutError(f"Document {doc_id} not ready after {timeout}s")

def test_upload_and_process():
    print("=== Testing Upload and Processing ===")
    pdf = _minimal_pdf_bytes()
    r = requests.post(
        f"{BASE}/api/documents",
        files={"file": ("test_doc.pdf", io.BytesIO(pdf), "application/pdf")},
        timeout=TIMEOUT,
    )
    assert r.status_code == 202, f"Upload failed: {r.text}"
    data = r.json()
    doc_id = data["doc_id"]
    print(f"Uploaded document: {doc_id}, status: {data['status']}")

    # Wait for processing
    rec = wait_for_ready(doc_id, timeout=120)
    assert rec["status"] == "ready", f"Pipeline error: {rec.get('error')}"
    print(f"Document ready: blocks={rec['blocks_count']}, chunks={rec['chunks_count']}")
    return doc_id

def test_document_chat(doc_id: str):
    print("\n=== Testing Document Chat (Flow 1) ===")
    r = requests.post(
        f"{BASE}/api/chat/document",
        json={
            "doc_id": doc_id,
            "messages": [{"role": "user", "content": "What is this document about?"}],
            "k": 3,
        },
        timeout=30,
    )
    assert r.status_code == 200, f"Document chat failed: {r.text}"
    data = r.json()
    print(f"Answer: {data['content'][:200]}...")
    print(f"Retrieved {len(data['retrieved_docs'])} documents")
    for i, doc in enumerate(data["retrieved_docs"][:2]):
        print(f"  Doc {i}: page={doc['page']}, label={doc['label']}, score={doc['score']:.2f}")
    assert "content" in data and isinstance(data["content"], str) and len(data["content"]) > 0
    assert "retrieved_docs" in data and isinstance(data["retrieved_docs"], list)
    return data

def test_knowledge_chat():
    print("\n=== Testing Knowledge Base Chat (Flow 2) ===")
    r = requests.post(
        f"{BASE}/api/chat/knowledge",
        json={"messages": [{"role": "user", "content": "What information is available?"}], "k": 3},
        timeout=30,
    )
    assert r.status_code == 200, f"Knowledge chat failed: {r.text}"
    data = r.json()
    print(f"Answer: {data['content'][:200]}...")
    print(f"Retrieved {len(data['retrieved_docs'])} documents")
    for i, doc in enumerate(data["retrieved_docs"][:2]):
        print(f"  Doc {i}: page={doc.get('page', 'N/A')}, label={doc['label']}, score={doc['score']:.2f}")
    assert "content" in data and isinstance(data["content"], str) and len(data["content"]) > 0
    assert "retrieved_docs" in data and isinstance(data["retrieved_docs"], list)
    return data

def test_crud():
    print("\n=== Testing CRUD Operations ===")
    # List
    r = requests.get(f"{BASE}/api/documents", timeout=TIMEOUT)
    assert r.status_code == 200
    docs = r.json()
    print(f"Current documents: {len(docs)}")

    # Upload a new doc
    pdf = _minimal_pdf_bytes()
    r = requests.post(
        f"{BASE}/api/documents",
        files={"file": ("test_doc2.pdf", io.BytesIO(pdf), "application/pdf")},
        timeout=TIMEOUT,
    )
    assert r.status_code == 202
    doc_id = r.json()["doc_id"]
    print(f"Uploaded second document: {doc_id}")

    # Get
    r = requests.get(f"{BASE}/api/documents/{doc_id}", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert data["doc_id"] == doc_id
    print(f"Retrieved document: {data['filename']}")

    # Delete
    r = requests.delete(f"{BASE}/api/documents/{doc_id}", timeout=TIMEOUT)
    assert r.status_code == 204
    print(f"Deleted document: {doc_id}")

    # Verify deletion
    r = requests.get(f"{BASE}/api/documents/{doc_id}", timeout=TIMEOUT)
    assert r.status_code == 404
    print("CRUD test passed")

if __name__ == "__main__":
    try:
        test_crud()
        doc_id = test_upload_and_process()
        test_document_chat(doc_id)
        test_knowledge_chat()
        print("\n=== All tests passed! ===")
    except Exception as e:
        print(f"\n=== Test failed: {e} ===")
        raise