from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from globalbot.api.store import DocumentRecord, UPLOAD_DIR, store

router = APIRouter(prefix="/api/pipeline/files", tags=["pipeline"])

ALLOWED_MIME = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


def _process_doc(rec: DocumentRecord) -> None:
    try:
        rec.status = "processing"
        rec.save()

        # BUG FIX: _rag_instance thuộc factory.py, không phải rag/core.py
        from globalbot.backend.model.ade import ADEAgent, load_document
        from globalbot.backend.llms.factory import _llm_instance, _rag_instance
        from globalbot.backend.rag.chunker import chunk_blocks
        import json

        pages = load_document(rec.file_path)
        agent = ADEAgent(llm=_llm_instance(), vlm=None, verbose=False)
        blocks = agent.process_pages(pages)
        for b in blocks:
            b["doc_id"] = rec.doc_id

        blocks_path = UPLOAD_DIR / f"{rec.doc_id}_blocks.json"
        blocks_path.write_text(
            json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        chunks = chunk_blocks(blocks, doc_id=rec.doc_id)
        rag = _rag_instance()
        rag.add_documents(chunks)

        rec.pages = len(pages)
        rec.blocks_count = len(blocks)
        rec.chunks_count = len(chunks)
        rec.status = "ready"
        rec.error = None
        rec.save()

    except Exception as exc:
        rec.status = "error"
        rec.error = str(exc)
        rec.save()


@router.post("", status_code=202)
async def create_file(bg: BackgroundTasks, file: UploadFile = File(...)):
    mime = file.content_type or ""
    if mime not in ALLOWED_MIME:
        raise HTTPException(400, f"Unsupported type: {mime}")

    doc_id = str(uuid.uuid4())
    ext = ALLOWED_MIME[mime]
    filename = Path(file.filename).name
    dest = UPLOAD_DIR / f"{doc_id}{ext}"
    dest.write_bytes(await file.read())

    rec = DocumentRecord(
        doc_id=doc_id,
        filename=filename,
        file_path=str(dest),
        mime_type=mime,
    )
    rec.save()
    bg.add_task(_process_doc, rec)
    return {"doc_id": doc_id, "status": rec.status, "filename": filename}


@router.get("")
def list_files():
    return [r.to_dict() for r in store.list_all()]


@router.get("/{doc_id}")
def get_file(doc_id: str):
    rec = store.get(doc_id)
    if rec is None:
        raise HTTPException(404, "Document not found")
    return rec.to_dict()


@router.get("/{doc_id}/blocks")
def get_file_blocks(doc_id: str):
    import json

    blocks_path = UPLOAD_DIR / f"{doc_id}_blocks.json"
    if not blocks_path.exists():
        raise HTTPException(404, "Blocks not ready")
    return json.loads(blocks_path.read_text(encoding="utf-8"))


@router.get("/{doc_id}/original")
def get_original_file(doc_id: str):
    rec = store.get(doc_id)
    if rec is None:
        raise HTTPException(404, "Document not found")
    path = Path(rec.file_path)
    if not path.exists():
        raise HTTPException(404, "File not found on disk")
    return FileResponse(
        path=str(path),
        media_type=rec.mime_type,
        filename=rec.filename,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.delete("/{doc_id}", status_code=204)
def delete_file(doc_id: str):
    if not store.delete(doc_id):
        raise HTTPException(404, "Document not found")


@router.post("/{doc_id}/ask")
def ask_file(doc_id: str, payload: dict):
    # BUG FIX: import đúng vị trí
    from globalbot.backend.llms.factory import _rag_instance

    rec = store.get(doc_id)
    if rec is None:
        raise HTTPException(404, "Document not found")
    if rec.status != "ready":
        raise HTTPException(400, f"Document not ready: {rec.status}")

    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "query is required")

    k = int(payload.get("k", 5))
    rag = _rag_instance()
    result = rag.answer(query, k=k, doc_id_filter=doc_id)

    return JSONResponse({
        "query": query,
        "doc_id": doc_id,
        "answer": result["answer"],
        "retrieved_docs": result["retrieved_docs"],
        "flow": "upload -> chunk -> rag -> retrieval -> llm -> answer",
    })