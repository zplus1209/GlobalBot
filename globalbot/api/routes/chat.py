from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/chat", tags=["chat"])


class Message(BaseModel):
    role: str
    content: str


class DocAskRequest(BaseModel):
    doc_id: str
    messages: List[Message]
    k: int = 5


class KnowledgeAskRequest(BaseModel):
    messages: List[Message]
    k: int = 5
    doc_ids: Optional[List[str]] = None


def _last_user_msg(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


@router.post("/document")
def ask_document(req: DocAskRequest):
    from api.store import store
    from llms.factory import _rag_instance

    rec = store.get(req.doc_id)
    if rec is None:
        raise HTTPException(404, "Document not found")
    if rec.status != "ready":
        raise HTTPException(400, f"Document not ready: {rec.status}")

    query = _last_user_msg(req.messages)
    if not query:
        raise HTTPException(400, "No user message found")

    rag = _rag_instance()
    result = rag.answer(query, k=req.k, doc_id_filter=req.doc_id)

    return JSONResponse({
        "role": "assistant",
        "content": result["answer"],
        "retrieved_docs": result["retrieved_docs"],
        "doc_id": req.doc_id,
    })


@router.post("/knowledge")
def ask_knowledge(req: KnowledgeAskRequest):
    from llms.factory import _llm_instance, _rag_instance

    query = _last_user_msg(req.messages)
    if not query:
        raise HTTPException(400, "No user message found")

    rag = _rag_instance()
    result = rag.answer(query, k=req.k)

    return JSONResponse({
        "role": "assistant",
        "content": result["answer"],
        "retrieved_docs": result["retrieved_docs"],
    })
