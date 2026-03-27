from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from globalbot.api.store import ChatSessionRecord, chat_store

router = APIRouter(prefix="/api/knowledge/chats", tags=["knowledge-chat"])


def _to_title(text: str) -> str:
    compact = " ".join(text.split())
    return compact[:60] + ("..." if len(compact) > 60 else "")


def _ask_knowledge(messages: list[dict[str, Any]], k: int = 5) -> dict:
    from globalbot.backend.llms.factory import _rag_instance

    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    query = (user_msgs[-1] if user_msgs else "").strip()
    if not query:
        raise HTTPException(400, "No user message found")

    rag = _rag_instance()
    result = rag.answer(query, k=k)
    return {
        "query": query,
        "answer": result["answer"],
        "retrieved_docs": result["retrieved_docs"],
        "flow": "knowledge db -> retrieval -> llm -> answer",
    }


@router.post("")
def create_chat(payload: dict):
    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "query is required")

    k = int(payload.get("k", 5))
    user_msg = {"role": "user", "content": query}
    answer = _ask_knowledge([user_msg], k=k)
    assistant_msg = {"role": "assistant", "content": answer["answer"], "retrieved_docs": answer["retrieved_docs"]}

    rec = ChatSessionRecord(
        chat_id=str(uuid.uuid4()),
        mode="knowledge",
        title=_to_title(query),
        messages=[user_msg, assistant_msg],
    )
    rec.save()

    return JSONResponse({"chat_id": rec.chat_id, **answer, "messages": rec.messages})


@router.get("")
def list_chats():
    return [r.to_dict() for r in chat_store.list_all(mode="knowledge")]


@router.get("/{chat_id}")
def get_chat(chat_id: str):
    rec = chat_store.get(chat_id)
    if rec is None or rec.mode != "knowledge":
        raise HTTPException(404, "Chat not found")
    return rec.to_dict()


@router.post("/{chat_id}/messages")
def add_chat_message(chat_id: str, payload: dict):
    rec = chat_store.get(chat_id)
    if rec is None or rec.mode != "knowledge":
        raise HTTPException(404, "Chat not found")

    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "query is required")

    k = int(payload.get("k", 5))
    user_msg = {"role": "user", "content": query}
    base_messages = [m for m in rec.messages if m.get("role") in ("user", "assistant")]
    answer = _ask_knowledge(base_messages + [user_msg], k=k)
    assistant_msg = {"role": "assistant", "content": answer["answer"], "retrieved_docs": answer["retrieved_docs"]}

    rec.messages.extend([user_msg, assistant_msg])
    rec.save()

    return JSONResponse({"chat_id": rec.chat_id, **answer, "messages": rec.messages})


@router.delete("/{chat_id}", status_code=204)
def delete_chat(chat_id: str):
    if not chat_store.delete(chat_id):
        raise HTTPException(404, "Chat not found")
