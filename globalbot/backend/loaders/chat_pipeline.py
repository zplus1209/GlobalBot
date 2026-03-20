from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from globalbot.backend.base import BaseComponent, RetrievedDocument
from globalbot.backend.llms.chats.base import BaseChatLLM
from globalbot.backend.storages.retrieval import RAGRetriever


_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the question based ONLY on the provided context. "
    "If the context does not contain enough information, say so clearly. "
    "Do not make up information."
)

_DEFAULT_PROMPT = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class ChatMessage:
    def __init__(self, role: str, content: str, sources: Optional[List[RetrievedDocument]] = None):
        self.role = role
        self.content = content
        self.sources = sources or []

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    def __repr__(self) -> str:
        return f"ChatMessage(role={self.role!r}, content={self.content[:60]!r})"


class RAGChatPipeline(BaseComponent):
    retriever: RAGRetriever
    llm: BaseChatLLM
    system_prompt: str = _DEFAULT_SYSTEM
    user_prompt_template: str = _DEFAULT_PROMPT
    max_history: int = 10
    top_k: int = 5
    return_sources: bool = True
    history: List[Dict[str, str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _set_name(self) -> "RAGChatPipeline":
        if not self.name:
            self.name = f"rag_chat/{self.llm.name or type(self.llm).__name__}"
        return self

    def chat(self, question: str, **kwargs: Any) -> ChatMessage:
        self.log.info("rag_chat.query", question=question[:80])

        with self.timed("rag_chat.retrieve.start", "rag_chat.retrieve.end"):
            docs = self.retriever.run(question, **kwargs)

        context = self._build_context(docs)
        user_content = self.user_prompt_template.format(
            context=context,
            question=question,
        )

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-self.max_history:])
        messages.append({"role": "user", "content": user_content})

        with self.timed("rag_chat.llm.start", "rag_chat.llm.end"):
            response = self.llm.run(messages)

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response.text})

        return ChatMessage(
            role="assistant",
            content=response.text,
            sources=docs if self.return_sources else [],
        )

    def stream(self, question: str, **kwargs: Any):
        docs = self.retriever.run(question, **kwargs)
        context = self._build_context(docs)
        user_content = self.user_prompt_template.format(context=context, question=question)

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-self.max_history:])
        messages.append({"role": "user", "content": user_content})

        accumulated = ""
        for chunk in self.llm.stream(messages):
            accumulated = chunk.text
            yield chunk.text

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": accumulated})

    def reset_history(self) -> None:
        self.history.clear()

    def _build_context(self, docs: List[RetrievedDocument]) -> str:
        if not docs:
            return "No relevant context found."
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page_no", "")
            ref = f"[{i}] (source: {source}" + (f", page: {page}" if page else "") + ")"
            parts.append(f"{ref}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def run(self, question: str, **kwargs: Any) -> ChatMessage:
        return self.chat(question, **kwargs)