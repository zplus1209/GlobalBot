from __future__ import annotations

from typing import Any, ClassVar, List, Optional, Union

from pydantic import Field
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import (
    AIMessage as LCAIMessage,
    BaseMessage as LCBaseMessage,
    HumanMessage as LCHumanMessage,
    SystemMessage as LCSystemMessage,
)

from globalbot.backend.utils import make_doc_id


class TextMixin:
    _text_field: ClassVar[str] = "text"

    @property
    def text(self) -> str:
        field = self._text_field
        val = self.__dict__.get(field)
        if val is None:
            try:
                val = object.__getattribute__(self, field)
            except AttributeError:
                val = ""
        if isinstance(val, list):
            return " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in val
            )
        return str(val) if val is not None else ""

    def __str__(self) -> str:
        return self.text


class Document(TextMixin, LCDocument):
    _text_field: ClassVar[str] = "page_content"

    doc_id: str = Field(default="")
    score: Optional[float] = None
    embedding: Optional[List[float]] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def model_post_init(self, __context: Any) -> None:
        if not self.doc_id:
            text = self.__dict__.get("page_content", "") or ""
            source = self.metadata.get("source") if self.metadata else None
            object.__setattr__(self, "doc_id", make_doc_id(text, source=source))

    @classmethod
    def from_text(cls, text: str, metadata: Optional[dict] = None, **kwargs: Any) -> "Document":
        return cls(page_content=text, metadata=metadata or {}, **kwargs)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"Document(doc_id={self.doc_id!r}, text={preview!r})"


class RetrievedDocument(Document):
    score: float = 0.0

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"RetrievedDocument(score={self.score:.4f}, text={preview!r})"


class _MessageMixin(TextMixin):
    _text_field: ClassVar[str] = "content"
    message_id: str = Field(default="")

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"{self.__class__.__name__}(text={preview!r})"


class HumanMessage(_MessageMixin, LCHumanMessage):
    pass


class AIMessage(_MessageMixin, LCAIMessage):
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class SystemMessage(_MessageMixin, LCSystemMessage):
    pass


AnyMessage = Union[HumanMessage, AIMessage, SystemMessage, LCBaseMessage]


class ExtractorOutput(Document):
    data: Optional[Any] = None


def to_document(obj: Any) -> Document:
    if isinstance(obj, Document):
        return obj
    if isinstance(obj, LCDocument):
        return Document(page_content=obj.page_content, metadata=obj.metadata)
    if isinstance(obj, LCBaseMessage):
        text = obj.content if isinstance(obj.content, str) else str(obj.content)
        return Document(page_content=text, metadata={"role": obj.type})
    if isinstance(obj, str):
        return Document(page_content=obj)
    text = (
        getattr(obj, "text", None)
        or getattr(obj, "page_content", None)
        or getattr(obj, "content", None)
        or str(obj)
    )
    return Document(page_content=str(text))


def get_text(obj: Any) -> str:
    if hasattr(obj, "text"):
        return obj.text
    if hasattr(obj, "page_content"):
        return obj.page_content
    if hasattr(obj, "content"):
        c = obj.content
        return c if isinstance(c, str) else str(c)
    return str(obj)
