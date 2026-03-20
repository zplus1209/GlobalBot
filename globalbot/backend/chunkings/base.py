from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Optional

from globalbot.backend.base import BaseComponent, Document
from globalbot.backend.utils.hashing import make_chunk_id


class BaseChunker(BaseComponent):
    chunk_size: int = 500
    chunk_overlap: int = 100

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError

    def split_documents(self, docs: List[Document]) -> List[Document]:
        chunks = []
        for doc in docs:
            texts = self.split_text(doc.page_content)
            for idx, text in enumerate(texts):
                if not text.strip():
                    continue
                chunk_id = make_chunk_id(text, parent_doc_id=doc.doc_id, chunk_index=idx)
                chunks.append(Document(
                    page_content=text,
                    metadata={**doc.metadata, "parent_doc_id": doc.doc_id, "chunk_index": idx},
                    doc_id=chunk_id,
                ))
        self.log.debug("chunker.split", chunker=type(self).__name__, input=len(docs), output=len(chunks))
        return chunks

    def run(self, docs: List[Document], **kwargs: Any) -> List[Document]:
        return self.split_documents(docs)