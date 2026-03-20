from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from globalbot.backend.base import BaseComponent, Document
from globalbot.backend.embeddings.base import BaseEmbeddings
from globalbot.backend.storages.vectorstores.base import BaseVectorStore


class TextSplitter(BaseComponent):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n"

    def run(self, docs: List[Document], **kwargs: Any) -> List[Document]:
        chunks = []
        for doc in docs:
            text = doc.page_content
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                if chunk_text.strip():
                    from globalbot.backend.utils.hashing import make_chunk_id
                    chunk_id = make_chunk_id(chunk_text, parent_doc_id=doc.doc_id, chunk_index=len(chunks))
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata={**doc.metadata, "parent_doc_id": doc.doc_id},
                        doc_id=chunk_id,
                    ))
                start += self.chunk_size - self.chunk_overlap
        self.log.debug("splitter.run", input_docs=len(docs), output_chunks=len(chunks))
        return chunks


class RAGIndexer(BaseComponent):
    vectorstore: BaseVectorStore
    embeddings: BaseEmbeddings
    splitter: Optional[TextSplitter] = None
    batch_size: int = 32

    def model_post_init(self, __context: Any) -> None:
        if self.splitter is None:
            self.splitter = TextSplitter()
        if not self.name:
            self.name = f"indexer/{self.vectorstore.name}"

    def run(self, docs: List[Document], **kwargs: Any) -> Dict[str, Any]:
        self.log.info("indexer.start", n_docs=len(docs))

        chunks = self.splitter.run(docs)
        self.log.info("indexer.chunks", n_chunks=len(chunks))

        all_ids = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i: i + self.batch_size]
            texts = [c.page_content for c in batch]

            with self.timed("indexer.embed.start", "indexer.embed.end", batch=i // self.batch_size):
                embeddings = self.embeddings.run(texts)

            with self.timed("indexer.store.start", "indexer.store.end", batch=i // self.batch_size):
                ids = self.vectorstore.add(batch, embeddings)

            all_ids.extend(ids)
            self.log.info("indexer.batch_done", batch=i // self.batch_size + 1, stored=len(ids))

        self.log.info("indexer.done", total_stored=len(all_ids))
        return {"stored": len(all_ids), "ids": all_ids}