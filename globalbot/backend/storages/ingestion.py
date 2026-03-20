from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import model_validator

from globalbot.backend.base import BaseComponent, Document
from globalbot.backend.chunkings.base import BaseChunker
from globalbot.backend.embeddings.base import BaseEmbeddings
from globalbot.backend.storages.vectorstores.base import BaseVectorStore


class RAGIndexer(BaseComponent):
    vectorstore: BaseVectorStore
    embeddings: BaseEmbeddings
    chunker: Optional[BaseChunker] = None
    batch_size: int = 32

    @model_validator(mode="after")
    def _set_defaults(self) -> "RAGIndexer":
        if self.chunker is None:
            from globalbot.backend.chunkings.fixed import RecursiveChunker
            self.chunker = RecursiveChunker()
        if not self.name:
            self.name = f"indexer/{self.vectorstore.name}"
        return self

    def run(self, docs: List[Document], **kwargs: Any) -> Dict[str, Any]:
        self.log.info("indexer.start", n_docs=len(docs))

        chunks = self.chunker.run(docs)
        self.log.info("indexer.chunks", n_chunks=len(chunks))

        all_ids = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i: i + self.batch_size]
            texts = [c.page_content for c in batch]

            with self.timed("indexer.embed.start", "indexer.embed.end", batch=i // self.batch_size):
                embs = self.embeddings.run(texts)

            with self.timed("indexer.store.start", "indexer.store.end", batch=i // self.batch_size):
                ids = self.vectorstore.add(batch, embs)

            all_ids.extend(ids)
            self.log.info("indexer.batch_done", batch=i // self.batch_size + 1, stored=len(ids))

        self.log.info("indexer.done", total_stored=len(all_ids))
        return {"stored": len(all_ids), "ids": all_ids}