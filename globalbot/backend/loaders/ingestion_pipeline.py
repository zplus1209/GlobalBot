from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import model_validator

from globalbot.backend.base import BaseComponent, Document
from globalbot.backend.chunkings.base import BaseChunker
from globalbot.backend.embeddings.base import BaseEmbeddings
from globalbot.backend.loaders.mineru_loader import PyMuPDF4LLMParser, MinerUParser
from globalbot.backend.storages.vectorstores.base import BaseVectorStore


class PDFIngestionPipeline(BaseComponent):
    """
    PDF → parse → chunk → embed → store

    parser defaults to PyMuPDF4LLMParser (fast, no GPU needed).
    Pass parser=MinerUParser(...) for higher accuracy when MinerU is ready.
    """

    vectorstore: BaseVectorStore
    embeddings: BaseEmbeddings
    chunker: Optional[BaseChunker] = None
    parser: Optional[Any] = None          # PyMuPDF4LLMParser | MinerUParser
    batch_size: int = 32
    index_tables: bool = True
    index_images: bool = False
    table_collection: Optional[str] = None
    image_collection: Optional[str] = None

    _table_store: Optional[Any] = None
    _image_store: Optional[Any] = None

    @model_validator(mode="after")
    def _init_defaults(self) -> "PDFIngestionPipeline":
        if self.chunker is None:
            from globalbot.backend.chunkings.fixed import RecursiveChunker
            self.chunker = RecursiveChunker()
        if self.parser is None:
            self.parser = PyMuPDF4LLMParser()   # ← default
        if not self.name:
            self.name = f"pdf_pipeline/{self.vectorstore.name}"
        return self

    # ── public API ────────────────────────────────────────────────────────────

    def ingest(self, pdf_path: str) -> Dict[str, Any]:
        self.log.info("pdf_pipeline.ingest.start", path=pdf_path)

        parsed = self.parser.parse(pdf_path)
        docs_map = self.parser.to_documents(parsed)

        stats: Dict[str, Any] = {
            "source": pdf_path,
            "parsed": {
                "texts": len(docs_map["texts"]),
                "tables": len(docs_map["tables"]),
                "images": len(docs_map["images"]),
            },
            "indexed": {},
        }

        if docs_map["texts"]:
            chunks = self.chunker.run(docs_map["texts"])
            stats["indexed"]["text_chunks"] = self._embed_and_store(chunks, self.vectorstore)

        if self.index_tables and docs_map["tables"]:
            stats["indexed"]["tables"] = self._embed_and_store(
                docs_map["tables"], self._get_store("table")
            )

        if self.index_images and docs_map["images"]:
            stats["indexed"]["images"] = self._embed_and_store(
                docs_map["images"], self._get_store("image")
            )

        self.log.info("pdf_pipeline.ingest.done", **stats["indexed"])
        return stats

    def ingest_directory(self, dir_path: str, glob: str = "**/*.pdf") -> Dict[str, Any]:
        pdfs = list(Path(dir_path).glob(glob))
        self.log.info("pdf_pipeline.ingest_dir", n_files=len(pdfs))
        all_stats: Dict[str, Any] = {"total": len(pdfs), "files": [], "errors": []}
        for pdf in pdfs:
            try:
                all_stats["files"].append(self.ingest(str(pdf)))
            except Exception as e:
                self.log.error("pdf_pipeline.ingest.error", path=str(pdf), error=str(e)[:120])
                all_stats["errors"].append({"path": str(pdf), "error": str(e)})
        return all_stats

    def run(self, pdf_path: str, **kwargs: Any) -> Dict[str, Any]:
        return self.ingest(pdf_path)

    # ── internals ─────────────────────────────────────────────────────────────

    def _embed_and_store(self, docs: List[Document], store: BaseVectorStore) -> int:
        all_ids = []
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i: i + self.batch_size]
            embs = self.embeddings.run([d.page_content for d in batch])
            all_ids.extend(store.add(batch, embs))
        return len(all_ids)

    def _get_store(self, kind: str) -> BaseVectorStore:
        attr = f"_{kind}_store"
        if getattr(self, attr, None) is not None:
            return getattr(self, attr)

        collection = getattr(self, f"{kind}_collection", None)
        if collection:
            try:
                store = self.vectorstore.__class__(
                    **{
                        **self.vectorstore.model_dump(exclude={"name", "collection_name"}),
                        "collection_name": collection,
                    }
                )
            except Exception:
                store = self.vectorstore
        else:
            store = self.vectorstore

        object.__setattr__(self, attr, store)
        return store