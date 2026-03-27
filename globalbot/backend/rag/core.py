from __future__ import annotations

import json
from typing import Any, Literal, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# flat import — chạy được khi backend dir trong sys.path
from embeddings import SentenceTransformerEmbedding, EmbeddingConfig


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise document analysis assistant. "
        "Answer using ONLY the provided context. "
        "Cite page number and region type when referencing information. "
        "If the answer is not in the context, say so clearly.",
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


class RetrievedChunk:
    def __init__(self, document: Document, score: float) -> None:
        self.document = document
        self.score = score

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict:
        return self.document.metadata

    @property
    def bbox(self) -> list[int]:
        raw = self.metadata.get("bbox", "[]")
        try:
            return json.loads(raw) if isinstance(raw, str) else list(raw)
        except Exception:
            return []

    @property
    def page(self) -> int:
        try:
            return int(self.metadata.get("page", 0))
        except (ValueError, TypeError):
            return 0

    @property
    def label(self) -> str:
        return self.metadata.get("label", "text")

    def to_dict(self) -> dict:
        return {
            "content":      self.content,
            "score":        round(self.score, 4),
            "page":         self.page,
            "bbox":         self.bbox,
            "label":        self.label,
            "origin_label": self.metadata.get("origin_label", ""),
            "doc_id":       self.metadata.get("doc_id", ""),
            "image_path":   self.metadata.get("image_path", ""),
            "chunk_id":     self.metadata.get("chunk_id", ""),
        }


class RAG:
    def __init__(
        self,
        llm: Any,
        db_type: Literal["chromadb", "qdrant", "mongodb"] = "chromadb",
        embedding_name: str = "Alibaba-NLP/gte-multilingual-base",
        chromadb_path: str = "./chroma_db",
        collection_name: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        mongodb_uri: Optional[str] = None,
        mongodb_db: Optional[str] = None,
        mongodb_collection: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.db_type = db_type
        self.collection_name = collection_name or embedding_name.split("/")[-1]
        self._embedding = SentenceTransformerEmbedding(EmbeddingConfig(name=embedding_name))
        self._chain = _PROMPT | self.llm | StrOutputParser()

        if db_type == "chromadb":
            import chromadb
            client = chromadb.PersistentClient(path=chromadb_path)
            self._col = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        elif db_type == "qdrant":
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        elif db_type == "mongodb":
            import pymongo
            c = pymongo.MongoClient(mongodb_uri)
            self._mcol = c[mongodb_db][mongodb_collection]

        else:
            raise ValueError(f"Unsupported db_type: {db_type!r}")

    # ── embedding ──────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Trả về vector dưới dạng list[float] (tương thích ChromaDB)."""
        vec = self._embedding.encode(text)
        # SentenceTransformer trả về numpy array — chuyển sang list
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return list(vec)

    # ── write ──────────────────────────────────────────────────────────────

    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return

        if self.db_type == "chromadb":
            self._col.add(
                ids=[d.metadata["chunk_id"] for d in docs],
                documents=[d.page_content for d in docs],
                embeddings=[self._embed(d.page_content) for d in docs],
                metadatas=[d.metadata for d in docs],
            )
        # TODO: Qdrant / MongoDB không implement trong bản gốc

    # ── read ───────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 5,
        doc_id_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        vec = self._embed(query)

        if self.db_type == "chromadb":
            where = {"doc_id": doc_id_filter} if doc_id_filter else None

            # BUG FIX: n_results không được vượt quá số document trong collection
            count = self._col.count()
            n_results = min(k, count) if count > 0 else 0
            if n_results == 0:
                return []

            res = self._col.query(
                query_embeddings=[vec],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            return [
                RetrievedChunk(
                    Document(
                        page_content=res["documents"][0][i],
                        metadata=res["metadatas"][0][i],
                    ),
                    # ChromaDB trả về cosine distance (0=giống, 2=khác nhau nhất)
                    # Chuyển sang score 0-1: score = 1 - distance/2
                    max(0.0, 1.0 - res["distances"][0][i] / 2),
                )
                for i in range(len(res["ids"][0]))
            ]

        return []

    def answer(
        self,
        query: str,
        k: int = 5,
        doc_id_filter: Optional[str] = None,
    ) -> dict:
        chunks = self.retrieve(query, k=k, doc_id_filter=doc_id_filter)
        if not chunks:
            return {"answer": "No relevant information found.", "retrieved_docs": []}

        context = "\n\n".join(
            f"[Page {c.page} | {c.label}] {c.content}" for c in chunks
        )
        answer = self._chain.invoke({"context": context, "question": query})
        return {
            "answer":         answer,
            "retrieved_docs": [c.to_dict() for c in chunks],
        }