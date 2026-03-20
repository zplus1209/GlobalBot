from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import model_validator

from globalbot.backend.base import Document, RetrievedDocument
from globalbot.backend.storages.vectorstores.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    collection_name: str = "globalbot"
    persist_directory: Optional[str] = "./chroma_db"
    host: Optional[str] = None
    port: int = 8000

    _client: Any = None
    _collection: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "ChromaVectorStore":
        import chromadb

        if self.host:
            self._client = chromadb.HttpClient(host=self.host, port=self.port)
        else:
            self._client = chromadb.PersistentClient(path=self.persist_directory)

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if not self.name:
            self.name = f"chroma/{self.collection_name}"
        return self

    def add(self, docs: List[Document], embeddings: List[List[float]], **kwargs: Any) -> List[str]:
        self.log.debug("vectorstore.chroma.add", n=len(docs))
        ids = [doc.doc_id for doc in docs]
        metadatas = [doc.metadata or {} for doc in docs]
        documents = [doc.page_content for doc in docs]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        return ids

    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedDocument]:
        self.log.debug("vectorstore.chroma.query", top_k=top_k)
        query_kwargs: dict = dict(query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas", "distances"])
        if filters:
            query_kwargs["where"] = filters
        results = self._collection.query(**query_kwargs)

        retrieved = []
        for i, doc_id in enumerate(results["ids"][0]):
            score = 1.0 - results["distances"][0][i]
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            text = results["documents"][0][i] if results["documents"] else ""
            retrieved.append(RetrievedDocument(page_content=text, metadata=metadata, doc_id=doc_id, score=score))
        return retrieved

    def delete(self, ids: List[str], **kwargs: Any) -> None:
        self.log.debug("vectorstore.chroma.delete", n=len(ids))
        self._collection.delete(ids=ids)

    def drop(self) -> None:
        self.log.info("vectorstore.chroma.drop", collection=self.collection_name)
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()