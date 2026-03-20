from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import model_validator

from globalbot.backend.base import Document, RetrievedDocument
from globalbot.backend.storages.vectorstores.base import BaseVectorStore


class MongoDBVectorStore(BaseVectorStore):
    collection_name: str = "globalbot"
    connection_string: str = "mongodb://localhost:27017"
    database_name: str = "globalbot_db"
    index_name: str = "vector_index"
    vector_size: int = 1536
    vector_field: str = "embedding"
    text_field: str = "text"

    _client: Any = None
    _collection: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "MongoDBVectorStore":
        from pymongo import MongoClient

        self._client = MongoClient(self.connection_string)
        db = self._client[self.database_name]
        self._collection = db[self.collection_name]
        if not self.name:
            self.name = f"mongodb/{self.database_name}/{self.collection_name}"
        return self

    def add(self, docs: List[Document], embeddings: List[List[float]], **kwargs: Any) -> List[str]:
        self.log.debug("vectorstore.mongodb.add", n=len(docs))
        records = []
        for doc, emb in zip(docs, embeddings):
            record = {
                "_id": doc.doc_id,
                self.text_field: doc.page_content,
                self.vector_field: emb,
                **(doc.metadata or {}),
            }
            records.append(record)

        for record in records:
            self._collection.replace_one({"_id": record["_id"]}, record, upsert=True)
        return [doc.doc_id for doc in docs]

    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedDocument]:
        self.log.debug("vectorstore.mongodb.query", top_k=top_k)
        pipeline: list = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": self.vector_field,
                    "queryVector": embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    **({"filter": filters} if filters else {}),
                }
            },
            {
                "$project": {
                    "_id": 1,
                    self.text_field: 1,
                    "score": {"$meta": "vectorSearchScore"},
                    "metadata": "$$ROOT",
                }
            },
        ]
        results = list(self._collection.aggregate(pipeline))
        retrieved = []
        for r in results:
            meta = {k: v for k, v in r.get("metadata", {}).items() if k not in ("_id", self.text_field, self.vector_field)}
            retrieved.append(RetrievedDocument(
                page_content=r.get(self.text_field, ""),
                metadata=meta,
                doc_id=str(r["_id"]),
                score=r.get("score", 0.0),
            ))
        return retrieved

    def delete(self, ids: List[str], **kwargs: Any) -> None:
        self.log.debug("vectorstore.mongodb.delete", n=len(ids))
        self._collection.delete_many({"_id": {"$in": ids}})

    def drop(self) -> None:
        self.log.info("vectorstore.mongodb.drop", collection=self.collection_name)
        self._collection.drop()

    def count(self) -> int:
        return self._collection.count_documents({})