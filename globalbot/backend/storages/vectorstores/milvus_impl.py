from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import model_validator

from globalbot.backend.base import Document, RetrievedDocument
from globalbot.backend.storages.vectorstores.base import BaseVectorStore


class MilvusVectorStore(BaseVectorStore):
    collection_name: str = "globalbot"
    uri: str = "http://localhost:19530"
    token: Optional[str] = None
    vector_size: int = 1536
    metric_type: str = "COSINE"
    index_type: str = "HNSW"
    nlist: int = 128

    _client: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "MilvusVectorStore":
        from pymilvus import MilvusClient, DataType

        connect_kwargs: dict = dict(uri=self.uri)
        if self.token:
            connect_kwargs["token"] = self.token
        self._client = MilvusClient(**connect_kwargs)

        if not self._client.has_collection(self.collection_name):
            schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("doc_id", DataType.VARCHAR, max_length=128, is_primary=True)
            schema.add_field("text", DataType.VARCHAR, max_length=65535)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.vector_size)

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type=self.index_type,
                metric_type=self.metric_type,
                params={"M": 16, "efConstruction": 64},
            )
            self._client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
        if not self.name:
            self.name = f"milvus/{self.collection_name}"
        return self

    def add(self, docs: List[Document], embeddings: List[List[float]], **kwargs: Any) -> List[str]:
        self.log.debug("vectorstore.milvus.add", n=len(docs))
        data = []
        for doc, emb in zip(docs, embeddings):
            row: dict = {"doc_id": doc.doc_id, "text": doc.page_content, "embedding": emb}
            row.update(doc.metadata or {})
            data.append(row)
        self._client.upsert(collection_name=self.collection_name, data=data)
        return [doc.doc_id for doc in docs]

    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedDocument]:
        self.log.debug("vectorstore.milvus.query", top_k=top_k)
        expr = " && ".join(f'{k} == "{v}"' for k, v in filters.items()) if filters else ""
        results = self._client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=top_k,
            filter=expr or None,
            output_fields=["doc_id", "text"],
            search_params={"metric_type": self.metric_type, "params": {"ef": 64}},
        )
        retrieved = []
        for r in results[0]:
            entity = r["entity"]
            score = 1.0 - r["distance"] if self.metric_type == "COSINE" else r["distance"]
            metadata = {k: v for k, v in entity.items() if k not in ("doc_id", "text", "embedding")}
            retrieved.append(RetrievedDocument(
                page_content=entity.get("text", ""),
                metadata=metadata,
                doc_id=entity.get("doc_id", ""),
                score=score,
            ))
        return retrieved

    def delete(self, ids: List[str], **kwargs: Any) -> None:
        self.log.debug("vectorstore.milvus.delete", n=len(ids))
        expr = f'doc_id in {ids}'
        self._client.delete(collection_name=self.collection_name, filter=expr)

    def drop(self) -> None:
        self.log.info("vectorstore.milvus.drop", collection=self.collection_name)
        self._client.drop_collection(self.collection_name)

    def count(self) -> int:
        stats = self._client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))