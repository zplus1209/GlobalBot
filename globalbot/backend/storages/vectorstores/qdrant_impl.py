from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import model_validator

from globalbot.backend.base import Document, RetrievedDocument
from globalbot.backend.storages.vectorstores.base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    collection_name: str = "globalbot"
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    path: Optional[str] = None
    vector_size: int = 1536
    distance: str = "Cosine"

    _client: Any = None

    @model_validator(mode="after")
    def _init_client(self) -> "QdrantVectorStore":
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if self.path:
            self._client = QdrantClient(path=self.path)
        elif self.url:
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self._client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key)

        distance_map = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}
        dist = distance_map.get(self.distance, Distance.COSINE)

        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=dist),
            )
        if not self.name:
            self.name = f"qdrant/{self.collection_name}"
        return self

    def add(self, docs: List[Document], embeddings: List[List[float]], **kwargs: Any) -> List[str]:
        from qdrant_client.models import PointStruct

        self.log.debug("vectorstore.qdrant.add", n=len(docs))
        points = []
        for doc, emb in zip(docs, embeddings):
            payload = {"text": doc.page_content, **(doc.metadata or {})}
            points.append(PointStruct(id=doc.doc_id, vector=emb, payload=payload))

        self._client.upsert(collection_name=self.collection_name, points=points)
        return [doc.doc_id for doc in docs]

    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedDocument]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self.log.debug("vectorstore.qdrant.query", top_k=top_k)
        qdrant_filter = None
        if filters:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            qdrant_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        retrieved = []
        for r in results:
            payload = r.payload or {}
            text = payload.pop("text", "")
            retrieved.append(RetrievedDocument(
                page_content=text,
                metadata=payload,
                doc_id=str(r.id),
                score=r.score,
            ))
        return retrieved

    def delete(self, ids: List[str], **kwargs: Any) -> None:
        from qdrant_client.models import PointIdsList

        self.log.debug("vectorstore.qdrant.delete", n=len(ids))
        self._client.delete(collection_name=self.collection_name, points_selector=PointIdsList(points=ids))

    def drop(self) -> None:
        self.log.info("vectorstore.qdrant.drop", collection=self.collection_name)
        self._client.delete_collection(self.collection_name)
        from qdrant_client.models import Distance, VectorParams
        distance_map = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=distance_map.get(self.distance, Distance.COSINE)),
        )

    def count(self) -> int:
        return self._client.get_collection(self.collection_name).points_count