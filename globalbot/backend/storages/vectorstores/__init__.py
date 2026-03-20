from __future__ import annotations

from typing import Any

from globalbot.backend.storages.vectorstores.base import BaseVectorStore

VECTORSTORE_PROVIDERS = {"chroma", "qdrant", "milvus", "mongodb"}


def init_vectorstore(provider: str, **kwargs: Any) -> BaseVectorStore:
    provider = provider.lower()

    if provider == "chroma":
        from globalbot.backend.storages.vectorstores.chroma_impl import ChromaVectorStore
        return ChromaVectorStore(**kwargs)

    if provider == "qdrant":
        from globalbot.backend.storages.vectorstores.qdrant_impl import QdrantVectorStore
        return QdrantVectorStore(**kwargs)

    if provider == "milvus":
        from globalbot.backend.storages.vectorstores.milvus_impl import MilvusVectorStore
        return MilvusVectorStore(**kwargs)

    if provider == "mongodb":
        from globalbot.backend.storages.vectorstores.mongodb_impl import MongoDBVectorStore
        return MongoDBVectorStore(**kwargs)

    raise ValueError(f"Unknown vectorstore provider: {provider!r}. Available: {VECTORSTORE_PROVIDERS}")


__all__ = [
    "BaseVectorStore",
    "init_vectorstore",
    "VECTORSTORE_PROVIDERS",
]