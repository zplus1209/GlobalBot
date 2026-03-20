from __future__ import annotations

from typing import Any, List, Optional

from pydantic import model_validator

from globalbot.backend.embeddings.base import BaseEmbeddings


class FastEmbedEmbeddings(BaseEmbeddings):
    model: str = "BAAI/bge-small-en-v1.5"
    max_length: int = 512
    batch_size: int = 256
    cache_dir: Optional[str] = None

    _model: Any = None

    @model_validator(mode="after")
    def _init_model(self) -> "FastEmbedEmbeddings":
        from fastembed import TextEmbedding
        kwargs: dict = dict(model_name=self.model, max_length=self.max_length)
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self._model = TextEmbedding(**kwargs)
        if not self.name:
            self.name = f"fastembed/{self.model}"
        return self

    def _embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        self.log.debug("embeddings.fastembed.call", model=self.model, n=len(texts))
        embeddings = list(self._model.embed(texts, batch_size=self.batch_size))
        return [e.tolist() for e in embeddings]

    def _embed_query(self, text: str, **kwargs: Any) -> List[float]:
        self.log.debug("embeddings.fastembed.query", model=self.model)
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()