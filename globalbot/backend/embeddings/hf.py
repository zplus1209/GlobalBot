from __future__ import annotations

from typing import Any, List, Optional

from pydantic import model_validator

from globalbot.backend.embeddings.base import BaseEmbeddings


class HuggingFaceEmbeddings(BaseEmbeddings):
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize: bool = True
    batch_size: int = 32

    _model: Any = None

    @model_validator(mode="after")
    def _init_model(self) -> "HuggingFaceEmbeddings":
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_id, device=self.device)
        if not self.name:
            self.name = f"hf/{self.model_id}"
        return self

    def _embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        self.log.debug("embeddings.hf.call", model=self.model_id, n=len(texts))
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _embed_query(self, text: str, **kwargs: Any) -> List[float]:
        self.log.debug("embeddings.hf.query", model=self.model_id)
        vector = self._model.encode(
            [text],
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vector[0].tolist()