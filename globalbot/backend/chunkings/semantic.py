from __future__ import annotations

import numpy as np
from typing import Any, List, Optional

from pydantic import model_validator

from globalbot.backend.chunkings.base import BaseChunker
from globalbot.backend.embeddings.base import BaseEmbeddings


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _split_into_sentences(text: str, min_length: int = 30) -> List[str]:
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= min_length]


class ClusterSemanticChunker(BaseChunker):
    embeddings: BaseEmbeddings
    max_chunk_size: int = 400
    min_chunk_size: int = 50
    sentence_split_min_length: int = 30

    @model_validator(mode="after")
    def _set_name(self) -> "ClusterSemanticChunker":
        if not self.name:
            self.name = f"cluster_semantic/{self.max_chunk_size}"
        return self

    def split_text(self, text: str) -> List[str]:
        sentences = _split_into_sentences(text, self.sentence_split_min_length)
        if not sentences:
            return [text] if text.strip() else []
        if len(sentences) == 1:
            return sentences

        self.log.debug("cluster_semantic.embed_sentences", n=len(sentences))
        sentence_embeddings = self.embeddings.run(sentences)

        chunks: List[str] = []
        current_sentences: List[str] = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_len = len(sentence)

            if current_len + sentence_len > self.max_chunk_size:
                if current_len >= self.min_chunk_size:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = [sentence]
                    current_len = sentence_len
                else:
                    current_sentences.append(sentence)
                    current_len += sentence_len
                continue

            current_sim = _cosine_similarity(
                sentence_embeddings[i - 1],
                sentence_embeddings[i],
            )

            if current_sim < 0.5 and current_len >= self.min_chunk_size:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_len = sentence_len
            else:
                current_sentences.append(sentence)
                current_len += sentence_len

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return [c for c in chunks if c.strip()]