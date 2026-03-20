from __future__ import annotations

import re
from typing import Any, List, Optional

from pydantic import model_validator

from globalbot.backend.chunkings.base import BaseChunker
from globalbot.backend.llms.chats.base import BaseChatLLM

_SYSTEM_PROMPT = (
    "You are an expert at splitting text into semantically coherent chunks. "
    "When given text divided into numbered pieces, identify where the best split points are."
)

_USER_PROMPT_TEMPLATE = """\
The text has been divided into pieces, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number.
Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together.
Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2.
THE CHUNKS MUST BE IN ASCENDING ORDER.
Your response should be in the form: 'split_after: 3, 5'

{chunks_text}
"""


def _parse_split_ids(response: str, n_chunks: int, current_chunk: int) -> List[int]:
    match = re.search(r"split_after:\s*([\d,\s]+)", response)
    if not match:
        return []
    numbers = [int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()]
    numbers = sorted(set(n for n in numbers if current_chunk <= n < n_chunks))
    return numbers


class LLMChunker(BaseChunker):
    llm: BaseChatLLM
    initial_chunk_size: int = 50
    max_iterations: int = 3

    @model_validator(mode="after")
    def _set_name(self) -> "LLMChunker":
        if not self.name:
            self.name = f"llm/{type(self.llm).__name__}"
        return self

    def _make_small_chunks(self, text: str) -> List[str]:
        words = text.split()
        pieces = []
        for i in range(0, len(words), self.initial_chunk_size):
            piece = " ".join(words[i: i + self.initial_chunk_size])
            if piece.strip():
                pieces.append(piece)
        return pieces

    def _build_tagged_text(self, pieces: List[str], start_idx: int) -> str:
        parts = []
        for i, piece in enumerate(pieces):
            idx = start_idx + i + 1
            parts.append(f"<|start_chunk_{idx}|>{piece}<|end_chunk_{idx}|>")
        return "\n".join(parts)

    def _query_llm(self, tagged_text: str, current_chunk: int, n_chunks: int, invalid_response: Optional[str] = None) -> List[int]:
        prompt = _USER_PROMPT_TEMPLATE.format(chunks_text=tagged_text)
        if invalid_response:
            prompt += (
                f"\nThe previous response of '{invalid_response}' was invalid. "
                f"Splits must be in ascending order and >= {current_chunk}."
            )

        self.log.debug("llm_chunker.query", n_chunks=n_chunks)
        response = self.llm.chat(user=prompt, system=_SYSTEM_PROMPT)
        return _parse_split_ids(response, n_chunks, current_chunk)

    def split_text(self, text: str) -> List[str]:
        pieces = self._make_small_chunks(text)
        if not pieces:
            return []
        if len(pieces) == 1:
            return pieces

        n_chunks = len(pieces)
        split_ids: List[int] = []
        current_chunk = 1
        invalid_response: Optional[str] = None

        for _ in range(self.max_iterations):
            tagged = self._build_tagged_text(pieces, 0)
            ids = self._query_llm(tagged, current_chunk, n_chunks, invalid_response)
            if ids:
                split_ids = ids
                break
            invalid_response = str(ids)

        if not split_ids:
            split_ids = [n_chunks // 2]

        boundaries = [0] + split_ids + [n_chunks]
        chunks = []
        for i in range(len(boundaries) - 1):
            chunk_pieces = pieces[boundaries[i]: boundaries[i + 1]]
            if chunk_pieces:
                chunks.append(" ".join(chunk_pieces))

        return [c for c in chunks if c.strip()]