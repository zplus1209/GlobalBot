from __future__ import annotations

from typing import List, Optional

from pydantic import model_validator

from globalbot.backend.chunkings.base import BaseChunker


class RecursiveChunker(BaseChunker):
    chunk_size: int = 500
    chunk_overlap: int = 100
    separators: Optional[List[str]] = None
    keep_separator: bool = True
    is_separator_regex: bool = False

    @model_validator(mode="after")
    def _set_name(self) -> "RecursiveChunker":
        if not self.name:
            self.name = f"recursive/{self.chunk_size}/{self.chunk_overlap}"
        return self

    def split_text(self, text: str) -> List[str]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        separators = self.separators or ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            keep_separator=self.keep_separator,
            is_separator_regex=self.is_separator_regex,
        )
        return splitter.split_text(text)


class CharacterChunker(BaseChunker):
    chunk_size: int = 500
    chunk_overlap: int = 100
    separator: str = "\n\n"

    @model_validator(mode="after")
    def _set_name(self) -> "CharacterChunker":
        if not self.name:
            self.name = f"character/{self.chunk_size}/{self.chunk_overlap}"
        return self

    def split_text(self, text: str) -> List[str]:
        from langchain_text_splitters import CharacterTextSplitter

        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator,
        )
        return splitter.split_text(text)


class TokenChunker(BaseChunker):
    chunk_size: int = 512
    chunk_overlap: int = 50
    encoding_name: str = "cl100k_base"

    @model_validator(mode="after")
    def _set_name(self) -> "TokenChunker":
        if not self.name:
            self.name = f"token/{self.chunk_size}/{self.chunk_overlap}"
        return self

    def split_text(self, text: str) -> List[str]:
        from langchain_text_splitters import TokenTextSplitter

        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name=self.encoding_name,
        )
        return splitter.split_text(text)


class SentenceChunker(BaseChunker):
    chunk_size: int = 500
    chunk_overlap: int = 50

    @model_validator(mode="after")
    def _set_name(self) -> "SentenceChunker":
        if not self.name:
            self.name = f"sentence/{self.chunk_size}/{self.chunk_overlap}"
        return self

    def split_text(self, text: str) -> List[str]:
        from langchain_text_splitters import SentenceTransformersTokenTextSplitter

        splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_text(text)


class MarkdownChunker(BaseChunker):
    chunk_size: int = 500
    chunk_overlap: int = 50

    @model_validator(mode="after")
    def _set_name(self) -> "MarkdownChunker":
        if not self.name:
            self.name = f"markdown/{self.chunk_size}/{self.chunk_overlap}"
        return self

    def split_text(self, text: str) -> List[str]:
        from langchain_text_splitters import MarkdownTextSplitter

        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_text(text)