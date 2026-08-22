from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    token_count: int = 0
    char_count: int = 0
    chunk_index: int = 0
    strategy: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    @classmethod
    def create(
        cls,
        doc_id: str,
        chunk_index: int,
        text: str,
        strategy: str,
        metadata: dict[str, Any] | None = None,
    ) -> "Chunk":
        import hashlib
        chunk_id = hashlib.md5(f"{doc_id}:{chunk_index}:{text[:64]}".encode()).hexdigest()
        return cls(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            token_count=len(text.split()),
            char_count=len(text),
            chunk_index=chunk_index,
            strategy=strategy,
            metadata=metadata or {},
        )


class Document(BaseModel):
    doc_id: str
    text: str
    title: str = ""
    language: str = "en"
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunks: list[Chunk] = Field(default_factory=list)
