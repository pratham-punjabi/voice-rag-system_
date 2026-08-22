from __future__ import annotations

from typing import Any

from backend.app.ingestion.chunkers.base import ChunkingStrategy
from backend.app.models.chunk import Chunk


class FixedChunker(ChunkingStrategy):
    """Strategy A: Fixed token-window with configurable overlap."""

    name = "fixed"

    def __init__(self, chunk_size: int = 256, overlap: int = 32) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        words = text.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        step = max(1, self.chunk_size - self.overlap)
        idx = 0
        chunk_num = 0

        while idx < len(words):
            window = words[idx: idx + self.chunk_size]
            chunk_text = " ".join(window)
            if chunk_text.strip():
                chunks.append(self._make_chunk(doc_id, chunk_num, chunk_text, metadata))
                chunk_num += 1
            idx += step

        return chunks
