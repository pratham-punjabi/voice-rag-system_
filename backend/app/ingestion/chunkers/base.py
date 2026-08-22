from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.app.models.chunk import Chunk


class ChunkingStrategy(ABC):
    """Abstract base for all chunking strategies."""

    name: str = "base"

    @abstractmethod
    def chunk(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into chunks. Returns list of Chunk objects."""

    def _make_chunk(
        self,
        doc_id: str,
        index: int,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        return Chunk.create(
            doc_id=doc_id,
            chunk_index=index,
            text=text.strip(),
            strategy=self.name,
            metadata=metadata or {},
        )
