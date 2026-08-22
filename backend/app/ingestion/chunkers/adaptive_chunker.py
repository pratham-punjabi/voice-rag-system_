from __future__ import annotations

from typing import Any

from backend.app.ingestion.chunkers.base import ChunkingStrategy
from backend.app.ingestion.chunkers.fixed_chunker import FixedChunker
from backend.app.ingestion.chunkers.metadata_chunker import MetadataChunker
from backend.app.ingestion.chunkers.sentence_chunker import SentenceChunker
from backend.app.models.chunk import Chunk

_SHORT_DOC_THRESHOLD = 100    # tokens — keep as one chunk
_LONG_DOC_THRESHOLD = 1000    # tokens — use fixed chunker for speed


class AdaptiveChunker(ChunkingStrategy):
    """
    Strategy E: Choose chunking strategy based on document length,
    metadata availability, and semantic density.

    Decision tree:
      - Has metadata passages  → MetadataChunker
      - Short doc (< 100 tok)  → single chunk
      - Medium doc (< 1000 tok)→ SentenceChunker
      - Long doc (≥ 1000 tok)  → FixedChunker
    """

    name = "adaptive"

    def __init__(self, chunk_size: int = 256, overlap: int = 32) -> None:
        self._meta = MetadataChunker(max_chunk_tokens=chunk_size)
        self._sentence = SentenceChunker(sentences_per_chunk=5, overlap_sentences=1)
        self._fixed = FixedChunker(chunk_size=chunk_size, overlap=overlap)
        self.chunk_size = chunk_size

    def chunk(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        meta = metadata or {}
        token_count = len(text.split())

        # If dataset supplies explicit passage splits → honour them
        if meta.get("passages"):
            return self._meta.chunk(doc_id, text, meta)

        if token_count < _SHORT_DOC_THRESHOLD:
            if text.strip():
                return [self._make_chunk(doc_id, 0, text, meta)]
            return []

        if token_count < _LONG_DOC_THRESHOLD:
            return self._sentence.chunk(doc_id, text, meta)

        return self._fixed.chunk(doc_id, text, meta)
