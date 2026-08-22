from __future__ import annotations

from typing import Any

from backend.app.ingestion.chunkers.base import ChunkingStrategy
from backend.app.ingestion.chunkers.sentence_chunker import SentenceChunker
from backend.app.models.chunk import Chunk


class MetadataChunker(ChunkingStrategy):
    """
    Strategy D: Respect metadata boundaries — e.g. use passage_id or
    paragraph markers from the dataset to avoid splitting logically connected units.
    Falls back to SentenceChunker within each metadata section.
    """

    name = "metadata"

    def __init__(self, max_chunk_tokens: int = 256) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self._fallback = SentenceChunker()

    def chunk(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        meta = metadata or {}

        # If the dataset provides explicit passage boundaries, honour them
        passages: list[str] = meta.get("passages", [])
        if not passages:
            # Try splitting by double newline (paragraph boundaries)
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            passages = paragraphs if len(paragraphs) > 1 else []

        if not passages:
            return self._fallback.chunk(doc_id, text, metadata)

        chunks: list[Chunk] = []
        chunk_num = 0
        for passage in passages:
            token_count = len(passage.split())
            if token_count <= self.max_chunk_tokens:
                if passage.strip():
                    chunks.append(self._make_chunk(doc_id, chunk_num, passage, meta))
                    chunk_num += 1
            else:
                # Sub-chunk large passages
                sub = self._fallback.chunk(doc_id, passage, meta)
                for s in sub:
                    s.chunk_index = chunk_num
                    s.chunk_id = Chunk.create(doc_id, chunk_num, s.text, self.name).chunk_id
                    chunks.append(s)
                    chunk_num += 1

        return chunks
