from __future__ import annotations

import re
from typing import Any

from backend.app.ingestion.chunkers.base import ChunkingStrategy
from backend.app.models.chunk import Chunk

# Sentence boundary pattern (supports Hindi/Devanagari ।  and English .)
_SENT_SPLIT = re.compile(r"(?<=[.!?।\u0964])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


class SentenceChunker(ChunkingStrategy):
    """Strategy B: Group N sentences per chunk with sentence-boundary respect."""

    name = "sentence"

    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> None:
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def chunk(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        step = max(1, self.sentences_per_chunk - self.overlap_sentences)
        idx = 0
        chunk_num = 0

        while idx < len(sentences):
            window = sentences[idx: idx + self.sentences_per_chunk]
            chunk_text = " ".join(window)
            if chunk_text.strip():
                chunks.append(self._make_chunk(doc_id, chunk_num, chunk_text, metadata))
                chunk_num += 1
            idx += step

        return chunks
