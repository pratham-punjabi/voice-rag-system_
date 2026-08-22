from __future__ import annotations

import re
from typing import Any

import numpy as np

from backend.app.ingestion.chunkers.base import ChunkingStrategy
from backend.app.ingestion.chunkers.sentence_chunker import _split_sentences
from backend.app.models.chunk import Chunk


class SemanticChunker(ChunkingStrategy):
    """
    Strategy C: Detect semantic boundaries using cosine similarity between
    consecutive sentence embeddings. Groups semantically similar sentences.
    Falls back to sentence chunker if model unavailable.
    """

    name = "semantic"

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        similarity_threshold: float = 0.75,
        max_chunk_tokens: int = 256,
        min_chunk_sentences: int = 2,
    ) -> None:
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_sentences = min_chunk_sentences
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                pass
        return self._model

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def chunk(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        model = self._get_model()
        if model is None or len(sentences) < 3:
            # Fallback to sentence chunker
            from backend.app.ingestion.chunkers.sentence_chunker import SentenceChunker
            return SentenceChunker().chunk(doc_id, text, metadata)

        embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

        # Find split points where cosine similarity drops below threshold
        split_indices = [0]
        current_tokens = len(sentences[0].split())

        for i in range(1, len(sentences)):
            sim = self._cosine(embeddings[i - 1], embeddings[i])
            token_count = len(sentences[i].split())

            # Split if: semantic break OR chunk too large
            should_split = (
                sim < self.similarity_threshold
                and i - split_indices[-1] >= self.min_chunk_sentences
            ) or (current_tokens + token_count > self.max_chunk_tokens)

            if should_split:
                split_indices.append(i)
                current_tokens = token_count
            else:
                current_tokens += token_count

        split_indices.append(len(sentences))

        chunks: list[Chunk] = []
        for chunk_num, (start, end) in enumerate(
            zip(split_indices, split_indices[1:])
        ):
            chunk_text = " ".join(sentences[start:end])
            if chunk_text.strip():
                chunks.append(self._make_chunk(doc_id, chunk_num, chunk_text, metadata))

        return chunks
