from __future__ import annotations

from backend.app.ingestion.chunkers.adaptive_chunker import AdaptiveChunker
from backend.app.ingestion.chunkers.base import ChunkingStrategy
from backend.app.ingestion.chunkers.fixed_chunker import FixedChunker
from backend.app.ingestion.chunkers.metadata_chunker import MetadataChunker
from backend.app.ingestion.chunkers.semantic_chunker import SemanticChunker
from backend.app.ingestion.chunkers.sentence_chunker import SentenceChunker

_REGISTRY: dict[str, type[ChunkingStrategy]] = {
    "fixed": FixedChunker,
    "sentence": SentenceChunker,
    "semantic": SemanticChunker,
    "metadata": MetadataChunker,
    "adaptive": AdaptiveChunker,
}


def get_chunker(
    strategy: str = "adaptive",
    chunk_size: int = 256,
    overlap: int = 32,
) -> ChunkingStrategy:
    cls = _REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}. Choose from {list(_REGISTRY)}")
    if strategy == "fixed":
        return FixedChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "adaptive":
        return AdaptiveChunker(chunk_size=chunk_size, overlap=overlap)
    return cls()
