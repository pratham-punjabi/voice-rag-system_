from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base for all embedding providers."""

    @abstractmethod
    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns normalized 1-D array."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Batch embed documents. Returns (N, D) normalized array."""

    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""

    @abstractmethod
    async def warmup(self) -> None:
        """Warm up model (run dummy inference)."""
