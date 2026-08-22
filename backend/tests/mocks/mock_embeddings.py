from __future__ import annotations

import numpy as np

from backend.app.providers.base_embeddings import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic mock embedding provider.
    Uses MD5 hash of text as the random seed — same text always
    produces the same vector, different texts produce different vectors.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.embed_query_calls = 0
        self.embed_documents_calls = 0

    def dimension(self) -> int:
        return self._dim

    async def warmup(self) -> None:
        pass

    async def embed_query(self, text: str) -> np.ndarray:
        self.embed_query_calls += 1
        return self._text_to_vec(text)

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.embed_documents_calls += 1
        return np.stack([self._text_to_vec(t) for t in texts])

    def _text_to_vec(self, text: str) -> np.ndarray:
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.random(self._dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


class ZeroEmbeddingProvider(EmbeddingProvider):
    """Returns all-zero vectors — for testing edge cases."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def dimension(self) -> int:
        return self._dim

    async def warmup(self) -> None:
        pass

    async def embed_query(self, text: str) -> np.ndarray:
        return np.zeros(self._dim, dtype=np.float32)

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self._dim), dtype=np.float32)


class FixedEmbeddingProvider(EmbeddingProvider):
    """Always returns the same vector — for controlled retrieval tests."""

    def __init__(self, vector: np.ndarray) -> None:
        self._vec = vector.astype(np.float32)

    def dimension(self) -> int:
        return len(self._vec)

    async def warmup(self) -> None:
        pass

    async def embed_query(self, text: str) -> np.ndarray:
        return self._vec.copy()

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec.copy() for _ in texts])
