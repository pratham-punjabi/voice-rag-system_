from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from backend.app.providers.base_embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """
    High-performance multilingual embeddings using sentence-transformers.
    - Precomputes and caches document embeddings to disk
    - Query embeddings run in thread pool (CPU-bound)
    - Returns L2-normalized vectors
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        cache_dir: str = "data/indexes/embeddings",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._batch_size = batch_size
        self._device = device
        self._model: object | None = None
        self._query_cache: dict[str, np.ndarray] = {}

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        logger.info("Embedding model loaded. Dimension: %d", self.dimension())

    @property
    def _st_model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def dimension(self) -> int:
        return self._st_model.get_sentence_embedding_dimension()

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    async def warmup(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
        await self.embed_query("warmup query")
        logger.info("Embedding provider warmed up")

    async def embed_query(self, text: str) -> np.ndarray:
        key = self._cache_key(text)
        if key in self._query_cache:
            return self._query_cache[key]

        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(
            None,
            lambda: self._st_model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
        )
        result = vec[0].astype(np.float32)
        # Small LRU-style in-memory cache for queries
        if len(self._query_cache) > 512:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[key] = result
        return result

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(
            None,
            lambda: self._st_model.encode(
                texts,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
        )
        return vecs.astype(np.float32)

    def save_embeddings(self, embeddings: np.ndarray, name: str) -> Path:
        path = self._cache_dir / f"{name}.npy"
        np.save(str(path), embeddings)
        logger.info("Saved %d embeddings to %s", len(embeddings), path)
        return path

    def load_embeddings(self, name: str) -> np.ndarray | None:
        path = self._cache_dir / f"{name}.npy"
        if path.exists():
            arr = np.load(str(path))
            logger.info("Loaded %d embeddings from %s", len(arr), path)
            return arr
        return None
