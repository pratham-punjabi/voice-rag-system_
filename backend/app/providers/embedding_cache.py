from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Two-level embedding cache:
      L1 — in-memory dict (fast, capped at max_memory_items)
      L2 — disk pickle cache (persistent across restarts)

    Designed for query-time embeddings only.
    Document embeddings are stored as .npy files by the embedder directly.
    """

    def __init__(
        self,
        cache_dir: str = "data/indexes/embeddings/query_cache",
        max_memory_items: int = 512,
        use_disk: bool = True,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_memory = max_memory_items
        self._use_disk = use_disk
        self._memory: dict[str, np.ndarray] = {}
        self._hits_mem = 0
        self._hits_disk = 0
        self._misses = 0

    def _key(self, text: str) -> str:
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    def _disk_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.pkl"

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._key(text)

        # L1 — memory
        if key in self._memory:
            self._hits_mem += 1
            return self._memory[key]

        # L2 — disk
        if self._use_disk:
            path = self._disk_path(key)
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        vec = pickle.load(f)
                    self._promote_to_memory(key, vec)
                    self._hits_disk += 1
                    return vec
                except Exception:
                    path.unlink(missing_ok=True)

        self._misses += 1
        return None

    def set(self, text: str, vector: np.ndarray) -> None:
        key = self._key(text)
        self._promote_to_memory(key, vector)

        if self._use_disk:
            try:
                with open(self._disk_path(key), "wb") as f:
                    pickle.dump(vector, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                logger.warning("Disk cache write failed: %s", exc)

    def _promote_to_memory(self, key: str, vec: np.ndarray) -> None:
        if len(self._memory) >= self._max_memory:
            # Evict oldest key
            evict_key = next(iter(self._memory))
            del self._memory[evict_key]
        self._memory[key] = vec

    @property
    def stats(self) -> dict:
        total = self._hits_mem + self._hits_disk + self._misses
        return {
            "memory_items": len(self._memory),
            "hits_memory": self._hits_mem,
            "hits_disk": self._hits_disk,
            "misses": self._misses,
            "hit_rate": round((self._hits_mem + self._hits_disk) / max(1, total), 3),
        }

    def clear_memory(self) -> None:
        self._memory.clear()

    def clear_all(self) -> None:
        self._memory.clear()
        for f in self._cache_dir.glob("*.pkl"):
            f.unlink(missing_ok=True)
