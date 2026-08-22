from __future__ import annotations

import hashlib
import time
from typing import Any


class QueryCache:
    """Simple TTL LRU cache for query results."""

    def __init__(self, max_size: int = 256, ttl_seconds: int = 300) -> None:
        self._cache: dict[str, tuple[Any, float]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> Any | None:
        key = self._key(query)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        value, ts = entry
        if time.time() - ts > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        return value

    def set(self, query: str, value: Any) -> None:
        if len(self._cache) >= self._max_size:
            # Evict oldest
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        self._cache[self._key(query)] = (value, time.time())

    @property
    def stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
        }
