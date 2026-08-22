from __future__ import annotations

import numpy as np
import pytest

from backend.app.providers.embedding_cache import EmbeddingCache


class TestEmbeddingCache:

    def test_miss_on_empty(self, tmp_path):
        cache = EmbeddingCache(cache_dir=str(tmp_path), use_disk=False)
        assert cache.get("hello world") is None

    def test_set_and_get_memory(self, tmp_path):
        cache = EmbeddingCache(cache_dir=str(tmp_path), use_disk=False)
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cache.set("test query", vec)
        result = cache.get("test query")
        assert result is not None
        np.testing.assert_array_almost_equal(result, vec)

    def test_case_normalised_key(self, tmp_path):
        cache = EmbeddingCache(cache_dir=str(tmp_path), use_disk=False)
        vec = np.ones(4, dtype=np.float32)
        cache.set("What IS NLP", vec)
        result = cache.get("what is nlp")
        assert result is not None

    def test_disk_persistence(self, tmp_path):
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        cache1 = EmbeddingCache(cache_dir=str(tmp_path), use_disk=True)
        cache1.set("persist me", vec)

        # New cache instance — cold memory, should load from disk
        cache2 = EmbeddingCache(cache_dir=str(tmp_path), use_disk=True)
        result = cache2.get("persist me")
        assert result is not None
        np.testing.assert_array_almost_equal(result, vec)

    def test_memory_eviction_at_max(self, tmp_path):
        cache = EmbeddingCache(cache_dir=str(tmp_path), max_memory_items=3, use_disk=False)
        for i in range(5):
            cache.set(f"query {i}", np.ones(4, dtype=np.float32) * i)
        assert len(cache._memory) <= 3

    def test_stats_tracks_hits_and_misses(self, tmp_path):
        cache = EmbeddingCache(cache_dir=str(tmp_path), use_disk=False)
        vec = np.ones(4, dtype=np.float32)
        cache.set("q", vec)
        cache.get("q")       # hit
        cache.get("missing") # miss
        stats = cache.stats
        assert stats["hits_memory"] == 1
        assert stats["misses"] == 1

    def test_clear_memory(self, tmp_path):
        cache = EmbeddingCache(cache_dir=str(tmp_path), use_disk=False)
        cache.set("q", np.ones(4, dtype=np.float32))
        cache.clear_memory()
        assert len(cache._memory) == 0

    def test_clear_all_removes_disk_files(self, tmp_path):
        cache = EmbeddingCache(cache_dir=str(tmp_path), use_disk=True)
        cache.set("q", np.ones(4, dtype=np.float32))
        cache.clear_all()
        assert list(tmp_path.glob("*.pkl")) == []


class TestEmbeddingProviderMock:
    """Test the embedding provider interface contract via a mock."""

    @pytest.mark.asyncio
    async def test_embed_query_returns_1d_float32(self):
        from backend.app.providers.base_embeddings import EmbeddingProvider

        class MinimalEmbedder(EmbeddingProvider):
            def dimension(self): return 8
            async def embed_query(self, text):
                rng = np.random.default_rng(0)
                v = rng.random(8).astype(np.float32)
                return v / np.linalg.norm(v)
            async def embed_documents(self, texts):
                rng = np.random.default_rng(0)
                v = rng.random((len(texts), 8)).astype(np.float32)
                return v / np.linalg.norm(v, axis=1, keepdims=True)
            async def warmup(self): pass

        embedder = MinimalEmbedder()
        vec = await embedder.embed_query("test")
        assert vec.dtype == np.float32
        assert vec.ndim == 1
        assert vec.shape[0] == 8
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5  # normalised

    @pytest.mark.asyncio
    async def test_embed_documents_returns_2d(self):
        from backend.app.providers.base_embeddings import EmbeddingProvider

        class MinimalEmbedder(EmbeddingProvider):
            def dimension(self): return 8
            async def embed_query(self, text):
                return np.ones(8, dtype=np.float32) / np.sqrt(8)
            async def embed_documents(self, texts):
                rng = np.random.default_rng(0)
                return rng.random((len(texts), 8)).astype(np.float32)
            async def warmup(self): pass

        embedder = MinimalEmbedder()
        texts = ["a", "b", "c"]
        vecs = await embedder.embed_documents(texts)
        assert vecs.ndim == 2
        assert vecs.shape == (3, 8)
