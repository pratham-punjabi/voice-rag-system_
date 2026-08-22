from __future__ import annotations

import numpy as np
import pytest

from backend.app.retrieval.vector_store import FAISSVectorStore
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.cache import QueryCache


class TestFAISSVectorStore:

    @pytest.fixture
    def store_with_data(self, tmp_path, sample_embeddings, sample_ids):
        store = FAISSVectorStore(
            index_path=str(tmp_path / "faiss"),
            dimension=128,
            nlist=2,
        )
        store.build(sample_embeddings, sample_ids)
        return store

    def test_build_and_search(self, store_with_data, sample_embeddings):
        query_vec = sample_embeddings[0]
        results = store_with_data.search(query_vec, top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_search_returns_correct_top_result(self, store_with_data, sample_embeddings, sample_ids):
        # Query with first embedding — should retrieve itself as top result
        results = store_with_data.search(sample_embeddings[0], top_k=5)
        top_id = results[0][0]
        assert top_id == sample_ids[0]

    def test_scores_in_descending_order(self, store_with_data, sample_embeddings):
        results = store_with_data.search(sample_embeddings[0], top_k=5)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_save_and_load(self, tmp_path, sample_embeddings, sample_ids):
        path = str(tmp_path / "faiss")
        store = FAISSVectorStore(index_path=path, dimension=128, nlist=2)
        store.build(sample_embeddings, sample_ids)
        store.save()

        loaded = FAISSVectorStore(index_path=path, dimension=128)
        loaded.load()
        assert loaded.is_loaded

        r1 = store.search(sample_embeddings[0], top_k=3)
        r2 = loaded.search(sample_embeddings[0], top_k=3)
        assert [r[0] for r in r1] == [r[0] for r in r2]

    def test_top_k_respected(self, store_with_data, sample_embeddings):
        results = store_with_data.search(sample_embeddings[0], top_k=2)
        assert len(results) <= 2

    def test_load_missing_index_raises(self, tmp_path):
        store = FAISSVectorStore(index_path=str(tmp_path / "nonexistent"), dimension=128)
        from backend.app.core.exceptions import IndexNotFoundError
        with pytest.raises(IndexNotFoundError):
            store.load()


class TestQueryCache:

    def test_miss_on_empty(self):
        cache = QueryCache(max_size=10, ttl_seconds=60)
        assert cache.get("hello") is None

    def test_set_and_get(self):
        cache = QueryCache()
        cache.set("test query", {"answer": "42"})
        result = cache.get("test query")
        assert result == {"answer": "42"}

    def test_case_insensitive_key(self):
        cache = QueryCache()
        cache.set("WHAT IS NLP", "answer")
        assert cache.get("what is nlp") == "answer"

    def test_ttl_expiry(self):
        import time
        cache = QueryCache(max_size=10, ttl_seconds=0)
        cache.set("key", "value")
        time.sleep(0.01)
        assert cache.get("key") is None

    def test_max_size_eviction(self):
        cache = QueryCache(max_size=3)
        for i in range(5):
            cache.set(f"query_{i}", f"result_{i}")
        assert len(cache._cache) <= 3

    def test_stats_tracks_hits_and_misses(self):
        cache = QueryCache()
        cache.set("q", "r")
        cache.get("q")       # hit
        cache.get("missing") # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
