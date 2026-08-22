from __future__ import annotations

import pytest

from backend.app.retrieval.bm25 import BM25Index, tokenize


class TestTokenize:

    def test_lowercases(self):
        assert "hello" in tokenize("Hello World")

    def test_removes_punctuation(self):
        tokens = tokenize("hello, world!")
        assert "," not in tokens
        assert "!" not in tokens

    def test_filters_single_chars(self):
        tokens = tokenize("a b c hello")
        assert "a" not in tokens
        assert "b" not in tokens

    def test_unicode_words_kept(self):
        tokens = tokenize("मशीन लर्निंग")
        assert len(tokens) > 0

    def test_empty_string(self):
        assert tokenize("") == []


class TestBM25Index:

    @pytest.fixture
    def built_index(self):
        texts = [
            "machine learning is a subset of artificial intelligence",
            "neural networks are inspired by the human brain",
            "natural language processing enables text understanding",
            "deep learning uses multiple layers of neural networks",
            "retrieval augmented generation improves answer quality",
        ]
        ids = [f"doc_{i}" for i in range(len(texts))]
        idx = BM25Index(k1=1.5, b=0.75)
        idx.build(texts, ids)
        return idx, ids

    def test_build_sets_n(self, built_index):
        idx, ids = built_index
        assert idx._n == len(ids)

    def test_search_returns_list_of_tuples(self, built_index):
        idx, _ = built_index
        results = idx.search("machine learning", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_top_result_is_most_relevant(self, built_index):
        idx, _ = built_index
        results = idx.search("machine learning artificial intelligence", top_k=5)
        assert results[0][0] == "doc_0"

    def test_neural_network_query(self, built_index):
        idx, _ = built_index
        results = idx.search("neural networks", top_k=5)
        top_ids = [r[0] for r in results]
        assert "doc_1" in top_ids[:2] or "doc_3" in top_ids[:2]

    def test_scores_descending(self, built_index):
        idx, _ = built_index
        results = idx.search("deep learning neural", top_k=5)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_term_returns_empty(self, built_index):
        idx, _ = built_index
        results = idx.search("xyznonexistentterm", top_k=5)
        assert results == []

    def test_empty_query_returns_empty(self, built_index):
        idx, _ = built_index
        assert idx.search("", top_k=5) == []

    def test_top_k_respected(self, built_index):
        idx, _ = built_index
        results = idx.search("neural network learning", top_k=2)
        assert len(results) <= 2

    def test_only_positive_scores_returned(self, built_index):
        idx, _ = built_index
        results = idx.search("machine learning neural network", top_k=10)
        assert all(r[1] > 0 for r in results)

    def test_save_and_load_roundtrip(self, built_index, tmp_path):
        idx, _ = built_index
        path = str(tmp_path / "bm25.pkl")
        idx.save(path)

        loaded = BM25Index.load(path)
        assert loaded._n == idx._n

        r1 = idx.search("machine learning", top_k=3)
        r2 = loaded.search("machine learning", top_k=3)
        assert [r[0] for r in r1] == [r[0] for r in r2]

    def test_avgdl_computed(self, built_index):
        idx, _ = built_index
        assert idx._avgdl > 0

    def test_df_populated(self, built_index):
        idx, _ = built_index
        assert "machine" in idx._df
        assert idx._df["machine"] >= 1

    def test_build_with_empty_corpus(self):
        idx = BM25Index()
        idx.build([], [])
        assert idx._n == 0
        assert idx.search("test") == []
