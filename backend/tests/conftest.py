from __future__ import annotations

import pytest
import numpy as np


@pytest.fixture
def sample_texts():
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by the human brain structure.",
        "Natural language processing enables computers to understand text.",
        "Retrieval augmented generation combines search with generation.",
        "BERT is a transformer-based language model.",
    ]


@pytest.fixture
def sample_embeddings(sample_texts):
    """Deterministic fake embeddings for testing."""
    rng = np.random.default_rng(42)
    vecs = rng.random((len(sample_texts), 128)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def sample_ids(sample_texts):
    return [f"chunk_{i:03d}" for i in range(len(sample_texts))]
