from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from backend.app.core.exceptions import IndexNotFoundError, VectorDBUnavailableError

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Production FAISS vector store with IVFFlat index.
    - Build offline, load at startup
    - Batch insertion
    - Top-K retrieval with scores
    """

    def __init__(
        self,
        index_path: str = "data/indexes/faiss",
        dimension: int = 768,
        nlist: int = 100,
        nprobe: int = 10,
    ) -> None:
        self._index_path = Path(index_path)
        self._dimension = dimension
        self._nlist = nlist
        self._nprobe = nprobe
        self._index = None
        self._id_map: list[str] = []  # positional index → chunk_id

    def _faiss(self):
        try:
            import faiss
            return faiss
        except ImportError as exc:
            raise VectorDBUnavailableError("faiss-cpu not installed") from exc

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        faiss = self._faiss()
        n, d = embeddings.shape
        assert d == self._dimension, f"Dimension mismatch: got {d}, expected {self._dimension}"

        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        # Use Flat for small datasets, IVFFlat for larger ones
        if n < self._nlist * 39:  # FAISS rule: n > nlist * 39 for meaningful IVF
            logger.info("Using IndexFlatIP (dataset too small for IVF)")
            index = faiss.IndexFlatIP(d)
        else:
            logger.info("Using IndexIVFFlat nlist=%d", self._nlist)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, self._nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)

        index.add(embeddings)
        if hasattr(index, "nprobe"):
            index.nprobe = self._nprobe

        self._index = index
        self._id_map = ids
        logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, d)

    def save(self) -> None:
        import faiss
        self._index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path / "index.faiss"))
        with open(self._index_path / "id_map.json", "w") as f:
            json.dump(self._id_map, f)
        logger.info("FAISS index saved to %s", self._index_path)

    def load(self) -> None:
        import faiss
        idx_file = self._index_path / "index.faiss"
        id_file = self._index_path / "id_map.json"
        if not idx_file.exists():
            raise IndexNotFoundError(f"FAISS index not found at {idx_file}")
        self._index = faiss.read_index(str(idx_file))
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = self._nprobe
        with open(id_file) as f:
            self._id_map = json.load(f)
        logger.info("FAISS index loaded: %d vectors", self._index.ntotal)

    def search(self, query_vec: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        """Return [(chunk_id, score)] sorted descending."""
        if self._index is None:
            raise VectorDBUnavailableError("Index not loaded")
        import faiss
        vec = query_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._id_map):
                results.append((self._id_map[idx], float(score)))
        return results

    @property
    def is_loaded(self) -> bool:
        return self._index is not None
