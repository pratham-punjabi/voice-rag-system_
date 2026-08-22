from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.
    Returns merged list sorted by descending RRF score.

    Args:
        ranked_lists: Each list is [(doc_id, score), ...] in rank order.
        k: RRF constant (default 60 — from Cormack et al. 2009).
        weights: Per-list weights (default: equal).
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    assert len(weights) == len(ranked_lists)

    rrf_scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight * (1.0 / (k + rank))

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def normalize_scores(results: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Min-max normalize scores to [0, 1]."""
    if not results:
        return []
    scores = [s for _, s in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [(d, 1.0) for d, _ in results]
    return [(d, (s - min_s) / (max_s - min_s)) for d, s in results]


class HybridRetriever:
    """
    Combines dense vector search + BM25 keyword search via RRF.
    """

    def __init__(
        self,
        vector_store,
        bm25_index,
        alpha: float = 0.6,
        rrf_k: int = 60,
    ) -> None:
        self._vs = vector_store
        self._bm25 = bm25_index
        self.alpha = alpha          # dense weight
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        top_k: int = 20,
    ) -> list[tuple[str, float, float, float]]:
        """
        Returns [(chunk_id, fused_score, dense_score, bm25_score)].
        """
        dense_results = self._vs.search(query_vec, top_k=top_k)
        bm25_results = self._bm25.search(query, top_k=top_k)

        dense_norm = normalize_scores(dense_results)
        bm25_norm = normalize_scores(bm25_results)

        fused = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=self.rrf_k,
            weights=[self.alpha, 1.0 - self.alpha],
        )

        # Build score lookup maps
        dense_map = {d: s for d, s in dense_norm}
        bm25_map = {d: s for d, s in bm25_norm}

        return [
            (doc_id, fused_score, dense_map.get(doc_id, 0.0), bm25_map.get(doc_id, 0.0))
            for doc_id, fused_score in fused
        ][:top_k]
