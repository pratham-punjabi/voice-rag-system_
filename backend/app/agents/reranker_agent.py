from __future__ import annotations

import logging
import time

from backend.app.models.request_context import RequestContext
from backend.app.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class RerankerAgent:
    def __init__(
        self,
        reranker: CrossEncoderReranker,
        top_k: int = 5,
        enabled: bool = True,
    ) -> None:
        self._reranker = reranker
        self._top_k = top_k
        self._enabled = enabled

    async def process(self, ctx: RequestContext) -> RequestContext:
        t0 = time.perf_counter()
        docs = ctx.candidate_documents

        if not self._enabled or not self._reranker.available:
            ctx.reranked_documents = docs[: self._top_k]
            ctx.latency.reranking_ms = (time.perf_counter() - t0) * 1000
            return ctx

        candidates = [(d.chunk_id, d.text, d.score) for d in docs]
        try:
            ranked = await self._reranker.rerank(
                ctx.query_info.normalized_query, candidates
            )
            # Map back to RetrievedDocument objects
            score_map = {chunk_id: score for chunk_id, score in ranked}
            reranked = sorted(docs, key=lambda d: score_map.get(d.chunk_id, 0), reverse=True)
            for d in reranked:
                d.rerank_score = score_map.get(d.chunk_id)
            ctx.reranked_documents = reranked[: self._top_k]
        except Exception as exc:
            logger.warning("Reranker failed, using retrieval order: %s", exc)
            ctx.reranked_documents = docs[: self._top_k]

        ctx.latency.reranking_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Reranking complete",
            extra={
                "request_id": ctx.request_id,
                "n_reranked": len(ctx.reranked_documents),
                "latency_ms": ctx.latency.reranking_ms,
            },
        )
        return ctx
