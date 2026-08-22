from __future__ import annotations

import logging
import time

from backend.app.core.exceptions import LowConfidenceError, NoDocumentsRetrievedError
from backend.app.models.request_context import RequestContext

logger = logging.getLogger(__name__)


class ValidationAgent:
    """Validate retrieved evidence quality before generation."""

    def __init__(
        self,
        min_docs: int = 1,
        min_confidence: float = 0.35,
    ) -> None:
        self._min_docs = min_docs
        self._min_confidence = min_confidence

    async def process(self, ctx: RequestContext) -> RequestContext:
        t0 = time.perf_counter()
        docs = ctx.reranked_documents or ctx.candidate_documents

        if not docs:
            ctx.latency.validation_ms = (time.perf_counter() - t0) * 1000
            raise NoDocumentsRetrievedError()

        # Deduplicate by text similarity (exact dedup first)
        seen_texts: set[str] = set()
        unique_docs = []
        for d in docs:
            key = d.text[:128]
            if key not in seen_texts:
                seen_texts.add(key)
                unique_docs.append(d)

        top_score = unique_docs[0].score if unique_docs else 0.0
        avg_score = sum(d.score for d in unique_docs) / max(1, len(unique_docs))
        ctx.retrieval_confidence = float(top_score)

        if len(unique_docs) < self._min_docs or top_score < self._min_confidence:
            ctx.retrieval_passed_validation = False
            ctx.latency.validation_ms = (time.perf_counter() - t0) * 1000
            raise LowConfidenceError(
                f"Top score {top_score:.3f} below threshold {self._min_confidence}"
            )

        ctx.reranked_documents = unique_docs
        ctx.retrieval_passed_validation = True
        ctx.latency.validation_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Validation passed",
            extra={
                "request_id": ctx.request_id,
                "n_docs": len(unique_docs),
                "top_score": top_score,
                "avg_score": avg_score,
            },
        )
        return ctx
