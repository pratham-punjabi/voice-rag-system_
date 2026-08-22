from __future__ import annotations

import logging
import re
import time

from backend.app.models.request_context import RequestContext

logger = logging.getLogger(__name__)

_INSUFFICIENT = "i don't have enough information"
_REFUSAL_PHRASES = [
    "i cannot", "i can't", "i am unable", "i'm unable",
    "no information", "not found", "not in the",
]


def _overlap_ratio(answer: str, context: str) -> float:
    """Rough lexical overlap between answer and context."""
    answer_words = set(re.findall(r"\b\w+\b", answer.lower()))
    ctx_words = set(re.findall(r"\b\w+\b", context.lower()))
    if not answer_words:
        return 0.0
    return len(answer_words & ctx_words) / len(answer_words)


class GroundingAgent:
    """
    Post-generation hallucination check.
    Validates that the answer is grounded in retrieved evidence.
    """

    def __init__(self, overlap_threshold: float = 0.25) -> None:
        self._threshold = overlap_threshold

    async def process(self, ctx: RequestContext) -> RequestContext:
        t0 = time.perf_counter()
        result = ctx.generation_result
        answer = result.answer.lower()

        # Already a refusal — grounding trivially passes
        if _INSUFFICIENT in answer:
            result.grounded = True
            result.refused = True
            result.refusal_reason = "insufficient_evidence"
            ctx.grounding_passed = True
            ctx.latency.grounding_ms = (time.perf_counter() - t0) * 1000
            return ctx

        if any(phrase in answer for phrase in _REFUSAL_PHRASES):
            result.grounded = True
            result.refused = True
            result.refusal_reason = "model_refusal"
            ctx.grounding_passed = True
            ctx.latency.grounding_ms = (time.perf_counter() - t0) * 1000
            return ctx

        # Build combined context
        combined_ctx = " ".join(d.text for d in ctx.reranked_documents)
        overlap = _overlap_ratio(result.answer, combined_ctx)

        if overlap < self._threshold and not result.grounded:
            logger.warning(
                "Grounding check failed: overlap=%.3f threshold=%.3f",
                overlap, self._threshold,
                extra={"request_id": ctx.request_id},
            )
            # Replace answer with safe refusal
            result.answer = "I don't have enough information in the provided dataset to answer that question."
            result.confidence = 0.0
            result.grounded = False
            result.refused = True
            result.refusal_reason = "hallucination_detected"
            ctx.grounding_passed = False
        else:
            result.grounded = True
            ctx.grounding_passed = True

        ctx.latency.grounding_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Grounding check",
            extra={
                "request_id": ctx.request_id,
                "overlap": overlap,
                "passed": ctx.grounding_passed,
                "latency_ms": ctx.latency.grounding_ms,
            },
        )
        return ctx
