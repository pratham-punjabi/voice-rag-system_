from __future__ import annotations

import time
from typing import Any

from backend.app.models.request_context import RequestContext


class ResponseAgent:
    """Converts RequestContext into a clean frontend-friendly dict."""

    async def format(self, ctx: RequestContext) -> dict[str, Any]:
        t0 = time.perf_counter()

        # Compute total latency
        ctx.latency.total_ms = ctx.elapsed_ms()

        result = {
            "request_id": ctx.request_id,
            "status": ctx.pipeline_status,
            "transcript": ctx.transcript,
            "query": {
                "original": ctx.query_info.original_query,
                "normalized": ctx.query_info.normalized_query,
                "language": ctx.query_info.language,
                "intent": ctx.query_info.intent,
            },
            "answer": ctx.generation_result.answer,
            "confidence": round(ctx.generation_result.confidence, 3),
            "grounded": ctx.generation_result.grounded,
            "refused": ctx.generation_result.refused,
            "citations": ctx.generation_result.citations,
            "retrieval": {
                "n_candidates": len(ctx.candidate_documents),
                "n_final": len(ctx.reranked_documents),
                "top_score": round(ctx.retrieval_confidence, 4),
                "passed_validation": ctx.retrieval_passed_validation,
            },
            "latency_ms": {
                "stt": round(ctx.latency.stt_ms, 2),
                "query_processing": round(ctx.latency.query_processing_ms, 2),
                "guardrail": round(ctx.latency.guardrail_ms, 2),
                "embedding": round(ctx.latency.embedding_ms, 2),
                "dense_retrieval": round(ctx.latency.dense_retrieval_ms, 2),
                "bm25_retrieval": round(ctx.latency.bm25_retrieval_ms, 2),
                "fusion": round(ctx.latency.fusion_ms, 2),
                "reranking": round(ctx.latency.reranking_ms, 2),
                "validation": round(ctx.latency.validation_ms, 2),
                "generation": round(ctx.latency.generation_ms, 2),
                "grounding": round(ctx.latency.grounding_ms, 2),
                "total": round(ctx.latency.total_ms, 2),
            },
            "errors": ctx.errors,
        }

        ctx.latency.formatting_ms = (time.perf_counter() - t0) * 1000
        return result
