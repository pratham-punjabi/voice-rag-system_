from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from backend.app.agents.generation_agent import GenerationAgent
from backend.app.agents.grounding_agent import GroundingAgent
from backend.app.agents.guardrail_agent import GuardrailAgent
from backend.app.agents.query_agent import QueryAgent
from backend.app.agents.reranker_agent import RerankerAgent
from backend.app.agents.response_agent import ResponseAgent
from backend.app.agents.retrieval_agent import RetrievalAgent
from backend.app.agents.stt_agent import STTAgent
from backend.app.agents.validation_agent import ValidationAgent
from backend.app.core.exceptions import (
    InvalidQueryError,
    LowConfidenceError,
    NoDocumentsRetrievedError,
    PromptInjectionError,
    STTEmptyAudioError,
    UnsafeQueryError,
)
from backend.app.models.request_context import GenerationResult, RequestContext
from backend.app.monitoring.latency_recorder import latency_recorder
from backend.app.retrieval.cache import QueryCache

logger = logging.getLogger(__name__)

_INSUFFICIENT_ANSWER = "I don't have enough information to answer that question."


class Orchestrator:
    """
    Central coordinator for the voice RAG pipeline.

    Supports three data source modes:
    - api: Groq LLM with full knowledge (no dataset required)
    - dataset: ChromaDB vector store strictly
    - hybrid: ChromaDB first, Groq fallback if no relevant docs found

    Hot path (after STT):
      query_processing + guardrail  (parallel)
      → retrieval (optional in api mode)
      → reranking (optional)
      → validation
      → generation
      → grounding
      → response
    """

    def __init__(
        self,
        stt_agent: STTAgent | None,
        query_agent: QueryAgent,
        guardrail_agent: GuardrailAgent,
        retrieval_agent: RetrievalAgent,
        reranker_agent: RerankerAgent,
        validation_agent: ValidationAgent,
        generation_agent: GenerationAgent,
        grounding_agent: GroundingAgent,
        response_agent: ResponseAgent,
        query_cache: QueryCache | None = None,
        mode: Literal["api", "dataset", "hybrid"] = "hybrid",
    ) -> None:
        self._stt = stt_agent
        self._query = query_agent
        self._guardrail = guardrail_agent
        self._retrieval = retrieval_agent
        self._reranker = reranker_agent
        self._validation = validation_agent
        self._generation = generation_agent
        self._grounding = grounding_agent
        self._response = response_agent
        self._cache = query_cache
        self._mode = mode

    async def process_voice(self, audio_bytes: bytes, sample_rate: int = 16000) -> dict[str, Any]:
        ctx = RequestContext(audio_bytes=audio_bytes, audio_sample_rate=sample_rate)
        ctx.pipeline_status = "running"
        try:
            if self._stt is None:
                raise STTEmptyAudioError("No STT provider configured")
            ctx = await self._stt.process(ctx)
            return await self._run_text_pipeline(ctx)
        except Exception as exc:
            return await self._handle_pipeline_error(ctx, exc)

    async def process_text(self, query: str) -> dict[str, Any]:
        ctx = RequestContext()
        ctx.transcript = query
        ctx.pipeline_status = "running"
        try:
            return await self._run_text_pipeline(ctx)
        except Exception as exc:
            return await self._handle_pipeline_error(ctx, exc)

    async def _run_text_pipeline(self, ctx: RequestContext) -> dict[str, Any]:
        # ── Cache check ───────────────────────────────────────────────────
        if self._cache:
            cached = self._cache.get(ctx.transcript)
            if cached:
                cached["request_id"] = ctx.request_id
                cached["latency_ms"]["total"] = round(ctx.elapsed_ms(), 2)
                cached["cached"] = True
                return cached

        # ── Query processing + guardrail (parallel) ───────────────────────
        await asyncio.gather(self._query.process(ctx))
        await self._guardrail.process(ctx)

        # ── Retrieval ─────────────────────────────────────────────────────
        await self._retrieval.process(ctx)

        # ── Hybrid fallback: if no docs in hybrid/api mode, skip strict validation ──
        has_docs = bool(ctx.candidate_documents)

        # ── Reranking (only when docs exist) ─────────────────────────────
        if has_docs:
            await self._reranker.process(ctx)

        # ── Validation ────────────────────────────────────────────────────
        # In API/hybrid mode with no docs, skip doc-count validation
        if self._mode == "dataset" or has_docs:
            await self._validation.process(ctx)

        # ── Generation ────────────────────────────────────────────────────
        await self._generation.process(ctx)

        # ── Grounding ─────────────────────────────────────────────────────
        if has_docs:
            await self._grounding.process(ctx)

        ctx.pipeline_status = "success"
        result = await self._response.format(ctx)

        # Add mode metadata
        result["data_source_mode"] = self._mode
        result["retrieved_docs"] = len(ctx.candidate_documents)
        result["has_context"] = has_docs

        # Record latency
        latency_recorder.record("total", ctx.latency.total_ms)
        latency_recorder.record("retrieval", ctx.latency.dense_retrieval_ms)
        latency_recorder.record("reranking", ctx.latency.reranking_ms)
        latency_recorder.record("generation", ctx.latency.generation_ms)
        latency_recorder.record("grounding", ctx.latency.grounding_ms)
        latency_recorder.record("embedding", ctx.latency.embedding_ms)

        if self._cache and ctx.pipeline_status == "success":
            self._cache.set(ctx.transcript, result)

        return result

    async def _handle_pipeline_error(
        self, ctx: RequestContext, exc: Exception
    ) -> dict[str, Any]:
        user_facing = _INSUFFICIENT_ANSWER
        code = str(getattr(exc, "code", "PIPELINE_ERROR"))

        if isinstance(exc, (InvalidQueryError, STTEmptyAudioError)):
            user_facing = "Please speak or type a clear question."
            ctx.pipeline_status = "refused"
        elif isinstance(exc, (UnsafeQueryError, PromptInjectionError)):
            user_facing = "I can't help with that request."
            ctx.pipeline_status = "refused"
        elif isinstance(exc, (NoDocumentsRetrievedError, LowConfidenceError)):
            ctx.pipeline_status = "refused"
        else:
            ctx.pipeline_status = "failed"
            logger.exception("Pipeline error: %s", exc, extra={"request_id": ctx.request_id})

        ctx.generation_result = GenerationResult(
            answer=user_facing,
            confidence=0.0,
            grounded=False,
            refused=True,
            refusal_reason=str(code),
        )
        return await self._response.format(ctx)
