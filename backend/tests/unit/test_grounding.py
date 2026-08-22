from __future__ import annotations

import json
import pytest

from backend.app.models.request_context import (
    GenerationResult, RequestContext, RetrievedDocument,
)
from backend.app.agents.grounding_agent import GroundingAgent, _overlap_ratio
from backend.app.agents.generation_agent import _parse_response, _build_context


# ── Overlap ratio ─────────────────────────────────────────────────────────────

class TestOverlapRatio:

    def test_perfect_overlap(self):
        assert _overlap_ratio("the cat sat on mat", "the cat sat on mat") == 1.0

    def test_no_overlap(self):
        ratio = _overlap_ratio("quantum physics dark matter", "cooking pasta recipe")
        assert ratio < 0.1

    def test_partial_overlap(self):
        ratio = _overlap_ratio("machine learning model", "deep learning and machine translation")
        assert 0 < ratio < 1.0

    def test_empty_answer(self):
        assert _overlap_ratio("", "some context") == 0.0


# ── Context builder ───────────────────────────────────────────────────────────

class TestBuildContext:

    def test_basic_context(self):
        docs = [
            RetrievedDocument(doc_id="d1", chunk_id="c1", text="Machine learning is great.", score=0.9),
            RetrievedDocument(doc_id="d2", chunk_id="c2", text="Deep learning uses neural nets.", score=0.8),
        ]
        ctx = _build_context(docs, max_tokens=500)
        assert "Machine learning" in ctx
        assert "Passage 1" in ctx
        assert "Passage 2" in ctx

    def test_token_limit_respected(self):
        docs = [
            RetrievedDocument(
                doc_id=f"d{i}", chunk_id=f"c{i}",
                text=" ".join(["word"] * 300),
                score=0.9 - i * 0.01,
            )
            for i in range(20)
        ]
        ctx = _build_context(docs, max_tokens=200)
        # Should not include all 20 passages
        assert ctx.count("Passage") < 20

    def test_empty_docs(self):
        assert _build_context([], max_tokens=500) == ""


# ── Response parser ───────────────────────────────────────────────────────────

class TestParseResponse:

    def test_valid_json(self):
        raw = '{"answer": "Test answer.", "confidence": 0.9, "grounded": true}'
        result = _parse_response(raw)
        assert result["answer"] == "Test answer."
        assert result["confidence"] == 0.9
        assert result["grounded"] is True

    def test_json_embedded_in_text(self):
        raw = 'Here is the answer: {"answer": "The answer is 42.", "confidence": 0.7, "grounded": true} end.'
        result = _parse_response(raw)
        assert "42" in result["answer"]

    def test_malformed_raises(self):
        from backend.app.core.exceptions import LLMMalformedResponseError
        with pytest.raises(LLMMalformedResponseError):
            _parse_response("this is not json at all")


# ── Grounding Agent integration ───────────────────────────────────────────────

class TestGroundingAgent:

    @pytest.mark.asyncio
    async def test_insufficient_response_passes(self):
        ctx = RequestContext()
        ctx.reranked_documents = []
        ctx.generation_result = GenerationResult(
            answer="I don't have enough information in the provided dataset to answer that question.",
        )
        agent = GroundingAgent()
        ctx = await agent.process(ctx)
        assert ctx.grounding_passed is True
        assert ctx.generation_result.refused is True
        assert ctx.generation_result.refusal_reason == "insufficient_evidence"

    @pytest.mark.asyncio
    async def test_grounded_answer_passes(self):
        ctx = RequestContext()
        ctx.reranked_documents = [
            RetrievedDocument(
                doc_id="d1", chunk_id="c1",
                text="Machine learning is a powerful technique used in AI.",
                score=0.9,
            )
        ]
        ctx.generation_result = GenerationResult(
            answer="Machine learning is a powerful AI technique.",
            confidence=0.88,
            grounded=True,
        )
        agent = GroundingAgent(overlap_threshold=0.2)
        ctx = await agent.process(ctx)
        assert ctx.grounding_passed is True

    @pytest.mark.asyncio
    async def test_hallucinated_answer_replaced(self):
        ctx = RequestContext()
        ctx.reranked_documents = [
            RetrievedDocument(
                doc_id="d1", chunk_id="c1",
                text="Python is a programming language.",
                score=0.9,
            )
        ]
        ctx.generation_result = GenerationResult(
            answer="Quantum entanglement enables faster-than-light communication.",
            confidence=0.9,
            grounded=False,
        )
        agent = GroundingAgent(overlap_threshold=0.9)
        ctx = await agent.process(ctx)
        assert ctx.grounding_passed is False
        assert "don't have enough information" in ctx.generation_result.answer
        assert ctx.generation_result.refused is True
        assert ctx.generation_result.refusal_reason == "hallucination_detected"

    @pytest.mark.asyncio
    async def test_model_refusal_phrases_pass(self):
        ctx = RequestContext()
        ctx.reranked_documents = []
        ctx.generation_result = GenerationResult(
            answer="I cannot answer this question based on the provided context.",
        )
        agent = GroundingAgent()
        ctx = await agent.process(ctx)
        assert ctx.grounding_passed is True
        assert ctx.generation_result.refused is True

    @pytest.mark.asyncio
    async def test_latency_recorded(self):
        ctx = RequestContext()
        ctx.reranked_documents = []
        ctx.generation_result = GenerationResult(
            answer="I don't have enough information in the provided dataset to answer that question.",
        )
        agent = GroundingAgent()
        ctx = await agent.process(ctx)
        assert ctx.latency.grounding_ms >= 0
