from __future__ import annotations

import pytest
import numpy as np

from backend.app.retrieval.reranker import CrossEncoderReranker


class TestCrossEncoderReranker:

    @pytest.fixture
    def reranker(self):
        """Reranker with model loading skipped — tests fallback behaviour."""
        r = CrossEncoderReranker()
        # Don't call warmup — model won't load in test env without the weights
        return r

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self, reranker):
        result = await reranker.rerank("test query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_fallback_without_model_preserves_order(self, reranker):
        """When model is unavailable, scores from retrieval are preserved."""
        candidates = [
            ("chunk_001", "Machine learning is great", 0.9),
            ("chunk_002", "Neural networks are powerful", 0.7),
            ("chunk_003", "NLP processes text", 0.5),
        ]
        result = await reranker.rerank("machine learning", candidates)
        # Fallback returns original order
        assert [r[0] for r in result] == ["chunk_001", "chunk_002", "chunk_003"]

    @pytest.mark.asyncio
    async def test_returns_chunk_id_and_score_tuples(self, reranker):
        candidates = [
            ("chunk_A", "some text", 0.8),
            ("chunk_B", "other text", 0.6),
        ]
        result = await reranker.rerank("query", candidates)
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    @pytest.mark.asyncio
    async def test_single_candidate_returned(self, reranker):
        candidates = [("only_chunk", "only content", 1.0)]
        result = await reranker.rerank("query", candidates)
        assert len(result) == 1
        assert result[0][0] == "only_chunk"

    def test_available_false_without_warmup(self, reranker):
        assert reranker.available is False


class TestRerankerAgentIntegration:
    """Tests RerankerAgent behaviour with and without an active reranker."""

    @pytest.mark.asyncio
    async def test_disabled_reranker_uses_retrieval_order(self):
        from backend.app.agents.reranker_agent import RerankerAgent
        from backend.app.models.request_context import RequestContext, RetrievedDocument

        ctx = RequestContext()
        ctx.candidate_documents = [
            RetrievedDocument(doc_id="d1", chunk_id="c1", text="text1", score=0.9),
            RetrievedDocument(doc_id="d2", chunk_id="c2", text="text2", score=0.7),
            RetrievedDocument(doc_id="d3", chunk_id="c3", text="text3", score=0.5),
        ]
        ctx.query_info.normalized_query = "test"

        reranker = CrossEncoderReranker()
        agent = RerankerAgent(reranker=reranker, top_k=2, enabled=False)
        result = await agent.process(ctx)

        assert len(result.reranked_documents) == 2
        assert result.reranked_documents[0].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_top_k_respected(self):
        from backend.app.agents.reranker_agent import RerankerAgent
        from backend.app.models.request_context import RequestContext, RetrievedDocument

        ctx = RequestContext()
        ctx.candidate_documents = [
            RetrievedDocument(doc_id=f"d{i}", chunk_id=f"c{i}", text=f"text{i}", score=0.9 - i * 0.1)
            for i in range(10)
        ]
        ctx.query_info.normalized_query = "test"

        reranker = CrossEncoderReranker()
        agent = RerankerAgent(reranker=reranker, top_k=3, enabled=False)
        result = await agent.process(ctx)

        assert len(result.reranked_documents) <= 3

    @pytest.mark.asyncio
    async def test_latency_is_recorded(self):
        from backend.app.agents.reranker_agent import RerankerAgent
        from backend.app.models.request_context import RequestContext, RetrievedDocument

        ctx = RequestContext()
        ctx.candidate_documents = [
            RetrievedDocument(doc_id="d1", chunk_id="c1", text="content", score=0.8)
        ]
        ctx.query_info.normalized_query = "test"

        agent = RerankerAgent(reranker=CrossEncoderReranker(), top_k=5, enabled=False)
        result = await agent.process(ctx)
        assert result.latency.reranking_ms >= 0
