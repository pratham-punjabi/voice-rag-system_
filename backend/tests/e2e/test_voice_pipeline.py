from __future__ import annotations

"""
End-to-end test: audio bytes → STT → RAG pipeline → answer.
Uses mock STT and LLM so no external APIs needed.
"""

import asyncio
import wave
import io
import struct

import pytest

from backend.app.agents.generation_agent import GenerationAgent
from backend.app.agents.grounding_agent import GroundingAgent
from backend.app.agents.guardrail_agent import GuardrailAgent
from backend.app.agents.orchestrator import Orchestrator
from backend.app.agents.query_agent import QueryAgent
from backend.app.agents.reranker_agent import RerankerAgent
from backend.app.agents.response_agent import ResponseAgent
from backend.app.agents.retrieval_agent import RetrievalAgent
from backend.app.agents.stt_agent import STTAgent
from backend.app.agents.validation_agent import ValidationAgent
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.app.retrieval.vector_store import FAISSVectorStore

import numpy as np


def make_dummy_wav(duration_secs: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a valid WAV file with silence."""
    n_samples = int(duration_secs * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return buf.getvalue()


class MockSTT:
    async def transcribe(self, audio_bytes, sample_rate=16000):
        from backend.app.providers.base_stt import TranscriptionResult
        if not audio_bytes:
            from backend.app.core.exceptions import STTEmptyAudioError
            raise STTEmptyAudioError()
        return TranscriptionResult(
            transcript="What is machine learning?",
            language="en",
            confidence=0.98,
            duration_seconds=1.0,
        )

    async def health_check(self):
        return True


class MockEmbedder:
    def dimension(self): return 8
    async def warmup(self): pass
    async def embed_query(self, text):
        rng = np.random.default_rng(42)
        v = rng.random(8).astype(np.float32)
        return v / np.linalg.norm(v)
    async def embed_documents(self, texts):
        rng = np.random.default_rng(42)
        v = rng.random((len(texts), 8)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class MockVS:
    is_loaded = True
    def search(self, vec, top_k=5):
        return [(f"chunk_{i:03d}", 0.9 - i * 0.05) for i in range(min(top_k, 3))]


class MockBM25:
    def search(self, query, top_k=5):
        return [(f"chunk_{i:03d}", 10.0 - i) for i in range(min(top_k, 3))]


class MockLLM:
    async def complete(self, system_prompt, user_prompt, **kwargs):
        from backend.app.providers.base_llm import LLMResponse
        return LLMResponse(
            content='{"answer": "Machine learning enables computers to learn from data.", "confidence": 0.88, "grounded": true}',
            model="mock",
            prompt_tokens=80,
            completion_tokens=20,
            finish_reason="stop",
        )
    async def health_check(self): return True


_METADATA = {
    f"chunk_{i:03d}": {
        "chunk_id": f"chunk_{i:03d}",
        "doc_id": f"doc_{i:03d}",
        "text": f"Machine learning is a powerful technique. It enables learning from data. Passage {i}.",
        "title": f"Doc {i}",
        "language": "en",
    }
    for i in range(10)
}


def _build_orchestrator(with_stt: bool = True) -> Orchestrator:
    vs = MockVS()
    bm25 = MockBM25()
    embedder = MockEmbedder()
    llm = MockLLM()
    reranker = CrossEncoderReranker()

    retrieval = RetrievalAgent(
        vector_store=vs, bm25_index=bm25, embedder=embedder,
        metadata_path="nonexistent.json", top_k=5,
    )
    retrieval._metadata = _METADATA

    stt_agent = STTAgent(MockSTT()) if with_stt else None

    return Orchestrator(
        stt_agent=stt_agent,
        query_agent=QueryAgent(),
        guardrail_agent=GuardrailAgent(),
        retrieval_agent=retrieval,
        reranker_agent=RerankerAgent(reranker=reranker, top_k=3, enabled=False),
        validation_agent=ValidationAgent(min_docs=1, min_confidence=0.1),
        generation_agent=GenerationAgent(llm=llm),
        grounding_agent=GroundingAgent(overlap_threshold=0.0),
        response_agent=ResponseAgent(),
        query_cache=None,
    )


class TestE2EVoicePipeline:

    @pytest.mark.asyncio
    async def test_full_voice_pipeline(self):
        """Audio bytes → STT → RAG → answer."""
        orch = _build_orchestrator(with_stt=True)
        wav = make_dummy_wav(duration_secs=1.0)

        result = await orch.process_voice(wav)

        assert "request_id" in result
        assert result["transcript"] == "What is machine learning?"
        assert result["answer"]
        assert result["latency_ms"]["stt"] > 0
        assert result["latency_ms"]["total"] > 0

    @pytest.mark.asyncio
    async def test_empty_audio_refused(self):
        orch = _build_orchestrator(with_stt=True)
        result = await orch.process_voice(b"")
        assert result["refused"] is True

    @pytest.mark.asyncio
    async def test_text_pipeline_no_stt(self):
        """Text query skips STT, still completes full pipeline."""
        orch = _build_orchestrator(with_stt=False)
        result = await orch.process_text("What is machine learning?")
        assert result["latency_ms"]["stt"] == 0.0
        assert result["answer"]

    @pytest.mark.asyncio
    async def test_pipeline_produces_citations(self):
        orch = _build_orchestrator(with_stt=True)
        wav = make_dummy_wav()
        result = await orch.process_voice(wav)
        if result["status"] == "success":
            assert isinstance(result["citations"], list)

    @pytest.mark.asyncio
    async def test_pipeline_latency_all_components_recorded(self):
        orch = _build_orchestrator(with_stt=True)
        wav = make_dummy_wav()
        result = await orch.process_voice(wav)
        lat = result["latency_ms"]
        for key in ["stt", "query_processing", "embedding", "generation", "total"]:
            assert key in lat, f"Missing latency key: {key}"
