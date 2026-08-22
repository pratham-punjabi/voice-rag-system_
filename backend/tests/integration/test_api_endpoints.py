from __future__ import annotations

"""
Integration tests for FastAPI endpoints.
Uses httpx AsyncClient with the full app (mocked state).
"""

import io
import json
import struct
import wave

import pytest
from fastapi.testclient import TestClient

# We patch the lifespan so tests don't need real models
from unittest.mock import AsyncMock, MagicMock, patch


def make_wav_bytes(duration=0.5, sample_rate=16000) -> bytes:
    n = int(duration * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{n}h", *([100] * n)))
    return buf.getvalue()


MOCK_RESULT = {
    "request_id": "test-123",
    "status": "success",
    "transcript": "What is NLP?",
    "query": {"original": "What is NLP?", "normalized": "What is NLP?", "language": "en", "intent": "definition"},
    "answer": "NLP stands for Natural Language Processing.",
    "confidence": 0.88,
    "grounded": True,
    "refused": False,
    "citations": [],
    "retrieval": {"n_candidates": 5, "n_final": 3, "top_score": 0.91, "passed_validation": True},
    "latency_ms": {
        "stt": 0.0, "query_processing": 0.2, "guardrail": 0.1,
        "embedding": 10.0, "dense_retrieval": 4.0, "bm25_retrieval": 1.0,
        "fusion": 0.5, "reranking": 0.0, "validation": 0.3,
        "generation": 450.0, "grounding": 0.5, "total": 467.0,
    },
    "errors": [],
}


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    orch.process_text = AsyncMock(return_value=MOCK_RESULT)
    orch.process_voice = AsyncMock(return_value=MOCK_RESULT)
    return orch


@pytest.fixture
def app_client(mock_orchestrator):
    """Create a test client with mocked app state."""
    from backend.app.main import app

    # Inject mocked state directly
    app.state.orchestrator = mock_orchestrator
    app.state.query_cache = None
    app.state.vector_store = MagicMock(is_loaded=True)
    app.state.bm25_index = MagicMock()
    app.state.llm_provider = MagicMock()
    app.state.llm_provider.health_check = AsyncMock(return_value=True)
    app.state.stt_provider = MagicMock()
    app.state.stt_provider.health_check = AsyncMock(return_value=True)

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


class TestTextQueryEndpoint:

    def test_text_query_success(self, app_client, mock_orchestrator):
        resp = app_client.post(
            "/api/query/text",
            json={"query": "What is NLP?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == MOCK_RESULT["answer"]
        assert data["status"] == "success"
        mock_orchestrator.process_text.assert_called_once_with("What is NLP?")

    def test_text_query_empty_rejected(self, app_client):
        resp = app_client.post("/api/query/text", json={"query": ""})
        assert resp.status_code == 422  # Pydantic validation

    def test_text_query_too_long_rejected(self, app_client):
        resp = app_client.post("/api/query/text", json={"query": "x" * 501})
        assert resp.status_code == 422

    def test_text_query_strips_whitespace(self, app_client, mock_orchestrator):
        resp = app_client.post("/api/query/text", json={"query": "  What is NLP?  "})
        assert resp.status_code == 200
        # The stripped query should be passed
        call_arg = mock_orchestrator.process_text.call_args[0][0]
        assert call_arg == "What is NLP?"


class TestVoiceQueryEndpoint:

    def test_voice_upload_success(self, app_client, mock_orchestrator):
        wav = make_wav_bytes()
        resp = app_client.post(
            "/api/query",
            files={"audio": ("test.wav", wav, "audio/wav")},
            data={"sample_rate": "16000"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    def test_empty_audio_rejected(self, app_client):
        resp = app_client.post(
            "/api/query",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        assert resp.status_code == 400


class TestHealthEndpoint:

    def test_health_returns_200(self, app_client):
        resp = app_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "components" in data

    def test_health_fields_present(self, app_client):
        data = app_client.get("/api/health").json()
        for field in ["index_loaded", "bm25_loaded", "llm_available", "stt_available"]:
            assert field in data


class TestMetricsEndpoint:

    def test_metrics_returns_200(self, app_client):
        resp = app_client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "latency_percentiles" in data

    def test_config_returns_200(self, app_client):
        resp = app_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "embedding_model" in data
        assert "top_k" in data
