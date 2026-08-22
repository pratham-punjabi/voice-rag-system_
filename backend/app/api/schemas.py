from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class TextQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class VoiceQueryRequest(BaseModel):
    sample_rate: int = Field(default=16000, ge=8000, le=48000)


class CitationResponse(BaseModel):
    document_id: str
    chunk_id: str
    score: float
    rerank_score: float | None = None


class LatencyBreakdown(BaseModel):
    stt: float = 0.0
    query_processing: float = 0.0
    guardrail: float = 0.0
    embedding: float = 0.0
    dense_retrieval: float = 0.0
    bm25_retrieval: float = 0.0
    fusion: float = 0.0
    reranking: float = 0.0
    validation: float = 0.0
    generation: float = 0.0
    grounding: float = 0.0
    total: float = 0.0


class QueryResponse(BaseModel):
    request_id: str
    status: str
    transcript: str
    query: dict[str, Any]
    answer: str
    confidence: float
    grounded: bool
    refused: bool
    citations: list[dict[str, Any]]
    retrieval: dict[str, Any]
    latency_ms: dict[str, float]
    errors: list[dict[str, str]]


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    index_loaded: bool
    bm25_loaded: bool
    llm_available: bool
    stt_available: bool
    components: dict[str, str]


class MetricsResponse(BaseModel):
    latency_percentiles: dict[str, Any]
    cache_stats: dict[str, Any]
    total_requests: int


class ErrorResponse(BaseModel):
    error: str
    code: str
    request_id: str | None = None
