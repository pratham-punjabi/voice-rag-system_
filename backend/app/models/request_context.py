from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class LatencyMetrics(BaseModel):
    stt_ms: float = 0.0
    query_processing_ms: float = 0.0
    guardrail_ms: float = 0.0
    embedding_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    bm25_retrieval_ms: float = 0.0
    fusion_ms: float = 0.0
    reranking_ms: float = 0.0
    validation_ms: float = 0.0
    generation_ms: float = 0.0
    grounding_ms: float = 0.0
    formatting_ms: float = 0.0
    total_ms: float = 0.0


class RetrievedDocument(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryInfo(BaseModel):
    original_query: str = ""
    normalized_query: str = ""
    language: str = "en"
    is_valid: bool = True
    intent: str = ""
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    is_code_mixed: bool = False


class GenerationResult(BaseModel):
    answer: str = ""
    confidence: float = 0.0
    grounded: bool = False
    citations: list[dict[str, Any]] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""


class RequestContext(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = Field(default_factory=time.time)

    # Audio
    audio_bytes: bytes | None = None
    audio_sample_rate: int = 16000
    audio_duration_seconds: float = 0.0

    # STT
    transcript: str = ""
    stt_language: str = ""
    stt_confidence: float = 0.0

    # Query processing
    query_info: QueryInfo = Field(default_factory=QueryInfo)

    # Safety
    is_safe: bool = True
    safety_flags: list[str] = Field(default_factory=list)

    # Retrieval
    candidate_documents: list[RetrievedDocument] = Field(default_factory=list)
    reranked_documents: list[RetrievedDocument] = Field(default_factory=list)
    retrieval_passed_validation: bool = False
    retrieval_confidence: float = 0.0

    # Generation
    generation_result: GenerationResult = Field(default_factory=GenerationResult)
    grounding_passed: bool = False

    # Metrics
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)

    # Error tracking
    errors: list[dict[str, str]] = Field(default_factory=list)
    pipeline_status: str = "pending"  # pending | running | success | failed | refused

    def add_error(self, stage: str, code: str, message: str) -> None:
        self.errors.append({"stage": stage, "code": code, "message": message})

    def elapsed_ms(self) -> float:
        return (time.time() - self.created_at) * 1000

    class Config:
        arbitrary_types_allowed = True
