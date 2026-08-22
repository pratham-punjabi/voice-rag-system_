from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.api.dependencies import get_orchestrator

router = APIRouter(prefix="/api/query", tags=["query"])


class TextQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    mode: str | None = Field(None, description="Override data_source_mode: api|dataset|hybrid")


class TextQueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    confidence: float
    grounded: bool
    citations: list[dict] = []
    latency_ms: dict = {}
    data_source_mode: str = ""
    retrieved_docs: int = 0
    has_context: bool = False
    cached: bool = False


@router.post("/text", response_model=TextQueryResponse, summary="Text query")
async def query_text(
    body: TextQueryRequest,
    orchestrator=Depends(get_orchestrator),
):
    """
    Submit a text query to the RAG pipeline.

    Supports three modes (set by DATA_SOURCE_MODE env or per-request override):
    - **api**: Uses Groq LLM with full knowledge — no pre-indexed data needed
    - **dataset**: Uses ChromaDB vector store strictly
    - **hybrid**: ChromaDB first, Groq fallback if few/no docs found
    """
    result = await orchestrator.process_text(body.query)
    return TextQueryResponse(**result)


@router.post("/", response_model=TextQueryResponse, summary="Alias for text query")
async def query_text_alias(
    body: TextQueryRequest,
    orchestrator=Depends(get_orchestrator),
):
    result = await orchestrator.process_text(body.query)
    return TextQueryResponse(**result)
