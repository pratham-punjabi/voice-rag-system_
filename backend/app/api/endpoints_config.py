from __future__ import annotations

from fastapi import APIRouter

from backend.app.core.config import get_settings

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config")
async def get_config() -> dict:
    """Return active (non-secret) configuration for the running system."""
    s = get_settings()
    return {
        "embedding_model": s.embedding_model,
        "embedding_dimension": s.embedding_dimension,
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "chunking_strategy": s.chunking_strategy,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "top_k": s.top_k,
        "rerank_top_k": s.rerank_top_k,
        "enable_reranker": s.enable_reranker,
        "enable_hybrid_search": s.enable_hybrid_search,
        "enable_guardrails": s.enable_guardrails,
        "low_confidence_threshold": s.low_confidence_threshold,
        "max_context_tokens": s.max_context_tokens,
        "enable_query_cache": s.enable_query_cache,
        "dataset_name": s.dataset_name,
        "vector_db": s.vector_db,
    }
