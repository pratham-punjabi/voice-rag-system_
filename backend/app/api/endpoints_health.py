from __future__ import annotations

import time
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", summary="Health check")
async def health(request: Request):
    """System health including ChromaDB, LLM, and STT status."""
    app = request.app

    chroma_store = getattr(app.state, "chroma_store", None)
    chroma_chunks = chroma_store.count() if chroma_store else 0
    chroma_ok = chroma_chunks > 0 if chroma_store else False

    vector_store = getattr(app.state, "vector_store", None)
    faiss_ok = (vector_store is not None and vector_store.is_loaded) if vector_store else False

    bm25 = getattr(app.state, "bm25_index", None)
    # AFTER (fixed)
    bm25_ok = bm25 is not None and getattr(bm25, "_n", 0) > 0


    llm = getattr(app.state, "llm_provider", None)
    llm_ok = llm is not None

    stt = getattr(app.state, "stt_provider", None)
    stt_ok = stt is not None

    # Determine data source mode
    settings = None
    try:
        from backend.app.core.config import get_settings
        settings = get_settings()
    except Exception:
        pass

    mode = settings.data_source_mode if settings else "unknown"

    index_loaded = chroma_ok or faiss_ok or (mode in ("api", "hybrid"))

    return {
        "status": "healthy" if llm_ok else "degraded",
        "index_loaded": index_loaded,
        "chroma_loaded": chroma_ok,
        "chroma_chunks": chroma_chunks,
        "bm25_loaded": bm25_ok,
        "llm_available": llm_ok,
        "stt_available": stt_ok,
        "data_source_mode": mode,
        "components": {
            "chromadb": "ok" if chroma_ok else ("empty" if chroma_store else "unavailable"),
            "faiss": "ok" if faiss_ok else "unavailable",
            "bm25": "ok" if bm25_ok else "empty",
            "llm": "ok" if llm_ok else "not configured",
            "stt": "ok" if stt_ok else "using browser STT",
        },
        "timestamp": time.time(),
    }
