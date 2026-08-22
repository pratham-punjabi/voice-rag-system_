from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.endpoints_health import router as health_router
from backend.app.api.endpoints_ingest import router as ingest_router
from backend.app.api.endpoints_metrics import router as metrics_router
from backend.app.api.endpoints_query import router as query_router
from backend.app.api.endpoints_voice import router as voice_router
from backend.app.api.middleware import RateLimitMiddleware, RequestIDMiddleware
from backend.app.core.config import get_settings
from backend.app.core.exceptions import RAGBaseError
from backend.app.lifespan import lifespan

settings = get_settings()

app = FastAPI(
    title="Voice-Enabled Agentic RAG",
    description=(
        "Production RAG system with voice input, ChromaDB vector store, "
        "Groq API integration, hybrid retrieval, and grounded generation.\n\n"
        "**Modes:**\n"
        "- `api` — Groq LLM with full knowledge (no dataset needed)\n"
        "- `dataset` — ChromaDB vector store strictly\n"
        "- `hybrid` — ChromaDB first, Groq fallback (recommended)\n"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window,
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(query_router)
app.include_router(voice_router)
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(metrics_router)


# ── Global Exception Handlers ─────────────────────────────────────────────────
@app.exception_handler(RAGBaseError)
async def rag_error_handler(request: Request, exc: RAGBaseError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": exc.message, "code": exc.code},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "code": "INTERNAL_ERROR"},
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Voice RAG API v2.0",
        "docs": "/docs",
        "health": "/api/health",
        "ingest": "/api/ingest/status",
        "mode": settings.data_source_mode,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=False,
    )
