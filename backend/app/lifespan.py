from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI

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
from backend.app.core.config import get_settings
from backend.app.monitoring.logger import setup_logging
from backend.app.providers.embeddings import SentenceTransformerEmbeddings
from backend.app.providers.groq_llm import GroqLLMProvider
from backend.app.providers.openai_llm import OpenAILLMProvider
from backend.app.providers.sarvam_stt import SarvamSTTProvider
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.cache import QueryCache
from backend.app.retrieval.chroma_store import ChromaVectorStore
from backend.app.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    logger.info(
        "Starting Voice RAG system | mode=%s | llm_provider=%s | vector_db=%s",
        settings.data_source_mode,
        settings.llm_provider,
        settings.vector_db,
    )

    # ── Embedder ──────────────────────────────────────────────────────────
    embedder = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
        batch_size=settings.embedding_batch_size,
    )
    await embedder.warmup()
    app.state.embedder = embedder

    # ── ChromaDB (primary vector store) ───────────────────────────────────
    chroma_store = None
    if settings.use_chroma:
        chroma_store = ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
            dimension=settings.embedding_dimension,
        )
        count = chroma_store.count()
        if count > 0:
            logger.info("ChromaDB loaded: %d chunks in collection '%s'", count, settings.chroma_collection_name)
        else:
            logger.info(
                "ChromaDB empty — operating in API mode (use /api/ingest to add documents)"
            )
    app.state.chroma_store = chroma_store

    # ── FAISS (optional legacy fallback) ──────────────────────────────────
    vector_store = None
    if not settings.use_chroma:
        from backend.app.retrieval.vector_store import FAISSVectorStore
        vector_store = FAISSVectorStore(
            index_path=settings.vector_db_path,
            dimension=settings.embedding_dimension,
            nprobe=settings.faiss_nprobe,
        )
        try:
            vector_store.load()
            logger.info("FAISS index loaded")
        except Exception as exc:
            logger.warning("FAISS index not available: %s", exc)
    app.state.vector_store = vector_store

    # ── BM25 ──────────────────────────────────────────────────────────────
    bm25 = None
    bm25_path = Path(settings.bm25_index_path)
    if bm25_path.exists():
        bm25 = BM25Index.load(settings.bm25_index_path)
        logger.info("BM25 index loaded")
    else:
        logger.info("BM25 index not found — keyword search disabled")
        bm25 = BM25Index()  # empty fallback
    app.state.bm25_index = bm25

    # ── Reranker ──────────────────────────────────────────────────────────
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        max_length=settings.reranker_max_length,
    )
    if settings.enable_reranker:
        await reranker.warmup()
    app.state.reranker = reranker

    # ── LLM Provider (Groq preferred, fallback to generic OpenAI-compat) ──
    llm = None
    api_key = settings.effective_llm_api_key
    if api_key:
        if settings.llm_provider == "groq" or settings.groq_api_key:
            llm = GroqLLMProvider(
                api_key=api_key,
                model=settings.groq_model if settings.groq_api_key else settings.llm_model,
                base_url=settings.groq_base_url if settings.groq_api_key else settings.llm_base_url,
                timeout_seconds=settings.llm_timeout_seconds,
            )
            logger.info("Groq LLM provider initialized: model=%s", settings.groq_model)
        else:
            llm = OpenAILLMProvider(
                api_key=api_key,
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                timeout_seconds=settings.llm_timeout_seconds,
            )
            logger.info("OpenAI-compat LLM provider initialized: model=%s", settings.llm_model)

        healthy = await llm.health_check()
        logger.info("LLM health check: %s", "OK" if healthy else "FAILED")
    else:
        logger.warning("No LLM API key configured. Query answering will be disabled.")

    app.state.llm_provider = llm

    # ── STT Provider ──────────────────────────────────────────────────────
    stt_provider = None
    if settings.sarvam_api_key:
        stt_provider = SarvamSTTProvider(api_key=settings.sarvam_api_key)
        logger.info("Sarvam STT provider initialized")
    else:
        logger.info("No Sarvam STT key configured — using browser Web Speech API for STT")
    app.state.stt_provider = stt_provider

    # ── Cache ─────────────────────────────────────────────────────────────
    cache = (
        QueryCache(
            max_size=settings.query_cache_size,
            ttl_seconds=settings.query_cache_ttl,
        )
        if settings.enable_query_cache
        else None
    )
    app.state.query_cache = cache

    # ── Agents ────────────────────────────────────────────────────────────
    stt_agent = STTAgent(stt_provider) if stt_provider else None

    retrieval_agent = RetrievalAgent(
        embedder=embedder,
        top_k=settings.top_k,
        enable_hybrid=settings.enable_hybrid_search,
        hybrid_alpha=settings.hybrid_alpha,
        rrf_k=settings.rrf_k,
        mode=settings.data_source_mode,
        chroma_store=chroma_store,
        vector_store=vector_store,
        bm25_index=bm25,
        metadata_path=settings.metadata_path,
    )

    reranker_agent = RerankerAgent(
        reranker=reranker,
        top_k=settings.rerank_top_k,
        enabled=settings.enable_reranker,
    )

    generation_agent = GenerationAgent(
        llm=llm,
        max_context_tokens=settings.max_context_tokens,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        mode=settings.data_source_mode,
    )

    orchestrator = Orchestrator(
        stt_agent=stt_agent,
        query_agent=QueryAgent(),
        guardrail_agent=GuardrailAgent(),
        retrieval_agent=retrieval_agent,
        reranker_agent=reranker_agent,
        validation_agent=ValidationAgent(
            min_docs=settings.min_retrieval_docs,
            min_confidence=settings.low_confidence_threshold,
        ),
        generation_agent=generation_agent,
        grounding_agent=GroundingAgent(),
        response_agent=ResponseAgent(),
        query_cache=cache,
        mode=settings.data_source_mode,
    )
    app.state.orchestrator = orchestrator

    logger.info("All components initialised. Server ready.")
    logger.info(
        "Data source mode: %s | ChromaDB chunks: %d | BM25: %s",
        settings.data_source_mode,
        chroma_store.count() if chroma_store else 0,
        "loaded" if bm25 and bm25._n > 0 else "empty",
    )

    yield  # ── Application runs ────────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Shutting down...")
    if stt_provider:
        await stt_provider.close()
    if llm and hasattr(llm, "close"):
        await llm.close()
    logger.info("Shutdown complete.")
