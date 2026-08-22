from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Sarvam ───────────────────────────────────────────────────────────────
    sarvam_api_key: str = Field(default="", description="Sarvam API key")
    sarvam_stt_url: str = Field(default="wss://api.sarvam.ai/speech-to-text-translate/ws")
    sarvam_llm_url: str = Field(default="https://api.sarvam.ai/v1")

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["sarvam", "openai", "groq", "local"] = "groq"
    llm_api_key: str = Field(default="", description="LLM API key (Groq/OpenAI)")
    llm_model: str = "llama-3.3-70b-versatile"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1
    llm_timeout_seconds: int = 30

    # ── Groq API (primary) ───────────────────────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_dimension: int = 768
    embedding_batch_size: int = 64
    embedding_cache_dir: str = "data/indexes/embeddings"

    # ── Vector DB (ChromaDB) ─────────────────────────────────────────────────
    vector_db: Literal["chroma", "faiss"] = "chroma"
    chroma_persist_dir: str = "data/indexes/chromadb"
    chroma_collection_name: str = "rag_documents"

    # Keep FAISS settings for backward compat / fallback
    vector_db_path: str = "data/indexes/faiss"
    faiss_nlist: int = 100
    faiss_nprobe: int = 10

    # ── BM25 ─────────────────────────────────────────────────────────────────
    bm25_index_path: str = "data/indexes/bm25/bm25_index.pkl"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 20
    rerank_top_k: int = 5
    final_top_k: int = 5
    enable_reranker: bool = True
    enable_hybrid_search: bool = True
    hybrid_alpha: float = 0.6
    rrf_k: int = 60

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_max_length: int = 512

    # ── Guardrails ────────────────────────────────────────────────────────────
    enable_guardrails: bool = True
    low_confidence_threshold: float = 0.30
    min_retrieval_docs: int = 1
    max_context_tokens: int = 3000

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunking_strategy: Literal["fixed", "sentence", "semantic", "metadata", "adaptive"] = "adaptive"
    chunk_size: int = 256
    chunk_overlap: int = 32
    chunk_min_size: int = 64
    chunk_max_size: int = 512

    # ── Data Source Mode ──────────────────────────────────────────────────────
    # "api"     -> query Groq LLM directly (no pre-indexed data needed)
    # "dataset" -> use pre-indexed ChromaDB + optional dataset ingestion
    # "hybrid"  -> try ChromaDB first; fall back to Groq if no docs found
    data_source_mode: Literal["api", "dataset", "hybrid"] = "hybrid"

    # ── Dataset (optional, used only in dataset/hybrid mode) ─────────────────
    dataset_name: str = "ai4bharat/MSMARCO-XL"
    dataset_split: str = "train"
    dataset_language: str = "en"
    dataset_max_docs: int = 50000
    data_dir: str = "data"

    # ── Groq API RAG settings ─────────────────────────────────────────────────
    groq_rag_system_prompt: str = (
        "You are a precise, knowledgeable assistant. "
        "Answer questions accurately and concisely. "
        "If context is provided, use it and cite it. "
        "If you don't know something, say so clearly. "
        "Format answers in clear markdown. "
        "Return valid JSON: {\"answer\": \"...\", \"confidence\": 0.0-1.0, \"grounded\": true/false}"
    )
    groq_api_rag_max_tokens: int = 1024
    groq_api_rag_temperature: float = 0.1

    # ── API Server ────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    request_timeout_ms: int = 60000
    max_audio_size_bytes: int = 10 * 1024 * 1024
    max_query_length: int = 1000
    rate_limit_requests: int = 120
    rate_limit_window: int = 60

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Cache ─────────────────────────────────────────────────────────────────
    enable_query_cache: bool = True
    query_cache_size: int = 512
    query_cache_ttl: int = 600

    # ── Monitoring ────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    metrics_enabled: bool = True
    store_audio: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def effective_llm_api_key(self) -> str:
        return self.groq_api_key or self.llm_api_key

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.data_dir, "indexes", "metadata.json")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.data_dir, "processed")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.data_dir, "raw")

    @property
    def use_chroma(self) -> bool:
        return self.vector_db == "chroma"

    @property
    def dataset_mode_enabled(self) -> bool:
        return self.data_source_mode in ("dataset", "hybrid")

    @property
    def api_mode_enabled(self) -> bool:
        return self.data_source_mode in ("api", "hybrid")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
