from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Literal

from backend.app.models.request_context import RequestContext, RetrievedDocument
from backend.app.providers.base_embeddings import EmbeddingProvider
from backend.app.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Hybrid retrieval agent supporting:
    - ChromaDB (primary vector store with persistence)
    - FAISS (legacy fallback)
    - BM25 keyword search
    - Hybrid RRF fusion
    - No-index pass-through for pure API mode
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        top_k: int = 20,
        enable_hybrid: bool = True,
        hybrid_alpha: float = 0.6,
        rrf_k: int = 60,
        mode: Literal["api", "dataset", "hybrid"] = "hybrid",
        # ChromaDB
        chroma_store=None,
        # FAISS (optional fallback)
        vector_store=None,
        bm25_index: BM25Index | None = None,
        metadata_path: str = "data/indexes/metadata.json",
    ) -> None:
        self._embedder = embedder
        self._top_k = top_k
        self._enable_hybrid = enable_hybrid
        self._hybrid_alpha = hybrid_alpha
        self._rrf_k = rrf_k
        self._mode = mode
        self._chroma = chroma_store
        self._vs = vector_store
        self._bm25 = bm25_index
        self._metadata: dict[str, dict] = {}

        if metadata_path:
            self._load_metadata(metadata_path)

    def _load_metadata(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.info("Metadata file not found: %s (will use ChromaDB metadata)", path)
            return
        try:
            with open(p, encoding="utf-8") as f:
                records = json.load(f)
            self._metadata = {r["chunk_id"]: r for r in records}
            logger.info("Metadata loaded: %d chunks", len(self._metadata))
        except Exception as exc:
            logger.warning("Failed to load metadata: %s", exc)

    def _has_indexed_data(self) -> bool:
        """Check whether any vector store has data."""
        if self._chroma and self._chroma.is_loaded:
            return True
        if self._vs and self._vs.is_loaded:
            return True
        return False

    def _rrf_fuse(
        self,
        dense_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
    ) -> list[tuple[str, float, float, float]]:
        """Reciprocal Rank Fusion."""
        k = self._rrf_k
        scores: dict[str, float] = {}
        dense_map: dict[str, float] = {}
        bm25_map: dict[str, float] = {}

        for rank, (cid, score) in enumerate(dense_results):
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            dense_map[cid] = score

        for rank, (cid, score) in enumerate(bm25_results):
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
            bm25_map[cid] = score

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            (cid, score, dense_map.get(cid, 0.0), bm25_map.get(cid, 0.0))
            for cid, score in fused
        ]

    async def _search_chroma(self, query_vec, top_k: int) -> list[tuple[str, float, str, dict]]:
        """Search ChromaDB asynchronously."""
        if not self._chroma:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._chroma.search, query_vec, top_k
        )

    async def _search_faiss(self, query_vec, top_k: int) -> list[tuple[str, float]]:
        """Search FAISS asynchronously."""
        if not self._vs or not self._vs.is_loaded:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._vs.search, query_vec, top_k
        )

    async def _search_bm25(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Search BM25 asynchronously."""
        if not self._bm25:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._bm25.search, query, top_k
        )

    def _build_doc_from_chroma(
        self, chunk_id: str, score: float, doc_text: str, meta: dict
    ) -> RetrievedDocument:
        return RetrievedDocument(
            doc_id=meta.get("doc_id", chunk_id),
            chunk_id=chunk_id,
            text=doc_text or meta.get("text", ""),
            score=score,
            dense_score=score,
            bm25_score=0.0,
            metadata={k: v for k, v in meta.items() if k not in {"chunk_id", "doc_id", "text"}},
        )

    def _build_doc_from_metadata(
        self, chunk_id: str, score: float, dense_score: float, bm25_score: float
    ) -> RetrievedDocument:
        meta = self._metadata.get(chunk_id, {})
        return RetrievedDocument(
            doc_id=meta.get("doc_id", chunk_id),
            chunk_id=chunk_id,
            text=meta.get("text", ""),
            score=score,
            dense_score=dense_score,
            bm25_score=bm25_score,
            metadata={k: v for k, v in meta.items() if k not in {"chunk_id", "doc_id", "text"}},
        )

    async def process(self, ctx: RequestContext) -> RequestContext:
        query = ctx.query_info.normalized_query

        # In pure API mode with no indexed data, skip retrieval
        if self._mode == "api" and not self._has_indexed_data():
            logger.info("API mode: no indexed data, skipping retrieval")
            ctx.candidate_documents = []
            return ctx

        # Embed query
        t_embed = time.perf_counter()
        query_vec = await self._embedder.embed_query(query)
        ctx.latency.embedding_ms = (time.perf_counter() - t_embed) * 1000

        t0 = time.perf_counter()
        docs: list[RetrievedDocument] = []

        # ── ChromaDB path ──────────────────────────────────────────────────
        if self._chroma and self._chroma.is_loaded:
            if self._enable_hybrid and self._bm25:
                chroma_task = self._search_chroma(query_vec, self._top_k)
                bm25_task = self._search_bm25(query, self._top_k)
                chroma_results, bm25_results = await asyncio.gather(chroma_task, bm25_task)

                # Convert chroma results to (id, score) for fusion
                dense_pairs = [(cid, score) for cid, score, _, _ in chroma_results]
                fused = self._rrf_fuse(dense_pairs, bm25_results)

                # Re-attach text from chroma results
                chroma_map = {cid: (txt, meta) for cid, _, txt, meta in chroma_results}
                for cid, fused_score, dense_score, bm25_score in fused[: self._top_k]:
                    if cid in chroma_map:
                        txt, meta = chroma_map[cid]
                        doc = self._build_doc_from_chroma(cid, fused_score, txt, meta)
                        doc.dense_score = dense_score
                        doc.bm25_score = bm25_score
                    else:
                        doc = self._build_doc_from_metadata(cid, fused_score, dense_score, bm25_score)
                    docs.append(doc)
            else:
                chroma_results = await self._search_chroma(query_vec, self._top_k)
                for cid, score, txt, meta in chroma_results:
                    docs.append(self._build_doc_from_chroma(cid, score, txt, meta))

        # ── FAISS fallback path ────────────────────────────────────────────
        elif self._vs and self._vs.is_loaded:
            if self._enable_hybrid and self._bm25:
                dense_task = self._search_faiss(query_vec, self._top_k)
                bm25_task = self._search_bm25(query, self._top_k)
                dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)
                fused = self._rrf_fuse(dense_results, bm25_results)
                for cid, fused_score, dense_score, bm25_score in fused[: self._top_k]:
                    docs.append(self._build_doc_from_metadata(cid, fused_score, dense_score, bm25_score))
            else:
                faiss_results = await self._search_faiss(query_vec, self._top_k)
                for cid, score in faiss_results:
                    docs.append(self._build_doc_from_metadata(cid, score, score, 0.0))

        ctx.latency.dense_retrieval_ms = (time.perf_counter() - t0) * 1000
        ctx.candidate_documents = docs

        logger.info(
            "Retrieval complete",
            extra={
                "request_id": ctx.request_id,
                "mode": self._mode,
                "n_candidates": len(docs),
                "top_score": docs[0].score if docs else 0,
                "embed_ms": ctx.latency.embedding_ms,
                "retrieval_ms": ctx.latency.dense_retrieval_ms,
            },
        )
        return ctx
