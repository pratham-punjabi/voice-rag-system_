from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.app.ingestion.chunkers.factory import get_chunker
from backend.app.ingestion.dataset_loader import load_dataset_streaming, row_to_document
from backend.app.models.chunk import Chunk

logger = logging.getLogger(__name__)


class IngestionStats:
    def __init__(self) -> None:
        self.total_docs = 0
        self.total_chunks = 0
        self.duplicate_docs = 0
        self.empty_docs = 0
        self.chunk_sizes: list[int] = []
        self.start_time = time.time()

    def report(self) -> dict[str, Any]:
        elapsed = time.time() - self.start_time
        return {
            "total_documents": self.total_docs,
            "total_chunks": self.total_chunks,
            "duplicate_documents": self.duplicate_docs,
            "empty_documents": self.empty_docs,
            "avg_chunk_tokens": sum(self.chunk_sizes) / max(1, len(self.chunk_sizes)),
            "min_chunk_tokens": min(self.chunk_sizes, default=0),
            "max_chunk_tokens": max(self.chunk_sizes, default=0),
            "processing_time_seconds": elapsed,
            "throughput_docs_per_second": self.total_docs / max(1, elapsed),
        }


async def run_ingestion(settings: Settings) -> IngestionStats:
    """
    Full offline ingestion pipeline:
    1. Load + inspect dataset (HuggingFace or local files)
    2. Clean + deduplicate
    3. Chunk with selected strategy
    4. Embed all chunks
    5. Build ChromaDB index (primary) + optionally BM25
    6. Save metadata JSON
    """
    from backend.app.providers.embeddings import SentenceTransformerEmbeddings
    from backend.app.retrieval.chroma_store import ChromaVectorStore
    from backend.app.retrieval.bm25 import BM25Index

    stats = IngestionStats()
    Path(settings.processed_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.embedding_cache_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Load Dataset ──────────────────────────────────────────────────────
    logger.info("Step 1/6: Loading dataset '%s'...", settings.dataset_name)
    ds, schema = load_dataset_streaming(
        settings.dataset_name,
        split=settings.dataset_split,
        language_filter=settings.dataset_language,
    )

    # ── 2. Clean + Deduplicate ───────────────────────────────────────────────
    logger.info("Step 2/6: Cleaning and chunking...")
    chunker = get_chunker(
        strategy=settings.chunking_strategy,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    seen_ids: set[str] = set()
    all_chunks: list[Chunk] = []
    chunk_metadata: list[dict[str, Any]] = []

    for i, row in enumerate(ds):
        if settings.dataset_max_docs and i >= settings.dataset_max_docs:
            logger.info("Reached dataset_max_docs limit (%d), stopping.", settings.dataset_max_docs)
            break

        doc = row_to_document(row, schema)
        doc_id = doc["doc_id"] or f"doc_{i}"
        text = doc["text"]

        if not text or len(text.strip()) < 10:
            stats.empty_docs += 1
            continue

        if doc_id in seen_ids:
            stats.duplicate_docs += 1
            continue
        seen_ids.add(doc_id)
        stats.total_docs += 1

        chunks = chunker.chunk(doc_id, text, doc.get("metadata", {}))
        for chunk in chunks:
            if chunk.text:
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "title": doc.get("title", ""),
                    "language": doc.get("language", "en"),
                    "chunk_index": chunk.chunk_index,
                    "strategy": chunk.strategy,
                    "metadata": chunk.metadata,
                })
                stats.chunk_sizes.append(chunk.token_count)
                stats.total_chunks += 1

        if (i + 1) % 1000 == 0:
            logger.info("Processed %d docs, %d chunks so far", i + 1, stats.total_chunks)

    logger.info("Total chunks to embed: %d", len(all_chunks))

    # ── 3. Embed ─────────────────────────────────────────────────────────────
    logger.info("Step 3/6: Generating embeddings...")
    embedder = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
        batch_size=settings.embedding_batch_size,
    )
    texts = [c.text for c in all_chunks]
    embeddings = await embedder.embed_documents(texts)

    # ── 4. Build ChromaDB Index ───────────────────────────────────────────────
    logger.info("Step 4/6: Building ChromaDB index...")
    chroma_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        dimension=settings.embedding_dimension,
    )

    ids = [c.chunk_id for c in all_chunks]
    documents = [c.text for c in all_chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "strategy": c.strategy,
            "text": c.text[:500],  # Store snippet in metadata too
        }
        for c in all_chunks
    ]

    chroma_store.upsert(ids, embeddings, documents, metadatas)
    logger.info("ChromaDB index built with %d chunks", len(all_chunks))

    # ── 5. Build BM25 Index ───────────────────────────────────────────────────
    logger.info("Step 5/6: Building BM25 index...")
    Path(settings.bm25_index_path).parent.mkdir(parents=True, exist_ok=True)
    bm25 = BM25Index()
    bm25.build(texts, ids)
    bm25.save(settings.bm25_index_path)

    # ── 6. Save Metadata ──────────────────────────────────────────────────────
    logger.info("Step 6/6: Saving metadata...")
    meta_path = Path(settings.metadata_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)

    report = stats.report()
    report.update({
        "embedding_dimension": int(embeddings.shape[1]),
        "index_size": len(all_chunks),
        "chroma_collection": settings.chroma_collection_name,
        "chroma_persist_dir": settings.chroma_persist_dir,
        "metadata_path": str(meta_path),
        "schema": {
            "text_field": schema.text_field,
            "id_field": schema.id_field,
            "language_field": schema.language_field,
            "total_rows": schema.total_rows,
        },
    })

    report_path = Path(settings.processed_dir) / "ingestion_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Ingestion complete: %s", report)
    return stats


async def ingest_documents(
    documents: list[dict],
    settings: Settings,
    replace: bool = False,
) -> dict[str, Any]:
    """
    Ingest a list of raw documents (from API or files) into ChromaDB.
    Each document: {"id": str, "text": str, "title": str, "metadata": dict}
    """
    from backend.app.providers.embeddings import SentenceTransformerEmbeddings
    from backend.app.retrieval.chroma_store import ChromaVectorStore
    from backend.app.retrieval.bm25 import BM25Index

    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    chunker = get_chunker(
        strategy=settings.chunking_strategy,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    all_chunks: list[Chunk] = []
    chunk_meta: list[dict] = []

    for doc in documents:
        doc_id = doc.get("id", f"doc_{len(all_chunks)}")
        text = doc.get("text", "").strip()
        if not text:
            continue
        chunks = chunker.chunk(doc_id, text, doc.get("metadata", {}))
        for c in chunks:
            if c.text:
                all_chunks.append(c)
                chunk_meta.append({
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "title": doc.get("title", ""),
                })

    if not all_chunks:
        return {"status": "no_chunks", "total_chunks": 0}

    embedder = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
        batch_size=settings.embedding_batch_size,
    )
    texts = [c.text for c in all_chunks]
    embeddings = await embedder.embed_documents(texts)

    chroma_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        dimension=settings.embedding_dimension,
    )

    if replace:
        chroma_store.delete_collection()

    ids = [c.chunk_id for c in all_chunks]
    metadatas = [{"doc_id": c.doc_id, "text": c.text[:500]} for c in all_chunks]
    chroma_store.upsert(ids, embeddings, texts, metadatas)

    return {
        "status": "success",
        "total_documents": len(documents),
        "total_chunks": len(all_chunks),
        "chroma_collection": settings.chroma_collection_name,
    }
