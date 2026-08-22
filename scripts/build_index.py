#!/usr/bin/env python3
"""
Rebuild FAISS and BM25 indexes from already-processed chunks.
Use this when you want to change index parameters without re-embedding.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --nlist 200 --nprobe 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main(nlist: int = 100, nprobe: int = 10) -> None:
    from backend.app.core.config import get_settings
    from backend.app.monitoring.logger import setup_logging
    from backend.app.providers.embeddings import SentenceTransformerEmbeddings
    from backend.app.retrieval.bm25 import BM25Index
    from backend.app.retrieval.vector_store import FAISSVectorStore

    settings = get_settings()
    setup_logging(settings.log_level, "text")
    logger = logging.getLogger(__name__)

    meta_path = Path(settings.metadata_path)
    if not meta_path.exists():
        print(f"✗ Metadata not found at {meta_path}. Run ingestion first.")
        sys.exit(1)

    print(f"\n  Loading metadata from {meta_path}…")
    with open(meta_path, encoding="utf-8") as f:
        records = json.load(f)

    print(f"  Found {len(records):,} chunks")

    texts = [r["text"] for r in records]
    ids = [r["chunk_id"] for r in records]

    # Re-embed (or load cached)
    embedder = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
    )

    cached = embedder.load_embeddings("chunks")
    if cached is not None and len(cached) == len(records):
        print(f"  Using cached embeddings ({len(cached):,} vectors)")
        embeddings = cached
    else:
        print(f"  Re-computing embeddings…")
        t0 = time.time()
        embeddings = await embedder.embed_documents(texts)
        embedder.save_embeddings(embeddings, "chunks")
        print(f"  Embedded in {time.time()-t0:.1f}s")

    # FAISS
    print(f"  Building FAISS index (nlist={nlist})…")
    t0 = time.time()
    vs = FAISSVectorStore(
        index_path=settings.vector_db_path,
        dimension=embeddings.shape[1],
        nlist=nlist,
        nprobe=nprobe,
    )
    vs.build(embeddings, ids)
    vs.save()
    print(f"  FAISS built in {time.time()-t0:.1f}s → {settings.vector_db_path}")

    # BM25
    print(f"  Building BM25 index…")
    t0 = time.time()
    bm25 = BM25Index(k1=settings.bm25_k1, b=settings.bm25_b)
    bm25.build(texts, ids)
    bm25.save(settings.bm25_index_path)
    print(f"  BM25 built in {time.time()-t0:.1f}s → {settings.bm25_index_path}")

    print(f"\n  ✅ Indexes ready. Start the server: make dev-backend\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main(nlist=args.nlist, nprobe=args.nprobe))
