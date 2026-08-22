#!/usr/bin/env python3
"""
Warm up all models (embedder + reranker) to eliminate cold-start latency.
Run once before benchmarking or in production startup.

Usage:
    python scripts/warmup.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> None:
    from backend.app.core.config import get_settings
    from backend.app.monitoring.logger import setup_logging
    from backend.app.providers.embeddings import SentenceTransformerEmbeddings
    from backend.app.retrieval.reranker import CrossEncoderReranker

    settings = get_settings()
    setup_logging("INFO", "text")

    print("\n  Warming up models…\n")

    # Embedder
    print("  [1/2] Loading embedding model…")
    embedder = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
    )
    await embedder.warmup()
    print(f"        ✓  Dimension: {embedder.dimension()}")

    # Reranker
    if settings.enable_reranker:
        print("  [2/2] Loading reranker model…")
        reranker = CrossEncoderReranker(model_name=settings.reranker_model)
        await reranker.warmup()
        status = "✓  Ready" if reranker.available else "⚠  Unavailable (will fallback)"
        print(f"        {status}")
    else:
        print("  [2/2] Reranker disabled (ENABLE_RERANKER=false)")

    print("\n  ✅ Warmup complete. Models are in memory.\n")


if __name__ == "__main__":
    asyncio.run(main())
