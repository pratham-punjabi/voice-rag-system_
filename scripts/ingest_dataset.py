#!/usr/bin/env python3
"""
Run the full offline ingestion pipeline.

Usage:
    python -m scripts.ingest_dataset
    python scripts/ingest_dataset.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.config import get_settings
from backend.app.ingestion.indexer import run_ingestion
from backend.app.monitoring.logger import setup_logging


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level, "text")

    print("=" * 60)
    print("  VOICE RAG — DATASET INGESTION")
    print("=" * 60)
    print(f"  Dataset  : {settings.dataset_name}")
    print(f"  Split    : {settings.dataset_split}")
    print(f"  Max docs : {settings.dataset_max_docs:,}")
    print(f"  Strategy : {settings.chunking_strategy}")
    print(f"  Chunk sz : {settings.chunk_size} tokens (overlap {settings.chunk_overlap})")
    print("=" * 60)

    stats = await run_ingestion(settings)
    report = stats.report()

    print("\n✅  Ingestion complete!")
    print(f"  Documents     : {report['total_documents']:,}")
    print(f"  Chunks        : {report['total_chunks']:,}")
    print(f"  Duplicates    : {report['duplicate_documents']:,}")
    print(f"  Empty docs    : {report['empty_documents']:,}")
    print(f"  Avg chunk tok : {report['avg_chunk_tokens']:.1f}")
    print(f"  Min/Max tok   : {report['min_chunk_tokens']} / {report['max_chunk_tokens']}")
    print(f"  Time          : {report['processing_time_seconds']:.1f}s")
    print(f"  Throughput    : {report['throughput_docs_per_second']:.1f} docs/s")


if __name__ == "__main__":
    asyncio.run(main())
