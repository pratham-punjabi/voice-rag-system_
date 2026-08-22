#!/usr/bin/env python3
"""
Offline comparison of all 5 chunking strategies.

Usage:
    python scripts/compare_chunkers.py
    python scripts/compare_chunkers.py --docs 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_TEXTS = [
    (
        "doc_001",
        """Machine learning is a subset of artificial intelligence that enables systems to learn
        from data without being explicitly programmed. It focuses on developing algorithms that
        can access data and use it to learn for themselves. The process begins with data collection,
        followed by preprocessing and feature engineering. Models are then trained using various
        algorithms such as linear regression, decision trees, or neural networks. After training,
        models are evaluated on held-out test sets using metrics like accuracy, F1, or BLEU.
        Finally, models are deployed to production and monitored for data drift.""",
    ),
    (
        "doc_002",
        """Natural language processing (NLP) is a subfield of linguistics, computer science,
        and artificial intelligence concerned with the interactions between computers and human
        language. It focuses on how to program computers to process and analyze large amounts of
        natural language data. Key tasks include tokenization, part-of-speech tagging, named entity
        recognition, sentiment analysis, machine translation, and question answering.
        Modern NLP systems are based on transformer architectures like BERT, GPT, and T5.
        These models are pre-trained on large corpora and fine-tuned for specific downstream tasks.""",
    ),
    (
        "doc_003",
        "Short document.",  # Edge case: very short
    ),
    (
        "doc_004",
        " ".join([f"word{i}" for i in range(800)]),  # Very long
    ),
]


def evaluate_strategy(strategy_name: str, chunk_size: int = 256, overlap: int = 32) -> dict[str, Any]:
    from backend.app.ingestion.chunkers.factory import get_chunker

    chunker = get_chunker(strategy_name, chunk_size=chunk_size, overlap=overlap)

    all_chunks = []
    t0 = time.perf_counter()

    for doc_id, text in SAMPLE_TEXTS:
        chunks = chunker.chunk(doc_id, text)
        all_chunks.extend(chunks)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not all_chunks:
        return {"strategy": strategy_name, "n_chunks": 0, "error": "no chunks produced"}

    sizes = [c.token_count for c in all_chunks]
    return {
        "strategy": strategy_name,
        "n_chunks": len(all_chunks),
        "avg_tokens": round(sum(sizes) / len(sizes), 1),
        "min_tokens": min(sizes),
        "max_tokens": max(sizes),
        "std_tokens": round((sum((s - sum(sizes) / len(sizes)) ** 2 for s in sizes) / len(sizes)) ** 0.5, 1),
        "total_tokens": sum(sizes),
        "processing_ms": round(elapsed_ms, 2),
        "chunks_per_doc": round(len(all_chunks) / len(SAMPLE_TEXTS), 2),
    }


def main(n_docs: int = 100) -> None:
    strategies = ["fixed", "sentence", "semantic", "metadata", "adaptive"]

    print("\n" + "=" * 70)
    print("  CHUNKING STRATEGY COMPARISON")
    print("=" * 70)
    print(f"  Sample docs  : {len(SAMPLE_TEXTS)}")
    print(f"  Chunk size   : 256 tokens (overlap=32)")
    print()

    results = []
    for strategy in strategies:
        print(f"  Evaluating: {strategy}...", end=" ", flush=True)
        try:
            result = evaluate_strategy(strategy)
            results.append(result)
            print(f"✓  {result['n_chunks']} chunks, avg={result['avg_tokens']} tok, {result['processing_ms']}ms")
        except Exception as exc:
            print(f"✗  {exc}")
            results.append({"strategy": strategy, "error": str(exc)})

    print("\n" + "-" * 70)
    print(f"  {'Strategy':<12} {'Chunks':>7} {'AvgTok':>7} {'MinTok':>7} {'MaxTok':>7} {'ms':>8}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"  {r['strategy']:<12} {'ERROR':>7}")
        else:
            print(f"  {r['strategy']:<12} {r['n_chunks']:>7} {r['avg_tokens']:>7} "
                  f"{r['min_tokens']:>7} {r['max_tokens']:>7} {r['processing_ms']:>8}")
    print("=" * 70)

    # Save report
    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "chunking_comparison.json"
    with open(report_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\n  Report saved → {report_path}\n")

    # Recommendation
    valid = [r for r in results if "error" not in r]
    if valid:
        best = min(valid, key=lambda r: abs(r["avg_tokens"] - 200))
        print(f"  ✅ Recommended strategy for balanced chunk size: {best['strategy'].upper()}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=100)
    args = parser.parse_args()
    main(args.docs)
