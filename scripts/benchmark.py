#!/usr/bin/env python3
"""
Latency benchmark for the RAG pipeline.

Usage:
    python scripts/benchmark.py --queries 100
    python scripts/benchmark.py --queries 50 --warmup 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


SAMPLE_QUERIES = [
    "What is machine learning?",
    "How does neural network training work?",
    "What are the applications of natural language processing?",
    "Explain the transformer architecture",
    "What is retrieval augmented generation?",
    "How does BERT work?",
    "What is transfer learning?",
    "Explain attention mechanism in deep learning",
    "What is the difference between supervised and unsupervised learning?",
    "How does gradient descent optimization work?",
    "What are convolutional neural networks used for?",
    "Explain the concept of embeddings in NLP",
    "What is fine-tuning a language model?",
    "How does semantic search differ from keyword search?",
    "What is the role of tokenization in NLP?",
    "Explain reinforcement learning from human feedback",
    "What is a vector database?",
    "How does BM25 ranking work?",
    "What is cross-encoder reranking?",
    "Explain the concept of hallucination in language models",
]


def percentile_stats(data: list[float]) -> dict:
    if not data:
        return {}
    arr = np.array(data)
    return {
        "count": len(data),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p70_ms": float(np.percentile(arr, 70)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "p100_ms": float(np.max(arr)),
    }


async def run_benchmark(n_queries: int = 100, n_warmup: int = 5) -> None:
    from backend.app.core.config import get_settings
    from backend.app.monitoring.logger import setup_logging

    settings = get_settings()
    setup_logging("WARNING", "text")

    from backend.app.agents.generation_agent import GenerationAgent
    from backend.app.agents.grounding_agent import GroundingAgent
    from backend.app.agents.guardrail_agent import GuardrailAgent
    from backend.app.agents.orchestrator import Orchestrator
    from backend.app.agents.query_agent import QueryAgent
    from backend.app.agents.reranker_agent import RerankerAgent
    from backend.app.agents.response_agent import ResponseAgent
    from backend.app.agents.retrieval_agent import RetrievalAgent
    from backend.app.agents.validation_agent import ValidationAgent
    from backend.app.providers.embeddings import SentenceTransformerEmbeddings
    from backend.app.providers.openai_llm import OpenAILLMProvider
    from backend.app.retrieval.bm25 import BM25Index
    from backend.app.retrieval.cache import QueryCache
    from backend.app.retrieval.reranker import CrossEncoderReranker
    from backend.app.retrieval.vector_store import FAISSVectorStore

    print("Loading components...")

    embedder = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
    )
    await embedder.warmup()

    vs = FAISSVectorStore(index_path=settings.vector_db_path, dimension=settings.embedding_dimension)
    try:
        vs.load()
    except Exception as e:
        print(f"⚠️  No FAISS index loaded ({e}). Retrieval will return empty results.")

    bm25 = BM25Index()
    bm25_path = Path(settings.bm25_index_path)
    if bm25_path.exists():
        bm25 = BM25Index.load(settings.bm25_index_path)

    reranker = CrossEncoderReranker(model_name=settings.reranker_model)
    if settings.enable_reranker:
        await reranker.warmup()

    llm = OpenAILLMProvider(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )

    orchestrator = Orchestrator(
        stt_agent=None,
        query_agent=QueryAgent(),
        guardrail_agent=GuardrailAgent(),
        retrieval_agent=RetrievalAgent(
            vector_store=vs,
            bm25_index=bm25,
            embedder=embedder,
            metadata_path=settings.metadata_path,
            top_k=settings.top_k,
            enable_hybrid=settings.enable_hybrid_search,
        ),
        reranker_agent=RerankerAgent(reranker=reranker, top_k=settings.rerank_top_k,
                                     enabled=settings.enable_reranker),
        validation_agent=ValidationAgent(min_confidence=settings.low_confidence_threshold),
        generation_agent=GenerationAgent(llm=llm, max_context_tokens=settings.max_context_tokens),
        grounding_agent=GroundingAgent(),
        response_agent=ResponseAgent(),
        query_cache=None,  # Disable cache for honest benchmarking
    )

    # Build query list by repeating samples
    queries = (SAMPLE_QUERIES * ((n_queries + len(SAMPLE_QUERIES)) // len(SAMPLE_QUERIES)))[:n_queries]

    # Warmup
    print(f"Warming up ({n_warmup} queries)...")
    for q in queries[:n_warmup]:
        await orchestrator.process_text(q)

    # Benchmark
    print(f"Benchmarking {n_queries} queries...")
    measurements: dict[str, list[float]] = {
        "total": [], "embedding": [], "dense_retrieval": [],
        "bm25_retrieval": [], "reranking": [], "generation": [],
        "grounding": [], "query_processing": [], "guardrail": [],
    }

    t_start = time.perf_counter()
    for i, query in enumerate(queries, 1):
        result = await orchestrator.process_text(query)
        lat = result.get("latency_ms", {})
        measurements["total"].append(lat.get("total", 0))
        measurements["embedding"].append(lat.get("embedding", 0))
        measurements["dense_retrieval"].append(lat.get("dense_retrieval", 0))
        measurements["bm25_retrieval"].append(lat.get("bm25_retrieval", 0))
        measurements["reranking"].append(lat.get("reranking", 0))
        measurements["generation"].append(lat.get("generation", 0))
        measurements["grounding"].append(lat.get("grounding", 0))
        measurements["query_processing"].append(lat.get("query_processing", 0))
        measurements["guardrail"].append(lat.get("guardrail", 0))

        if i % 10 == 0:
            print(f"  [{i}/{n_queries}] last total: {lat.get('total', 0):.1f}ms")

    elapsed = time.perf_counter() - t_start
    throughput = n_queries / elapsed

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RAG PERFORMANCE BENCHMARK")
    print("=" * 65)
    print(f"  Queries     : {n_queries}")
    print(f"  Warmup      : {n_warmup}")
    print(f"  Wall time   : {elapsed:.1f}s")
    print(f"  Throughput  : {throughput:.2f} req/s")
    print()

    stats = {k: percentile_stats(v) for k, v in measurements.items()}

    component_order = [
        ("total", "TOTAL PIPELINE"),
        ("embedding", "Embedding"),
        ("dense_retrieval", "Dense Retrieval"),
        ("bm25_retrieval", "BM25 Retrieval"),
        ("reranking", "Reranking"),
        ("generation", "LLM Generation"),
        ("grounding", "Grounding"),
        ("query_processing", "Query Processing"),
        ("guardrail", "Guardrail"),
    ]

    for key, label in component_order:
        s = stats[key]
        if not s:
            continue
        print(f"  {label}")
        print(f"    P50={s['p50_ms']:.1f}ms  P70={s['p70_ms']:.1f}ms  "
              f"P90={s['p90_ms']:.1f}ms  P95={s['p95_ms']:.1f}ms  "
              f"P99={s['p99_ms']:.1f}ms  P100={s['p100_ms']:.1f}ms")
        print(f"    mean={s['mean_ms']:.1f}ms  min={s['min_ms']:.1f}ms  max={s['max_ms']:.1f}ms")
        print()

    # Bottleneck
    by_p95 = sorted(
        [(k, stats[k].get("p95_ms", 0)) for k in measurements if k != "total"],
        key=lambda x: x[1], reverse=True,
    )
    print(f"  ⚡ Slowest component (P95): {by_p95[0][0]} ({by_p95[0][1]:.1f}ms)")
    print("=" * 65)

    # Save raw results
    out = {
        "config": {
            "n_queries": n_queries,
            "n_warmup": n_warmup,
            "model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "reranker_enabled": settings.enable_reranker,
            "hybrid_search": settings.enable_hybrid_search,
        },
        "throughput_rps": throughput,
        "wall_time_seconds": elapsed,
        "stats": stats,
        "raw": measurements,
    }
    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Benchmark")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run_benchmark(n_queries=args.queries, n_warmup=args.warmup))
