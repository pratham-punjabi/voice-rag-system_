from __future__ import annotations

"""
Standalone schema inspection utility.
Can be run directly to explore a dataset before ingestion.

Usage:
    python -m backend.app.ingestion.schema_inspector
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def print_schema_report(dataset_name: str, split: str = "train", n_samples: int = 5) -> None:
    """Download a small sample and print a detailed schema report."""
    from datasets import load_dataset
    from backend.app.ingestion.dataset_loader import inspect_schema

    print(f"\n{'='*60}")
    print(f"  Schema Inspection: {dataset_name} [{split}]")
    print(f"{'='*60}")

    try:
        ds = load_dataset(dataset_name, split=split, streaming=False)
    except Exception:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

    schema = inspect_schema(ds)

    print(f"\n  Total rows       : {schema.total_rows:,}")
    print(f"  All fields       : {schema.all_fields}")
    print(f"\n  Detected mappings:")
    print(f"    text_field     : {schema.text_field}")
    print(f"    id_field       : {schema.id_field}")
    print(f"    title_field    : {schema.title_field}")
    print(f"    language_field : {schema.language_field}")
    print(f"    query_field    : {schema.query_field}")
    print(f"    relevance_field: {schema.relevance_field}")
    print(f"    sample_languages: {schema.sample_languages}")

    print(f"\n  Sample rows ({n_samples}):")
    for i, row in enumerate(ds.select(range(n_samples))):
        print(f"\n  Row {i}:")
        for k, v in row.items():
            val_str = str(v)
            if len(val_str) > 120:
                val_str = val_str[:120] + "…"
            print(f"    {k:20s}: {val_str}")

    # Field value stats
    print(f"\n  Field statistics (first 1000 rows):")
    sample = ds.select(range(min(1000, len(ds))))

    text_lens = [len(str(row.get(schema.text_field, ""))) for row in sample]
    empty = sum(1 for l in text_lens if l == 0)
    print(f"    {schema.text_field:20s}: avg_len={sum(text_lens)//max(1,len(text_lens))}, "
          f"min={min(text_lens)}, max={max(text_lens)}, empty={empty}")

    if schema.language_field:
        lang_counts: dict[str, int] = {}
        for row in sample:
            lang = str(row.get(schema.language_field, "unknown"))
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        top_langs = sorted(lang_counts.items(), key=lambda x: -x[1])[:5]
        print(f"    {schema.language_field:20s}: top_values={top_langs}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "ai4bharat/MSMARCO-XL"
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    print_schema_report(name, split)
