#!/usr/bin/env python3
"""
Download and cache the MSMARCO-XL dataset without running full ingestion.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --dataset ai4bharat/MSMARCO-XL --split train
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset to local HF cache")
    parser.add_argument("--dataset", default="ai4bharat/MSMARCO-XL")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    print(f"\n  Downloading: {args.dataset} [{args.split}]")
    print("  This may take several minutes on first run…\n")

    t0 = time.time()
    try:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, split=args.split)
        elapsed = time.time() - t0
        print(f"  ✅ Dataset downloaded successfully!")
        print(f"     Rows     : {len(ds):,}")
        print(f"     Fields   : {list(ds.features.keys())}")
        print(f"     Time     : {elapsed:.1f}s")
        print(f"     Cache    : ~/.cache/huggingface/datasets/\n")
    except Exception as exc:
        print(f"  ✗ Download failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
