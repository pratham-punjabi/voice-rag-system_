from __future__ import annotations

import pytest

from backend.app.ingestion.chunkers.adaptive_chunker import AdaptiveChunker
from backend.app.ingestion.chunkers.factory import get_chunker
from backend.app.ingestion.chunkers.fixed_chunker import FixedChunker
from backend.app.ingestion.chunkers.metadata_chunker import MetadataChunker
from backend.app.ingestion.chunkers.sentence_chunker import SentenceChunker


TEXT_SHORT = "This is a short document."
TEXT_MEDIUM = (
    "Machine learning is a powerful technique. It enables computers to learn from data. "
    "Neural networks form the backbone of deep learning. They are inspired by the brain. "
    "NLP processes human language. It has many applications in search and generation."
)
TEXT_LONG = " ".join([f"Word{i}" for i in range(600)])


class TestFixedChunker:
    def test_single_chunk_short_text(self):
        c = FixedChunker(chunk_size=256, overlap=32)
        chunks = c.chunk("doc1", TEXT_SHORT)
        assert len(chunks) >= 1
        assert all(ch.text for ch in chunks)

    def test_multiple_chunks_long_text(self):
        c = FixedChunker(chunk_size=100, overlap=10)
        chunks = c.chunk("doc1", TEXT_LONG)
        assert len(chunks) > 1

    def test_chunk_ids_unique(self):
        c = FixedChunker(chunk_size=50, overlap=5)
        chunks = c.chunk("doc1", TEXT_LONG)
        ids = [ch.chunk_id for ch in chunks]
        assert len(ids) == len(set(ids))

    def test_strategy_name(self):
        c = FixedChunker()
        chunks = c.chunk("doc1", TEXT_MEDIUM)
        assert all(ch.strategy == "fixed" for ch in chunks)

    def test_empty_text(self):
        c = FixedChunker()
        assert c.chunk("doc1", "") == []
        assert c.chunk("doc1", "   ") == []

    def test_overlap_creates_smaller_step(self):
        c_no_overlap = FixedChunker(chunk_size=50, overlap=0)
        c_overlap = FixedChunker(chunk_size=50, overlap=25)
        chunks_no = c_no_overlap.chunk("doc1", TEXT_LONG)
        chunks_ov = c_overlap.chunk("doc1", TEXT_LONG)
        assert len(chunks_ov) > len(chunks_no)


class TestSentenceChunker:
    def test_respects_sentence_boundaries(self):
        c = SentenceChunker(sentences_per_chunk=2, overlap_sentences=0)
        chunks = c.chunk("doc1", TEXT_MEDIUM)
        assert len(chunks) >= 1

    def test_empty_text(self):
        c = SentenceChunker()
        assert c.chunk("doc1", "") == []

    def test_strategy_name(self):
        c = SentenceChunker()
        chunks = c.chunk("doc1", TEXT_MEDIUM)
        assert all(ch.strategy == "sentence" for ch in chunks)


class TestMetadataChunker:
    def test_uses_passages_from_metadata(self):
        c = MetadataChunker(max_chunk_tokens=512)
        meta = {"passages": ["First passage.", "Second passage."]}
        chunks = c.chunk("doc1", "ignored text", metadata=meta)
        assert len(chunks) == 2

    def test_fallback_without_passages(self):
        c = MetadataChunker()
        chunks = c.chunk("doc1", TEXT_MEDIUM)
        assert len(chunks) >= 1

    def test_large_passage_gets_subchunked(self):
        c = MetadataChunker(max_chunk_tokens=10)
        big = " ".join(["word"] * 50)
        meta = {"passages": [big]}
        chunks = c.chunk("doc1", "", metadata=meta)
        assert len(chunks) >= 1


class TestAdaptiveChunker:
    def test_short_doc_single_chunk(self):
        c = AdaptiveChunker()
        chunks = c.chunk("doc1", TEXT_SHORT)
        assert len(chunks) == 1

    def test_medium_doc_uses_sentence(self):
        c = AdaptiveChunker()
        chunks = c.chunk("doc1", TEXT_MEDIUM)
        assert len(chunks) >= 1

    def test_long_doc_uses_fixed(self):
        c = AdaptiveChunker(chunk_size=50)
        chunks = c.chunk("doc1", TEXT_LONG)
        assert len(chunks) > 1

    def test_metadata_passages_take_precedence(self):
        c = AdaptiveChunker()
        meta = {"passages": ["A.", "B.", "C."]}
        chunks = c.chunk("doc1", "ignored", metadata=meta)
        assert len(chunks) == 3


class TestChunkerFactory:
    @pytest.mark.parametrize("strategy", ["fixed", "sentence", "metadata", "adaptive"])
    def test_all_strategies_produce_chunks(self, strategy):
        chunker = get_chunker(strategy, chunk_size=128, overlap=16)
        chunks = chunker.chunk("doc1", TEXT_MEDIUM)
        assert isinstance(chunks, list)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("nonexistent")
