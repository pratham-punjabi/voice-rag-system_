from __future__ import annotations

import pytest

from backend.app.agents.query_agent import (
    QueryAgent,
    _normalize,
    _detect_language,
    _detect_intent,
    _extract_keywords,
    _extract_entities,
    _is_valid_query,
)
from backend.app.models.request_context import RequestContext


class TestNormalize:

    def test_removes_filler_words(self):
        result = _normalize("um so like what is machine learning you know")
        assert "um" not in result.lower()
        assert "like" not in result.lower()
        assert "you know" not in result.lower()

    def test_collapses_whitespace(self):
        result = _normalize("what   is    NLP")
        assert "  " not in result

    def test_capitalises_first_letter(self):
        result = _normalize("what is bert?")
        assert result[0].isupper()

    def test_strips_leading_trailing(self):
        result = _normalize("  what is bert  ")
        assert result == result.strip()

    def test_preserves_entity_words(self):
        result = _normalize("tell me about BERT and GPT")
        assert "bert" in result.lower() or "gpt" in result.lower()

    def test_handles_empty(self):
        assert _normalize("") == ""

    def test_handles_only_fillers(self):
        result = _normalize("um uh er ah")
        # All fillers removed — result is empty or near-empty
        assert len(result.strip()) < 5


class TestLanguageDetection:

    @pytest.mark.parametrize("text,expected_lang,expected_mixed", [
        ("What is machine learning?", "en", False),
        ("मशीन लर्निंग क्या है?", "hi", False),
        ("Machine learning kya hai yaar", "en", False),   # all latin
    ])
    def test_language_detected(self, text, expected_lang, expected_mixed):
        lang, is_mixed = _detect_language(text)
        assert lang == expected_lang

    def test_devanagari_detected(self):
        lang, _ = _detect_language("यह एक परीक्षण है")
        assert lang == "hi"

    def test_mixed_script_detected(self):
        _, is_mixed = _detect_language("मुझे NLP के बारे में बताओ")
        assert is_mixed is True

    def test_english_not_mixed(self):
        _, is_mixed = _detect_language("What is BERT?")
        assert is_mixed is False


class TestIntentDetection:

    @pytest.mark.parametrize("text,expected", [
        ("What is retrieval augmented generation?", "definition"),
        ("define machine learning", "definition"),
        ("How to fine-tune a language model?", "procedural"),
        ("steps to build a RAG pipeline", "procedural"),
        ("Who invented the transformer?", "factual"),
        ("When was BERT released?", "factual"),
        ("Why does attention work so well?", "causal"),
        ("reason behind transformer success", "causal"),
        ("Compare BERT vs GPT", "comparative"),
        ("difference between supervised and unsupervised", "comparative"),
        ("Tell me about embeddings", "general"),
    ])
    def test_intent_detection(self, text, expected):
        assert _detect_intent(text) == expected


class TestKeywordExtraction:

    def test_extracts_content_words(self):
        kws = _extract_keywords("What is natural language processing used for?")
        assert any(k in kws for k in ["natural", "language", "processing"])

    def test_filters_stopwords(self):
        kws = _extract_keywords("What is the answer to this question?")
        assert "what" not in kws
        assert "the" not in kws
        assert "is" not in kws

    def test_max_10_keywords(self):
        long_text = " ".join([f"keyword{i}" for i in range(50)])
        kws = _extract_keywords(long_text)
        assert len(kws) <= 10

    def test_empty_text(self):
        assert _extract_keywords("") == []


class TestEntityExtraction:

    def test_proper_nouns_extracted(self):
        entities = _extract_entities("Google released BERT for NLP tasks")
        assert any("BERT" in e or "Google" in e for e in entities)

    def test_max_5_entities(self):
        text = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa"
        entities = _extract_entities(text)
        assert len(entities) <= 5

    def test_empty_text(self):
        assert _extract_entities("") == []


class TestIsValidQuery:

    @pytest.mark.parametrize("text", [
        "What is machine learning?",
        "How does BERT work in NLP?",
        "Explain retrieval augmented generation",
        "मशीन लर्निंग क्या है",
    ])
    def test_valid_queries(self, text):
        assert _is_valid_query(text) is True

    @pytest.mark.parametrize("text", [
        "",
        " ",
        "hi",
        "ok",
        "123",
        "!!??",
    ])
    def test_invalid_queries(self, text):
        assert _is_valid_query(text) is False


class TestQueryAgentE2E:

    @pytest.mark.asyncio
    async def test_processes_english_query(self):
        agent = QueryAgent()
        ctx = RequestContext()
        ctx.transcript = "What is machine learning?"

        result = await agent.process(ctx)

        assert result.query_info.normalized_query
        assert result.query_info.language == "en"
        assert result.query_info.is_valid is True
        assert result.query_info.intent
        assert result.latency.query_processing_ms >= 0

    @pytest.mark.asyncio
    async def test_marks_empty_as_invalid(self):
        agent = QueryAgent()
        ctx = RequestContext()
        ctx.transcript = ""

        result = await agent.process(ctx)
        assert result.query_info.is_valid is False

    @pytest.mark.asyncio
    async def test_latency_recorded(self):
        agent = QueryAgent()
        ctx = RequestContext()
        ctx.transcript = "What is NLP?"

        result = await agent.process(ctx)
        assert result.latency.query_processing_ms >= 0

    @pytest.mark.asyncio
    async def test_keywords_extracted(self):
        agent = QueryAgent()
        ctx = RequestContext()
        ctx.transcript = "What is natural language processing?"

        result = await agent.process(ctx)
        assert len(result.query_info.keywords) > 0
