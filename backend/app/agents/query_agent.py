from __future__ import annotations

import logging
import re
import time
import unicodedata

from backend.app.models.request_context import QueryInfo, RequestContext

logger = logging.getLogger(__name__)

# Filler words to strip (English + Hindi common fillers)
_FILLERS = frozenset({
    "um", "uh", "er", "ah", "like", "you know", "i mean", "so", "basically",
    "kind of", "sort of", "actually", "literally", "hmm", "umm", "uhh",
    "acha", "haan", "matlab", "toh", "na", "yaar",
})

_LANG_DETECT_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LANG_DETECT_LATIN = re.compile(r"[a-zA-Z]")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


def _detect_language(text: str) -> tuple[str, bool]:
    """Return (language_code, is_code_mixed)."""
    has_devanagari = bool(_LANG_DETECT_DEVANAGARI.search(text))
    has_latin = bool(_LANG_DETECT_LATIN.search(text))

    if has_devanagari and has_latin:
        return "hi-en", True   # Hinglish / code-mixed
    elif has_devanagari:
        return "hi", False
    else:
        return "en", False


def _normalize(text: str) -> str:
    """Clean and normalize query text deterministically."""
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # Lowercase for filler removal
    lower = text.lower()
    # Remove filler words at word boundaries
    for filler in _FILLERS:
        lower = re.sub(r"\b" + re.escape(filler) + r"\b", " ", lower)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", lower).strip()
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    return text


def _extract_keywords(text: str) -> list[str]:
    stopwords = {
        "what", "who", "when", "where", "how", "why", "is", "are", "was",
        "were", "the", "a", "an", "in", "on", "at", "to", "for", "of",
        "and", "or", "tell", "me", "about", "please", "can", "you",
    }
    words = re.findall(r"\b[a-zA-Z\u0900-\u097F]{3,}\b", text.lower())
    return [w for w in words if w not in stopwords][:10]


def _extract_entities(text: str) -> list[str]:
    return list(set(_ENTITY_RE.findall(text)))[:5]


def _detect_intent(text: str) -> str:
    text_l = text.lower()
    if any(w in text_l for w in ["what is", "define", "meaning", "explain"]):
        return "definition"
    if any(w in text_l for w in ["how to", "steps", "process", "procedure"]):
        return "procedural"
    if any(w in text_l for w in ["who", "when", "where"]):
        return "factual"
    if any(w in text_l for w in ["why", "reason", "cause"]):
        return "causal"
    if any(w in text_l for w in ["compare", "difference", "vs", "versus"]):
        return "comparative"
    return "general"


def _is_valid_query(text: str) -> bool:
    if not text or len(text.strip()) < 3:
        return False
    if len(text.split()) < 2:
        return False
    if re.fullmatch(r"[^a-zA-Z\u0900-\u097F]+", text):
        return False
    return True


class QueryAgent:
    """
    Deterministic query understanding — no LLM required.
    Fast: ~0.1ms per query.
    """

    async def process(self, ctx: RequestContext) -> RequestContext:
        t0 = time.perf_counter()
        raw = ctx.transcript.strip()

        normalized = _normalize(raw)
        language, is_code_mixed = _detect_language(raw)
        is_valid = _is_valid_query(normalized)
        keywords = _extract_keywords(normalized)
        entities = _extract_entities(raw)
        intent = _detect_intent(normalized) if is_valid else "unknown"

        ctx.query_info = QueryInfo(
            original_query=raw,
            normalized_query=normalized,
            language=language,
            is_valid=is_valid,
            intent=intent,
            keywords=keywords,
            entities=entities,
            is_code_mixed=is_code_mixed,
        )

        ctx.latency.query_processing_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Query processed",
            extra={
                "request_id": ctx.request_id,
                "language": language,
                "intent": intent,
                "is_valid": is_valid,
                "latency_ms": ctx.latency.query_processing_ms,
            },
        )
        return ctx
