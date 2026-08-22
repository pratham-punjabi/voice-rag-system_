from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")


def normalize_text(text: str) -> str:
    """Apply unicode normalisation and basic cleanup."""
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG_RE.sub(" ", text)          # strip HTML tags
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_useful_text(text: str, min_chars: int = 20, min_words: int = 5) -> bool:
    """Return True if the text has sufficient content."""
    if not text or len(text) < min_chars:
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    # Reject purely numeric or symbolic content
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars < len(text) * 0.3:
        return False
    return True


def content_hash(text: str) -> str:
    """Deterministic hash for deduplication (normalized content)."""
    normalized = _WHITESPACE_RE.sub(" ", text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


class DataCleaner:
    """
    Stateful cleaner that tracks seen content hashes for deduplication.
    """

    def __init__(self, min_chars: int = 20, min_words: int = 5) -> None:
        self._seen_hashes: set[str] = set()
        self._seen_ids: set[str] = set()
        self.min_chars = min_chars
        self.min_words = min_words
        # Counters
        self.n_processed = 0
        self.n_empty = 0
        self.n_duplicates = 0
        self.n_accepted = 0

    def clean(self, doc: dict[str, Any]) -> dict[str, Any] | None:
        """
        Clean and validate a document dict.
        Returns cleaned doc or None if it should be skipped.
        """
        self.n_processed += 1
        text = normalize_text(str(doc.get("text", "")))

        # Empty / too short
        if not is_useful_text(text, self.min_chars, self.min_words):
            self.n_empty += 1
            return None

        # Duplicate content
        h = content_hash(text)
        if h in self._seen_hashes:
            self.n_duplicates += 1
            return None
        self._seen_hashes.add(h)

        # Duplicate ID
        doc_id = str(doc.get("doc_id", ""))
        if doc_id and doc_id in self._seen_ids:
            self.n_duplicates += 1
            return None
        if doc_id:
            self._seen_ids.add(doc_id)

        self.n_accepted += 1
        return {**doc, "text": text}

    def report(self) -> dict[str, Any]:
        return {
            "processed": self.n_processed,
            "accepted": self.n_accepted,
            "empty_rejected": self.n_empty,
            "duplicate_rejected": self.n_duplicates,
            "acceptance_rate": round(self.n_accepted / max(1, self.n_processed), 3),
        }
