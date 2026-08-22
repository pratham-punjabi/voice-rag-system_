from __future__ import annotations

import re
import unicodedata

MAX_QUERY_LENGTH = 500
MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024

# Prompt injection patterns
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|prompt)", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:an?\s+)?(?:different|new|evil|unrestricted)", re.I),
    re.compile(r"forget\s+(everything|all)\s+(you\s+)?(know|were\s+told)", re.I),
    re.compile(r"system\s*:\s*you\s+are", re.I),
    re.compile(r"<\s*system\s*>", re.I),
    re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+are|an?)\s+(?:a\s+)?(?:jailbreak|DAN|unrestricted)", re.I),
    re.compile(r"print\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"reveal\s+(your\s+)?(instructions?|prompt|system)", re.I),
]

_UNSAFE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(how\s+to\s+make\s+(?:bomb|weapon|poison|drug))\b", re.I),
    re.compile(r"\b(kill|harm|hurt)\s+(yourself|myself|others)\b", re.I),
]


def sanitize_text(text: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Normalize and truncate user text input."""
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove null bytes and control chars (except newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()
    # Truncate
    return text[:max_length]


def check_prompt_injection(text: str) -> bool:
    """Return True if possible prompt injection detected."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def check_unsafe_content(text: str) -> bool:
    """Return True if query contains obviously unsafe content."""
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def validate_audio_size(size_bytes: int) -> bool:
    return size_bytes <= MAX_AUDIO_SIZE_BYTES


def mask_api_key(key: str) -> str:
    """Return masked version for logging."""
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]
