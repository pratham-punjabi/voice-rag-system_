from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    transcript: str
    language: str
    confidence: float
    duration_seconds: float
    raw: dict | None = None


class SpeechToTextProvider(ABC):
    """Abstract base for all STT providers."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe raw PCM/WAV audio bytes."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the STT service is reachable."""
