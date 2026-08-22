from __future__ import annotations

from backend.app.providers.base_stt import SpeechToTextProvider, TranscriptionResult


class MockSTTProvider(SpeechToTextProvider):
    """
    Drop-in mock for SpeechToTextProvider.
    Returns a configurable transcript without hitting any external API.
    """

    def __init__(
        self,
        transcript: str = "What is machine learning?",
        language: str = "en",
        confidence: float = 0.97,
        duration_seconds: float = 2.0,
        should_fail: bool = False,
        fail_with: type[Exception] | None = None,
    ) -> None:
        self._transcript = transcript
        self._language = language
        self._confidence = confidence
        self._duration = duration_seconds
        self._should_fail = should_fail
        self._fail_with = fail_with
        self.call_count = 0

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        self.call_count += 1

        if not audio_bytes:
            from backend.app.core.exceptions import STTEmptyAudioError
            raise STTEmptyAudioError()

        if self._should_fail:
            exc_class = self._fail_with or Exception
            raise exc_class("Mock STT failure")

        return TranscriptionResult(
            transcript=self._transcript,
            language=self._language,
            confidence=self._confidence,
            duration_seconds=self._duration,
            raw={"mock": True},
        )

    async def health_check(self) -> bool:
        return not self._should_fail


class AlwaysFailSTTProvider(SpeechToTextProvider):
    """Always raises STTConnectionError — for testing fallback behaviour."""

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        from backend.app.core.exceptions import STTConnectionError
        raise STTConnectionError("Mock STT always fails")

    async def health_check(self) -> bool:
        return False


class EmptyTranscriptSTTProvider(SpeechToTextProvider):
    """Returns empty transcript — for testing empty-transcript handling."""

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        from backend.app.core.exceptions import STTEmptyTranscriptError
        raise STTEmptyTranscriptError()

    async def health_check(self) -> bool:
        return True
