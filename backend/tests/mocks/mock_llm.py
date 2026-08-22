from __future__ import annotations

from backend.app.providers.base_llm import LLMProvider, LLMResponse
from backend.app.providers.base_stt import SpeechToTextProvider, TranscriptionResult


class MockSTTProvider(SpeechToTextProvider):
    def __init__(self, transcript: str = "What is machine learning?", language: str = "en"):
        self._transcript = transcript
        self._language = language

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        if not audio_bytes:
            from backend.app.core.exceptions import STTEmptyAudioError
            raise STTEmptyAudioError()
        return TranscriptionResult(
            transcript=self._transcript,
            language=self._language,
            confidence=0.95,
            duration_seconds=2.0,
        )

    async def health_check(self) -> bool:
        return True


class MockLLMProvider(LLMProvider):
    def __init__(
        self,
        answer: str = "This is a test answer based on the context.",
        confidence: float = 0.85,
        grounded: bool = True,
        fail: bool = False,
    ):
        self._answer = answer
        self._confidence = confidence
        self._grounded = grounded
        self._fail = fail

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> LLMResponse:
        if self._fail:
            from backend.app.core.exceptions import LLMUnavailableError
            raise LLMUnavailableError("Mock LLM failure")
        import json
        content = json.dumps({
            "answer": self._answer,
            "confidence": self._confidence,
            "grounded": self._grounded,
        })
        return LLMResponse(
            content=content,
            model="mock-model",
            prompt_tokens=100,
            completion_tokens=50,
            finish_reason="stop",
        )

    async def health_check(self) -> bool:
        return not self._fail
