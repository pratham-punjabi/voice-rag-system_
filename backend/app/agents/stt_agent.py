from __future__ import annotations

import logging
import time

from backend.app.core.exceptions import STTEmptyAudioError, STTEmptyTranscriptError
from backend.app.core.retry import with_retry
from backend.app.models.request_context import RequestContext
from backend.app.providers.base_stt import SpeechToTextProvider

logger = logging.getLogger(__name__)


class STTAgent:
    """
    Wraps the SpeechToTextProvider with retry, VAD, and clean extraction.
    Never exposes API keys; works only with the provider abstraction.
    """

    def __init__(self, provider: SpeechToTextProvider) -> None:
        self._provider = provider

    async def process(self, ctx: RequestContext) -> RequestContext:
        if not ctx.audio_bytes:
            raise STTEmptyAudioError()

        t0 = time.perf_counter()
        try:
            result = await self._transcribe_with_retry(ctx.audio_bytes, ctx.audio_sample_rate)
            ctx.transcript = result.transcript
            ctx.stt_language = result.language
            ctx.stt_confidence = result.confidence
            ctx.audio_duration_seconds = result.duration_seconds
        except Exception as exc:
            ctx.add_error("stt", getattr(exc, "code", "STT_ERROR"), str(exc))
            raise
        finally:
            ctx.latency.stt_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "STT completed",
                extra={
                    "request_id": ctx.request_id,
                    "transcript_len": len(ctx.transcript),
                    "language": ctx.stt_language,
                    "latency_ms": ctx.latency.stt_ms,
                },
            )
        return ctx

    @with_retry(max_attempts=2, base_delay=0.3)
    async def _transcribe_with_retry(self, audio: bytes, rate: int):
        return await self._provider.transcribe(audio, sample_rate=rate)
