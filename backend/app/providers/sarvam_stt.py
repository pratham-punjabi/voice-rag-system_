from __future__ import annotations

import asyncio
import io
import logging
import wave

import aiohttp

from backend.app.core.exceptions import (
    STTConnectionError,
    STTEmptyTranscriptError,
    STTInvalidAPIKeyError,
    STTTimeoutError,
)
from backend.app.providers.base_stt import SpeechToTextProvider, TranscriptionResult

logger = logging.getLogger(__name__)

_SARVAM_REST_URL = "https://api.sarvam.ai/speech-to-text"


def _to_wav_bytes(audio_bytes: bytes, sample_rate: int) -> bytes:
    """
    Wrap raw audio bytes in a proper WAV container if not already WAV.
    Browser MediaRecorder produces audio/webm (Opus). Sarvam's REST API
    requires a valid WAV file, so we must convert.

    Strategy:
      - If the bytes already start with the RIFF header, pass through.
      - Otherwise treat as raw 16-bit signed PCM (mono) and wrap in a WAV
        container. This matches what browsers send after getUserMedia with
        sampleRate=16000 and channelCount=1.
    """
    if audio_bytes[:4] == b"RIFF":
        return audio_bytes  # Already a WAV — pass through unchanged

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    return buf.getvalue()


class SarvamSTTProvider(SpeechToTextProvider):
    """
    Sarvam Speech-to-Text via REST API.
    Supports: Hindi, English, Hinglish/code-mixed.
    """

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 20,
        language_code: str = "en-IN",
    ) -> None:
        if not api_key:
            raise STTInvalidAPIKeyError("Sarvam API key is required")
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._language_code = language_code
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"api-subscription-key": self._api_key},
                timeout=self._timeout,
            )
        return self._session

    async def transcribe(
        self, audio_bytes: bytes, sample_rate: int = 16000
    ) -> TranscriptionResult:
        if not audio_bytes:
            from backend.app.core.exceptions import STTEmptyAudioError
            raise STTEmptyAudioError()

        # FIX 1: Convert to proper WAV before sending.
        # Browsers record as audio/webm (Opus). Sending WebM to Sarvam
        # causes it to return an empty transcript instead of an error.
        wav_bytes = _to_wav_bytes(audio_bytes, sample_rate)
        logger.debug(
            "Audio prepared for STT: input=%d bytes, wav=%d bytes, "
            "already_wav=%s, sample_rate=%d",
            len(audio_bytes), len(wav_bytes),
            audio_bytes[:4] == b"RIFF", sample_rate,
        )

        session = await self._get_session()

        # FIX 2: Removed dead `audio_b64 = base64.b64encode(...)` that
        # was computed but never used.
        payload = {
            "model": "saarika:v2",
            "language_code": self._language_code,
            "with_timestamps": False,
        }

        try:
            form = aiohttp.FormData()
            form.add_field(
                "file",
                wav_bytes,
                filename="audio.wav",
                content_type="audio/wav",
            )
            for k, v in payload.items():
                form.add_field(k, str(v))

            async with session.post(_SARVAM_REST_URL, data=form) as resp:
                if resp.status == 401:
                    raise STTInvalidAPIKeyError()
                # FIX 3: 422 was previously unhandled and fell through to
                # raise_for_status() with no useful message logged.
                if resp.status == 422:
                    body = await resp.text()
                    logger.error(
                        "Sarvam rejected audio (422 Unprocessable Entity): %s", body
                    )
                    raise STTConnectionError(
                        f"Sarvam rejected audio format: {body[:200]}"
                    )
                if resp.status == 429:
                    from backend.app.core.exceptions import RateLimitError
                    raise RateLimitError("Sarvam rate limit exceeded")
                if resp.status >= 500:
                    raise STTConnectionError(f"Sarvam server error: {resp.status}")
                resp.raise_for_status()
                data = await resp.json()

        except aiohttp.ServerTimeoutError as exc:
            raise STTTimeoutError() from exc
        except aiohttp.ClientConnectionError as exc:
            raise STTConnectionError(str(exc)) from exc

        transcript = data.get("transcript", "").strip()
        if not transcript:
            logger.warning(
                "Sarvam returned empty transcript. Full response: %s", data
            )
            raise STTEmptyTranscriptError()

        language_code = data.get("language_code", self._language_code)
        return TranscriptionResult(
            transcript=transcript,
            language=language_code,
            confidence=data.get("confidence", 0.9),
            duration_seconds=data.get("duration", 0.0),
            raw=data,
        )

    async def health_check(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(
                "https://api.sarvam.ai/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status < 500
        except Exception:
            return False

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()