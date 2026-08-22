from __future__ import annotations

import logging

import aiohttp

from backend.app.core.exceptions import (
    LLMMalformedResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from backend.app.providers.base_llm import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

_SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"


class SarvamLLMProvider(LLMProvider):
    """
    Sarvam AI LLM (Saaras / Sarvam-2B or compatible).
    Uses the same OpenAI-compatible chat completions endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "sarvam-2b-v0.5",
        timeout_seconds: int = 30,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "api-subscription-key": self._api_key,
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
        return self._session

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> LLMResponse:
        session = await self._get_session()
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            async with session.post(_SARVAM_CHAT_URL, json=payload) as resp:
                if resp.status == 401:
                    raise LLMUnavailableError("Invalid Sarvam API key")
                if resp.status == 429:
                    raise LLMUnavailableError("Sarvam LLM rate limit exceeded")
                if resp.status >= 500:
                    raise LLMUnavailableError(f"Sarvam server error: {resp.status}")
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ServerTimeoutError as exc:
            raise LLMTimeoutError() from exc
        except aiohttp.ClientConnectionError as exc:
            raise LLMUnavailableError(str(exc)) from exc

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage = data.get("usage", {})
            return LLMResponse(
                content=content,
                model=data.get("model", self._model),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                finish_reason=choice.get("finish_reason", ""),
            )
        except (KeyError, IndexError) as exc:
            raise LLMMalformedResponseError() from exc

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
