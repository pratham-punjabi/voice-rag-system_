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


class LocalLLMProvider(LLMProvider):
    """
    OpenAI-compatible local LLM provider.
    Works with Ollama (`ollama serve`) or llama.cpp server.

    Ollama example:
        LLM_PROVIDER=local
        LLM_BASE_URL=http://localhost:11434/v1
        LLM_MODEL=llama3.2

    llama.cpp example:
        LLM_PROVIDER=local
        LLM_BASE_URL=http://localhost:8080/v1
        LLM_MODEL=mistral-7b
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        timeout_seconds: int = 60,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"},
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
            "stream": False,
        }
        try:
            async with session.post(
                f"{self._base_url}/chat/completions",
                json=payload,
            ) as resp:
                if resp.status >= 500:
                    raise LLMUnavailableError(f"Local LLM server error: {resp.status}")
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ServerTimeoutError as exc:
            raise LLMTimeoutError() from exc
        except aiohttp.ClientConnectionError as exc:
            raise LLMUnavailableError(
                f"Cannot connect to local LLM at {self._base_url}: {exc}"
            ) from exc

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
                f"{self._base_url}/models",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status < 500
        except Exception:
            return False

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
