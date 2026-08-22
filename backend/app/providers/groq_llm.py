from __future__ import annotations

import json
import logging

import aiohttp

from backend.app.core.exceptions import (
    LLMMalformedResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from backend.app.providers.base_llm import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

# Current active Groq models (August 2026)
# Source: https://console.groq.com/docs/models
GROQ_MODELS = {
    # ── Production ────────────────────────────────────────────────────────
    "openai/gpt-oss-120b",       # 500 t/s, 131K ctx — flagship
    "openai/gpt-oss-20b",        # 1000 t/s, 131K ctx — fastest
    # ── Preview ───────────────────────────────────────────────────────────
    "qwen/qwen3.6-27b",          # 500 t/s, vision capable
    "openai/gpt-oss-safeguard-20b",
    # ── Community / still active ──────────────────────────────────────────
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-versatile",   # still active per search results
    "llama-3.1-8b-instant",      # still active per search results
    "deepseek-r1-distill-llama-70b",
}

# Best default: fast, cheap, 131K context
DEFAULT_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqLLMProvider(LLMProvider):
    """
    Groq API provider (OpenAI-compatible chat completions).

    Recommended production models (August 2026):
      openai/gpt-oss-20b    — fastest (1000 t/s), cheapest
      openai/gpt-oss-120b   — best quality (500 t/s)
      qwen/qwen3.6-27b      — vision + text (500 t/s)
      llama-3.3-70b-versatile — good reasoning, community tier
      llama-3.1-8b-instant  — ultra-fast, lightweight

    Set GROQ_MODEL in .env to any of the above.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = GROQ_BASE_URL,
        timeout_seconds: int = 30,
    ) -> None:
        if not api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env")
        self._api_key = api_key
        self._model = model.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

        if self._model not in GROQ_MODELS:
            logger.warning(
                "Model '%s' not in known active Groq models. "
                "Known models: %s. "
                "If this is a new model, it may still work — "
                "check https://console.groq.com/docs/models",
                self._model, sorted(GROQ_MODELS),
            )

        logger.info("GroqLLMProvider: model=%s url=%s", self._model, self._base_url)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        return self._session

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> LLMResponse:
        session = await self._get_session()
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        logger.debug("Groq POST %s model=%s", url, self._model)

        try:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                raw_body = await resp.text()
                logger.debug("Groq response status=%d body_preview=%s", resp.status, raw_body[:300])

                if resp.status == 401:
                    raise LLMUnavailableError(
                        "Invalid Groq API key. Check GROQ_API_KEY in .env"
                    )
                if resp.status == 404:
                    raise LLMUnavailableError(
                        f"Groq 404 — model '{self._model}' not found or deprecated. "
                        f"Current active models: {sorted(GROQ_MODELS)}. "
                        f"Update GROQ_MODEL in .env. "
                        f"Full list: https://console.groq.com/docs/models"
                    )
                if resp.status == 429:
                    raise LLMUnavailableError(
                        "Groq rate limit exceeded. Wait a moment and retry, "
                        "or upgrade your plan at https://console.groq.com"
                    )
                if resp.status >= 500:
                    raise LLMUnavailableError(
                        f"Groq server error {resp.status}: {raw_body[:200]}"
                    )
                if resp.status >= 400:
                    raise LLMUnavailableError(
                        f"Groq API error {resp.status}: {raw_body[:200]}"
                    )

                try:
                    data = json.loads(raw_body)
                except json.JSONDecodeError as exc:
                    raise LLMMalformedResponseError() from exc

        except aiohttp.ServerTimeoutError as exc:
            raise LLMTimeoutError() from exc
        except aiohttp.ClientConnectionError as exc:
            raise LLMUnavailableError(f"Cannot connect to Groq: {exc}") from exc
        except (LLMUnavailableError, LLMTimeoutError, LLMMalformedResponseError):
            raise
        except Exception as exc:
            raise LLMUnavailableError(f"Unexpected Groq error: {exc}") from exc

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
            logger.error("Groq malformed response: %s", data)
            raise LLMMalformedResponseError() from exc

    async def health_check(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(
                f"{self._base_url}/models",
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Groq health check failed: %d %s", resp.status, body[:200])
                return resp.status == 200
        except Exception as exc:
            logger.warning("Groq health check error: %s", exc)
            return False

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()