from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Callable
from typing import Any, TypeVar

from backend.app.core.exceptions import (
    LLMTimeoutError,
    STTConnectionError,
    STTTimeoutError,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Errors that should NOT be retried
_NON_RETRYABLE = (
    "STT_INVALID_API_KEY",
    "INVALID_QUERY",
    "UNSAFE_QUERY",
    "PROMPT_INJECTION",
    "INVALID_DATASET_SCHEMA",
    "RATE_LIMIT_EXCEEDED",
)


def is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code in _NON_RETRYABLE:
        return False
    if isinstance(exc, (STTConnectionError, STTTimeoutError, LLMTimeoutError)):
        return True
    # Retry generic transient HTTP errors
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return False


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 0.2,
    max_delay: float = 5.0,
    jitter: bool = True,
    **kwargs: Any,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc) or attempt == max_attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            logger.warning(
                "Retryable error on attempt %d/%d: %s. Retrying in %.2fs",
                attempt, max_attempts, exc, delay,
            )
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.2,
    max_delay: float = 5.0,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await retry_async(
                func, *args,
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                **kwargs,
            )
        return wrapper  # type: ignore[return-value]
    return decorator
