from __future__ import annotations

import logging
import time

from backend.app.core.exceptions import InvalidQueryError, PromptInjectionError, UnsafeQueryError
from backend.app.core.security import check_prompt_injection, check_unsafe_content
from backend.app.models.request_context import RequestContext

logger = logging.getLogger(__name__)


class GuardrailAgent:
    """
    Pre-retrieval safety checks.
    Deterministic — no LLM needed.
    """

    async def process(self, ctx: RequestContext) -> RequestContext:
        t0 = time.perf_counter()

        query = ctx.query_info.normalized_query
        flags: list[str] = []

        try:
            # 1. Validity
            if not ctx.query_info.is_valid:
                raise InvalidQueryError(f"Query too short or empty: {query!r}")

            # 2. Prompt injection
            if check_prompt_injection(query) or check_prompt_injection(ctx.transcript):
                flags.append("PROMPT_INJECTION")
                raise PromptInjectionError()

            # 3. Unsafe content
            if check_unsafe_content(query):
                flags.append("UNSAFE_CONTENT")
                raise UnsafeQueryError()

            ctx.is_safe = True

        except (InvalidQueryError, PromptInjectionError, UnsafeQueryError) as exc:
            ctx.is_safe = False
            ctx.safety_flags = flags
            ctx.add_error("guardrail", exc.code, exc.message)
            raise
        finally:
            ctx.latency.guardrail_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "Guardrail check",
                extra={
                    "request_id": ctx.request_id,
                    "safe": ctx.is_safe,
                    "flags": flags,
                    "latency_ms": ctx.latency.guardrail_ms,
                },
            )

        return ctx
