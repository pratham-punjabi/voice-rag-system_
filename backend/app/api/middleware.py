from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter per IP."""

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60) -> None:
        super().__init__(app)
        self._max = max_requests
        self._window = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = self._buckets[ip]

        # Evict expired timestamps
        self._buckets[ip] = [t for t in bucket if now - t < self._window]

        if len(self._buckets[ip]) >= self._max:
            return Response(
                content='{"error":"Rate limit exceeded","code":"RATE_LIMIT_EXCEEDED"}',
                status_code=429,
                media_type="application/json",
            )

        self._buckets[ip].append(now)
        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request."""

    async def dispatch(self, request: Request, call_next):
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
