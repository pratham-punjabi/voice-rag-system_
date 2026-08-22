from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Shared connector settings
_CONNECTOR_LIMIT = 100       # total connections
_CONNECTOR_LIMIT_PER_HOST = 20
_KEEPALIVE_TIMEOUT = 30      # seconds


class ConnectionPool:
    """
    Global aiohttp connector pool shared across all HTTP providers.
    Keeps TCP connections alive across requests — eliminates connection
    setup latency on the hot path.

    Usage:
        pool = ConnectionPool()
        await pool.start()
        session = pool.get_session(base_headers={"Authorization": "Bearer ..."})
        # use session ...
        await pool.close()
    """

    def __init__(self) -> None:
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._connector is None or self._connector.closed:
                self._connector = aiohttp.TCPConnector(
                    limit=_CONNECTOR_LIMIT,
                    limit_per_host=_CONNECTOR_LIMIT_PER_HOST,
                    keepalive_timeout=_KEEPALIVE_TIMEOUT,
                    enable_cleanup_closed=True,
                    force_close=False,
                )
                logger.info(
                    "HTTP connection pool started (limit=%d, per_host=%d)",
                    _CONNECTOR_LIMIT,
                    _CONNECTOR_LIMIT_PER_HOST,
                )

    def get_session(
        self,
        base_headers: Optional[dict] = None,
        timeout_seconds: int = 30,
    ) -> aiohttp.ClientSession:
        """
        Create a new ClientSession backed by the shared connector.
        Each provider gets its own session (with its own auth headers)
        but they share the underlying TCP connection pool.
        """
        if self._connector is None or self._connector.closed:
            raise RuntimeError("ConnectionPool not started. Call await pool.start() first.")

        return aiohttp.ClientSession(
            connector=self._connector,
            connector_owner=False,   # pool owns the connector lifetime
            headers=base_headers or {},
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        )

    async def close(self) -> None:
        async with self._lock:
            if self._connector and not self._connector.closed:
                await self._connector.close()
                logger.info("HTTP connection pool closed")

    @property
    def is_open(self) -> bool:
        return self._connector is not None and not self._connector.closed


# Application-level singleton
connection_pool = ConnectionPool()
