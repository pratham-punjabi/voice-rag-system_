from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestRecord:
    request_id: str
    timestamp: float
    transcript: str
    query: str
    language: str
    n_retrieved: int
    top_score: float
    latency_total_ms: float
    latency_stt_ms: float
    latency_retrieval_ms: float
    latency_generation_ms: float
    latency_guardrail_ms: float
    success: bool
    refused: bool
    error_category: str = ""
    grounded: bool = False


class RequestTracker:
    """
    In-process store for recent request records.
    Provides a rolling window for observability / admin dashboards.
    Does NOT store audio or sensitive content.
    """

    def __init__(self, max_records: int = 1000) -> None:
        self._records: deque[RequestRecord] = deque(maxlen=max_records)
        self._total = 0
        self._failures = 0
        self._refusals = 0

    def record(self, rec: RequestRecord) -> None:
        self._records.append(rec)
        self._total += 1
        if not rec.success:
            self._failures += 1
        if rec.refused:
            self._refusals += 1

    def recent(self, n: int = 50) -> list[RequestRecord]:
        records = list(self._records)
        return records[-n:]

    def summary(self) -> dict[str, Any]:
        recs = list(self._records)
        if not recs:
            return {"total": 0}
        latencies = [r.latency_total_ms for r in recs]
        return {
            "total_requests": self._total,
            "recent_window": len(recs),
            "failures": self._failures,
            "refusals": self._refusals,
            "success_rate": round(
                (self._total - self._failures) / max(1, self._total), 3
            ),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
        }


# Singleton
request_tracker = RequestTracker()
