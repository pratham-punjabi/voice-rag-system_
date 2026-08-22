from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.monitoring.latency_recorder import latency_recorder
from backend.app.monitoring.metrics_store import metrics_store
from backend.app.monitoring.request_tracker import request_tracker

router = APIRouter(prefix="/api", tags=["metrics"])


@router.get("/metrics")
async def get_metrics(request: Request) -> dict:
    """
    Return real-time latency percentiles and system metrics.
    All values in milliseconds.
    """
    cache = getattr(request.app.state, "query_cache", None)

    return {
        "latency_percentiles": latency_recorder.all_summaries(),
        "request_summary": request_tracker.summary(),
        "component_stats": metrics_store.all_stats(),
        "cache_stats": cache.stats if cache else {},
    }
