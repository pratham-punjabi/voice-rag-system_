from __future__ import annotations

import time
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator, Generator

import numpy as np


@dataclass
class ComponentStats:
    name: str
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=10000))

    def record(self, ms: float) -> None:
        self.samples.append(ms)

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        return float(np.percentile(list(self.samples), p))

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {}
        arr = list(self.samples)
        return {
            "count": len(arr),
            "mean_ms": float(np.mean(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p70_ms": float(np.percentile(arr, 70)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "p100_ms": float(np.max(arr)),
        }


class LatencyRecorder:
    """Global in-process latency store."""

    def __init__(self) -> None:
        self._components: dict[str, ComponentStats] = {}

    def _get(self, name: str) -> ComponentStats:
        if name not in self._components:
            self._components[name] = ComponentStats(name=name)
        return self._components[name]

    def record(self, component: str, ms: float) -> None:
        self._get(component).record(ms)

    @contextmanager
    def measure(self, component: str) -> Generator[None, None, None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(component, (time.perf_counter() - t0) * 1000)

    @asynccontextmanager
    async def ameasure(self, component: str) -> AsyncGenerator[None, None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(component, (time.perf_counter() - t0) * 1000)

    def all_summaries(self) -> dict[str, dict[str, float]]:
        return {name: stats.summary() for name, stats in self._components.items()}

    def reset(self) -> None:
        self._components.clear()


# Singleton
latency_recorder = LatencyRecorder()
