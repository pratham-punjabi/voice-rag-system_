from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ComponentMetrics:
    name: str
    values: list[float] = field(default_factory=list)

    def add(self, v: float) -> None:
        self.values.append(v)

    def stats(self) -> dict[str, float]:
        if not self.values:
            return {}
        a = np.array(self.values)
        return {
            "count": len(self.values),
            "mean": float(np.mean(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "p50": float(np.percentile(a, 50)),
            "p70": float(np.percentile(a, 70)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "p100": float(np.max(a)),
        }


class MetricsStore:
    """Collects and exposes per-component latency metrics."""

    def __init__(self) -> None:
        self._start = time.time()
        self._components: dict[str, ComponentMetrics] = {}

    def record(self, component: str, value_ms: float) -> None:
        if component not in self._components:
            self._components[component] = ComponentMetrics(name=component)
        self._components[component].add(value_ms)

    def all_stats(self) -> dict[str, Any]:
        return {
            "uptime_seconds": round(time.time() - self._start, 1),
            "components": {
                name: comp.stats()
                for name, comp in self._components.items()
                if comp.values
            },
        }

    def reset(self) -> None:
        self._components.clear()
        self._start = time.time()


# Singleton
metrics_store = MetricsStore()
