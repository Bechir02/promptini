"""Tiny in-process metrics (C8): counters + observations, thread-safe."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


class Metrics:
    def __init__(self):
        self._counts: dict[str, int] = defaultdict(int)
        self._obs: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counts[name] += n

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._obs[name].append(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counts": dict(self._counts),
                "observations": {
                    k: {"count": len(v), "avg": (sum(v) / len(v) if v else 0.0)}
                    for k, v in self._obs.items()
                },
            }


metrics = Metrics()
