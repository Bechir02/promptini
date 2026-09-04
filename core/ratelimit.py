"""Sliding-window rate limiter (C7). Thread-safe, in-process, per-key."""

from __future__ import annotations

import threading
import time
from typing import Callable


class RateLimiter:
    def __init__(self, max_calls: int, per_seconds: float, time_fn: Callable[[], float] = time.time):
        self.max_calls = max_calls
        self.per = per_seconds
        self._time = time_fn
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """True if ``key`` is under the limit; records the call when allowed."""
        if self.max_calls <= 0:
            return True
        now = self._time()
        cutoff = now - self.per
        with self._lock:
            q = self._hits.setdefault(key, [])
            while q and q[0] < cutoff:
                q.pop(0)
            if len(q) >= self.max_calls:
                return False
            q.append(now)
            return True
