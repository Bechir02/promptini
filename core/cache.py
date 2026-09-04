"""Tiny thread-safe LRU cache for pipeline results (C4).

Caching identical (prompt, model, depth, language) transforms avoids repeat LLM
calls — cutting cost, latency, and rate-limit pressure. In-process only.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Hashable, Optional


class LRUCache:
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._d: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            if key not in self._d:
                return None
            self._d.move_to_end(key)
            return self._d[key]

    def set(self, key: Hashable, value: Any) -> None:
        if self.maxsize <= 0:
            return
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        return len(self._d)
