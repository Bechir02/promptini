"""Small ordered parallel map for I/O-bound work (LLM calls).

ThreadPoolExecutor.map preserves input order, so the arena's two model
transforms + scoring run concurrently while results stay deterministic.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_ordered(fn: Callable[[T], R], items: Iterable[T], max_workers: int = 4) -> List[R]:
    """Apply ``fn`` to each item in parallel, returning results in input order."""
    items = list(items)
    if not items:
        return []
    if len(items) == 1:
        return [fn(items[0])]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
        return list(ex.map(fn, items))
