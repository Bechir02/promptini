"""Tests for the ordered parallel map used by the arena."""

import time

from core.concurrency import map_ordered


def test_order_preserved():
    assert map_ordered(lambda x: x * x, [1, 2, 3, 4]) == [1, 4, 9, 16]


def test_empty_and_single():
    assert map_ordered(lambda x: x, []) == []
    assert map_ordered(lambda x: x + 1, [41]) == [42]


def test_runs_concurrently():
    def slow(x):
        time.sleep(0.2)
        return x
    start = time.time()
    out = map_ordered(slow, [1, 2, 3, 4], max_workers=4)
    elapsed = time.time() - start
    assert out == [1, 2, 3, 4]
    assert elapsed < 0.6, f"expected concurrent (<0.6s), got {elapsed:.2f}s"
