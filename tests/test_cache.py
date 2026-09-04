"""LRU pipeline cache (C4)."""

from core.cache import LRUCache


def test_get_set_and_eviction():
    c = LRUCache(2)
    c.set("a", 1); c.set("b", 2)
    assert c.get("a") == 1
    c.set("c", 3)          # "a" just used, so "b" is evicted
    assert c.get("b") is None
    assert c.get("a") == 1 and c.get("c") == 3


def test_maxsize_zero_disables():
    c = LRUCache(0)
    c.set("a", 1)
    assert c.get("a") is None and len(c) == 0


def test_missing_key_returns_none():
    assert LRUCache(4).get("nope") is None
