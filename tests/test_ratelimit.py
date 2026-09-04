"""Sliding-window rate limiter (C7)."""

from core.ratelimit import RateLimiter


def test_allows_then_blocks_within_window():
    now = {"t": 1000.0}
    rl = RateLimiter(max_calls=3, per_seconds=60, time_fn=lambda: now["t"])
    assert rl.allow("ip") and rl.allow("ip") and rl.allow("ip")
    assert not rl.allow("ip")  # 4th within the window


def test_refills_after_window():
    now = {"t": 0.0}
    rl = RateLimiter(max_calls=2, per_seconds=10, time_fn=lambda: now["t"])
    assert rl.allow("ip") and rl.allow("ip")
    assert not rl.allow("ip")
    now["t"] = 11.0
    assert rl.allow("ip")


def test_keys_are_independent():
    now = {"t": 0.0}
    rl = RateLimiter(max_calls=1, per_seconds=60, time_fn=lambda: now["t"])
    assert rl.allow("a")
    assert rl.allow("b")
    assert not rl.allow("a")


def test_zero_disables():
    rl = RateLimiter(max_calls=0, per_seconds=60)
    assert all(rl.allow("x") for _ in range(100))
