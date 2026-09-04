"""In-process metrics (C8)."""

from core.metrics import Metrics


def test_counts_and_observations():
    m = Metrics()
    m.incr("requests"); m.incr("requests"); m.incr("errors")
    m.observe("latency_s", 1.0); m.observe("latency_s", 3.0)
    snap = m.snapshot()
    assert snap["counts"]["requests"] == 2
    assert snap["counts"]["errors"] == 1
    assert snap["observations"]["latency_s"]["count"] == 2
    assert snap["observations"]["latency_s"]["avg"] == 2.0
