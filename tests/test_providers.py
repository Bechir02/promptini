"""Tests for provider resilience (retry + fallback chain)."""

from types import SimpleNamespace

import pytest

# llm imports the provider SDKs at module load — skip if unavailable locally.
pytest.importorskip("groq")
pytest.importorskip("cerebras")

from llm import _is_transient, _retry, build_provider_order


def test_is_transient_classifies():
    assert _is_transient(Exception("429 Too Many Requests"))
    assert _is_transient(Exception("Service temporarily unavailable (503)"))
    assert _is_transient(Exception("connection reset by peer"))
    assert not _is_transient(Exception("401 invalid api key"))
    assert not _is_transient(Exception("model not found"))


def test_retry_succeeds_after_transient():
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("503 unavailable")
        return "ok"
    assert _retry(flaky, retries=3, base_delay=0.0) == "ok"
    assert calls["n"] == 3


def test_retry_does_not_retry_nontransient():
    calls = {"n": 0}
    def bad():
        calls["n"] += 1
        raise Exception("401 invalid key")
    with pytest.raises(Exception):
        _retry(bad, retries=3, base_delay=0.0)
    assert calls["n"] == 1


def test_provider_order_from_keys():
    s = SimpleNamespace(groq_api_key="a", cerebras_api_key="", openrouter_api_key="c",
                        together_api_key="", google_api_key="")
    assert build_provider_order(s) == ["groq", "openrouter"]

    s2 = SimpleNamespace(groq_api_key="", cerebras_api_key="b", openrouter_api_key="",
                         together_api_key="t", google_api_key="g")
    assert build_provider_order(s2) == ["cerebras", "together", "google"]

    empty = SimpleNamespace(groq_api_key="", cerebras_api_key="", openrouter_api_key="",
                            together_api_key="", google_api_key="")
    assert build_provider_order(empty) == []
