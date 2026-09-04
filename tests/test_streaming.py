"""Streaming transform logic (B8) — deterministic, provider calls mocked."""

import pytest

pytest.importorskip("groq")
pytest.importorskip("cerebras")

import llm


def _fake(tokens):
    def gen(system_prompt, user_prompt, model=None):
        for t in tokens:
            yield t
    return gen


def test_stream_accumulates_and_labels(monkeypatch):
    monkeypatch.setattr(llm, "build_provider_order", lambda s: ["groq"])
    monkeypatch.setattr(llm, "_STREAMERS", {"groq": _fake(["Hel", "lo ", "world"])})
    out = list(llm.transform_prompt_stream("x", "claude", "code_generation", "concise", []))
    texts = [t for t, _ in out]
    assert texts == ["Hel", "Hello ", "Hello world"]      # monotonic accumulation
    assert out[-1][1].startswith("groq")                  # provider label


def test_stream_falls_back_to_next_provider(monkeypatch):
    def boom(system_prompt, user_prompt, model=None):
        raise Exception("429 rate limit")
        yield  # pragma: no cover
    monkeypatch.setattr(llm, "build_provider_order", lambda s: ["groq", "cerebras"])
    monkeypatch.setattr(llm, "_STREAMERS", {"groq": boom, "cerebras": _fake(["ok"])})
    out = list(llm.transform_prompt_stream("x", "claude", "code_generation", "concise", []))
    assert out[-1][0] == "ok"
    assert out[-1][1].startswith("cerebras")


def test_stream_falls_back_to_nonstreaming_when_no_streamer(monkeypatch):
    monkeypatch.setattr(llm, "build_provider_order", lambda s: ["openrouter"])
    monkeypatch.setattr(llm, "transform_prompt", lambda *a, **k: ("FULL", "openrouter (m)", {}))
    out = list(llm.transform_prompt_stream("x", "gpt-4", "writing", "concise", []))
    assert out == [("FULL", "openrouter (m)")]
