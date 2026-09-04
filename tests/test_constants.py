"""Single-source-of-truth invariants (W8) and config behavior (W7)."""

import os

from core.constants import SCORE_WEIGHTS, MODELS, ALLOWED_MODELS, DEPTHS


def test_score_weights_sum_to_one():
    assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9


def test_llm_judge_carries_most_weight():
    # The rubric judge should outweigh any single heuristic check.
    assert SCORE_WEIGHTS["llm_judge"] == max(SCORE_WEIGHTS.values())


def test_allowed_models_matches_models_list():
    assert ALLOWED_MODELS == frozenset(MODELS)


def test_config_reads_env(monkeypatch):
    from core import config
    config._settings = None  # reset singleton
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    s = config.get_settings()
    assert s.groq_api_key == "test-key-123"
    assert s.has_any_provider is True
    config._settings = None


def test_require_provider_raises_when_unset(monkeypatch):
    from core import config
    config._settings = None
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    s = config.get_settings()
    import pytest
    with pytest.raises(RuntimeError):
        s.require_provider()
    config._settings = None
