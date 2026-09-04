"""Tests for retrieval input validation (W2 hardening)."""

from core.retrieval import validate_filter_inputs


def test_valid_inputs_pass_through():
    assert validate_filter_inputs("claude", "debugging", 7.0) == ("claude", "debugging", 7.0)


def test_unknown_model_falls_back_to_general():
    model, _, _ = validate_filter_inputs("evil'; DROP TABLE", "debugging", 7.0)
    assert model == "general"


def test_unknown_task_type_falls_back_to_general():
    _, task, _ = validate_filter_inputs("claude", "nonsense", 7.0)
    assert task == "general"


def test_non_numeric_min_quality_falls_back():
    _, _, q = validate_filter_inputs("claude", "debugging", "not-a-number")
    assert isinstance(q, float)


def test_injection_attempt_neutralized():
    # An injection-shaped value must never survive validation.
    model, task, q = validate_filter_inputs("' OR '1'='1", "'; DELETE", "1;2")
    assert model == "general"
    assert task == "general"
    assert q == 7.0
