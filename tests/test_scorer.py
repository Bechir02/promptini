"""Tests for the heuristic scoring functions.

scorer.py imports the provider SDKs (groq/cerebras) at module load, so these
tests skip when those deps are not installed locally; CI installs the full
requirements and runs them.
"""

import pytest

pytest.importorskip("groq")
pytest.importorskip("cerebras")

from scorer import (  # noqa: E402
    score_structure,
    score_specificity,
    score_task_coverage,
    score_improvement,
)


def test_structure_full_marks_for_claude_code():
    out = "<role>x</role><task>y</task><constraints>z</constraints><output_format>w</output_format>"
    res = score_structure(out, "claude-code")
    assert res["score"] == 10.0
    assert res["missing"] == []


def test_structure_penalizes_missing_sections():
    res = score_structure("<role>only</role>", "claude-code")
    assert res["score"] < 10.0
    assert "<task>" in res["missing"]


def test_specificity_penalizes_vague_words():
    vague = score_specificity("x", "this should be good and nice and appropriate")
    crisp = score_specificity("x", "return a DataFrame; raise ValueError on missing file")
    assert crisp["score"] > vague["score"]


def test_specificity_flags_missing_placeholders():
    res = score_specificity("use {csv_path} here", "no placeholder at all")
    assert "csv_path" in res["note"]


def test_task_coverage_counts_elements():
    res = score_task_coverage("error root cause fix prevent", "debugging")
    assert res["score"] == 10.0


def test_improvement_rewards_reasonable_expansion():
    raw = "fix bug"
    out = " ".join(["word"] * 20)  # ~10x — flagged as long, not ideal
    res = score_improvement(raw, out)
    assert 0.0 <= res["score"] <= 10.0
