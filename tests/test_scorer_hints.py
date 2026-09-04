"""Actionable fix hints in the score card (B5)."""

import pytest

pytest.importorskip("groq")
pytest.importorskip("cerebras")

from scorer import collect_fix_hints, format_score_html


def _mk():
    return {
        "overall": 6.0, "grade": "C — Acceptable",
        "breakdown": {
            "structure":   {"score": 5.0, "missing": ["<role>", "<task>"], "note": ""},
            "specificity": {"score": 6.0, "vague_words": ["good", "nice"], "note": ""},
            "model_aware": {"score": 7.0, "note": ""},
            "task_cover":  {"score": 5.0, "missing": ["error", "return"], "note": ""},
            "improvement": {"score": 8.0, "note": ""},
            "llm_judge":   {"score": 6.0, "note": "x"},
        },
    }


def test_hints_collected():
    h = collect_fix_hints(_mk())
    assert any("Add sections" in x for x in h)
    assert any("vague" in x.lower() for x in h)
    assert any("task elements" in x.lower() for x in h)


def test_hints_rendered_and_escaped():
    html = format_score_html(_mk(), "some reasoning")
    assert "Make it better" in html
    assert "&lt;role&gt;" in html  # escaped, not raw tag
