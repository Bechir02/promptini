"""Chain (decompose-into-steps) mode (B6)."""

import pytest

pytest.importorskip("groq")
pytest.importorskip("cerebras")

from llm import build_system_prompt


def test_chain_mode_injects_directive():
    sp = build_system_prompt("claude", "code_generation", "standard", [], chain=True)
    assert "CHAIN MODE" in sp
    assert "Step 1:" in sp


def test_chain_off_by_default():
    sp = build_system_prompt("claude", "code_generation", "standard", [])
    assert "CHAIN MODE" not in sp
