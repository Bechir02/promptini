"""Tests for Tunisian Derja input understanding + output-language selection."""

import pytest

from core.templates import (
    build_language_block,
    normalize_language,
    LANGUAGE_GUIDANCE,
    DERJA_INPUT_NOTE,
    validate_templates,
)
from core.constants import OUTPUT_LANGUAGES, ALLOWED_LANGUAGES, DEFAULT_LANGUAGE


def test_default_language_is_allowed():
    assert DEFAULT_LANGUAGE in ALLOWED_LANGUAGES


def test_every_language_has_guidance():
    for lang in OUTPUT_LANGUAGES:
        assert LANGUAGE_GUIDANCE.get(lang), f"missing guidance for {lang}"


def test_normalize_unknown_and_empty_fall_back():
    assert normalize_language("klingon") == DEFAULT_LANGUAGE
    assert normalize_language("") == DEFAULT_LANGUAGE
    assert normalize_language(None) == DEFAULT_LANGUAGE
    assert normalize_language("  DERJA  ") == "derja"


def test_language_block_is_specific():
    for lang in OUTPUT_LANGUAGES:
        block = build_language_block(lang)
        assert isinstance(block, str) and len(block) > 10
    assert "Derja" in build_language_block("derja")
    assert "Arabic" in build_language_block("arabic")
    assert "French" in build_language_block("french")


def test_derja_note_mentions_arabizi():
    assert "Arabizi" in DERJA_INPUT_NOTE or "franco-arabe" in DERJA_INPUT_NOTE


def test_validate_templates_clean():
    assert validate_templates() == []


def test_system_prompt_includes_language_and_derja():
    # llm imports the provider SDKs at module load — skip if unavailable locally.
    pytest.importorskip("groq")
    pytest.importorskip("cerebras")
    from llm import build_system_prompt

    sp = build_system_prompt("claude", "code_generation", "standard", [], language="derja")
    assert "OUTPUT LANGUAGE" in sp
    assert "Derja" in sp
    assert "Arabizi" in sp or "franco-arabe" in sp
    # Default still works and preserves the intent rules.
    sp_en = build_system_prompt("gpt-4", "writing", "concise", [])
    assert "OUTPUT LANGUAGE" in sp_en
    assert "English" in sp_en
