"""YAML template overrides (B7)."""

import pytest

from core.templates import apply_overrides, load_yaml_overrides, OUTPUT_FORMATS, TASK_GUIDANCE


def test_apply_overrides_merges_and_preserves():
    of_before = dict(OUTPUT_FORMATS["general"])
    tg_before = TASK_GUIDANCE["general"]
    try:
        apply_overrides({
            "output_formats": {"general": {"description": "OVR"}},
            "task_guidance":  {"general": "NEW GUIDANCE"},
        })
        assert OUTPUT_FORMATS["general"]["description"] == "OVR"
        assert OUTPUT_FORMATS["general"]["format"] == of_before["format"]  # preserved
        assert TASK_GUIDANCE["general"] == "NEW GUIDANCE"
    finally:
        OUTPUT_FORMATS["general"] = of_before
        TASK_GUIDANCE["general"] = tg_before


def test_load_yaml_missing_returns_empty(tmp_path):
    assert load_yaml_overrides(str(tmp_path / "nope.yaml")) == {}


def test_load_yaml_reads_file(tmp_path):
    pytest.importorskip("yaml")
    p = tmp_path / "t.yaml"
    p.write_text("task_guidance:\n  general: 'from yaml'\n")
    data = load_yaml_overrides(str(p))
    assert data.get("task_guidance", {}).get("general") == "from yaml"
