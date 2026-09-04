"""Template coverage linter (S5)."""

from core.templates import validate_templates, OUTPUT_FORMATS, TASK_GUIDANCE
from core.constants import MODELS, TASK_TYPES


def test_templates_are_complete():
    problems = validate_templates()
    assert problems == [], f"Template coverage problems: {problems}"


def test_every_model_has_a_format():
    for m in MODELS:
        assert m in OUTPUT_FORMATS


def test_every_task_type_has_guidance():
    for t in TASK_TYPES:
        assert t in TASK_GUIDANCE
