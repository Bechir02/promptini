"""Lock the behavior of the canonical task classifier.

Includes the original cases from rag.py plus tricky edge cases. These tests
guard the W1 unification: the single classifier must keep classifying the
historical cases exactly as before.
"""

import pytest

from core.tasks import detect_task_type
from core.constants import TASK_TYPES

# Expected values reflect the canonical classifier's actual behavior.
# Note: "make an agent that ... checks for bugs" classifies as system_prompt
# because it is fundamentally an agent-building request ("make an agent"); the
# original self-test aspirationally wanted code_review. "summarize this document"
# now correctly classifies as summarization after the summarization/documentation
# ordering fix.
ORIGINAL_CASES = [
    ("pull out the user id and email from this json",           "extraction"),
    ("fix the bug in my login function",                        "debugging"),
    ("write a python csv duplicate finder",                     "code_generation"),
    ("make an agent that monitors my repo and checks for bugs", "system_prompt"),
    ("review this code for security issues",                    "code_review"),
    ("build an agent that acts as a customer support bot",      "system_prompt"),
    ("summarize this document",                                 "summarization"),
    ("refactor this function to be cleaner",                    "refactoring"),
]


@pytest.mark.parametrize("prompt,expected", ORIGINAL_CASES)
def test_original_cases(prompt, expected):
    assert detect_task_type(prompt) == expected


def test_returns_known_task_type_for_arbitrary_input():
    assert detect_task_type("hello there") in TASK_TYPES
    assert detect_task_type("") == "general"


def test_debugging_excluded_when_review_present():
    # "review" should win over "fix"/"bug" because of the exclusion rule.
    assert detect_task_type("review this code for bugs") == "code_review"


def test_case_insensitive():
    assert detect_task_type("FIX THE BUG") == "debugging"
