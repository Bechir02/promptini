# -*- coding: utf-8 -*-
"""Classification of the new task types (B2)."""

from core.tasks import detect_task_type

CASES = [
    ("translate this paragraph to French", "translation"),
    ("traduire ce texte en anglais", "translation"),
    ("ترجم هذا النص للانجليزية", "translation"),
    ("localize this app for the Tunisian market", "localization"),
    ("clean the data and drop duplicates", "data_cleaning"),
    ("handle missing values in this dataframe", "data_cleaning"),
    ("write a sql query to join orders and customers", "sql"),
    ("select * from users where active", "sql"),
    ("write ad copy for a new coffee brand", "marketing_copy"),
    ("give me a slogan for my startup", "marketing_copy"),
]

def test_new_task_types():
    for prompt, expected in CASES:
        got = detect_task_type(prompt)
        assert got == expected, f"{prompt!r} -> {got} != {expected}"


def test_existing_english_cases_unbroken():
    # sanity: a few of the original cases must not be stolen by new branches
    assert detect_task_type("write a python csv duplicate finder") == "code_generation"
    assert detect_task_type("refactor this function to be cleaner") == "refactoring"
    assert detect_task_type("summarize this document") == "summarization"
