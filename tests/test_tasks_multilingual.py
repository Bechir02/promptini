# -*- coding: utf-8 -*-
"""Multilingual task classification (French + Arabic/Derja) + Arabizi normalizer."""

from core.tasks import detect_task_type, normalize_arabizi


FRENCH = [
    ("corrige le bug dans ma fonction login", "debugging"),
    ("resume ce document en 3 points", "summarization"),
    ("ecris un article de blog sur l'IA", "writing"),
    ("analyse ces donnees et compare les tendances", "analysis"),
    ("cree une fonction python qui lit un csv", "code_generation"),
    ("refactorise ce code pour le simplifier", "refactoring"),
    ("extraire le nom et l'email de ce json", "extraction"),
]

ARABIC = [
    ("لخص هذا المستند في ثلاث نقاط", "summarization"),
    ("صحح الخطأ في هذه الدالة", "debugging"),
    ("اكتب كود بايثون يقرأ ملف csv", "code_generation"),
    ("حلل هذه البيانات وقارن النتائج", "analysis"),
    ("استخرج الاسم والبريد من هذا json", "extraction"),
]


def test_french_classification():
    for prompt, expected in FRENCH:
        assert detect_task_type(prompt) == expected, f"{prompt!r} -> {detect_task_type(prompt)} != {expected}"


def test_arabic_classification():
    for prompt, expected in ARABIC:
        assert detect_task_type(prompt) == expected, f"{prompt!r} -> {detect_task_type(prompt)} != {expected}"


def test_normalize_arabizi_collapses_and_maps():
    assert normalize_arabizi("Loooool") == "lool"
    assert normalize_arabizi("9alleb") == "qalleb"
    # digit-letters that map to empty just drop out
    assert "3" not in normalize_arabizi("3andi mochkla")
