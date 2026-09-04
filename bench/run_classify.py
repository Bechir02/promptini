# -*- coding: utf-8 -*-
"""Benchmark: multilingual task-classification accuracy.

Ground-truth labeled prompts across English, French, Arabic (script) and
Tunisian Derja written in Arabizi/franco-arabe (Latin). Run:  python -m bench.run_classify
"""

from collections import defaultdict

from core.tasks import detect_task_type

# (language, prompt, expected_task_type)
CASES = [
    # ── English (baseline) ──
    ("en", "fix the null pointer bug in checkout", "debugging"),
    ("en", "summarize this contract in 5 bullet points", "summarization"),
    ("en", "write a python function to parse a csv", "code_generation"),
    ("en", "extract the invoice number and total from this text", "extraction"),
    ("en", "translate this paragraph to Spanish", "translation"),
    ("en", "refactor this class to reduce complexity", "refactoring"),
    ("en", "review this PR for security issues", "code_review"),
    ("en", "write ad copy for a coffee shop", "marketing_copy"),
    ("en", "write a sql query to get monthly revenue", "sql"),
    ("en", "clean the data and remove duplicates", "data_cleaning"),

    # ── French ──
    ("fr", "corrige le bug dans la fonction de paiement", "debugging"),
    ("fr", "resume ce rapport en 5 points", "summarization"),
    ("fr", "ecris une fonction python pour lire un csv", "code_generation"),
    ("fr", "extrais le numero de facture et le total", "extraction"),
    ("fr", "traduis ce texte en anglais", "translation"),
    ("fr", "refactorise cette classe pour la simplifier", "refactoring"),
    ("fr", "fais une revue de ce code pour la securite", "code_review"),
    ("fr", "ecris un slogan pour un cafe", "marketing_copy"),
    ("fr", "ecris une requete sql pour le chiffre d affaires mensuel", "sql"),
    ("fr", "nettoie les donnees et supprime les doublons", "data_cleaning"),

    # ── Arabic (script) ──
    ("ar", "صحح الخطأ في دالة الدفع", "debugging"),
    ("ar", "لخص هذا التقرير في خمس نقاط", "summarization"),
    ("ar", "اكتب دالة بايثون لقراءة ملف csv", "code_generation"),
    ("ar", "استخرج رقم الفاتورة والمجموع", "extraction"),
    ("ar", "ترجم هذا النص إلى الإنجليزية", "translation"),
    ("ar", "حسن هذا الكود وبسطه", "refactoring"),
    ("ar", "راجع هذا الكود للأمان", "code_review"),
    ("ar", "اكتب شعارا لمقهى", "marketing_copy"),
    ("ar", "اكتب استعلام sql للإيرادات الشهرية", "sql"),
    ("ar", "نظف البيانات واحذف التكرارات", "data_cleaning"),

    # ── Tunisian Derja in Arabizi / franco-arabe (Latin) ──
    ("derja", "salla7li el bug fi login", "debugging"),
    ("derja", "lakhesli el contrat f 5 points", "summarization"),
    ("derja", "3malli fonction python bech ta9ra csv", "code_generation"),
    ("derja", "7awelli hedha lel anglais", "translation"),
    ("derja", "na77i el duplicates mel data", "data_cleaning"),
    ("derja", "raja3li el code hedha 3al securite", "code_review"),
    ("derja", "a3mel slogan l cafe mte3i", "marketing_copy"),
    ("derja", "ekteb requete sql mte3 revenue", "sql"),
    ("derja", "fassarli el code hedha", "documentation"),
    ("derja", "na9es el complexite w naddaf el code", "refactoring"),
    ("derja", "estakhrejli el email w el tel men hedha", "extraction"),
    ("derja", "3awenni na3mel agent bech yjaweb el clients", "system_prompt"),
]


def run():
    by_lang = defaultdict(lambda: [0, 0])   # lang -> [correct, total]
    fails = []
    for lang, prompt, expected in CASES:
        got = detect_task_type(prompt)
        ok = (got == expected)
        by_lang[lang][1] += 1
        if ok:
            by_lang[lang][0] += 1
        else:
            fails.append((lang, prompt, expected, got))

    print("── Task-classification accuracy ─────────────────────")
    total_c = total_n = 0
    for lang in ["en", "fr", "ar", "derja"]:
        c, n = by_lang[lang]
        total_c += c; total_n += n
        print(f"  {lang:6s} {c:2d}/{n:2d}  {100*c/n:5.1f}%")
    print(f"  {'ALL':6s} {total_c:2d}/{total_n:2d}  {100*total_c/total_n:5.1f}%")

    if fails:
        print("\n── Misclassified ───────────────────────────────────")
        for lang, prompt, expected, got in fails:
            print(f"  [{lang}] {prompt[:44]:44s} exp {expected} -> {got}")
    return total_c, total_n


if __name__ == "__main__":
    run()
