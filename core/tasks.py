"""Canonical task-type classifier (multilingual).

Single implementation used by BOTH the live pipeline (rag.py) and the corpus
builder (fetch_corpus.py). English keyword behavior is unchanged (locked by
tests); French and Arabic/Derja keywords are added so Tunisian users writing in
Derja, franco-arabe, or French get correctly-classified prompts too.

Kept dependency-free and pure so it is trivially testable.
"""

from __future__ import annotations

import re
import unicodedata

from .constants import DEFAULT_TASK_TYPE

# ── Light normalization for matching ─────────────────────────────────────────
_ARABIZI_MAP = {"3": "", "7": "", "9": "q", "2": "", "5": "kh", "8": "gh", "6": "t"}


def normalize_arabizi(text: str) -> str:
    """Light normalization of Latin-script Derja (Arabizi).

    Lowercases, collapses elongations (``loool`` -> ``lool``) and maps the common
    Arabizi digit-letters so franco-arabe still matches Latin keywords. This is a
    heuristic aid, not a full transliteration.
    """
    t = text.lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    for digit, repl in _ARABIZI_MAP.items():
        t = t.replace(digit, repl)
    return t


def _normalize(text: str) -> str:
    """Lowercase + strip Arabic diacritics/tatweel so keyword matching is robust."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.lower().replace("ـ", "")  # tatweel


def detect_task_type(raw_prompt: str) -> str:
    """Classify a prompt into one of the known task types via keyword rules.

    Order matters: earlier branches win. Keywords span English, French and
    Arabic/Derja. Returns ``"general"`` as fallback.
    """
    p = _normalize(raw_prompt)

    if any(w in p for w in [
        # en
        "extract", "pull out", "get fields", "fetch fields", "retrieve fields",
        "parse json", "parse xml", "pull the", "grab the fields",
        # fr
        "extraire", "extrais", "extraction",
        # ar
        "استخرج", "استخراج", "جيب لي", "هات لي", "استخلص",
    ]):
        return "extraction"

    if any(w in p for w in [
        # en
        "system prompt", "persona", "act as", "you are a", "build an agent",
        "make an agent", "create an agent", "design an agent",
        # fr
        "tu es un", "comporte-toi", "agent qui", "assistant qui", "prompt systeme",
        # ar
        "برومبت", "شخصية", "تصرف ك", "انت مساعد", "اعمل ايجنت", "اصنع ايجنت", "صمم ايجنت",
    ]):
        return "system_prompt"

    if any(w in p for w in [
        # en
        "review", "audit", "evaluate", "assess", "critique", "check for bugs",
        "check for issues", "check for errors", "monitor", "scan for", "look for issues",
        # fr
        "revise", "reviser", "revue", "audite", "verifie", "verifier", "evalue",
        # ar
        "راجع", "مراجعة", "دقق", "افحص", "قيم", "تحقق من",
    ]) and not any(w in p for w in ["fix", "debug", "corrige", "صلح", "صحح"]):
        return "code_review"

    if any(w in p for w in [
        # en
        "fix", "debug", "error", "bug", "issue", "broken", "crash", "exception",
        "not working", "fails",
        # fr
        "corrige", "corriger", "debogue", "deboguer", "erreur", "plante", "ne marche pas",
        # ar
        "صلح", "صحح", "خطأ", "باڨ", "ما يخدمش", "تعطل", "ما يمشيش", "علاش ما",
    ]) and not any(w in p for w in [
        "review", "monitor", "check for", "agent", "scan", "راجع", "audite",
    ]):
        return "debugging"

    if any(w in p for w in [
        "translate", "translation", "traduire", "traduis", "traduction",
        "ترجم", "ترجمة", "ترجملي",
    ]):
        return "translation"

    if any(w in p for w in [
        "localize", "localise", "localization", "localisation", "localiser",
        "adapt for", "adapt this for", "توطين", "اقلمة", "تعريب",
    ]):
        return "localization"

    if any(w in p for w in [
        "clean the data", "clean up the data", "data cleaning", "deduplicate",
        "drop duplicates", "remove duplicates", "missing values", "impute",
        "normalize the data", "nettoyer les donnees", "nettoyage des donnees",
        "تنظيف البيانات", "نظف البيانات",
    ]):
        return "data_cleaning"

    if any(w in p for w in [
        # en
        "refactor", "clean up", "improve", "optimize", "restructure", "simplify",
        "rewrite", "dry", "boilerplate", "modularize", "decouple",
        # fr
        "refactorise", "refactoriser", "nettoie", "ameliore", "ameliorer", "optimise",
        "simplifie", "reecris", "reecrire",
        # ar
        "نظف", "حسن", "بسط", "اعد كتابة", "اعادة هيكلة", "ريفاكتور",
    ]):
        return "refactoring"

    # Summarization before documentation (the word "document" would shadow
    # "summarize this document").
    if any(w in p for w in [
        # en
        "summarize", "summary", "tldr", "brief", "overview", "recap", "condense",
        "main points", "key takeaways", "gist", "abstract",
        # fr
        "resume", "resumer", "synthese", "synthetise", "en bref",
        # ar
        "لخص", "تلخيص", "اختصر", "ملخص", "باختصار", "اهم النقاط",
    ]):
        return "summarization"

    if any(w in p for w in [
        # en
        "document", "docs", "docstring", "readme", "comment", "explain this code",
        # fr
        "documente", "documentation", "commente", "explique le code",
        # ar
        "وثق", "توثيق", "اشرح الكود", "تعليقات",
    ]):
        return "documentation"

    if any(w in p for w in [
        # en
        "analyze", "analysis", "compare", "research", "investigate", "study", "examine",
        # fr
        "analyse", "analyser", "compare", "comparer", "etudie", "etudier", "examine",
        # ar
        "حلل", "تحليل", "قارن", "ادرس", "ابحث", "افحص",
    ]):
        return "analysis"

    if any(w in p for w in [
        "marketing copy", "ad copy", "advert", "advertisement", "slogan",
        "tagline", "landing page copy", "call to action", "product description",
        "publicite", "annonce publicitaire", "وصف منتج", "اعلان", "شعار",
    ]):
        return "marketing_copy"

    if any(w in p for w in [
        # en
        "story", "essay", "blog post", "article", "creative", "poem", "write about",
        "draft a", "composing", "narrative", "script a",
        # fr
        "ecris", "ecrire", "redige", "rediger", "redaction", "essai", "histoire",
        "poeme", "article de blog",
        # ar
        "اكتب مقال", "قصة", "مقال", "انشئ نص", "تدوينة", "قصيدة",
    ]):
        return "writing"

    if any(w in p for w in [
        "sql", "sql query", "select from", "select * from", "joins", "jointure",
        "requete sql", "قاعدة بيانات", "استعلام",
    ]):
        return "sql"

    if any(w in p for w in [
        # en
        "write", "create", "build", "implement", "generate", "code", "function",
        "class", "script", "program", "develop",
        # fr
        "cree", "creer", "genere", "generer", "fonction", "classe", "programme",
        "implemente", "implementer", "developpe", "developper",
        # ar
        "اكتب كود", "اعمل فنكشن", "برمج", "انشئ", "ولد", "سكريبت", "دالة", "كلاس",
    ]):
        return "code_generation"

    return DEFAULT_TASK_TYPE
