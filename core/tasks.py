"""Canonical task-type classifier.

This is the single implementation used by BOTH the live pipeline (rag.py) and
the corpus builder (fetch_corpus.py). Previously each file had its own slightly
different keyword lists, so corpus labels and live-query labels could disagree
and silently hurt retrieval. This version is the superset (the richer rag.py
ruleset) so live behavior is unchanged.

Kept dependency-free and pure so it is trivially testable.
"""

from __future__ import annotations

from .constants import DEFAULT_TASK_TYPE


def detect_task_type(raw_prompt: str) -> str:
    """Classify a prompt into one of the known task types via keyword rules.

    Order matters: earlier branches win. Returns ``"general"`` as fallback.
    """
    prompt_lower = raw_prompt.lower()

    if any(w in prompt_lower for w in [
        "extract", "pull out", "get fields", "fetch fields",
        "retrieve fields", "parse json", "parse xml",
        "pull the", "grab the fields",
    ]):
        return "extraction"

    if any(w in prompt_lower for w in [
        "system prompt", "persona", "act as",
        "you are a", "build an agent", "make an agent",
        "create an agent", "design an agent",
    ]):
        return "system_prompt"

    if any(w in prompt_lower for w in [
        "review", "audit", "evaluate", "assess", "critique",
        "check for bugs", "check for issues", "check for errors",
        "monitor", "scan for", "look for issues",
    ]) and not any(w in prompt_lower for w in ["fix", "debug"]):
        return "code_review"

    if any(w in prompt_lower for w in [
        "fix", "debug", "error", "bug", "issue", "broken",
        "crash", "exception", "not working", "fails",
    ]) and not any(w in prompt_lower for w in [
        "review", "monitor", "check for", "agent", "scan",
    ]):
        return "debugging"

    if any(w in prompt_lower for w in [
        "refactor", "clean up", "improve", "optimize",
        "restructure", "simplify", "rewrite", "dry",
        "boilerplate", "modularize", "decouple",
    ]):
        return "refactoring"

    # Summarization is checked before documentation: the word "document" in the
    # documentation triggers would otherwise shadow prompts like
    # "summarize this document".
    if any(w in prompt_lower for w in [
        "summarize", "summary", "tldr", "brief",
        "overview", "recap", "condense", "main points",
        "key takeaways", "gist", "abstract",
    ]):
        return "summarization"

    if any(w in prompt_lower for w in [
        "document", "docs", "docstring", "readme",
        "comment", "explain this code",
    ]):
        return "documentation"

    if any(w in prompt_lower for w in [
        "analyze", "analysis", "compare", "research",
        "investigate", "study", "examine",
    ]):
        return "analysis"

    if any(w in prompt_lower for w in [
        "story", "essay", "blog post", "article",
        "creative", "poem", "write about", "draft a",
        "composing", "narrative", "script a",
    ]):
        return "writing"

    if any(w in prompt_lower for w in [
        "write", "create", "build", "implement", "generate",
        "code", "function", "class", "script", "program", "develop",
    ]):
        return "code_generation"

    return DEFAULT_TASK_TYPE
