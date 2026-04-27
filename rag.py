import logging
import time

from ingest import retrieve
from llm import transform_prompt

logger = logging.getLogger(__name__)
ALLOWED_MODELS = {"claude-code", "gpt-4", "cursor", "gemini", "general"}
ALLOWED_DEPTHS = {"concise", "standard", "comprehensive"}


def detect_task_type(raw_prompt: str) -> str:
    """
    Keyword-based task type detector.
    Order matters — more specific checks run first.
    """
    prompt_lower = raw_prompt.lower()

    # Extraction first — very specific fields/data pulling language
    if any(w in prompt_lower for w in [
        "extract", "pull out", "get fields", "fetch fields",
        "retrieve fields", "parse json", "parse xml",
        "pull the", "grab the fields"
    ]):
        return "extraction"

    # System prompt / agent persona
    if any(w in prompt_lower for w in [
        "system prompt", "persona", "act as",
        "you are a", "build an agent", "make an agent",
        "create an agent", "design an agent"
    ]):
        return "system_prompt"

    # Code review — before debugging to catch "check for bugs"
    if any(w in prompt_lower for w in [
        "review", "audit", "evaluate", "assess", "critique",
        "check for bugs", "check for issues", "check for errors",
        "monitor", "scan for", "look for issues"
    ]):
        return "code_review"

    # Debugging — only if NOT about reviewing/monitoring
    if any(w in prompt_lower for w in [
        "fix", "debug", "error", "bug", "issue", "broken",
        "crash", "exception", "not working", "fails"
    ]) and not any(w in prompt_lower for w in [
        "review", "monitor", "check for", "agent", "scan"
    ]):
        return "debugging"

    # Refactoring
    if any(w in prompt_lower for w in [
        "refactor", "clean up", "improve", "optimize",
        "restructure", "simplify", "rewrite"
    ]):
        return "refactoring"

    # Documentation
    if any(w in prompt_lower for w in [
        "document", "docs", "docstring", "readme",
        "comment", "explain this code"
    ]):
        return "documentation"

    # Analysis
    if any(w in prompt_lower for w in [
        "analyze", "analysis", "compare", "research",
        "investigate", "study", "examine"
    ]):
        return "analysis"

    # Summarization
    if any(w in prompt_lower for w in [
        "summarize", "summary", "tldr", "brief",
        "overview", "recap", "condense"
    ]):
        return "summarization"

    # Creative writing
    if any(w in prompt_lower for w in [
        "story", "essay", "blog post", "article",
        "creative", "poem", "write about"
    ]):
        return "writing"

    # Code generation — broad, so runs late
    if any(w in prompt_lower for w in [
        "write", "create", "build", "implement", "generate",
        "code", "function", "class", "script", "program", "develop"
    ]):
        return "code_generation"

    return "general"


def run_pipeline(
    raw_prompt:   str,
    target_model: str = "general",
    depth:        str = "standard",
    top_k:        int = 3,
) -> dict:
    """
    Full RAG pipeline:
    1. Detect task type from raw prompt
    2. Retrieve relevant exemplars from LanceDB
    3. Transform prompt using LLM with exemplars as context
    4. Return result dict with all metadata
    """

    if target_model not in ALLOWED_MODELS:
        logger.warning("Invalid target_model '%s' — defaulting to general", target_model)
        target_model = "general"
    if depth not in ALLOWED_DEPTHS:
        logger.warning("Invalid depth '%s' — defaulting to standard", depth)
        depth = "standard"

    # Step 1 — detect task type
    task_type = detect_task_type(raw_prompt)
    logger.info("Detected task type: %s", task_type)
    start_time = time.perf_counter()

    # Step 2 — retrieve exemplars
    try:
        exemplars = retrieve(
            query        = raw_prompt,
            target_model = target_model,
            task_type    = task_type,
            top_k        = top_k,
        )
        logger.info("Retrieved %d exemplars.", len(exemplars))
    except Exception as e:
        logger.exception("Retrieval failed — proceeding without exemplars.")
        exemplars = []

    # Step 3 — transform
    try:
        transformed, provider = transform_prompt(
            raw_prompt   = raw_prompt,
            target_model = target_model,
            task_type    = task_type,
            depth        = depth,
            exemplars    = exemplars,
        )
        elapsed = time.perf_counter() - start_time
        logger.info("Transformation completed with provider %s in %.2fs", provider, elapsed)
        return {
            "transformed": transformed,
            "provider":    provider,
            "task_type":   task_type,
            "exemplars":   exemplars,
            "error":       None,
        }

    except Exception as e:
        logger.exception("Transformation failed.")
        return {
            "transformed": "",
            "provider":    "none",
            "task_type":   task_type,
            "exemplars":   exemplars,
            "error":       str(e),
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        # (prompt, expected_task_type)
        ("pull out the user id and email from this json",           "extraction"),
        ("fix the bug in my login function",                        "debugging"),
        ("write a python csv duplicate finder",                     "code_generation"),
        ("make an agent that monitors my repo and checks for bugs", "code_review"),
        ("review this code for security issues",                    "code_review"),
        ("build an agent that acts as a customer support bot",      "system_prompt"),
        ("summarize this document",                                 "summarization"),
        ("refactor this function to be cleaner",                    "refactoring"),
    ]

    print("── Task Detection Tests ─────────────────────────────")
    all_passed = True
    for prompt, expected in tests:
        detected = detect_task_type(prompt)
        status   = "✅" if detected == expected else "❌"
        if detected != expected:
            all_passed = False
        print(f"{status} '{prompt[:50]}' → {detected} (expected: {expected})")

    print(f"\n{'✅ All tests passed!' if all_passed else '❌ Some tests failed — review above'}")