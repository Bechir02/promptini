import logging
import time

from ingest import retrieve
from llm import transform_prompt

logger = logging.getLogger(__name__)

ALLOWED_MODELS = {
    "claude-code", "claude", "gpt-4", "cursor",
    "gemini", "llama", "mistral", "copilot", "general"
}
ALLOWED_DEPTHS = {"concise", "standard", "comprehensive"}


def detect_task_type(raw_prompt: str) -> str:
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
        "summarize", "summary", "tldr", "brief",
        "overview", "recap", "condense", "main points",
        "key takeaways", "gist", "abstract",
    ]):
        return "summarization"

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

    return "general"


def run_pipeline(
    raw_prompt:   str,
    target_model: str = "general",
    depth:        str = "standard",
    top_k:        int = 3,
) -> dict:

    if target_model not in ALLOWED_MODELS:
        logger.warning("Invalid target_model '%s' — defaulting to general", target_model)
        target_model = "general"
    if depth not in ALLOWED_DEPTHS:
        logger.warning("Invalid depth '%s' — defaulting to standard", depth)
        depth = "standard"

    task_type  = detect_task_type(raw_prompt)
    start_time = time.perf_counter()
    logger.info("Task type: %s | Model: %s | Depth: %s", task_type, target_model, depth)

    try:
        exemplars = retrieve(
            query        = raw_prompt,
            target_model = target_model,
            task_type    = task_type,
            top_k        = top_k,
        )
        logger.info("Retrieved %d exemplars.", len(exemplars))
    except Exception:
        logger.exception("Retrieval failed — proceeding without exemplars.")
        exemplars = []

    try:
        transformed, provider, usage = transform_prompt(
            raw_prompt   = raw_prompt,
            target_model = target_model,
            task_type    = task_type,
            depth        = depth,
            exemplars    = exemplars,
        )
        elapsed = time.perf_counter() - start_time
        logger.info("Done — provider: %s | %.2fs", provider, elapsed)
        return {
            "transformed": transformed,
            "provider":    provider,
            "task_type":   task_type,
            "exemplars":   exemplars,
            "usage":       usage,
            "error":       None,
        }

    except Exception as e:
        logger.exception("Transformation failed.")
        return {
            "transformed": "",
            "provider":    "none",
            "task_type":   task_type,
            "exemplars":   exemplars,
            "usage":       {},
            "error":       str(e),
        }


if __name__ == "__main__":
    tests = [
        ("pull out the user id and email from this json",           "extraction"),
        ("fix the bug in my login function",                        "debugging"),
        ("write a python csv duplicate finder",                     "code_generation"),
        ("make an agent that monitors my repo and checks for bugs", "code_review"),
        ("review this code for security issues",                    "code_review"),
        ("build an agent that acts as a customer support bot",      "system_prompt"),
        ("summarize this document",                                 "summarization"),
        ("refactor this function to be cleaner",                    "refactoring"),
    ]

    print("── Task Detection Tests ──────────────────────────────")
    all_passed = True
    for prompt, expected in tests:
        detected = detect_task_type(prompt)
        status   = "✅" if detected == expected else "❌"
        if detected != expected:
            all_passed = False
        print(f"{status} '{prompt[:50]}' → {detected} (expected: {expected})")
    print(f"\n{'✅ All passed!' if all_passed else '❌ Some failed'}")