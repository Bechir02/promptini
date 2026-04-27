from ingest import retrieve
from llm import transform_prompt

def detect_task_type(raw_prompt: str) -> str:
    """
    Keyword-based task type detector.
    Order matters — more specific checks run first.
    """
    prompt_lower = raw_prompt.lower()

    # Extraction first — before code_generation to avoid false matches
    if any(w in prompt_lower for w in [
        "extract", "parse", "pull out", "get fields", "find all",
        "pull", "grab", "fetch fields", "get the", "retrieve fields"
    ]):
        return "extraction"

    if any(w in prompt_lower for w in [
        "fix", "debug", "error", "bug", "issue", "broken",
        "crash", "exception", "not working", "fails"
    ]):
        return "debugging"

    if any(w in prompt_lower for w in [
        "review", "check", "audit", "evaluate", "assess", "critique"
    ]):
        return "code_review"

    if any(w in prompt_lower for w in [
        "refactor", "clean", "improve", "optimize",
        "restructure", "simplify"
    ]):
        return "refactoring"

    if any(w in prompt_lower for w in [
        "document", "docs", "docstring", "readme",
        "comment", "explain"
    ]):
        return "documentation"

    if any(w in prompt_lower for w in [
        "analyze", "analysis", "compare", "research",
        "investigate", "study"
    ]):
        return "analysis"

    if any(w in prompt_lower for w in [
        "summarize", "summary", "tldr", "brief",
        "overview", "recap"
    ]):
        return "summarization"

    if any(w in prompt_lower for w in [
        "system prompt", "persona", "act as",
        "you are", "role", "agent"
    ]):
        return "system_prompt"

    if any(w in prompt_lower for w in [
        "write", "create", "build", "implement", "generate", "code",
        "function", "class", "script", "program", "develop"
    ]):
        return "code_generation"

    if any(w in prompt_lower for w in [
        "story", "essay", "blog", "article",
        "creative", "poem", "write"
    ]):
        return "writing"

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

    # Step 1 — detect task type
    task_type = detect_task_type(raw_prompt)
    print(f"Detected task type: {task_type}")

    # Step 2 — retrieve exemplars
    try:
        exemplars = retrieve(
            query        = raw_prompt,
            target_model = target_model,
            task_type    = task_type,
            top_k        = top_k,
        )
        print(f"Retrieved {len(exemplars)} exemplars.")
    except Exception as e:
        print(f"Retrieval failed: {e} — proceeding without exemplars.")
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
        return {
            "transformed": transformed,
            "provider":    provider,
            "task_type":   task_type,
            "exemplars":   exemplars,
            "error":       None,
        }

    except Exception as e:
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
        ("pull out the user id and email from this json", "general"),
        ("fix the bug in my login function", "cursor"),
        ("write a python csv duplicate finder", "claude-code"),
    ]
    for prompt, model in tests:
        detected = detect_task_type(prompt)
        print(f"'{prompt[:50]}' → {detected}")