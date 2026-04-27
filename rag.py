from ingest import retrieve
from llm import transform_prompt

def detect_task_type(raw_prompt: str) -> str:
    """
    Simple keyword-based task type detector.
    Used to improve retrieval when user hasn't specified task type.
    """
    prompt_lower = raw_prompt.lower()

    if any(w in prompt_lower for w in [
        "write", "create", "build", "implement", "generate", "code",
        "function", "class", "script", "program", "develop"
    ]):
        return "code_generation"

    if any(w in prompt_lower for w in [
        "review", "check", "audit", "evaluate", "assess", "critique"
    ]):
        return "code_review"

    if any(w in prompt_lower for w in [
        "fix", "debug", "error", "bug", "issue", "broken", "crash", "exception"
    ]):
        return "debugging"

    if any(w in prompt_lower for w in [
        "refactor", "clean", "improve", "optimize", "restructure", "simplify"
    ]):
        return "refactoring"

    if any(w in prompt_lower for w in [
        "document", "docs", "docstring", "readme", "comment", "explain"
    ]):
        return "documentation"

    if any(w in prompt_lower for w in [
        "analyze", "analysis", "compare", "research", "investigate", "study"
    ]):
        return "analysis"

    if any(w in prompt_lower for w in [
        "summarize", "summary", "tldr", "brief", "overview", "recap"
    ]):
        return "summarization"

    if any(w in prompt_lower for w in [
        "extract", "parse", "pull out", "get fields", "find all", "retrieve"
    ]):
        return "extraction"

    if any(w in prompt_lower for w in [
        "write", "story", "essay", "blog", "article", "creative", "poem"
    ]):
        return "writing"

    if any(w in prompt_lower for w in [
        "system prompt", "persona", "act as", "you are", "role", "agent"
    ]):
        return "system_prompt"

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

    Returns:
        {
            "transformed": str,
            "provider":    str,
            "task_type":   str,
            "exemplars":   list[dict],
            "error":       str or None,
        }
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
    result = run_pipeline(
        raw_prompt   = "write me a python function that reads a csv and finds duplicates",
        target_model = "claude-code",
        depth        = "standard",
    )

    print("\n── Result ──────────────────────────────────────────")
    print(f"Provider  : {result['provider']}")
    print(f"Task type : {result['task_type']}")
    print(f"Exemplars : {len(result['exemplars'])}")
    print(f"Error     : {result['error']}")
    print("\n── Transformed Prompt ──────────────────────────────")
    print(result["transformed"])