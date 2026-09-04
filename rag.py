import logging
import time

from ingest import retrieve
from llm import transform_prompt
from core.constants import ALLOWED_MODELS, ALLOWED_DEPTHS, DEFAULT_MODEL, DEFAULT_DEPTH, DEFAULT_LANGUAGE
from core.tasks import detect_task_type  # canonical classifier (re-exported)
from core.templates import normalize_language
from core.cache import LRUCache

logger = logging.getLogger(__name__)

# Bounded in-process cache of successful pipeline results.
_PIPELINE_CACHE = LRUCache(128)

__all__ = ["detect_task_type", "run_pipeline"]


def run_pipeline(
    raw_prompt:   str,
    target_model: str = "general",
    depth:        str = "standard",
    language:     str = DEFAULT_LANGUAGE,
    top_k:        int = 3,
) -> dict:

    if target_model not in ALLOWED_MODELS:
        logger.warning("Invalid target_model '%s' — defaulting to %s", target_model, DEFAULT_MODEL)
        target_model = DEFAULT_MODEL
    if depth not in ALLOWED_DEPTHS:
        logger.warning("Invalid depth '%s' — defaulting to %s", depth, DEFAULT_DEPTH)
        depth = DEFAULT_DEPTH
    language = normalize_language(language)

    cache_key = (raw_prompt, target_model, depth, language, top_k)
    cached = _PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        logger.info("Pipeline cache hit.")
        return dict(cached)

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
            language     = language,
        )
        elapsed = time.perf_counter() - start_time
        logger.info("Done — provider: %s | %.2fs", provider, elapsed)
        result = {
            "transformed": transformed,
            "provider":    provider,
            "task_type":   task_type,
            "exemplars":   exemplars,
            "usage":       usage,
            "error":       None,
        }
        _PIPELINE_CACHE.set(cache_key, result)
        return result

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
        ("make an agent that monitors my repo and checks for bugs", "system_prompt"),
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