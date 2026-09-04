# Phases 1–3 — Done (checkpoint before redesign)

All 28 tests pass (1 scorer test skips locally because the provider SDKs aren't
installed here; it runs in CI). No UI/visual changes yet — that is Phase 4.

## New `core/` package (single source of truth)
- `core/constants.py` — MODELS, DEPTHS, TASK_TYPES, retrieval defaults, scoring weights (asserted to sum to 1.0). Removes the duplicated lists that lived in app/rag/llm/scorer.
- `core/tasks.py` — the one canonical `detect_task_type`, imported by both `rag.py` and `fetch_corpus.py` (was duplicated with divergent keyword lists = W1).
- `core/config.py` — typed `pydantic-settings` config; reads only from the environment (never opens a secrets file); `require_provider()` fails fast with a clear message (W7).
- `core/templates.py` — the prompt-engineering dicts (OUTPUT_FORMATS / TASK_GUIDANCE / ANTI_GENERIC) extracted from llm.py + a `validate_templates()` coverage linter (S5).
- `core/fusion.py` — Reciprocal Rank Fusion for hybrid retrieval (S3).
- `core/retrieval.py` — pure `validate_filter_inputs` so it's testable without a DB (W2).

## Fixes
- **W1** task-detection unified; also fixed a real ordering bug ("summarize this document" now → summarization, was → documentation).
- **W2** `retrieve()` now allow-list-validates `target_model`/`task_type` and casts `min_quality` before any value reaches a LanceDB filter string.
- **W3** scoring reweighted so the LLM judge dominates (0.40) and keyword-heuristics are capped — keyword stuffing can no longer inflate a score.
- **W4** `call_llm` now honors the requested provider/model (e.g. `"groq/<model>"`); the dead, wrong `"llama-3.1-70b-versatile"` string in `judge_battle` is gone.
- **W7/W8** config + constants centralized; `llm.py`/`ingest.py`/`app.py` read model names, paths, and params from `core.config`.
- **S3** retrieval now fuses vector + full-text rankings with RRF instead of "vector-first then FTS".
- **S4** embedding cache key now hashes the embedding-model name + actual embedded text, so editing a prompt or switching models correctly invalidates (was id-only).

## Tests & CI
- `tests/` with coverage for task detection, RRF, retrieval validation (incl. injection-shaped inputs), template completeness, constants/config, and scorer heuristics.
- `.github/workflows/ci.yml` runs the template linter + pytest on push/PR.
- `requirements.txt` gains `pydantic` + `pydantic-settings`.

## Deliberately deferred
- Embedding-based task classifier (optional W1 upgrade) — current keyword classifier is locked under tests; can revisit if you want higher accuracy.
- Repo hygiene (untracking the 6 MB `prompts.json`, vsix cleanup) — Phase 5.

## Behavior safety
Live query classification is unchanged except the one intentional summarization
fix. The corpus builder now shares the same classifier, so the next corpus
rebuild will be label-consistent with live queries (the existing committed
`prompts.json` / LanceDB store are untouched until you rebuild).
