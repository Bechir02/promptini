# Prompt Forge — Deep-Dive & Improvement Plan

**Author:** prepared for review before any code changes
**Date:** 2026-05-30
**Decision captured:** deliver this plan first → implement after approval. Design direction is my recommendation (below). All four priorities in scope: code quality & correctness, smarter core, visual polish, production-readiness.

---

## 0. How to read this document

Three parts:

1. **Deep-dive** — every strength and weakness, with file/line detail and why it matters.
2. **Improvement plan** — for each strength, how to make it stronger; for each weakness, the concrete fix.
3. **Phased roadmap** — the order I propose to build in, so you can approve phase by phase.

Nothing here has been changed in your code yet.

---

## PART 1 — DEEP DIVE

### The system in one paragraph

You paste a messy prompt and pick up to two target models plus a "depth." The app classifies the task by keywords (`rag.detect_task_type`), retrieves similar high-quality example prompts from a LanceDB vector store of 8,435 curated prompts (`ingest.retrieve`), builds a large model-and-task-specific system prompt (`llm.build_system_prompt`), and rewrites your prompt with Groq's Llama-3.3-70B (falling back to Cerebras). Each result is scored on six dimensions (`scorer.score_transformation`), and if you picked two models an LLM judge declares a winner. A VS Code extension wraps the Gradio UI.

---

### Strengths (deep)

**S1 — Clean module separation.** Retrieval (`ingest.py`), transformation (`llm.py`), scoring (`scorer.py`), orchestration (`rag.py`), and UI (`app.py`) each own one concern with narrow interfaces. `run_pipeline` is the single seam everything passes through. This makes the codebase easy to reason about and to test in isolation.

**S2 — Defensive fallback chains everywhere.** Retrieval degrades exact-match → model-only → general (`ingest.retrieve`, lines 182–221). The LLM layer tries Groq then Cerebras (`llm.transform_prompt`, 426–440). The scorer falls back to a heuristic score if the judge LLM fails (`scorer.score_with_llm`, 303–311). `run_pipeline` wraps retrieval and transformation in try/except so the UI never hard-crashes. This is real production thinking.

**S3 — Hybrid retrieval.** `retrieve` combines vector search and full-text (FTS) search and merges/dedupes by id (lines 188–217). Hybrid beats either method alone for short, keyword-heavy queries.

**S4 — Embedding cache.** `_load_embedding_cache` / `_save_embedding_cache` hash the corpus by prompt ids (`_get_cache_key`) and reuse vectors on disk, so only the first build is slow. Smart cold-start optimization.

**S5 — Detailed model-aware prompt engineering.** `OUTPUT_FORMATS` and `TASK_GUIDANCE` in `llm.py` encode genuinely thoughtful, per-model formatting rules (XML for Claude, markdown for GPT-4, minimal/action-first for Cursor, etc.) plus an `ANTI_GENERIC` rules block. This is the real domain value of the product.

**S6 — Quality-filtered corpus pipeline.** `fetch_corpus.py` scrapes ~35 repos, scores each candidate (`quality_score`), drops jailbreaks and low-signal text, dedupes by content prefix, and tags license + source per prompt. Provenance and licensing are tracked — unusually disciplined for a scraped dataset.

---

### Weaknesses (deep)

**W1 — Task detection is brittle and duplicated.** `detect_task_type` exists in BOTH `rag.py` (lines 16–87) and `fetch_corpus.py` (283–347), with *different* keyword lists. So the corpus is labeled by one ruleset and live queries by another — the `task_type` filter in retrieval can therefore miss matching exemplars. The logic is also order-dependent keyword matching, so "write a summary of this bug fix" classifies unpredictably.

**W2 — Filter strings built by f-string interpolation.** `ingest.retrieve` (lines 170–179) builds LanceDB `where` clauses by interpolating `target_model`/`task_type`/`min_quality` directly. Today the values are validated against allow-lists in `rag.run_pipeline`, so it's safe *now*, but `retrieve` is a public function with no validation of its own — any future caller passing a raw string is an injection/break risk.

**W3 — Heuristic scorer measures formatting, not quality.** `scorer.py` rewards the presence of substrings (`<role>`, "error", digits) and word-count ratio (`score_improvement`). A verbose prompt stuffed with the right keywords scores high regardless of actual quality. The six sub-scores also overlap (structure vs model-awareness check nearly the same tags), inflating correlated signal.

**W4 — Dead/incorrect model argument.** `scorer.judge_battle` calls `call_llm(prompt, model="groq/llama-3.1-70b-versatile", ...)` but `llm.call_llm` ignores the `model` argument entirely and the configured model is `llama-3.3-70b-versatile`. The parameter is dead code and the string is wrong — a clear leftover bug.

**W5 — Repo hygiene.** `prompts.json` (6 MB, 8,435 records) is git-tracked, bloating clones and diffs. (`.env`, `lancedb_store/`, and `*.vsix` ARE correctly gitignored — good.) Thirteen built `.vsix` artifacts sit in `vscode-extension/` on disk.

**W6 — No real test suite.** The only checks are print-based asserts in `rag.py`'s `__main__`. No tests for retrieval correctness, scoring, the fallback chains, or the corpus parsers. Refactoring is risky without them.

**W7 — Secrets / config handling is ad hoc.** API keys are read with `os.environ.get(..., "")` and failures surface only at call time as exceptions. No central config, no startup validation, no clear separation of dev vs deployed config.

**W8 — Duplicated constants & magic values.** The model list lives in `app.MODELS`, `rag.ALLOWED_MODELS`, `llm.OUTPUT_FORMATS`, and `scorer` maps — four places to keep in sync. `min_quality=7.0`, weights, and depth word-budgets are scattered literals.

**W9 — UI/UX gaps.** `forge` returns `[""] * 11` on empty input with no user feedback; long LLM calls have no loading state; the "Copy/Save" relies on `document.activeElement` which is fragile; the score panel concatenates raw text. The cream/orange theme is custom but not a recognized design system.

---

## PART 2 — IMPROVEMENT PLAN

### Make each strength stronger

**S1 → ** Introduce a thin `core/` package (`core/retrieval.py`, `core/transform.py`, `core/scoring.py`, `core/tasks.py`, `core/config.py`) and a `PipelineResult` dataclass instead of loose dicts, so interfaces are typed and discoverable. Add `__all__` and type hints throughout.

**S2 → ** Make fallbacks observable: structured logging with a `provider`/`fallback_reason` field, and a small retry-with-backoff wrapper around network calls so transient 429/5xx don't trigger an unnecessary provider switch.

**S3 → ** Upgrade hybrid retrieval to proper score fusion (Reciprocal Rank Fusion) instead of "vector hits first, then FTS." Normalize and combine ranks so a strong keyword match can outrank a mediocre vector match. Make `top_k` and the vector/FTS blend configurable.

**S4 → ** Version the cache key by embedding-model name + prompt *content* hash (not just ids), so changing the model or editing a prompt's text correctly invalidates. Store cache under a `.cache/` dir and add a `--rebuild` flag.

**S5 → ** Externalize `OUTPUT_FORMATS`/`TASK_GUIDANCE` into a `templates/` directory (YAML or per-model files) so prompt-engineering iteration doesn't require touching Python. Add a tiny linter that checks every model has all required keys. This is the product's crown jewel — make it editable by non-developers.

**S6 → ** Add a `corpus stats` report (counts by model/task/quality, license breakdown), checksum the output, and make `fetch_corpus` incremental (skip unchanged repos via ETag). Add a schema validation pass on `prompts.json`.

### Fix each weakness

**W1 → ** Single source of truth: move `detect_task_type` into `core/tasks.py`, import it in both `rag` and `fetch_corpus`. Re-label the corpus with the unified function (one-time migration). Optionally back the classifier with embedding-similarity to task descriptions instead of keyword lists, with keyword matching as a fast pre-filter. Add a labeled test set.

**W2 → ** Validate inputs inside `retrieve` itself (allow-list + cast `min_quality` to float), and/or use LanceDB parameterized/escaped filters. Never trust the caller.

**W3 → ** Rebalance scoring: keep heuristics as cheap "lint" checks but cap their combined weight; lean more on a rubric-driven LLM judge with a fixed JSON schema and few-shot calibration. Remove overlapping signals (merge structure + model-awareness). Replace the word-count "improvement" score with a semantic check that the rewrite preserves intent (embedding similarity to the original) AND adds constraints.

**W4 → ** Make `call_llm` actually honor a `model` parameter (map friendly names → provider+model), fix the model string, and route `judge_battle` through it correctly. Add a unit test that asserts the requested model is used.

**W5 → ** Stop tracking `prompts.json` in git; publish it as a release asset / HF dataset and download on build, OR keep it but document it as a data artifact. Add `vscode-extension/*.vsix` cleanup; keep only the latest in a `dist/`.

**W6 → ** Add `pytest`: unit tests for `detect_task_type` (table-driven, the existing cases plus tricky ones), `retrieve` (fixture DB), each `score_*` function, the fallback chains (mock providers), and corpus parsers. Wire a GitHub Actions CI to run them.

**W7 → ** Central `core/config.py` using `pydantic-settings`: typed settings, startup validation that fails fast with a clear message if required keys are missing, and a `/health` style readiness check in the app. Keep reading from env/Space secrets.

**W8 → ** Define `MODELS`, depths, weights, and thresholds once (in `core/config.py` or a `constants.py`) and import everywhere. Derive `app.MODELS` and `rag.ALLOWED_MODELS` from the same source.

**W9 → ** Covered by the redesign below.

---

## PART 3 — DESIGN RECOMMENDATION

**My recommendation: restyle Gradio in-place (Option A), not a separate front-end.**

Reasoning: the VS Code extension embeds the Gradio app via webview and talks to it through `postMessage` events (`copyText`, `savePrompt`, `setPrompt`, `syncLibrary`). A standalone React front-end would mean re-implementing all of that plumbing, the server, and the extension bridge — high effort, high regression risk, for the same user-visible result. Gradio's theming + custom CSS can fully achieve a Claude-grade look, and it keeps one codebase. If you later outgrow Gradio, the clean `core/` package from Part 2 makes a front-end swap cheap — so we lose nothing by waiting.

**"Claude design" target — what I'll apply:**

- **Palette:** move from cream/orange to Anthropic's restrained, warm-neutral system — off-white/paper background, near-black ink text, a single warm clay/terracotta accent used sparingly (Anthropic's signature is calm, low-saturation, lots of whitespace). Semantic colors for score grades (green/amber/red) kept muted.
- **Typography:** a clean humanist sans for UI (e.g. the Styrene/Inter family feel) and a refined mono for prompt output; tighter heading tracking, generous line-height. Replace the current DM Sans/DM Mono mix with a more deliberate scale.
- **Layout & components:** softer elevation (single subtle shadow, not double), consistent 8px spacing grid, rounded-but-not-bubbly radii, clear input→output visual hierarchy, a real "Arena" split view with aligned panels.
- **States & feedback:** proper loading/skeleton state during the LLM call, toast confirmations for copy/save (replace the fragile `activeElement` hack), empty-state messaging instead of silent blank returns, and a redesigned score card (compact bars per dimension + grade chip) instead of concatenated text.
- **Accessibility:** focus rings, color-contrast on the muted palette, keyboard-operable buttons.

I'll keep all existing `elem_classes` hooks and the `postMessage` contract intact so the extension keeps working.

---

## PART 4 — PHASED ROADMAP (proposed build order)

Each phase is independently reviewable. I'll stop after each for your sign-off if you prefer.

**Phase 1 — Safety net & correctness (low risk, high value)**
1. Add `pytest` + CI; capture current behavior in tests first.
2. Fix W4 (dead/incorrect model arg) and W2 (validate filters in `retrieve`).
3. Unify `detect_task_type` (W1) into `core/tasks.py`; keep behavior identical, prove with tests.

**Phase 2 — Structure & config (refactor under test)**
4. Create `core/` package, `core/config.py` (pydantic-settings), single source of truth for constants (W8, W7).
5. Move templates to `templates/` (S5) with a key-completeness linter.

**Phase 3 — Smarter core (measurable quality)**
6. RRF score fusion in retrieval (S3) + configurable blend.
7. Rebalanced scoring + intent-preservation check (W3); content-hash cache key (S4).
8. Optional embedding-based task classifier (W1 upgrade), gated behind the test set.

**Phase 4 — Design / UX (the visible upgrade)**
9. Claude-design Gradio restyle (Part 3): palette, type, layout, loading states, redesigned score card, toasts.
10. Empty-state and error messaging.

**Phase 5 — Hygiene & ship**
11. Repo cleanup (W5), corpus stats + schema validation (S6), README/architecture docs, deploy checklist.

---

## Open questions for you

1. Approve restyle-in-place for the design (my rec), or do you specifically want a separate front-end?
2. Should I proceed phase-by-phase with a check-in after each phase, or run Phases 1–3 straight through and check in before the redesign?
3. Are Groq + Cerebras the only providers, or should the config support adding more (e.g. an OpenAI/Anthropic key) as additional fallbacks?
