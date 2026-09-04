# Scale & Extend — Roadmap

Full scan of Prompt Forge Rag with concrete, prioritized options to (A) deepen
Tunisian Derja support, (B) extend the prompting system, and (C) make it scale.
Each item names the file(s) it touches and is independently shippable with tests.

---

## Shipped this session (increment 1)

- **Derja / Arabizi input understanding** — `llm.build_system_prompt` always
  injects `DERJA_INPUT_NOTE` (Arabizi digit map 3=ع 7=ح 9=ق…, franco-arabe,
  code-switching), so a rough prompt in Tunisian Derja is understood, not refused.
- **Output-language selector** — `auto / english / arabic / derja / french`
  threaded through `core.constants → core.templates → llm → rag → app` UI.
- **Tests** — `tests/test_language.py` (normalization, coverage, injection),
  all 41 tests green.

---

## A. Tunisian Derja — make it first-class

| # | Item | Touches | Effort | Impact |
|---|---|---|---|---|
| A1 | ✅ Input understanding + output language | done | — | high |
| A2 | **Derja-aware retrieval** — swap `bge-small` (English) for a multilingual embedder (`bge-m3` or `paraphrase-multilingual-MiniLM`) so Derja/Arabic input retrieves relevant exemplars | `core.config`, `ingest` | M | high |
| A3 | **Derja/Arabic task classifier** — `detect_task_type` is English-keyword only, so Derja → `general`; add an Arabic/Derja keyword layer (or embedding classifier) | `core.tasks` | S | med |
| A4 | **Derja/MENA exemplar corpus** — curate Tunisian SMB prompts (invoicing, support, marketing) in Derja; seed synthetically, human-review | `fetch_corpus`, data | L | high (MENA) |
| A5 | **RTL + Arabic UI** — right-to-left layout + Arabic labels when Arabic/Derja is selected | `app.py` CSS | S | med |
| A6 | **Arabizi→Arabic normalizer** — transliterate before retrieval/classification so Latin-script Derja matches Arabic exemplars | `core` new module | S | med |

## B. Prompting system — extend what we have

| # | Item | Touches | Effort | Impact |
|---|---|---|---|---|
| B1 | **More target models** — o1/o3, DeepSeek, Qwen, Grok, v0, Bolt, Windsurf, Kimi. `validate_templates()` already enforces coverage | `core.constants`, `core.templates`, `scorer` | S each | high |
| B2 | **More task types** — translation, localization, data-cleaning, SQL, agent/tooling, marketing-copy (close to your day-to-day) | `core.constants`, `core.templates`, `core.tasks` | S each | high |
| B3 | **Domain hint** — optional industry dimension (finance/credit for Cedar Rose, SMB) that enriches `TASK_GUIDANCE` | `core`, `app` | M | med |
| B4 | **Judge calibration** — few-shot + strict JSON schema in `score_with_llm`/`judge_battle` for consistent, less-noisy scores | `scorer` | S | med |
| B5 | **Actionable lint** — surface the score "why" (missing sections, vague words you already compute) as fix suggestions in the UI | `scorer`, `app` | S | med |
| B6 | **Chaining / decompose mode** — break a big task into an ordered prompt sequence | `rag`, `llm`, `app` | M | med |
| B7 | **Editable templates (YAML)** — move `OUTPUT_FORMATS`/`TASK_GUIDANCE` to YAML + tiny admin view so non-devs iterate | `core.templates` | M | med |
| B8 | **Streaming output** — Groq SSE for faster perceived latency | `llm`, `app` | M | med |

## C. Scaling — make it hold up

| # | Item | Touches | Effort | Impact |
|---|---|---|---|---|
| C1 | **Cold-start / persistent index** — ship prebuilt cache (LFS) or a persistent-disk host; biggest free-tier UX lever (see `docs/DEPLOYMENT.md`) | repo/host | S | high |
| C2 | **Parallelize the arena** — `forge()` runs the two transforms + scores + judge serially (3–6 LLM calls back-to-back); run concurrently (thread pool/asyncio) → ~2× faster | `app.py`, `rag` | S | high |
| C3 | **Provider resilience** — retry-with-backoff on 429/5xx + more free providers (OpenRouter, Together, Google AI Studio) behind the existing `call_llm` routing | `llm`, `core.config` | M | high |
| C4 | **Response cache** — cache identical `(prompt, model, depth, language)` results (in-proc LRU + optional disk) to cut cost, latency, and rate-limit pressure | `rag`/`llm` | S | med |
| C5 | **Incremental corpus builder** — `fetch_corpus` refetches everything with no ETag; add conditional requests + a manifest so refreshes are cheap and the corpus can grow | `fetch_corpus` | M | med |
| C6 | **Semantic dedup** — corpus dedup is a 200-char prefix; add embedding near-dup detection so quality holds as it grows | `fetch_corpus` | S | med |
| C7 | **Rate limiting** — per-IP token bucket on the public Space to protect your free provider quotas | `app.py` | S | med |
| C8 | **Observability** — counters for provider used / fallback rate / latency, surfaced in logs or a `/health` view | `rag`, `llm` | S | low |
| C9 | **Engine API** — expose `run_pipeline` via a small FastAPI endpoint so the VS Code extension (and other clients) call the engine directly, not only through the Gradio iframe | new `api.py` | M | med |

---

## Recommended next build order

1. **C2 — parallelize the arena** (fast, high felt-impact, low risk).
2. **C3 — provider resilience + more free providers** (reliability + scale headroom).
3. **A2 + A3 — Derja-aware retrieval + classifier** (makes Derja first-class, not just understood).
4. **B1 + B2 — more models + task types** (breadth).
5. **A4 — Derja/MENA corpus** (highest local value, most effort — do once the plumbing above is in place).

*S = hours · M = a day · L = multi-day. Effort/impact are rough.*
