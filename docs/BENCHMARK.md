# Benchmark

## 1. Multilingual task classification

`detect_task_type` is the backbone of retrieval and templating, so it is
benchmarked against a labeled set of 42 prompts across four languages
(`bench/run_classify.py` — reproduce with `python -m bench.run_classify`).

| Language | Baseline | After Derja/Arabizi enhancement |
|---|---|---|
| English | 10/10 · 100% | 10/10 · 100% |
| French | 7/10 · 70% | 10/10 · 100% |
| Arabic (script) | 10/10 · 100% | 10/10 · 100% |
| **Tunisian Derja / Arabizi** | **4/12 · 33%** | **12/12 · 100%** |
| **Overall** | **31/42 · 73.8%** | **42/42 · 100%** |

**What moved the needle**
- Match an Arabizi-digit-folded copy of the text (`salla7`→`sala7`, `7awel`, `na9es`…).
- Per-task Derja/franco-arabe keywords (`salla7`, `lakhes`, `raja3`, `fassar`,
  `na77i`, `estakhrej`, `na3mel agent`, …).
- French fixes: `data_cleaning` "nettoie les données"/"doublons"; removed bare
  "ecris/ecrire" from *writing* (it was stealing *code_generation*).
- English behavior unchanged (locked by the existing test suite).

## 2. Coverage & correctness (this sandbox)

- Template linter: **15 models × 16 task types × 5 output languages**, all covered.
- Test suite: **76 tests** green (classifier, RRF fusion, retrieval validation
  incl. injection shapes, cache, rate limiter, metrics, providers/retry,
  streaming logic, templates, scorer).
- Streaming: verified with deterministic mocks (accumulation, provider fallback,
  non-streaming fallback) — real token streaming runs where LLM egress is open.

## 3. LLM output-quality (run live)

Quality of the actual transform needs a provider key and open egress, so it runs
on your machine / HF, not in this sandbox:

```bash
python -m bench.llm_eval     # scores sample EN/FR/AR/Derja prompts per model
```

It reports average score by language and by model and saves JSON under
`bench/results/`. Use it to track quality as you change templates or models.

## 4. What good looks like next
- Add A2 (multilingual embedder, `PF_EMBEDDING_MODEL=BAAI/bge-m3`) and re-run
  `bench/llm_eval.py` to measure the retrieval lift for Derja/Arabic input.
- Grow `data/derja_seed.json` and watch the by-language averages.
