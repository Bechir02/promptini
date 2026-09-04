# -*- coding: utf-8 -*-
"""Live LLM-quality benchmark (run where LLM egress is open — your machine / HF).

Scores the transform output for sample multilingual prompts across a few target
models and reports average quality per language and per model. Needs the full
deps + a provider key.

    python -m bench.llm_eval            # console report + saves JSON
"""

import json, time, statistics
from collections import defaultdict
from pathlib import Path

from rag import run_pipeline
from scorer import score_transformation

# messy prompts in each language (as a real user would type them)
SAMPLES = [
    ("en",    "write a python function that dedupes rows in a csv and handles missing files"),
    ("en",    "fix my login it throws a null error sometimes"),
    ("fr",    "corrige le bug de paiement qui plante parfois"),
    ("fr",    "ecris une fonction pour lire un csv et gerer les fichiers manquants"),
    ("ar",    "اكتب دالة بايثون تنظف صفوف csv وتتعامل مع الملفات المفقودة"),
    ("ar",    "لخص هذا العقد في خمس نقاط لصاحب مشروع صغير"),
    ("derja", "3malli fonction python bech tnaddaf csv w thandli les fichiers li mahomch mawjoudin"),
    ("derja", "salla7li el login yatih null error mara mara"),
]

MODELS = ["claude-code", "gpt-4", "general"]


def run():
    by_lang = defaultdict(list)
    by_model = defaultdict(list)
    rows = []
    for lang, prompt in SAMPLES:
        for model in MODELS:
            t0 = time.time()
            res = run_pipeline(raw_prompt=prompt, target_model=model, depth="standard")
            dt = time.time() - t0
            if res.get("error") or not res.get("transformed"):
                print(f"  [{lang}/{model}] ERROR: {res.get('error')}")
                continue
            sc = score_transformation(prompt, res["transformed"], model, res["task_type"])
            overall = sc["overall"]
            by_lang[lang].append(overall)
            by_model[model].append(overall)
            rows.append({"lang": lang, "model": model, "task_type": res["task_type"],
                         "overall": overall, "provider": res["provider"], "latency_s": round(dt, 2)})
            print(f"  [{lang:5s}/{model:11s}] {overall:4.1f}/10  {res['task_type']:15s} {dt:4.1f}s")

    def avg(xs): return round(statistics.mean(xs), 2) if xs else 0.0
    print("\n── Average quality by language ──")
    for lang in ["en", "fr", "ar", "derja"]:
        print(f"  {lang:6s} {avg(by_lang[lang]):4.2f}/10   (n={len(by_lang[lang])})")
    print("── Average quality by model ──")
    for m in MODELS:
        print(f"  {m:12s} {avg(by_model[m]):4.2f}/10")

    out = Path("bench/results"); out.mkdir(parents=True, exist_ok=True)
    fp = out / f"llm_eval_{int(time.time())}.json"
    fp.write_text(json.dumps({
        "rows": rows,
        "avg_by_language": {k: avg(v) for k, v in by_lang.items()},
        "avg_by_model": {k: avg(v) for k, v in by_model.items()},
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved {fp}")


if __name__ == "__main__":
    run()
