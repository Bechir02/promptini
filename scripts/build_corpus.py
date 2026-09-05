#!/usr/bin/env python3
"""
build_corpus.py — Generate a big, sophisticated exemplar corpus with Groq.

For every (task_type x target_model) combo it asks Groq to synthesize realistic
Tunisian-Derja/Arabizi "rough ideas" and their ideal OPTIMIZED prompts (formatted
per model, following the restraint rules). Validated, de-duplicated, and merged
into data/exemplars.json — which the app already few-shots from, matched by
task + model.

Run from the repo root (needs GROQ_API_KEY in .env):
    python scripts/build_corpus.py --per-combo 2
    python scripts/build_corpus.py --per-combo 3 --models claude-code,gpt-4,cursor,gemini,copilot,general
    python scripts/build_corpus.py --replace          # start a fresh corpus
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.constants import MODELS as ALL_MODELS
from core.templates import OUTPUT_FORMATS, TASK_GUIDANCE

ALL_TASKS = list(TASK_GUIDANCE.keys())
OUT = Path(__file__).resolve().parent.parent / "data" / "exemplars.json"

SYS = ("You generate high-quality training exemplars for a Tunisian-Derja "
       "prompt-optimization tool. You output STRICT JSON only — no prose, no code fences.")


def gen_prompt(task, model, n, fmt_desc, task_guide):
    return f"""Generate {n} DIVERSE exemplars for task="{task}", target model="{model}".

Each exemplar = a realistic MESSY "rough idea" a Tunisian developer or user would
actually type, written in Tunisian Derja or Arabizi (mix Arabic script and Latin
with numerals like 3=ع, 7=ح, 9=ق), AND its ideal OPTIMIZED prompt.

The optimized prompt MUST:
- Be formatted for {model}: {fmt_desc[:280]}
- Match the task focus: {task_guide[:220]}
- RESTRAINT: include ONLY what the rough idea states or clearly implies. NEVER invent
  constraints, edge-cases, file sizes, or custom error types. Keep it tight — a simple
  ask = role + task + 1-3 real constraints. Under 120 words.
- Preserve the user's exact intent.

Return ONLY a JSON array, nothing else:
[{{"input":"<derja rough idea>","output":"<optimized prompt>"}}]"""


def parse_json_array(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return []
    return []


def valid(ex):
    i, o = (ex.get("input") or "").strip(), (ex.get("output") or "").strip()
    if not i or not o or i == o:
        return False
    if not (40 <= len(o) <= 1400):
        return False
    if not re.search(r"[؀-ۿ]|[2356789]", i):   # some Derja/Arabizi signal
        return False
    return True


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-combo", type=int, default=2)
    ap.add_argument("--models", default="claude-code,claude,gpt-4,gemini,cursor,general")
    ap.add_argument("--tasks", default=",".join(ALL_TASKS))
    ap.add_argument("--gen-model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    ap.add_argument("--replace", action="store_true", help="start fresh (drop existing corpus)")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    key = os.getenv("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY not set (put it in .env).")
    from groq import Groq
    client = Groq(api_key=key)

    models = [m.strip() for m in args.models.split(",") if m.strip() in ALL_MODELS]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in ALL_TASKS]
    print(f"Generating with {args.gen_model} — {len(tasks)} tasks x {len(models)} models "
          f"x {args.per_combo} = up to {len(tasks)*len(models)*args.per_combo} exemplars.\n")

    corpus = [] if args.replace else (json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else [])
    seen = {norm(e.get("input")) for e in corpus}
    added = 0

    for task in tasks:
        fmt = OUTPUT_FORMATS.get(task) or OUTPUT_FORMATS.get("general", {})
        fmt_desc = (OUTPUT_FORMATS.get("general", {}) or {}).get("description", "")
        # per-model format description:
        for model in models:
            mfmt = OUTPUT_FORMATS.get(model, OUTPUT_FORMATS.get("general", {}))
            desc = mfmt.get("description", fmt_desc)
            guide = TASK_GUIDANCE.get(task, TASK_GUIDANCE.get("general", ""))
            try:
                resp = client.chat.completions.create(
                    model=args.gen_model,
                    messages=[{"role": "system", "content": SYS},
                              {"role": "user", "content": gen_prompt(task, model, args.per_combo, desc, guide)}],
                    temperature=0.8, max_tokens=1800,
                )
                items = parse_json_array(resp.choices[0].message.content)
            except Exception as e:
                print(f"  ! {task}/{model}: {e}")
                time.sleep(args.sleep)
                continue

            kept = 0
            for ex in items:
                if not valid(ex):
                    continue
                if norm(ex["input"]) in seen:
                    continue
                seen.add(norm(ex["input"]))
                corpus.append({
                    "task_type": task, "target_model": model, "lang": "derja",
                    "input": ex["input"].strip(), "output": ex["output"].strip(),
                    "quality_score": 8.8, "source": "groq-generated",
                })
                kept += 1
                added += 1
            print(f"  {task:16s} / {model:11s} +{kept}")
            time.sleep(args.sleep)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Corpus now has {len(corpus)} exemplars (+{added} new) -> {OUT}")
    print("  The app few-shots from this automatically, matched by task + model.")


if __name__ == "__main__":
    main()
