#!/usr/bin/env python3
"""
build_derja_lexicon.py — Mine a Tunisian Derja lexicon from the unified corpus.

Source : hamzabouajila/tunisian-derja-unified-raw-corpus  (CC-BY-SA-4.0, commercial-OK)
Output : data/derja_lexicon.json  { arabizi_map, curated_glossary, top_terms }

The top_terms are the most frequent Derja words in real usage — inject a slice of
them into the system prompt so the model primes on authentic Derja vocabulary.

Run from the repo root:
    pip install -U datasets            # if not already installed
    python scripts/build_derja_lexicon.py --max-rows 200000 --top 500
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DATASET = "hamzabouajila/tunisian-derja-unified-raw-corpus"
OUT = Path(__file__).resolve().parent.parent / "data" / "derja_lexicon.json"

# Arabizi (Latin+numerals) -> Arabic phoneme map, curated from TArC conventions.
ARABIZI_MAP = {
    "2": "ء", "3": "ع", "4": "غ", "5": "خ", "6": "ط",
    "7": "ح", "8": "غ", "9": "ق", "'": "ء",
}

# High-confidence Derja glossary (hand-curated — meanings can't be auto-derived
# from a monolingual corpus). Extend freely; these prime the model on real Derja.
CURATED_GLOSSARY = {
    "برشا": "a lot / very", "شويّة": "a little", "توا": "now", "وقتاش": "when",
    "علاش": "why", "شكون": "who", "كيفاش": "how", "قدّاش": "how much",
    "فمّا": "there is / there are", "ماكاش": "there isn't", "يزّي": "enough / stop",
    "عيّشك": "please / thank you", "بالك": "maybe / watch out", "ديجا": "already",
    "مريڤل": "great / excellent", "باهي": "okay / good", "نجّم": "I can / it's possible",
    "أعمل": "make / do", "حلّ": "solve / open", "صلّح": "fix", "نقّي": "clean / pick",
    "زيد": "add / more", "نقّص": "reduce", "بدّل": "change", "لقّى": "find",
    "رجّع": "return / give back", "أكتب": "write", "أقرا": "read", "خدمة": "work / job",
    "برنامج": "program", "ملفّ": "file", "كود": "code", "خطأ": "error / bug",
    "صفحة": "page", "زر": "button", "شبكة": "network", "قاعدة بيانات": "database",
}

# Common MSA/Derja function words to drop from the frequency list.
STOPWORDS = set("""
في من على الى إلى عن مع هذا هذه ذلك التي الذي و أو ثم كان يكون قد لا ما
كل بعض غير بين عند لكن حتى إذا اذا كما أن إن أنا انت هو هي نحن هم كي لي
لك له لها بش باش انا اني الي هك هكا هاو هاذا هاذي متاع متاعي متاعك
""".split())

_AR = re.compile(r"[؀-ۿ]+")


def clean_tokens(text: str):
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # urls
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", " ", text)  # emoji
    for tok in _AR.findall(text):
        tok = tok.strip("ـ")  # tatweel
        if len(tok) >= 3 and tok not in STOPWORDS:
            yield tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=200_000,
                    help="rows to stream (whole corpus ~860k)")
    ap.add_argument("--top", type=int, default=500, help="top terms to keep")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing dependency. Run:  pip install -U datasets")

    print(f"Streaming {DATASET} (up to {args.max_rows:,} rows)…")
    ds = load_dataset(DATASET, split="train", streaming=True)
    counter = Counter()
    n = 0
    for row in ds:
        text = row.get("text") or ""
        counter.update(clean_tokens(text))
        n += 1
        if n % 20_000 == 0:
            print(f"  …{n:,} rows, {len(counter):,} unique terms")
        if n >= args.max_rows:
            break

    top_terms = [{"word": w, "count": c} for w, c in counter.most_common(args.top)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": DATASET,
        "license": "CC-BY-SA-4.0",
        "rows_scanned": n,
        "arabizi_map": ARABIZI_MAP,
        "curated_glossary": CURATED_GLOSSARY,
        "top_terms": top_terms,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✓ Wrote {OUT}")
    print(f"  {len(top_terms)} top terms, {len(CURATED_GLOSSARY)} glossed, "
          f"{len(ARABIZI_MAP)} Arabizi mappings.")
    print("  Top 25:", " ".join(t["word"] for t in top_terms[:25]))
    print("\nNext: I can wire a curated slice of these into the system prompt.")


if __name__ == "__main__":
    main()
