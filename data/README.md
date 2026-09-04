# Derja / MENA seed corpus (A4)

`derja_seed.json` is a small, curated set of Tunisian-SMB prompt exemplars in
Derja, Arabic and French — a starting point to grow the corpus for the MENA
market. Same schema as `prompts.json`.

## Merge into the corpus

```python
import json
base = json.load(open("prompts.json", encoding="utf-8"))
seed = json.load(open("data/derja_seed.json", encoding="utf-8"))
ids = {p["id"] for p in base}
base += [p for p in seed if p["id"] not in ids]
json.dump(base, open("prompts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
```

Then rebuild the index: `python ingest.py` (re-embeds).

> Retrieval value is unlocked by A2 (a multilingual embedder — set
> `PF_EMBEDDING_MODEL=BAAI/bge-m3`), since the default embedder is English-only.
> Expand this file with real, reviewed examples before relying on it in production.
