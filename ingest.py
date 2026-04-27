import os
import json
import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
PROMPTS_FILE = "prompts.json"
DB_PATH      = "lancedb_store"
TABLE_NAME   = "prompts"
MODEL_NAME   = "BAAI/bge-small-en-v1.5"

# ── Load embedding model ─────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

def embed(text: str) -> list[float]:
    """Embed a single string using BGE-small.
    BGE models perform best with a query instruction prefix."""
    prefixed = f"Represent this sentence for retrieval: {text}"
    vector   = embedder.encode(prefixed, normalize_embeddings=True)
    return vector.tolist()

# ── Build or rebuild the index ───────────────────────────────────────────────
def build_index(force: bool = False):
    db_dir = Path(DB_PATH)

    # Skip rebuild if index already exists and force=False
    if db_dir.exists() and not force:
        print(f"Index already exists at '{DB_PATH}'. Skipping rebuild.")
        print("Pass force=True to rebuild from scratch.")
        return

    print(f"Building index from '{PROMPTS_FILE}'...")

    # Load prompts
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print(f"Loaded {len(prompts)} prompts.")

    # Build rows for LanceDB
    rows = []
    for i, p in enumerate(prompts):
        # Text we embed = task_type + target_model + first 300 chars of prompt
        embed_text = (
            f"{p.get('task_type', '')} "
            f"{p.get('target_model', '')} "
            f"{p.get('prompt', '')[:300]}"
        )
        vector = embed(embed_text)

        rows.append({
            "id":           p.get("id", f"prompt_{i}"),
            "target_model": p.get("target_model", "general"),
            "task_type":    p.get("task_type", "general"),
            "prompt":       p.get("prompt", ""),
            "source_repo":  p.get("source_repo", ""),
            "license":      p.get("license", ""),
            "quality_score":float(p.get("quality_score", 5)),
            "vector":       vector,
        })

        print(f"  Embedded [{i+1}/{len(prompts)}] {p.get('id', '?')}")

    # Write to LanceDB
    db    = lancedb.connect(DB_PATH)

    # Drop existing table if rebuilding
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
        print(f"Dropped existing table '{TABLE_NAME}'.")

    table = db.create_table(TABLE_NAME, data=rows)
    print(f"Created table '{TABLE_NAME}' with {len(rows)} rows.")
    print("Index build complete.")

# ── Retrieve function (used by rag.py) ───────────────────────────────────────
def get_table():
    """Return the LanceDB table, building index first if needed."""
    db_dir = Path(DB_PATH)
    if not db_dir.exists():
        print("Index not found — building now...")
        build_index()
    db = lancedb.connect(DB_PATH)
    return db.open_table(TABLE_NAME)

def retrieve(
    query:        str,
    target_model: str  = "general",
    task_type:    str  = "general",
    top_k:        int  = 3,
    min_quality:  float = 7.0,
) -> list[dict]:
    """
    Retrieve top_k relevant prompts for a given query.
    Filters by target_model and min quality_score before vector search.
    Falls back to general if no model-specific results found.
    """
    table = get_table()

    # Embed the query
    query_vector = embed(query)

    # Build metadata filter
    # Try model-specific first
    model_filter = (
        f"target_model = '{target_model}' "
        f"AND quality_score >= {min_quality}"
    )

    try:
        results = (
            table.search(query_vector)
                 .where(model_filter)
                 .limit(top_k)
                 .to_list()
        )
    except Exception:
        results = []

    # Fallback — if not enough results, widen to general
    if len(results) < top_k:
        general_filter = f"quality_score >= {min_quality}"
        try:
            fallback = (
                table.search(query_vector)
                     .where(general_filter)
                     .limit(top_k)
                     .to_list()
            )
            # Merge, deduplicate by id
            seen = {r["id"] for r in results}
            for r in fallback:
                if r["id"] not in seen:
                    results.append(r)
                    seen.add(r["id"])
                    if len(results) >= top_k:
                        break
        except Exception:
            pass

    # Clean up — remove vector from returned results (not needed downstream)
    for r in results:
        r.pop("vector", None)

    return results[:top_k]


# ── Run directly to build index ───────────────────────────────────────────────
if __name__ == "__main__":
    build_index(force=True)
    print("\nTest retrieval:")
    hits = retrieve(
        query        = "write a python function with error handling",
        target_model = "claude-code",
        task_type    = "code_generation",
        top_k        = 3,
    )
    for h in hits:
        print(f"  → [{h['target_model']}] {h['id']} | score: {h['quality_score']}")