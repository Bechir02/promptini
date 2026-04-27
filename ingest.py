import functools
import logging
import os
import json
import lancedb
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
PROMPTS_FILE = "prompts.json"
DB_PATH      = "lancedb_store"
TABLE_NAME   = "prompts"
MODEL_NAME   = "BAAI/bge-small-en-v1.5"

# ── Load embedding model ─────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

@functools.lru_cache(maxsize=256)
def _embed_cached(text: str) -> tuple[float, ...]:
    prefixed = f"Represent this sentence for retrieval: {text}"
    vector   = embedder.encode(prefixed, normalize_embeddings=True)
    return tuple(vector.tolist())

def embed(text: str) -> list[float]:
    """Embed a single string using BGE-small with an in-process cache."""
    return list(_embed_cached(text))

# ── Build or rebuild the index ───────────────────────────────────────────────
def build_index(force: bool = False):
    db_dir = Path(DB_PATH)

    if db_dir.exists() and not force:
        print(f"Index already exists at '{DB_PATH}'. Skipping rebuild.")
        return

    print(f"Building index from '{PROMPTS_FILE}'...")

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print(f"Loaded {len(prompts)} prompts.")

    rows = []
    for i, p in enumerate(prompts):
        # Use cached embedding if available, else compute
        if "embedding" in p:
            vector = p["embedding"]
        else:
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

    db = lancedb.connect(DB_PATH)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)

    table = db.create_table(TABLE_NAME, data=rows)
    print(f"Created table '{TABLE_NAME}' with {len(rows)} rows.")
    print("Index build complete.")

# ── Retrieve function (used by rag.py) ───────────────────────────────────────
def get_table():
    """Return the LanceDB table, building index first if needed."""
    db_dir = Path(DB_PATH)
    if not db_dir.exists():
        logger.info("Index not found — building now...")
        build_index()

    db = lancedb.connect(DB_PATH)
    if TABLE_NAME not in db.table_names():
        logger.warning("LanceDB table missing — rebuilding index.")
        build_index(force=True)
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

    # Embed the query with task+model context
    query_vector = embed(f"{task_type} {target_model} {query}")

    exact_filter = (
        f"target_model = '{target_model}' "
        f"AND task_type = '{task_type}' "
        f"AND quality_score >= {min_quality}"
    )
    model_filter = (
        f"target_model = '{target_model}' "
        f"AND quality_score >= {min_quality}"
    )
    general_filter = f"quality_score >= {min_quality}"

    results = []
    for metadata_filter in (exact_filter, model_filter, general_filter):
        if len(results) >= top_k:
            break
        try:
            candidates = (
                table.search(query_vector)
                     .where(metadata_filter)
                     .limit(top_k)
                     .to_list()
            )
        except Exception as exc:
            logger.warning("LanceDB search failed for filter '%s': %s", metadata_filter, exc)
            candidates = []

        # Merge, deduplicate by id
        seen = {r["id"] for r in results}
        for r in candidates:
            if r["id"] not in seen:
                results.append(r)
                seen.add(r["id"])
                if len(results) >= top_k:
                    break

        # If exact or model-specific search returns enough, keep it
        if len(results) >= top_k:
            break

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