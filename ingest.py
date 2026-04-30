import functools
import hashlib
import logging
import os
import json
import pickle
import lancedb
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
PROMPTS_FILE    = "prompts.json"
DB_PATH         = "lancedb_store"
TABLE_NAME      = "prompts"
MODEL_NAME      = "BAAI/bge-small-en-v1.5"
EMBEDDING_CACHE = "embedding_cache.pkl"

# ── Load embedding model ─────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME)
        print("Model loaded.")
    return _model

@functools.lru_cache(maxsize=512)
def _embed_cached(text: str) -> tuple[float, ...]:
    prefixed = f"Represent this sentence for retrieval: {text}"
    vector   = get_model().encode(prefixed, normalize_embeddings=True)
    return tuple(vector.tolist())

def embed(text: str) -> list[float]:
    """Embed a single string using BGE-small with an in-process cache."""
    return list(_embed_cached(text))

# ── Persistent embedding cache ───────────────────────────────────────────────
def _get_cache_key(prompts: list[dict]) -> str:
    """Generate a hash of prompts.json content so cache invalidates on changes."""
    content = json.dumps([p.get('id', '') for p in prompts], sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

def _load_embedding_cache(prompts: list[dict]) -> list | None:
    """Load cached embeddings from disk if they match the current prompts."""
    cache_path = Path(EMBEDDING_CACHE)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if cached.get('key') == _get_cache_key(prompts):
            print(f"✅ Loaded {len(cached['vectors'])} cached embeddings from disk.")
            return cached['vectors']
        else:
            print("Cache key mismatch — regenerating embeddings.")
            return None
    except Exception as e:
        print(f"Cache load failed: {e} — regenerating.")
        return None

def _save_embedding_cache(prompts: list[dict], vectors: list):
    """Save embeddings to disk for fast cold starts."""
    try:
        with open(EMBEDDING_CACHE, 'wb') as f:
            pickle.dump({'key': _get_cache_key(prompts), 'vectors': vectors}, f)
        print(f"💾 Saved {len(vectors)} embeddings to cache.")
    except Exception as e:
        print(f"Cache save failed: {e}")

# ── Build or rebuild the index ───────────────────────────────────────────────
def build_index(force: bool = False):
    db_dir = Path(DB_PATH)

    if db_dir.exists() and not force:
        print(f"Index already exists. Skipping rebuild.")
        return

    print(f"Building index from '{PROMPTS_FILE}'...")

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print(f"Loaded {len(prompts)} prompts.")

    # Try to load cached embeddings first
    cached_vectors = _load_embedding_cache(prompts)
    
    if cached_vectors is not None:
        vectors = cached_vectors
    else:
        print("Embedding in batch (this is slow on first run only)...")
        # Build all texts first
        texts = [
            f"Represent this sentence for retrieval: "
            f"{p.get('task_type','')} "
            f"{p.get('target_model','')} "
            f"{p.get('prompt','')[:300]}"
            for p in prompts
        ]

        # Batch embed all at once — much faster than one by one
        vectors = get_model().encode(
            texts,
            normalize_embeddings = True,
            batch_size           = 64,
            show_progress_bar    = True,
        )
        vectors = [v.tolist() for v in vectors]
        _save_embedding_cache(prompts, vectors)

    rows = []
    for i, (p, vector) in enumerate(zip(prompts, vectors)):
        rows.append({
            "id":           p.get("id", f"prompt_{i}"),
            "target_model": p.get("target_model", "general"),
            "task_type":    p.get("task_type", "general"),
            "prompt":       p.get("prompt", ""),
            "source_repo":  p.get("source_repo", ""),
            "license":      p.get("license", ""),
            "quality_score":float(p.get("quality_score", 5)),
            "vector":       vector if isinstance(vector, list) else vector.tolist(),
        })

    db = lancedb.connect(DB_PATH)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)

    table = db.create_table(TABLE_NAME, data=rows)
    table.create_fts_index("prompt", replace=True)
    print(f"Created table '{TABLE_NAME}' with {len(rows)} rows and FTS index.")
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
        
        candidates = []
        try:
            # 1. Vector search
            v_hits = (
                table.search(query_vector)
                     .where(metadata_filter)
                     .limit(top_k)
                     .to_list()
            )
            candidates.extend(v_hits)
            
            # 2. FTS search (Keyword matching)
            # We search for the raw query string
            f_hits = (
                table.search(query)
                     .where(metadata_filter)
                     .limit(top_k)
                     .to_list()
            )
            candidates.extend(f_hits)
            
        except Exception as exc:
            logger.warning("LanceDB search failed for filter '%s': %s", metadata_filter, exc)

        # Merge, deduplicate by id, prioritize vector hits slightly by order
        seen = {r["id"] for r in results}
        for r in candidates:
            if r["id"] not in seen:
                results.append(r)
                seen.add(r["id"])
                if len(results) >= top_k:
                    break

        if len(results) >= top_k:
            break

    # Clean up
    for r in results:
        r.pop("vector", None)
        r.pop("_distance", None) # Remove search distance/score
        r.pop("_score", None)

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