import functools
import hashlib
import logging
import json
import pickle
import lancedb
from sentence_transformers import SentenceTransformer
from pathlib import Path

from core.config import get_settings
from core.constants import DEFAULT_TOP_K, DEFAULT_MIN_QUALITY
from core.fusion import fuse_hits
from core.retrieval import validate_filter_inputs

logger = logging.getLogger(__name__)
_settings = get_settings()

# ── Config ──────────────────────────────────────────────────────────────────
PROMPTS_FILE    = _settings.prompts_file
DB_PATH         = _settings.db_path
TABLE_NAME      = "prompts"
MODEL_NAME      = _settings.embedding_model
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
    """Hash the embedding inputs so the cache invalidates correctly.

    Includes the embedding model name AND the actual text that gets embedded
    (id + task_type + target_model + prompt prefix) — not just ids. Editing a
    prompt's text or switching the embedding model now busts the cache, which
    the previous id-only key failed to do.
    """
    payload = [
        (
            p.get("id", ""),
            p.get("task_type", ""),
            p.get("target_model", ""),
            p.get("prompt", "")[:300],
        )
        for p in prompts
    ]
    content = json.dumps([MODEL_NAME, payload], sort_keys=True)
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
    top_k:        int  = DEFAULT_TOP_K,
    min_quality:  float = DEFAULT_MIN_QUALITY,
) -> list[dict]:
    """
    Retrieve top_k relevant prompts for a given query.

    Uses hybrid search (vector + full-text) fused with Reciprocal Rank Fusion,
    filtered by target_model / task_type / quality. Falls back exact -> model ->
    general until ``top_k`` results are collected. Inputs are validated against
    allow-lists before being interpolated into filters.
    """
    target_model, task_type, min_quality = validate_filter_inputs(
        target_model, task_type, min_quality
    )

    table = get_table()

    # Embed the query with task+model context
    query_vector = embed(f"{task_type} {target_model} {query}")

    # Values below are allow-list validated, so interpolation is safe.
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

    results: list[dict] = []
    seen: set = set()

    for metadata_filter in (exact_filter, model_filter, general_filter):
        if len(results) >= top_k:
            break

        try:
            v_hits = (
                table.search(query_vector)
                     .where(metadata_filter)
                     .limit(top_k * 2)
                     .to_list()
            )
            f_hits = (
                table.search(query)
                     .where(metadata_filter)
                     .limit(top_k * 2)
                     .to_list()
            )
        except Exception as exc:
            logger.warning("LanceDB search failed for filter '%s': %s", metadata_filter, exc)
            continue

        # Reciprocal Rank Fusion of the two rankings (no score calibration needed).
        fused = fuse_hits([v_hits, f_hits], id_key="id")
        for r in fused:
            if r["id"] not in seen:
                results.append(r)
                seen.add(r["id"])
                if len(results) >= top_k:
                    break

    # Clean up internal fields before returning.
    for r in results:
        r.pop("vector", None)
        r.pop("_distance", None)
        r.pop("_score", None)
        r.pop("rrf_score", None)

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