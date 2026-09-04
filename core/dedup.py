"""Near-duplicate detection via character shingles + Jaccard (C6).

Stronger than the corpus builder's 200-char prefix dedup and dependency-free
(no embeddings) — catches reworded near-copies. It is O(n^2), so it is an
offline corpus-build step, not a hot path.
"""

from __future__ import annotations

from typing import Any


def char_shingles(text: str, k: int = 5) -> set[str]:
    t = " ".join(text.lower().split())
    if len(t) <= k:
        return {t} if t else set()
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def dedupe_near(items: list[dict], key: str = "prompt",
                threshold: float = 0.85, k: int = 5) -> list[dict]:
    """Keep the first of each near-duplicate group (Jaccard >= threshold)."""
    kept: list[dict] = []
    sigs: list[set] = []
    for it in items:
        sh = char_shingles(str(it.get(key, "")), k)
        if any(jaccard(sh, s) >= threshold for s in sigs):
            continue
        kept.append(it)
        sigs.append(sh)
    return kept
