"""Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Pure, dependency-free ranking math so it can be unit tested without a database.
RRF combines several ranked lists into one by summing 1/(k + rank) across the
lists in which each item appears. It needs no score calibration between the
vector and full-text rankers, which is exactly why it is a robust default for
hybrid search.
"""

from __future__ import annotations

from typing import Any, Hashable, Sequence

from .constants import RRF_K


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[Hashable, float]]:
    """Fuse several ranked lists of ids into one ranking.

    Args:
        ranked_lists: each inner sequence is a list of ids ordered best-first.
        k: RRF damping constant (larger flattens the contribution of rank).
        weights: optional per-list weight (defaults to 1.0 each).

    Returns:
        ``[(id, fused_score), ...]`` sorted by score descending.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    scores: dict[Hashable, float] = {}
    for lst, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + weight * (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def fuse_hits(
    hit_lists: Sequence[Sequence[dict[str, Any]]],
    id_key: str = "id",
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """RRF over lists of result dicts, returning the deduped dicts in fused order.

    The first dict seen for an id is kept (so its metadata is preserved) and an
    ``rrf_score`` field is attached.
    """
    id_lists = [[h[id_key] for h in hits] for hits in hit_lists]
    fused = reciprocal_rank_fusion(id_lists, k=k, weights=weights)

    first_seen: dict[Hashable, dict[str, Any]] = {}
    for hits in hit_lists:
        for h in hits:
            first_seen.setdefault(h[id_key], h)

    ordered: list[dict[str, Any]] = []
    for item_id, score in fused:
        row = dict(first_seen[item_id])
        row["rrf_score"] = score
        ordered.append(row)
    return ordered
