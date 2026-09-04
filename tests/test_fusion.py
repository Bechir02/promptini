"""Tests for Reciprocal Rank Fusion."""

from core.fusion import reciprocal_rank_fusion, fuse_hits


def test_item_in_both_lists_ranks_first():
    a = ["x", "y", "z"]
    b = ["y", "w", "x"]
    fused = reciprocal_rank_fusion([a, b])
    ids = [item for item, _ in fused]
    # x and y appear in both lists, so they should outrank singletons z and w.
    assert set(ids[:2]) == {"x", "y"}
    # y is rank1+rank0, x is rank0+rank2 -> y's combined RRF score is higher.
    assert ids[0] == "y"


def test_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


def test_weights_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_fuse_hits_dedupes_and_keeps_metadata():
    v = [{"id": "1", "prompt": "alpha"}, {"id": "2", "prompt": "beta"}]
    f = [{"id": "2", "prompt": "beta"}, {"id": "3", "prompt": "gamma"}]
    fused = fuse_hits([v, f])
    ids = [r["id"] for r in fused]
    assert ids[0] == "2"             # appears in both -> top
    assert sorted(ids) == ["1", "2", "3"]   # deduped
    assert all("rrf_score" in r for r in fused)
    assert all("prompt" in r for r in fused)
