"""Near-duplicate dedup (C6)."""

from core.dedup import char_shingles, jaccard, dedupe_near


def test_jaccard_bounds():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_dedupe_near_removes_reworded_copy():
    items = [
        {"prompt": "Write a python function that finds duplicate rows in a csv"},
        {"prompt": "Write a python function that finds duplicate rows in a CSV"},  # near-dup
        {"prompt": "Compose a haiku about the sea and the wind at dawn"},          # distinct
    ]
    out = dedupe_near(items, threshold=0.85)
    assert len(out) == 2
    assert out[0]["prompt"].endswith("csv")
    assert "haiku" in out[1]["prompt"]


def test_shingles_short_text():
    assert char_shingles("hi", k=5) == {"hi"}
    assert char_shingles("", k=5) == set()
