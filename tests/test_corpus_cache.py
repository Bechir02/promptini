"""Incremental corpus cache (C5)."""

from core.corpus_cache import CorpusCache, not_modified


def test_not_modified():
    assert not_modified(304)
    assert not not_modified(200)


def test_roundtrip(tmp_path):
    p = tmp_path / "cache.json"
    c = CorpusCache(str(p))
    c.update("owner/repo", "etag-abc", [{"id": "x"}])
    c.save()

    c2 = CorpusCache(str(p))
    assert c2.etag("owner/repo") == "etag-abc"
    assert c2.prompts("owner/repo") == [{"id": "x"}]
    assert c2.etag("missing/repo") is None
