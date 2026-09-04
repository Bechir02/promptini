"""Incremental corpus-build cache (C5): per-repo ETag + parsed prompts.

Lets fetch_corpus send conditional requests and reuse a repo's parsed prompts
when GitHub returns 304 Not Modified, so refreshes are cheap. Build artifact —
gitignored, not shipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def not_modified(status_code: int) -> bool:
    return status_code == 304


class CorpusCache:
    def __init__(self, path: str = ".corpus_cache.json"):
        self.path = Path(path)
        self._data: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                self._data = {}

    def etag(self, repo: str) -> Optional[str]:
        return self._data.get(repo, {}).get("etag")

    def prompts(self, repo: str) -> Optional[list]:
        return self._data.get(repo, {}).get("prompts")

    def update(self, repo: str, etag: Optional[str], prompts: list) -> None:
        self._data[repo] = {"etag": etag, "prompts": prompts}

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._data))
        except Exception:
            pass
