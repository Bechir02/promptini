"""Curated few-shot exemplar bank — matched by task type + target model, zero ML."""
import json
import functools
import pathlib

_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "exemplars.json"


@functools.lru_cache(maxsize=1)
def _by_task() -> dict:
    try:
        data = json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict = {}
    for ex in data:
        out.setdefault(ex.get("task_type", "general"), []).append(ex)
    return out


def get_exemplars(task_type: str, target_model: str | None = None, k: int = 2) -> list[dict]:
    """Up to k curated exemplars for a task type, preferring the target model."""
    bank = _by_task()
    items = bank.get(task_type) or bank.get("general") or []
    if target_model:
        same = [e for e in items if e.get("target_model") == target_model]
        rest = [e for e in items if e.get("target_model") != target_model]
        items = same + rest
    return items[:k]
