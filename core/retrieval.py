"""Pure retrieval helpers (no DB / ML imports) so they stay unit-testable."""

from __future__ import annotations

import logging

from .constants import ALLOWED_MODELS, TASK_TYPES, DEFAULT_MIN_QUALITY, DEFAULT_MODEL, DEFAULT_TASK_TYPE

logger = logging.getLogger(__name__)

_ALLOWED_TASK_TYPES = frozenset(TASK_TYPES)


def validate_filter_inputs(
    target_model: str, task_type: str, min_quality: float
) -> tuple[str, str, float]:
    """Validate filter inputs before they are interpolated into a query string.

    Unknown models/task types fall back to ``general``; an unparseable quality
    threshold falls back to the default. Never trusts the caller — this is
    defense in depth for the LanceDB ``where`` clauses built in ingest.retrieve.
    """
    if target_model not in ALLOWED_MODELS:
        logger.warning("retrieve(): unknown target_model %r — using %r", target_model, DEFAULT_MODEL)
        target_model = DEFAULT_MODEL
    if task_type not in _ALLOWED_TASK_TYPES:
        logger.warning("retrieve(): unknown task_type %r — using %r", task_type, DEFAULT_TASK_TYPE)
        task_type = DEFAULT_TASK_TYPE
    try:
        min_quality = float(min_quality)
    except (TypeError, ValueError):
        logger.warning("retrieve(): invalid min_quality %r — using default", min_quality)
        min_quality = DEFAULT_MIN_QUALITY
    return target_model, task_type, min_quality
