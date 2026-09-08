"""The inference vector a forward pass actually ran at, persisted for backward.

The backward pass replays the forward pass's detection cache, and the cache key
includes the batch size. If backward re-resolves independently it can pick a
different size, miss the key, and abort the run -- measured on courtship. The
GUI and headless paths issue the two passes separately and share no in-memory
state, so the inference-cache directory is the only seam available.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from hydra_suite.core.inference.autotune.models import InferenceTuningSettings

logger = logging.getLogger(__name__)

APPLIED_VECTOR_FILENAME = "applied_inference_vector.json"


def write_applied_vector(cache_dir, settings: InferenceTuningSettings) -> None:
    """Record the vector this pass ran at, next to the cache it wrote."""
    path = Path(cache_dir) / APPLIED_VECTOR_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    except Exception:
        # Never fail a run over provenance: the backward pass falls back to
        # configured values, which is the pre-existing untuned behaviour.
        logger.warning("Could not persist the applied inference vector", exc_info=True)


def read_applied_vector(cache_dir) -> InferenceTuningSettings | None:
    """Return the forward pass's vector, or None to fall back to config."""
    path = Path(cache_dir) / APPLIED_VECTOR_FILENAME
    if not path.is_file():
        return None
    try:
        return InferenceTuningSettings.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except Exception:
        logger.warning("Ignoring an unreadable applied inference vector", exc_info=True)
        return None
