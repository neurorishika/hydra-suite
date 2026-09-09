"""Persist the inference execution vector a forward pass actually ran with.

Batch sizes are part of the inference cache key (``cache/keys.py::_batch_term``)
because batching changes the numbers a stage produces and is NOT re-applied at
replay time. That is correct and must stay. What is not correct is a replay
pass -- the backward pass, or the confidence optimizer's read-only production
replay -- reconstructing that key from the project's *configured* batch sizes
when the forward pass ran at *tuned* ones: the key then lacks ``|batch=N`` and
the replay refuses with "Cached tracking replay requires valid inference
caches" even though the cache it needs is sitting right there.

The fix is to read rather than re-derive. The forward pass owns the cache
directory, so it records the effective vector next to the caches it wrote, and
a replay pass loads that record and resolves its keys from it. A replay then
cannot disagree with the forward pass by construction.

Invariants:

* **Absent record == configured vector.** Any cache written before this module
  existed, and every run whose overlay changed nothing, has no record, and a
  replay of it resolves exactly as it did before. No existing cache is
  invalidated.
* **A forward pass that overrides nothing REMOVES the record.** Otherwise a
  tuned run followed by an untuned one would leave a stale ``det=4`` record
  pointing at a cache written at the default batch -- the very mismatch this
  module exists to prevent, with the sign flipped.
* **An unreadable record fails loudly, not quietly.** It degrades to "no
  record", which means the replay resolves at the configured vector and hits
  the existing hard refusal. It never guesses.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .models import InferenceRuntimeOverlay, InferenceTuningSettings

if TYPE_CHECKING:
    from hydra_suite.core.inference.config import InferenceConfig

logger = logging.getLogger(__name__)

REPLAY_VECTOR_FILENAME = "inference_vector.json"
REPLAY_VECTOR_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class ReplayVector:
    """The execution vector a forward pass wrote its caches under."""

    effective: InferenceTuningSettings
    requested: InferenceTuningSettings
    status: str = ""
    profile_id: str | None = None

    def apply(self, config: "InferenceConfig") -> "InferenceConfig":
        """Return a detached config carrying the recorded execution vector.

        Mirrors :meth:`InferenceRuntimeOverlay.apply` exactly, including its
        ``disable_tile_autotune`` rule, so the replay's config is the same
        object the forward pass built.
        """
        return self.effective.apply(
            config,
            disable_tile_autotune=self.effective != self.requested,
        )


def replay_vector_path(cache_dir: str | os.PathLike[str]) -> Path:
    return Path(cache_dir) / REPLAY_VECTOR_FILENAME


def write_replay_vector(
    cache_dir: str | os.PathLike[str],
    overlay: InferenceRuntimeOverlay | None,
) -> None:
    """Record ``overlay`` beside the caches, or clear a stale record.

    Writing only happens when the overlay actually overrides the configured
    vector; a no-op overlay (autotuner off, ``fallback``, ``record``, kept
    baseline, ...) deletes any previous record so "absent == configured"
    stays true.
    """
    path = replay_vector_path(cache_dir)
    if overlay is None or overlay.effective == overlay.requested:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning(
                "Could not clear a stale inference vector record at %s; a later "
                "replay may resolve the wrong cache key.",
                path,
                exc_info=True,
            )
        return
    payload = {
        "schema": REPLAY_VECTOR_SCHEMA,
        "effective": overlay.effective.to_dict(),
        "requested": overlay.requested.to_dict(),
        "status": str(overlay.status),
        "profile_id": overlay.profile_id,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, path)
    except OSError:
        logger.warning(
            "Could not record the effective inference vector at %s", path, exc_info=True
        )
        try:
            tmp.unlink()
        except OSError:
            pass


def load_replay_vector(
    cache_dir: str | os.PathLike[str],
) -> ReplayVector | None:
    """Return the recorded vector, or ``None`` when there is nothing usable.

    ``None`` means "resolve at the configured vector", which is the pre-existing
    behavior; a mismatch that survives that still hits the caller's loud
    refusal.
    """
    path = replay_vector_path(cache_dir)
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("Could not read the inference vector record at %s", path)
        return None
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("inference vector record is not an object")
        schema = int(payload.get("schema", 0))
        if schema != REPLAY_VECTOR_SCHEMA:
            raise ValueError(f"unsupported inference vector schema {schema}")
        effective = InferenceTuningSettings.from_dict(payload["effective"])
        requested_raw = payload.get("requested")
        requested = (
            InferenceTuningSettings.from_dict(requested_raw)
            if isinstance(requested_raw, dict)
            else effective
        )
    except Exception:
        logger.warning(
            "Ignoring an unreadable inference vector record at %s; cache keys "
            "will resolve at the configured batch sizes.",
            path,
            exc_info=True,
        )
        return None
    return ReplayVector(
        effective=effective,
        requested=requested,
        status=str(payload.get("status", "")),
        profile_id=payload.get("profile_id"),
    )
