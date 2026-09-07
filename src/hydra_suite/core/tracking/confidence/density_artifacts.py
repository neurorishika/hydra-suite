"""Provenance-addressed confidence-density region sidecars.

Density regions affect assignment costs, so a region file is an inference
artifact rather than a generic ``confidence_regions.json`` convenience file.
Its address must change whenever the frame range, filtering, ROI/arena layout,
or density algorithm inputs change.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_ARTIFACT_VERSION = 1
_FILTERING_KEYS = (
    "DETECTION_METHOD",
    "YOLO_CONFIDENCE_THRESHOLD",
    "YOLO_IOU_THRESHOLD",
    "YOLO_TARGET_CLASSES",
    "MAX_TARGETS",
    "MIN_OBJECT_SIZE",
    "MAX_OBJECT_SIZE",
    "ENABLE_SIZE_FILTERING",
    "ADVANCED_CONFIG",
)
_DENSITY_KEYS = (
    "ENABLE_CONFIDENCE_DENSITY_MAP",
    "REFERENCE_BODY_SIZE",
    "RESIZE_FACTOR",
)


def _canonical(value: Any) -> Any:
    """Produce a JSON-safe, content-addressed value without ndarray truncation."""

    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": {
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
                "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
            }
        }
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        # Parameters should normally be ordered collections, but make an
        # accidental set deterministic rather than letting hash randomization
        # change an artifact address across otherwise-identical processes.
        return sorted((_canonical(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def density_regions_cache_key(
    params: Mapping[str, Any],
    start_frame: int,
    end_frame: int,
    *,
    source_signature: str | None = None,
) -> str:
    """Return a key for density regions computed over an exact replay domain."""

    density = {
        str(key): params.get(key) for key in params if str(key).startswith("DENSITY_")
    }
    density.update({key: params.get(key) for key in _DENSITY_KEYS})
    filtering = {key: params.get(key) for key in _FILTERING_KEYS}
    payload = {
        "version": _ARTIFACT_VERSION,
        "frame_range": [int(start_frame), int(end_frame)],
        "filtering": filtering,
        "density": density,
        # Both masks alter the spatial population.  ``_canonical`` hashes the
        # content instead of relying on an unstable ndarray repr.
        "roi_mask": params.get("ROI_MASK"),
        "arena_labels": params.get("ARENA_LABELS"),
        "n_arenas": params.get("N_ARENAS", 1),
        "animals_per_arena": params.get("ANIMALS_PER_ARENA"),
        # The cache directory can retain sidecars from an earlier detector
        # generation. Bind regions to the raw-detection cache contract so a
        # changed model/crop/extraction setting cannot reuse stale regions.
        "source_signature": source_signature,
    }
    encoded = json.dumps(
        _canonical(payload), sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def density_regions_cache_path(
    cache_dir: str | Path,
    params: Mapping[str, Any],
    start_frame: int,
    end_frame: int,
    *,
    source_signature: str | None = None,
) -> Path:
    """Return the keyed density-sidecar path inside an inference cache dir."""

    key = density_regions_cache_key(
        params,
        start_frame,
        end_frame,
        source_signature=source_signature,
    )
    return Path(cache_dir) / f"confidence_regions-{key}.json"
