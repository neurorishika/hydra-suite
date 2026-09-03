"""Read-only opener for a modern detection.npz, independent of run config."""

from __future__ import annotations

from pathlib import Path

from .base import CacheKey
from .store import CNNCacheHandle, DetectionCacheHandle


def open_detection_cache_reader(path: str | Path) -> DetectionCacheHandle:
    """Open an existing ``detection.npz`` for reading.

    Uses a path-only key and ``require_key=False``: validity gates on file
    existence and the stored ``written_frames`` set, not on a run-config
    match. Intended for consumers (RefineKit overlays, dataset exporters)
    that only need the geometry already on disk, regardless of which run
    produced it.
    """
    key = CacheKey(schema_version=0, model_path="", model_mtime=0.0, config_hash="")
    return DetectionCacheHandle(
        path=Path(path),
        key=key,
        require_key=False,
        # Without this, closing the returned handle overwrites the cache with an
        # empty one keyed by the placeholder above.
        read_only=True,
    )


def open_cnn_cache_reader(path: str | Path, label: str) -> CNNCacheHandle:
    """Open a CNN cache for bounded path-only reads, including legacy NPZ."""
    key = CacheKey(schema_version=0, model_path="", model_mtime=0.0, config_hash="")
    return CNNCacheHandle(
        path=Path(path),
        key=key,
        label=str(label),
        require_key=False,
        read_only=True,
    )
