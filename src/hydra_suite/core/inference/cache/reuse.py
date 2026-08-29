"""Shared cache-reuse helper: read a fully-covered on-disk detection cache, or
recompute the whole requested frame set fresh in one batched pass and persist
it as a new complete cache session.

No incremental merge -- matches this codebase's existing all-or-nothing cache
convention (see ``core/tracking/worker.py``'s backward-pass cache handling,
and ``InferenceRunner.detection_cache_covers_frame_range`` /
``caches_all_valid`` in ``core/inference/runner.py``): a cache is either fully
usable as-is, or the entire requested set is recomputed and rewritten. This is
the ONE shared implementation of that pattern for callers outside
``InferenceRunner`` itself -- do not copy-paste it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..result import OBBResult
from .base import CacheKey
from .keys import bgsub_detection_cache_key, detection_cache_key, with_video_signature
from .store import DetectionCacheHandle

# Placeholder key for callers that cannot offer a real OBBConfig-derived key
# (e.g. a minimal test double implementing only `detect_batch_raw`). Mirrors
# `open_detection_cache_reader`'s convention: validity then gates on the
# cache file's existence and its written-frame bookkeeping alone, not on a
# run-config match.
_FALLBACK_KEY = CacheKey(
    schema_version=0, model_path="", model_mtime=0.0, config_hash=""
)


def _cache_key_for(runner: object) -> tuple[CacheKey, bool]:
    """Return ``(key, require_key)`` for validating/stamping this call's cache.

    A real ``InferenceRunner`` carries either a full ``config.obb`` or a
    ``config.bgsub`` plus a video signature (``runner._video_sig``) and an
    arena ROI mask (``runner._roi_mask``).  Its detections are therefore keyed
    exactly like the runner's own on-disk detection cache (see ``_open_caches``
    in ``core/inference/runner.py``), including the source video's size/mtime
    fingerprint.

    A caller that only implements ``detect_batch_raw`` (no usable detection
    config) has none of that to offer; this then falls back to a placeholder
    key with ``require_key=False``, so validity is judged by cache existence
    and written-frame coverage only.  This is intentionally limited to test
    doubles and generic consumers -- a real bg-sub runner must not take it.
    """
    config = getattr(runner, "config", None)
    obb_config = getattr(config, "obb", None) if config is not None else None
    roi_mask = getattr(runner, "_roi_mask", None)
    video_sig = getattr(runner, "_video_sig", "")
    if obb_config is not None:
        key = detection_cache_key(obb_config, roi_mask)
        return with_video_signature(key, video_sig), True
    bgsub_config = getattr(config, "bgsub", None) if config is not None else None
    if bgsub_config is not None:
        key = bgsub_detection_cache_key(bgsub_config)
        return with_video_signature(key, video_sig), True
    return _FALLBACK_KEY, False


def open_raw_detection_cache_reader(
    runner: object, cache_dir: Path
) -> DetectionCacheHandle:
    """Open one validated, read-only raw-detection cache handle for *runner*.

    A ``DetectionCacheHandle`` memoizes the arrays it reads.  Exporters that
    process their input in many chunks should retain this reader for the whole
    operation, otherwise each chunk reloads and decompresses the full
    ``detection.npz`` file.  A miss remains a miss for the whole session, but
    is still recomputed by :func:`get_or_compute_raw` without modifying a
    borrowed cache.
    """
    key, require_key = _cache_key_for(runner)
    return DetectionCacheHandle(
        path=Path(cache_dir) / "detection.npz",
        key=key,
        require_key=require_key,
        read_only=True,
    )


def get_or_compute_raw(
    runner: object,
    cache_dir: Path,
    frames: "list[np.ndarray]",
    frame_indices: "list[int]",
    *,
    write: bool = True,
    cache_reader: DetectionCacheHandle | None = None,
) -> "dict[int, OBBResult]":
    """Return raw (unfiltered) per-frame OBB detections for ``frame_indices``.

    Reads ``<cache_dir>/detection.npz`` if it exists, is valid, and covers
    every requested frame index -- a pure read, zero calls to
    ``runner.detect_batch_raw``. Otherwise recomputes the WHOLE requested set
    fresh in a single ``detect_batch_raw`` call (not just the missing
    subset) and persists it as one new complete cache write.

    ``write=False`` makes the miss path read-only: the whole requested set is
    still recomputed and returned, but NOTHING is written -- no write handle is
    constructed and ``<cache_dir>/detection.npz`` is never touched. This is for
    callers that merely *borrow* a cache file another subsystem owns (notably
    TrackerKit's dataset export, which points at tracking's own
    ``.inference_cache_<stem>/detection.npz``). Persisting there would be
    destructive, not merely wasteful: ``DetectionCacheHandle.close()`` rewrites
    the file from the current call's buffer alone (this repo's documented
    no-merge convention), so one miss would replace tracking's complete
    detection cache with just the frames this call happened to ask for --
    breaking backward/replay tracking, which requires a full-range cache. The
    cache-HIT path is identical either way (it is already a pure read), so
    ``write=False`` gives up nothing on the common, fast path.

    ``runner`` must implement ``detect_batch_raw(frames, frame_indices=...)
    -> list[OBBResult]`` (``InferenceRunner.detect_batch_raw``, Task 2).
    """
    frame_indices = list(frame_indices)
    cache_path = Path(cache_dir) / "detection.npz"
    read_handle = cache_reader or open_raw_detection_cache_reader(runner, cache_dir)
    if read_handle.is_valid():
        cached = {idx: read_handle.read_frame(idx) for idx in frame_indices}
        if all(result is not None for result in cached.values()):
            return cached

    raw_results = runner.detect_batch_raw(frames, frame_indices=frame_indices)

    if not write:
        return dict(zip(frame_indices, raw_results))

    key, require_key = _cache_key_for(runner)
    write_handle = DetectionCacheHandle(
        path=cache_path,
        key=key,
        require_key=require_key,
        read_only=False,
    )
    for idx, result in zip(frame_indices, raw_results):
        write_handle.write_frame(idx, result=result)
    write_handle.close()

    return dict(zip(frame_indices, raw_results))
