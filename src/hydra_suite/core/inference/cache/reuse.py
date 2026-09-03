"""Shared cache-reuse helper with bounded incremental chunk resume."""

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

    Reads every available requested frame from ``<cache_dir>/detection.npz``.
    A chunked cache miss recomputes and appends only missing frames. A partial
    legacy cache is recomputed as one complete new-format session because
    legacy files cannot be appended without destructive migration.

    ``write=False`` makes the miss path read-only: missing requested frames are
    recomputed and returned, but NOTHING is written -- no write handle is
    constructed and ``<cache_dir>/detection.npz`` is never touched. This is for
    callers that merely *borrow* a cache file another subsystem owns (notably
    TrackerKit's dataset export, which points at tracking's own
    ``.inference_cache_<stem>/detection.npz``). Persisting there would be
    destructive for legacy stores. Chunked stores can append safely, but a
    borrower still must not mutate a cache owned by tracking. The cache-hit path
    is identical either way, so ``write=False`` gives up nothing there.

    ``runner`` must implement ``detect_batch_raw(frames, frame_indices=...)
    -> list[OBBResult]`` (``InferenceRunner.detect_batch_raw``, Task 2).
    """
    frame_indices = list(frame_indices)
    frames = list(frames)
    if len(frames) != len(frame_indices):
        raise ValueError("frames and frame_indices must have the same length")
    if any(not isinstance(idx, (int, np.integer)) for idx in frame_indices):
        raise ValueError("frame indices must be integers")
    if any(int(idx) < 0 for idx in frame_indices):
        raise ValueError("frame indices must be nonnegative")
    if len({int(idx) for idx in frame_indices}) != len(frame_indices):
        raise ValueError("frame indices must be unique")
    if any(right <= left for left, right in zip(frame_indices, frame_indices[1:])):
        raise ValueError("frame indices must be strictly increasing")
    cache_path = Path(cache_dir) / "detection.npz"
    read_handle = cache_reader or open_raw_detection_cache_reader(runner, cache_dir)
    cached: dict[int, OBBResult | None] = {}
    missing_indices = list(frame_indices)
    missing_frames = list(frames)
    structurally_valid = read_handle.is_valid()
    reusable = read_handle.is_reusable() if structurally_valid else False
    replacement_required = structurally_valid and not reusable
    if reusable:
        cached = {idx: read_handle.read_frame(idx) for idx in frame_indices}
        missing_indices = [idx for idx in frame_indices if cached[idx] is None]
        if not missing_indices:
            return {idx: cached[idx] for idx in frame_indices}
        if not read_handle.is_legacy:
            frame_by_index = dict(zip(frame_indices, frames))
            missing_frames = [frame_by_index[idx] for idx in missing_indices]
        else:
            # Preserve the old file until a complete replacement is ready.
            cached = {}
            missing_indices = list(frame_indices)
            missing_frames = list(frames)
            replacement_required = True

    raw_results = runner.detect_batch_raw(missing_frames, frame_indices=missing_indices)
    if len(raw_results) != len(missing_indices):
        raise ValueError("detect_batch_raw must return one result per frame")
    if any(
        int(result.frame_idx) != int(idx)
        for idx, result in zip(missing_indices, raw_results)
    ):
        raise ValueError(
            "detect_batch_raw result.frame_idx must match its requested frame"
        )

    if not write:
        computed = dict(zip(missing_indices, raw_results))
        return {
            idx: cached.get(idx) if cached.get(idx) is not None else computed[idx]
            for idx in frame_indices
        }

    key, require_key = _cache_key_for(runner)
    write_handle = DetectionCacheHandle(
        path=cache_path,
        key=key,
        require_key=require_key,
        read_only=False,
        write_mode="fresh" if replacement_required else "resume",
    )
    for idx, result in zip(missing_indices, raw_results):
        write_handle.write_frame(idx, result=result)
    write_handle.close()

    computed = dict(zip(missing_indices, raw_results))
    return {
        idx: cached.get(idx) if cached.get(idx) is not None else computed[idx]
        for idx in frame_indices
    }
