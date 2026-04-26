"""Path builders for video-related artifact files (caches, logs, detection outputs)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable


def _video_stem(video_path: str | os.PathLike[str]) -> str:
    stem = Path(video_path).stem.strip()
    return stem or "video"


def _normalize_base_dir(path: str | os.PathLike[str] | None) -> Path | None:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def candidate_artifact_base_dirs(
    video_path: str | os.PathLike[str],
    preferred_base_dirs: Iterable[str | os.PathLike[str] | None] | None = None,
) -> list[Path]:
    """Return a deduplicated, ordered list of candidate base directories for artifact storage.

    The order is: video's parent directory, any preferred dirs supplied by the caller,
    then the system temp directory as a final fallback.
    """
    candidates: list[Path] = []
    seen: set[str] = set()
    raw_candidates: list[str | os.PathLike[str] | None] = [
        Path(video_path).expanduser().parent,
        *(preferred_base_dirs or []),
        tempfile.gettempdir(),
    ]
    for raw in raw_candidates:
        candidate = _normalize_base_dir(raw)
        if candidate is None:
            continue
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def choose_writable_artifact_base_dir(
    video_path: str | os.PathLike[str],
    preferred_base_dirs: Iterable[str | os.PathLike[str] | None] | None = None,
) -> Path:
    """Return the first writable candidate directory for artifact storage.

    Iterates through ``candidate_artifact_base_dirs`` and returns the first
    existing, writable directory; falls back to the system temp dir if none qualify.
    """
    for candidate in candidate_artifact_base_dirs(video_path, preferred_base_dirs):
        if candidate.exists() and candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate
    return Path(tempfile.gettempdir())


def build_video_cache_dir(
    video_path: str | os.PathLike[str],
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create: bool = False,
) -> Path:
    """Return the cache directory path for a video, named ``<stem>_caches/``.

    Placed under ``artifact_base_dir`` if supplied, otherwise next to the video file.
    If ``create`` is True the directory is created on disk.
    """
    base_dir = (
        _normalize_base_dir(artifact_base_dir) or Path(video_path).expanduser().parent
    )
    expected_cache_dir_name = f"{_video_stem(video_path)}_caches"
    cache_dir = (
        base_dir
        if base_dir.name == expected_cache_dir_name
        else base_dir / expected_cache_dir_name
    )
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def build_video_log_dir(
    video_path: str | os.PathLike[str],
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create: bool = False,
) -> Path:
    """Return the log directory path for a video, named ``<stem>_logs/``.

    Placed under ``artifact_base_dir`` if supplied, otherwise next to the video file.
    If ``create`` is True the directory is created on disk.
    """
    base_dir = (
        _normalize_base_dir(artifact_base_dir) or Path(video_path).expanduser().parent
    )
    log_dir = base_dir / f"{_video_stem(video_path)}_logs"
    if create:
        log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def build_detection_cache_path(
    video_path: str | os.PathLike[str],
    model_id: str,
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create_dir: bool = False,
) -> Path:
    """Return the canonical ``.npz`` path for a detection cache inside the video's cache dir.

    The filename is ``<stem>_detection_cache_<model_id>.npz`` under ``<stem>_caches/``.
    """
    cache_dir = build_video_cache_dir(
        video_path,
        artifact_base_dir=artifact_base_dir,
        create=create_dir,
    )
    return cache_dir / f"{_video_stem(video_path)}_detection_cache_{model_id}.npz"


def build_legacy_detection_cache_path(
    video_path: str | os.PathLike[str],
    model_id: str,
    artifact_base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the legacy flat ``.npz`` path for a detection cache placed directly in the base dir.

    Used only to locate caches written by older versions of the tracker that did not
    use the ``<stem>_caches/`` subdirectory layout.
    """
    base_dir = (
        _normalize_base_dir(artifact_base_dir) or Path(video_path).expanduser().parent
    )
    return base_dir / f"{_video_stem(video_path)}_detection_cache_{model_id}.npz"


def find_existing_detection_cache_path(
    video_path: str | os.PathLike[str],
    model_id: str,
    artifact_base_dirs: Iterable[str | os.PathLike[str] | None] | None = None,
) -> Path | None:
    """Locate an existing detection cache ``.npz`` for the given video and model.

    Checks current-layout paths first across all candidate directories, then falls back
    to legacy flat paths.  Returns ``None`` if no cache is found anywhere.
    """
    base_dirs = artifact_base_dirs or candidate_artifact_base_dirs(video_path)

    for base_dir in base_dirs:
        current = build_detection_cache_path(
            video_path,
            model_id,
            artifact_base_dir=base_dir,
        )
        if current.exists():
            return current

    for base_dir in base_dirs:
        legacy = build_legacy_detection_cache_path(
            video_path,
            model_id,
            artifact_base_dir=base_dir,
        )
        if legacy.exists():
            return legacy
    return None


def build_optimizer_detection_cache_path(
    video_path: str | os.PathLike[str],
    model_name: str,
    resize_percent: int,
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create_dir: bool = False,
) -> Path:
    """Return the ``.npz`` path for a parameter-optimizer detection cache.

    The filename encodes the model name and resize percentage:
    ``<stem>_<model_name>_r<resize_percent>_opt_cache.npz`` inside ``<stem>_caches/``.
    """
    cache_dir = build_video_cache_dir(
        video_path,
        artifact_base_dir=artifact_base_dir,
        create=create_dir,
    )
    stem = _video_stem(video_path)
    return cache_dir / f"{stem}_{model_name}_r{int(resize_percent)}_opt_cache.npz"


def build_individual_properties_cache_path(
    video_path: str | os.PathLike[str],
    properties_id: str,
    start_frame: int,
    end_frame: int,
    detection_cache_path: str | os.PathLike[str] | None = None,
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create_dir: bool = False,
) -> Path:
    """Return the ``.npz`` path for a per-individual pose/properties cache.

    The filename is ``<stem>_pose_cache_<properties_id>_<start>_<end>.npz``.
    When ``detection_cache_path`` is supplied the cache is placed alongside it;
    otherwise it goes into the standard ``<stem>_caches/`` directory.
    """
    if detection_cache_path:
        base_dir = Path(detection_cache_path).expanduser().parent
        if create_dir:
            base_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = build_video_cache_dir(
            video_path,
            artifact_base_dir=artifact_base_dir,
            create=create_dir,
        )
    stem = _video_stem(video_path)
    return base_dir / (
        f"{stem}_pose_cache_{properties_id}_" f"{int(start_frame)}_{int(end_frame)}.npz"
    )


def build_detected_properties_cache_path(
    video_path: str | os.PathLike[str],
    properties_id: str,
    start_frame: int,
    end_frame: int,
    detection_cache_path: str | os.PathLike[str] | None = None,
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create_dir: bool = False,
) -> Path:
    """Return the ``.npz`` path for detected-frame heading/export metadata."""
    if detection_cache_path:
        base_dir = Path(detection_cache_path).expanduser().parent
        if create_dir:
            base_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = build_video_cache_dir(
            video_path,
            artifact_base_dir=artifact_base_dir,
            create=create_dir,
        )
    stem = _video_stem(video_path)
    return base_dir / (
        f"{stem}_detected_props_cache_{properties_id}_"
        f"{int(start_frame)}_{int(end_frame)}.npz"
    )


def build_apriltag_cache_path(
    video_path: str | os.PathLike[str],
    apriltag_id: str,
    start_frame: int,
    end_frame: int,
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create_dir: bool = False,
) -> Path:
    """Return the ``.npz`` path for an AprilTag detection cache.

    The filename is ``<stem>_apriltag_cache_<apriltag_id>_<start>_<end>.npz``
    inside ``<stem>_caches/``.
    """
    cache_dir = build_video_cache_dir(
        video_path,
        artifact_base_dir=artifact_base_dir,
        create=create_dir,
    )
    stem = _video_stem(video_path)
    return cache_dir / (
        f"{stem}_apriltag_cache_{apriltag_id}_"
        f"{int(start_frame)}_{int(end_frame)}.npz"
    )


def find_existing_apriltag_cache_path(
    video_path: str | os.PathLike[str],
    apriltag_id: str,
    start_frame: int,
    end_frame: int,
    artifact_base_dirs: Iterable[str | os.PathLike[str] | None] | None = None,
) -> Path | None:
    """Locate an existing AprilTag cache ``.npz`` for the given video, tag ID, and frame range.

    Searches across all candidate base directories and returns the first match,
    or ``None`` if no cache is found.
    """
    base_dirs = artifact_base_dirs or candidate_artifact_base_dirs(video_path)
    for base_dir in base_dirs:
        current = build_apriltag_cache_path(
            video_path,
            apriltag_id,
            start_frame,
            end_frame,
            artifact_base_dir=base_dir,
        )
        if current.exists():
            return current
    return None


def build_classify_cache_path(
    video_path: str | os.PathLike[str],
    classify_id: str,
    label: str,
    start_frame: int,
    end_frame: int,
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create_dir: bool = False,
) -> Path:
    """Return the ``.npz`` path for a classification embedding cache.

    The filename is ``<stem>_classify_cache_<safe_label>_<classify_id>_<start>_<end>.npz``
    inside ``<stem>_caches/``.  Non-alphanumeric characters in ``label`` are replaced with
    underscores to produce a safe filename component.
    """
    import re as _re

    cache_dir = build_video_cache_dir(
        video_path,
        artifact_base_dir=artifact_base_dir,
        create=create_dir,
    )
    stem = _video_stem(video_path)
    safe_label = _re.sub(r"[^\w-]", "_", label)
    return cache_dir / (
        f"{stem}_classify_cache_{safe_label}_{classify_id}_"
        f"{int(start_frame)}_{int(end_frame)}.npz"
    )


def find_existing_classify_cache_path(
    video_path: str | os.PathLike[str],
    classify_id: str,
    label: str,
    start_frame: int,
    end_frame: int,
    artifact_base_dirs: Iterable[str | os.PathLike[str] | None] | None = None,
) -> Path | None:
    """Locate an existing classification cache ``.npz`` for the given video, classifier, label, and frame range.

    Searches across all candidate base directories and returns the first match,
    or ``None`` if no cache is found.
    """
    base_dirs = artifact_base_dirs or candidate_artifact_base_dirs(video_path)
    for base_dir in base_dirs:
        current = build_classify_cache_path(
            video_path,
            classify_id,
            label,
            start_frame,
            end_frame,
            artifact_base_dir=base_dir,
        )
        if current.exists():
            return current
    return None


def build_autotune_state_path(
    detection_cache_path: str | os.PathLike[str],
) -> Path:
    """Return the ``.autotune_state.json`` path that lives alongside a detection cache.

    Derived by replacing the ``.npz`` extension of ``detection_cache_path`` with
    ``.autotune_state.json``.
    """
    return Path(detection_cache_path).expanduser().with_suffix(".autotune_state.json")


def build_tracking_session_log_path(
    video_path: str | os.PathLike[str],
    timestamp: str,
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create_dir: bool = False,
) -> Path:
    """Return the ``.log`` path for a tracking session log file.

    The filename is ``<stem>_tracking_<timestamp>.log`` inside ``<stem>_logs/``.
    """
    log_dir = build_video_log_dir(
        video_path,
        artifact_base_dir=artifact_base_dir,
        create=create_dir,
    )
    return log_dir / f"{_video_stem(video_path)}_tracking_{timestamp}.log"


def iter_detection_cache_candidates(
    video_path: str | os.PathLike[str],
    artifact_base_dirs: Iterable[str | os.PathLike[str] | None] | None = None,
    include_legacy: bool = True,
) -> list[Path]:
    """Return all ``.npz`` cache files matching ``<stem>*_cache*.npz`` for a video, sorted newest-first.

    Searches both the current ``<stem>_caches/`` subdirectory layout and,
    when ``include_legacy`` is True, the flat base directories used by older versions.
    """
    stem = _video_stem(video_path)
    pattern = f"{stem}*_cache*.npz"
    found: dict[str, Path] = {}
    base_dirs = artifact_base_dirs or candidate_artifact_base_dirs(video_path)

    for base_dir in base_dirs:
        normalized_base_dir = _normalize_base_dir(base_dir)
        if normalized_base_dir is None:
            continue

        cache_dir = build_video_cache_dir(
            video_path, artifact_base_dir=normalized_base_dir
        )
        if cache_dir.exists():
            for path in cache_dir.glob(pattern):
                found[str(path.resolve())] = path

        if include_legacy and normalized_base_dir.exists():
            for path in normalized_base_dir.glob(pattern):
                found[str(path.resolve())] = path

    return sorted(
        found.values(),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )


__all__ = [
    "build_apriltag_cache_path",
    "build_autotune_state_path",
    "build_classify_cache_path",
    "build_detected_properties_cache_path",
    "build_detection_cache_path",
    "build_individual_properties_cache_path",
    "build_optimizer_detection_cache_path",
    "build_tracking_session_log_path",
    "build_video_cache_dir",
    "build_video_log_dir",
    "candidate_artifact_base_dirs",
    "choose_writable_artifact_base_dir",
    "find_existing_apriltag_cache_path",
    "find_existing_classify_cache_path",
    "find_existing_detection_cache_path",
    "iter_detection_cache_candidates",
]
