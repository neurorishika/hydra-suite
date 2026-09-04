"""Small, torch-free helpers for completed SAM3 adapter artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

COMPLETION_VERSION = 1
_COMPLETION_SUFFIX = ".complete.json"


def completion_path(artifact_path: Path) -> Path:
    """Return the marker written only after child-side structural validation."""

    return artifact_path.with_name(artifact_path.name + _COMPLETION_SUFFIX)


def artifact_sha256(artifact_path: Path) -> str:
    """Hash an artifact with bounded reads."""

    digest = hashlib.sha256()
    with artifact_path.open("rb") as artifact_file:
        while chunk := artifact_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_completion_marker(artifact_path: Path) -> Path:
    """Atomically record the exact child-validated artifact bytes."""

    marker = completion_path(artifact_path)
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    payload: dict[str, Any] = {
        "version": COMPLETION_VERSION,
        "size_bytes": artifact_path.stat().st_size,
        "sha256": artifact_sha256(artifact_path),
    }
    try:
        with temporary.open("w", encoding="utf-8") as marker_file:
            json.dump(payload, marker_file, sort_keys=True)
            marker_file.flush()
            os.fsync(marker_file.fileno())
        os.replace(temporary, marker)
        _fsync_directory(marker.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return marker


def validate_completion(artifact_path: Path) -> str | None:
    """Return an error when the final bytes lack a matching completion marker."""

    marker = completion_path(artifact_path)
    if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
        return f"did not write a non-empty {artifact_path.name}"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"did not write a valid {marker.name} validation marker"
    if not isinstance(payload, dict) or payload.get("version") != COMPLETION_VERSION:
        return f"wrote an unsupported {marker.name} validation marker"
    if payload.get("size_bytes") != artifact_path.stat().st_size:
        return f"failed {artifact_path.name} completion size validation"
    if payload.get("sha256") != artifact_sha256(artifact_path):
        return f"failed {artifact_path.name} completion hash validation"
    return None


def remove_artifact(artifact_path: Path, *, remove_staging: bool = False) -> None:
    """Remove final output and, only when proven safe, private staging files."""

    artifact_path.unlink(missing_ok=True)
    marker = completion_path(artifact_path)
    marker.unlink(missing_ok=True)
    if not remove_staging:
        return
    for staged in artifact_path.parent.glob(f".{artifact_path.name}.*.validated.tmp"):
        staged.unlink(missing_ok=True)
    for staged_marker in marker.parent.glob(f".{marker.name}.*.tmp"):
        staged_marker.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
