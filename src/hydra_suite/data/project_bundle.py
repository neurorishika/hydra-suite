"""Shared project-bundle layout and manifest helpers."""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

DEFAULT_BUNDLE_MANIFEST_FILENAME = "hydra_project.json"
DEFAULT_BUNDLE_STATE_DIRNAME = "state"
DEFAULT_BUNDLE_ARTIFACTS_DIRNAME = "artifacts"
DEFAULT_BUNDLE_HISTORY_DIRNAME = "history"
SUPPORTED_BUNDLE_VERSION = 1

logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    """Return an RFC3339 UTC timestamp without fractional seconds."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class ProjectBundlePaths:
    """Canonical filesystem layout for a project bundle."""

    root_dir: Path
    manifest_path: Path
    state_dir: Path
    artifacts_dir: Path
    history_dir: Path


@dataclass
class ProjectBundleManifest:
    """Top-level project-bundle manifest shared by all kits."""

    kit: str
    state_path: str
    bundle_version: int = 1
    display_name: str = ""
    database_path: str = ""
    artifacts_dir: str = DEFAULT_BUNDLE_ARTIFACTS_DIRNAME
    history_dir: str = DEFAULT_BUNDLE_HISTORY_DIRNAME
    created_at: str = ""
    updated_at: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a JSON-compatible dictionary."""
        return {
            "bundle_version": int(self.bundle_version),
            "kit": str(self.kit),
            "display_name": str(self.display_name),
            "state_path": str(self.state_path),
            "database_path": str(self.database_path),
            "artifacts_dir": str(self.artifacts_dir),
            "history_dir": str(self.history_dir),
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectBundleManifest":
        """Deserialize a manifest dictionary with sensible defaults."""
        return cls(
            bundle_version=int(data.get("bundle_version", 1)),
            kit=str(data.get("kit", "")).strip(),
            display_name=str(data.get("display_name", "")).strip(),
            state_path=str(data.get("state_path", "")).strip(),
            database_path=str(data.get("database_path", "")).strip(),
            artifacts_dir=str(
                data.get("artifacts_dir", DEFAULT_BUNDLE_ARTIFACTS_DIRNAME)
            ).strip()
            or DEFAULT_BUNDLE_ARTIFACTS_DIRNAME,
            history_dir=str(
                data.get("history_dir", DEFAULT_BUNDLE_HISTORY_DIRNAME)
            ).strip()
            or DEFAULT_BUNDLE_HISTORY_DIRNAME,
            created_at=str(data.get("created_at", "")).strip(),
            updated_at=str(data.get("updated_at", "")).strip(),
            meta=dict(data.get("meta", {}) or {}),
        )


def bundle_paths(project_dir: Path) -> ProjectBundlePaths:
    """Return the canonical bundle paths for *project_dir*."""
    root_dir = project_dir.expanduser().resolve()
    return ProjectBundlePaths(
        root_dir=root_dir,
        manifest_path=root_dir / DEFAULT_BUNDLE_MANIFEST_FILENAME,
        state_dir=root_dir / DEFAULT_BUNDLE_STATE_DIRNAME,
        artifacts_dir=root_dir / DEFAULT_BUNDLE_ARTIFACTS_DIRNAME,
        history_dir=root_dir / DEFAULT_BUNDLE_HISTORY_DIRNAME,
    )


def ensure_project_bundle_layout(project_dir: Path) -> ProjectBundlePaths:
    """Create the canonical bundle directories if needed and return their paths."""
    paths = bundle_paths(project_dir)
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths.history_dir.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_bundle_subdirectories(
    project_dir: Path,
    relative_paths: list[str] | tuple[str, ...],
) -> dict[str, Path]:
    """Create extra subdirectories inside a bundle and return them by relative path."""
    paths = ensure_project_bundle_layout(project_dir)
    created: dict[str, Path] = {}
    for relative_path in relative_paths:
        subdir = paths.root_dir / Path(relative_path)
        subdir.mkdir(parents=True, exist_ok=True)
        created[str(relative_path)] = subdir
    return created


def ensure_bundle_subdirectory(project_dir: Path, relative_path: str) -> Path:
    """Create one extra subdirectory inside a bundle and return it."""
    return ensure_bundle_subdirectories(project_dir, (relative_path,))[relative_path]


def ensure_bundle_state_subdirectory(anchor_path: Path, name: str) -> Path:
    """Create a named subdirectory under a bundle's state directory and return it."""
    root_dir = bundle_root_for_path(anchor_path)
    relative_path = str(Path(DEFAULT_BUNDLE_STATE_DIRNAME) / name)
    return ensure_bundle_subdirectory(root_dir, relative_path)


def bundle_root_for_path(path: Path) -> Path:
    """Infer the bundle root from a path inside a project bundle."""
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        if resolved.name in {
            DEFAULT_BUNDLE_STATE_DIRNAME,
            DEFAULT_BUNDLE_ARTIFACTS_DIRNAME,
            DEFAULT_BUNDLE_HISTORY_DIRNAME,
        }:
            return resolved.parent
        return resolved
    if resolved.name == DEFAULT_BUNDLE_MANIFEST_FILENAME:
        return resolved.parent
    if resolved.parent.name in {
        DEFAULT_BUNDLE_STATE_DIRNAME,
        DEFAULT_BUNDLE_ARTIFACTS_DIRNAME,
        DEFAULT_BUNDLE_HISTORY_DIRNAME,
    }:
        return resolved.parent.parent
    return resolved.parent


def write_json_atomic(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Atomically write JSON to *path* using a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(encoded)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def load_project_bundle_manifest(
    project_dir: Path,
    *,
    strict: bool = False,
) -> ProjectBundleManifest | None:
    """Load the shared bundle manifest for *project_dir*, if present."""
    manifest_path = bundle_paths(project_dir).manifest_path
    if not manifest_path.exists():
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid project manifest: {manifest_path}")
        manifest = ProjectBundleManifest.from_dict(raw)
        if manifest.bundle_version != SUPPORTED_BUNDLE_VERSION:
            raise ValueError(
                f"Unsupported project bundle version {manifest.bundle_version}: {manifest_path}"
            )
        if not manifest.state_path:
            raise ValueError(f"Project manifest missing state_path: {manifest_path}")
        return manifest
    except Exception:
        if strict:
            raise
        logger.warning(
            "Ignoring invalid project manifest: %s", manifest_path, exc_info=True
        )
        return None


def save_project_bundle_manifest(
    project_dir: Path,
    manifest: ProjectBundleManifest,
) -> ProjectBundleManifest:
    """Persist *manifest* for *project_dir* with updated timestamps."""
    paths = ensure_project_bundle_layout(project_dir)
    existing = load_project_bundle_manifest(project_dir)
    now = _utc_timestamp()
    manifest_to_save = ProjectBundleManifest(
        bundle_version=int(manifest.bundle_version),
        kit=str(manifest.kit),
        display_name=str(manifest.display_name),
        state_path=str(manifest.state_path),
        database_path=str(manifest.database_path),
        artifacts_dir=str(manifest.artifacts_dir),
        history_dir=str(manifest.history_dir),
        created_at=(
            manifest.created_at
            or (existing.created_at if existing is not None else "")
            or now
        ),
        updated_at=now,
        meta=dict(manifest.meta),
    )
    write_json_atomic(paths.manifest_path, manifest_to_save.to_dict())
    return manifest_to_save


def _archive_root_prefix(members: list[str]) -> str:
    """Return the shared archive prefix that contains the bundle manifest."""
    if DEFAULT_BUNDLE_MANIFEST_FILENAME in members:
        return ""

    manifest_suffix = f"/{DEFAULT_BUNDLE_MANIFEST_FILENAME}"
    prefixes = [
        member[: -len(DEFAULT_BUNDLE_MANIFEST_FILENAME)]
        for member in members
        if member.endswith(manifest_suffix)
    ]
    for prefix in prefixes:
        if prefix and all(member.startswith(prefix) for member in members):
            return prefix
    raise ValueError("Archive does not contain a project bundle manifest")


def _validated_archive_relpath(member: str, prefix: str) -> Path:
    if prefix:
        if not member.startswith(prefix):
            raise ValueError(f"Archive entry escapes bundle root: {member}")
        member = member[len(prefix) :]

    candidate = Path(member)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ValueError(f"Unsafe archive entry: {member}")
    return candidate


def load_project_bundle_archive_manifest(
    archive_path: Path,
    *,
    strict: bool = False,
) -> ProjectBundleManifest | None:
    """Load a bundle manifest directly from a project archive."""
    archive_path = Path(archive_path).expanduser().resolve()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [
                info.filename
                for info in archive.infolist()
                if not info.is_dir() and info.filename
            ]
            prefix = _archive_root_prefix(members)
            manifest_member = f"{prefix}{DEFAULT_BUNDLE_MANIFEST_FILENAME}"
            raw = json.loads(archive.read(manifest_member).decode("utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"Invalid project manifest in archive: {archive_path}")
            manifest = ProjectBundleManifest.from_dict(raw)
            if manifest.bundle_version != SUPPORTED_BUNDLE_VERSION:
                raise ValueError(
                    "Unsupported project bundle version "
                    f"{manifest.bundle_version}: {archive_path}"
                )
            if not manifest.state_path:
                raise ValueError(
                    f"Project manifest missing state_path in archive: {archive_path}"
                )
            return manifest
    except Exception:
        if strict:
            raise
        logger.warning(
            "Ignoring invalid project archive manifest: %s",
            archive_path,
            exc_info=True,
        )
        return None


def export_project_bundle_archive(project_dir: Path, archive_path: Path) -> Path:
    """Zip a project bundle into *archive_path* and return the written archive."""
    project_dir = Path(project_dir).expanduser().resolve()
    archive_path = Path(archive_path).expanduser().resolve()
    manifest = load_project_bundle_manifest(project_dir, strict=True)
    if manifest is None:
        raise ValueError(f"Project bundle manifest not found: {project_dir}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for child in sorted(project_dir.rglob("*")):
            if child.is_dir() or child.resolve() == archive_path:
                continue
            archive.write(child, arcname=child.relative_to(project_dir).as_posix())
    return archive_path


def import_project_bundle_archive(
    archive_path: Path,
    destination_dir: Path,
    *,
    expected_kit: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Extract a project archive into *destination_dir* and return that path."""
    archive_path = Path(archive_path).expanduser().resolve()
    destination_dir = Path(destination_dir).expanduser().resolve()
    manifest = load_project_bundle_archive_manifest(archive_path, strict=True)
    if expected_kit and manifest.kit != expected_kit:
        raise ValueError(
            f"Project archive kit mismatch: expected {expected_kit}, got {manifest.kit}"
        )

    if destination_dir.exists():
        existing = list(destination_dir.iterdir())
        if existing and not overwrite:
            raise FileExistsError(
                f"Destination directory is not empty: {destination_dir}"
            )
    else:
        destination_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename
        ]
        prefix = _archive_root_prefix(members)
        for info in archive.infolist():
            if info.is_dir() or not info.filename:
                continue
            rel_path = _validated_archive_relpath(info.filename, prefix)
            target_path = destination_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source_handle:
                target_path.write_bytes(source_handle.read())

    load_project_bundle_manifest(destination_dir, strict=True)
    return destination_dir
