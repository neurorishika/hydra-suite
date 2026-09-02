"""Project-local recovery transactions for DetectKit dataset mutations.

Images and labels can belong to linked sources outside a DetectKit project.
Keeping recovery payloads below ``artifacts/recovery`` gives those mutations
the same predictable undo location as project-owned sources without depending
on platform Trash APIs or filesystem-volume boundaries.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from .utils import find_label_for_image

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "operation.json"


class DatasetRecoveryError(RuntimeError):
    """Raised when a recoverable mutation or its undo cannot complete safely."""


@dataclass(frozen=True)
class DatasetRecoveryEntry:
    """One original file and its project-local recovery payload."""

    original_path: Path
    recovery_path: Path
    role: str
    strategy: str


@dataclass(frozen=True)
class DatasetRecoveryOperation:
    """A completed or restored dataset recovery transaction."""

    operation_id: str
    action: str
    status: str
    item_count: int
    created_at: str
    entries: tuple[DatasetRecoveryEntry, ...]
    manifest_path: Path

    @property
    def summary(self) -> str:
        noun = "image" if self.action == "remove_images" else "label file"
        suffix = "" if self.item_count == 1 else "s"
        verb = "Removed" if self.action == "remove_images" else "Cleared"
        return f"{verb} {self.item_count} {noun}{suffix}"


def _recovery_root(project_dir: str | Path) -> Path:
    root = Path(project_dir).expanduser().resolve() / "artifacts" / "recovery"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dedupe_existing(paths: Iterable[str | Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().absolute()
        if path in seen:
            continue
        if not path.is_file():
            raise DatasetRecoveryError(f"Dataset file no longer exists: {path}")
        seen.add(path)
        result.append(path)
    return result


def _new_operation(
    project_dir: str | Path,
    *,
    action: str,
    item_count: int,
    files: list[tuple[Path, str, str]],
) -> DatasetRecoveryOperation:
    now = datetime.now(timezone.utc)
    operation_id = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
    operation_dir = _recovery_root(project_dir) / operation_id
    payload_dir = operation_dir / "files"
    payload_dir.mkdir(parents=True)
    entries = tuple(
        DatasetRecoveryEntry(
            original_path=original,
            recovery_path=payload_dir / f"{index:06d}-{original.name}",
            role=role,
            strategy=strategy,
        )
        for index, (original, role, strategy) in enumerate(files, start=1)
    )
    return DatasetRecoveryOperation(
        operation_id=operation_id,
        action=action,
        status="preparing",
        item_count=item_count,
        created_at=now.isoformat(),
        entries=entries,
        manifest_path=operation_dir / _MANIFEST_NAME,
    )


def _manifest_data(operation: DatasetRecoveryOperation) -> dict[str, object]:
    operation_dir = operation.manifest_path.parent
    project_dir = operation_dir.parents[2].resolve()

    def _stored_original(path: Path) -> dict[str, str]:
        try:
            relative = path.relative_to(project_dir)
        except ValueError:
            return {"scope": "external", "path": str(path)}
        return {"scope": "project", "path": relative.as_posix()}

    return {
        "version": 1,
        "operation_id": operation.operation_id,
        "action": operation.action,
        "status": operation.status,
        "item_count": operation.item_count,
        "created_at": operation.created_at,
        "entries": [
            {
                "original_path": _stored_original(entry.original_path),
                "recovery_path": entry.recovery_path.relative_to(
                    operation_dir
                ).as_posix(),
                "role": entry.role,
                "strategy": entry.strategy,
            }
            for entry in operation.entries
        ],
    }


def _write_manifest(operation: DatasetRecoveryOperation) -> None:
    manifest = operation.manifest_path
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(_manifest_data(operation), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)


def _finish_operation(operation: DatasetRecoveryOperation) -> DatasetRecoveryOperation:
    completed = replace(operation, status="active")
    _write_manifest(completed)
    return completed


def _cleanup_failed_operation(operation: DatasetRecoveryOperation) -> None:
    """Remove a fully rolled-back operation or retain any stranded payload.

    A second I/O failure can prevent rollback from restoring a file. In that
    case deleting the operation directory would turn a recoverable failure
    into data loss, so expose the stranded entries as a normal Undo operation.
    """
    recoverable: list[DatasetRecoveryEntry] = []
    for entry in operation.entries:
        if not entry.recovery_path.is_file():
            continue
        if entry.strategy == "move" and not entry.original_path.exists():
            recoverable.append(entry)
        elif entry.strategy == "copy" and (
            not entry.original_path.exists()
            or (
                entry.original_path.is_file()
                and entry.original_path.stat().st_size == 0
            )
        ):
            recoverable.append(entry)

    if recoverable:
        item_count = (
            sum(entry.role == "image" for entry in recoverable)
            if operation.action == "remove_images"
            else len(recoverable)
        )
        retained = replace(
            operation,
            status="active",
            item_count=item_count,
            entries=tuple(recoverable),
        )
        try:
            _write_manifest(retained)
        except OSError:
            logger.exception(
                "Could not update the manifest for stranded recovery payloads"
            )
        logger.error(
            "Retained %d dataset file(s) in recovery after rollback failed",
            len(recoverable),
        )
        return

    try:
        shutil.rmtree(operation.manifest_path.parent)
    except OSError:
        logger.warning(
            "Could not remove failed dataset recovery operation %s",
            operation.operation_id,
            exc_info=True,
        )


def remove_images_with_recovery(
    project_dir: str | Path,
    source_path: str | Path,
    image_paths: Iterable[str | Path],
) -> DatasetRecoveryOperation:
    """Move images and matching labels into a project-local recovery operation."""
    source_root = Path(source_path).expanduser().resolve()
    images_root = source_root / "images"
    allowed_root = images_root if images_root.is_dir() else source_root
    images = _dedupe_existing(image_paths)
    for image in images:
        try:
            image.resolve().relative_to(allowed_root.resolve())
        except ValueError as exc:
            raise DatasetRecoveryError(
                f"Refusing to remove an image outside the selected source: {image}"
            ) from exc
    if not images:
        raise DatasetRecoveryError("No existing images were selected for removal.")

    labels: list[Path] = []
    seen_labels: set[Path] = set()
    for image in images:
        label = find_label_for_image(image, str(source_root))
        if label is None:
            continue
        label = label.expanduser().absolute()
        if label not in seen_labels and label.is_file():
            seen_labels.add(label)
            labels.append(label)

    files = [(image, "image", "move") for image in images]
    files.extend((label, "label", "move") for label in labels)
    operation = _new_operation(
        project_dir,
        action="remove_images",
        item_count=len(images),
        files=files,
    )
    _write_manifest(operation)

    moved: list[DatasetRecoveryEntry] = []
    try:
        for entry in operation.entries:
            shutil.move(str(entry.original_path), str(entry.recovery_path))
            moved.append(entry)
        return _finish_operation(operation)
    except Exception as exc:
        for entry in reversed(moved):
            try:
                entry.original_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(entry.recovery_path), str(entry.original_path))
            except OSError:
                logger.exception("Failed to roll back dataset removal for %s", entry)
        _cleanup_failed_operation(operation)
        raise DatasetRecoveryError(
            f"Could not stage selected files for recovery: {exc}"
        ) from exc


def clear_labels_with_recovery(
    project_dir: str | Path,
    label_paths: Iterable[str | Path],
) -> DatasetRecoveryOperation:
    """Snapshot exact label bytes, then clear the originals transactionally."""
    labels = _dedupe_existing(label_paths)
    if not labels:
        raise DatasetRecoveryError(
            "No existing label files were selected for clearing."
        )

    operation = _new_operation(
        project_dir,
        action="clear_labels",
        item_count=len(labels),
        files=[(label, "label", "copy") for label in labels],
    )
    _write_manifest(operation)
    try:
        for entry in operation.entries:
            shutil.copy2(entry.original_path, entry.recovery_path)
        for entry in operation.entries:
            entry.original_path.write_bytes(b"")
        return _finish_operation(operation)
    except Exception as exc:
        for entry in operation.entries:
            if not entry.recovery_path.is_file():
                continue
            try:
                shutil.copy2(entry.recovery_path, entry.original_path)
            except OSError:
                logger.exception("Failed to roll back label clearing for %s", entry)
        _cleanup_failed_operation(operation)
        raise DatasetRecoveryError(
            f"Could not snapshot and clear labels: {exc}"
        ) from exc


def _load_operation(manifest_path: Path) -> DatasetRecoveryOperation:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        operation_dir = manifest_path.parent.resolve()
        project_dir = operation_dir.parents[2]
        entries: list[DatasetRecoveryEntry] = []
        for raw_entry in data["entries"]:
            recovery_path = (operation_dir / raw_entry["recovery_path"]).resolve()
            try:
                recovery_path.relative_to(operation_dir)
            except ValueError as exc:
                raise DatasetRecoveryError(
                    f"Recovery manifest points outside its operation: {manifest_path}"
                ) from exc
            stored_original = raw_entry["original_path"]
            if isinstance(stored_original, dict):
                if stored_original.get("scope") == "project":
                    original_path = project_dir / str(stored_original["path"])
                elif stored_original.get("scope") == "external":
                    original_path = Path(stored_original["path"])
                else:
                    raise DatasetRecoveryError(
                        f"Recovery manifest has an invalid path scope: {manifest_path}"
                    )
            else:
                # Version-1 development manifests briefly stored absolute strings.
                original_path = Path(stored_original)
            entries.append(
                DatasetRecoveryEntry(
                    original_path=original_path,
                    recovery_path=recovery_path,
                    role=str(raw_entry["role"]),
                    strategy=str(raw_entry["strategy"]),
                )
            )
        return DatasetRecoveryOperation(
            operation_id=str(data["operation_id"]),
            action=str(data["action"]),
            status=str(data["status"]),
            item_count=int(data["item_count"]),
            created_at=str(data["created_at"]),
            entries=tuple(entries),
            manifest_path=manifest_path.resolve(),
        )
    except DatasetRecoveryError:
        raise
    except Exception as exc:
        raise DatasetRecoveryError(
            f"Could not read recovery manifest {manifest_path}: {exc}"
        ) from exc


def latest_dataset_recovery(
    project_dir: str | Path,
) -> DatasetRecoveryOperation | None:
    """Return the newest operation that is still available to undo."""
    root = _recovery_root(project_dir)
    for operation_dir in sorted(root.iterdir(), reverse=True):
        manifest = operation_dir / _MANIFEST_NAME
        if not manifest.is_file():
            continue
        try:
            operation = _load_operation(manifest)
        except DatasetRecoveryError:
            logger.warning("Ignoring invalid dataset recovery manifest %s", manifest)
            continue
        if operation.status == "active":
            return operation
    return None


def undo_latest_dataset_recovery(
    project_dir: str | Path,
) -> DatasetRecoveryOperation:
    """Restore the newest active operation without overwriting later edits."""
    operation = latest_dataset_recovery(project_dir)
    if operation is None:
        raise DatasetRecoveryError("There is no dataset change available to undo.")

    label_existed: dict[Path, bool] = {}
    for entry in operation.entries:
        if not entry.recovery_path.is_file():
            raise DatasetRecoveryError(
                f"Recovery payload is missing for {entry.original_path}."
            )
        if entry.strategy == "move" and entry.original_path.exists():
            raise DatasetRecoveryError(
                f"Cannot restore {entry.original_path}: its original path already exists."
            )
        if entry.strategy == "copy":
            label_existed[entry.original_path] = entry.original_path.exists()
            if entry.original_path.exists() and not entry.original_path.is_file():
                raise DatasetRecoveryError(
                    f"Cannot restore {entry.original_path}: its original path already exists."
                )
            if (
                entry.original_path.is_file()
                and entry.original_path.stat().st_size != 0
            ):
                raise DatasetRecoveryError(
                    f"Cannot restore {entry.original_path}: it changed since labels were cleared."
                )

    restored: list[DatasetRecoveryEntry] = []
    try:
        for entry in operation.entries:
            entry.original_path.parent.mkdir(parents=True, exist_ok=True)
            if entry.strategy == "move":
                shutil.move(str(entry.recovery_path), str(entry.original_path))
            else:
                shutil.copy2(entry.recovery_path, entry.original_path)
            restored.append(entry)
    except Exception as exc:
        for entry in reversed(restored):
            try:
                if entry.strategy == "move":
                    entry.recovery_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(entry.original_path), str(entry.recovery_path))
                elif label_existed.get(entry.original_path, False):
                    entry.original_path.write_bytes(b"")
                else:
                    entry.original_path.unlink(missing_ok=True)
            except OSError:
                logger.exception(
                    "Failed to roll back dataset recovery undo for %s", entry
                )
        raise DatasetRecoveryError(f"Could not restore dataset files: {exc}") from exc

    restored_operation = replace(operation, status="restored")
    _write_manifest(restored_operation)
    return restored_operation
