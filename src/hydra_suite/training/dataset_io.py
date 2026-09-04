"""Bounded, streaming filesystem primitives for dataset preparation.

Dataset inputs are untrusted in size even when they live on a local disk.  The
helpers here deliberately avoid ``rglob``/``read_text`` and put every source
cardinality behind an explicit limit.  A small SQLite index provides stable
global ordering without retaining every pathname in the Python heap.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class DatasetIOLimits:
    max_files: int = 1_000_000
    max_depth: int = 32
    max_path_bytes: int = 16 * 1024
    max_label_bytes: int = 16 * 1024 * 1024
    max_label_lines: int = 1_000_000
    max_line_bytes: int = 1024 * 1024
    max_points_per_object: int = 100_000
    max_classes: int = 4096
    max_metadata_bytes: int = 16 * 1024 * 1024
    max_image_pixels: int = 250_000_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")


DEFAULT_DATASET_IO_LIMITS = DatasetIOLimits()


class DatasetLimitError(RuntimeError):
    """An input exceeded an explicit preparation cardinality/size limit."""


@contextmanager
def sorted_file_index(
    root: Path,
    *,
    suffixes: set[str] | frozenset[str],
    limits: DatasetIOLimits = DEFAULT_DATASET_IO_LIMITS,
) -> Iterator[sqlite3.Connection]:
    """Yield a disk-backed, lexically ordered recursive file index.

    Directory iteration remains streaming.  Global ordering is delegated to
    SQLite, so even a directory containing hundreds of thousands of entries
    cannot create a Python list of all paths.
    """

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Dataset directory not found: {root}")
    fd, db_name = tempfile.mkstemp(prefix="hydra-dataset-index-", suffix=".sqlite3")
    os.close(fd)
    db_path = Path(db_name)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE files (ordinal INTEGER PRIMARY KEY, rel TEXT NOT NULL UNIQUE)"
        )
        count = 0

        def visit(directory: Path, depth: int) -> None:
            nonlocal count
            if depth > limits.max_depth:
                raise DatasetLimitError(
                    f"Dataset directory depth exceeds cap {limits.max_depth}: {directory}"
                )
            try:
                entries = os.scandir(directory)
            except OSError as exc:
                raise RuntimeError(
                    f"Could not inspect dataset directory {directory}: {exc}"
                ) from exc
            with entries:
                for entry in entries:
                    path = Path(entry.path)
                    rel = path.relative_to(root).as_posix()
                    if len(os.fsencode(rel)) > limits.max_path_bytes:
                        raise DatasetLimitError(
                            f"Dataset path exceeds {limits.max_path_bytes} bytes: {path}"
                        )
                    if entry.is_dir(follow_symlinks=False):
                        visit(path, depth + 1)
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if path.suffix.lower() not in suffixes:
                        continue
                    count += 1
                    if count > limits.max_files:
                        raise DatasetLimitError(
                            f"Dataset contains more than {limits.max_files} matching files"
                        )
                    connection.execute(
                        "INSERT INTO files(ordinal, rel) VALUES (?, ?)", (count, rel)
                    )
            if count % 4096 == 0:
                connection.commit()

        visit(root, 0)
        connection.commit()
        connection.execute("CREATE INDEX files_rel ON files(rel)")
        yield connection
    finally:
        connection.close()
        db_path.unlink(missing_ok=True)


def iter_indexed_paths(connection: sqlite3.Connection, root: Path) -> Iterator[Path]:
    """Yield paths from :func:`sorted_file_index` in deterministic order."""

    cursor = connection.execute("SELECT rel FROM files ORDER BY rel")
    for (relative,) in cursor:
        yield root / str(relative)


def iter_bounded_text_lines(
    path: Path,
    *,
    limits: DatasetIOLimits = DEFAULT_DATASET_IO_LIMITS,
) -> Iterator[str]:
    """Decode a text file one bounded line at a time."""

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"Could not inspect text input {path}: {exc}") from exc
    if size > limits.max_label_bytes:
        raise DatasetLimitError(
            f"Text input exceeds {limits.max_label_bytes} bytes: {path}"
        )
    with path.open("rb") as stream:
        for line_number in range(1, limits.max_label_lines + 2):
            raw = stream.readline(limits.max_line_bytes + 1)
            if not raw:
                return
            if line_number > limits.max_label_lines:
                raise DatasetLimitError(
                    f"Text input exceeds {limits.max_label_lines} lines: {path}"
                )
            if len(raw) > limits.max_line_bytes:
                raise DatasetLimitError(
                    f"Line {line_number} exceeds {limits.max_line_bytes} bytes: {path}"
                )
            try:
                yield raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError as exc:
                raise DatasetLimitError(
                    f"Text input is not valid UTF-8 at line {line_number}: {path}"
                ) from exc


def read_bounded_text(
    path: Path,
    *,
    max_bytes: int,
) -> str:
    """Read a small metadata file only after bounding its encoded size."""

    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise DatasetLimitError(f"Metadata exceeds {max_bytes} bytes: {path}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetLimitError(f"Metadata is not valid UTF-8: {path}") from exc


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_output_directory(target: Path) -> Iterator[Path]:
    """Build beside *target* and atomically promote only on success."""

    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
    staging.mkdir()

    def sync_tree(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    sync_tree(path)
                elif entry.is_file(follow_symlinks=False):
                    fsync_file(path)
        fsync_directory(directory)

    try:
        yield staging
        sync_tree(staging)
        moved_old = False
        if target.exists():
            os.replace(target, backup)
            moved_old = True
        try:
            os.replace(staging, target)
            fsync_directory(target.parent)
        except BaseException:
            if moved_old and backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            # Promotion already succeeded. A stale private backup is preferable
            # to reporting failure after the new complete dataset is visible.
            shutil.rmtree(backup, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
