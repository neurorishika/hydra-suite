"""Dependency-light immutable-chunk storage for inference caches.

The canonical ``*.npz`` is a small manifest. Payload arrays live in immutable
NPZ chunks under a sibling directory and are published before the manifest
references them. Legacy monolithic NPZ files are detected but never rewritten
as a side effect of reading.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import CacheKey

CHUNK_FORMAT_VERSION = 1
DEFAULT_CHUNK_FRAMES = 64
_FORMAT_FIELD = "chunked_format_version"


@dataclass(frozen=True)
class ChunkEntry:
    name: str
    ranges: tuple[tuple[int, int], ...]
    byte_size: int
    sha256: str

    @classmethod
    def from_dict(cls, raw: dict) -> "ChunkEntry":
        return cls(
            name=str(raw["name"]),
            ranges=(
                tuple((int(start), int(end)) for start, end in raw["ranges"])
                if "ranges" in raw
                else _compress_frames(tuple(int(v) for v in raw["frames"]))
            ),
            byte_size=int(raw["byte_size"]),
            sha256=str(raw["sha256"]),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ranges": [list(bounds) for bounds in self.ranges],
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(
            frame
            for start, end in self.ranges
            for frame in range(int(start), int(end) + 1)
        )

    @property
    def first_frame(self) -> int:
        return self.ranges[0][0]

    def contains(self, frame_idx: int) -> bool:
        return any(start <= frame_idx <= end for start, end in self.ranges)


def _compress_frames(frames: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if not frames:
        return ()
    ordered = sorted({int(frame) for frame in frames})
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        ranges.append((start, previous))
        start = previous = frame
    ranges.append((start, previous))
    return tuple(ranges)


def _atomic_npz_save(path: Path, **arrays: np.ndarray) -> None:
    """Write an NPZ beside *path*, fsync it, then atomically replace *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some filesystems do not support directory fsync. The payload file
        # itself was still flushed before the atomic rename.
        pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class ChunkedArrayStore:
    """Atomic manifest and immutable NPZ payload chunks for one cache kind."""

    def __init__(
        self,
        path: Path,
        key: CacheKey,
        kind: str,
        *,
        require_key: bool = True,
    ) -> None:
        self.path = Path(path)
        self.key = key
        self.kind = str(kind)
        self.require_key = bool(require_key)
        self._loaded = False
        self._legacy = False
        self._valid = False
        self._session_id = ""
        self._entries: list[ChunkEntry] = []
        self._range_index: list[tuple[int, int, int, ChunkEntry]] = []
        self._range_starts: list[int] = []
        self._range_prefix_max_end: list[int] = []
        self._cached_entry: ChunkEntry | None = None
        self._cached_arrays: dict[str, np.ndarray] | None = None
        self._fresh_staged = False

    @property
    def is_legacy(self) -> bool:
        self._load_manifest()
        return self._legacy

    @property
    def chunk_directory(self) -> Path:
        return self.path.parent / f"{self.path.name}.chunks" / self._session_id

    def is_valid(self) -> bool:
        self._load_manifest()
        return self._valid

    def _load_manifest(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.is_file():
            return
        try:
            with np.load(self.path, allow_pickle=False) as raw:
                files = set(raw.files)
                if _FORMAT_FIELD not in files:
                    self._legacy = True
                    if self.require_key:
                        stored_key = str(raw["cache_key"][0])
                        self._valid = stored_key == self.key.as_string()
                    else:
                        self._valid = True
                    return
                version = int(raw[_FORMAT_FIELD][0])
                stored_kind = str(raw["cache_kind"][0])
                stored_key = str(raw["cache_key"][0])
                session_id = str(raw["session_id"][0])
                entries_raw = json.loads(str(raw["chunks_json"][0]))
            if version != CHUNK_FORMAT_VERSION or stored_kind != self.kind:
                return
            if self.require_key and stored_key != self.key.as_string():
                return
            entries = [ChunkEntry.from_dict(item) for item in entries_raw]
            if not session_id or Path(session_id).name != session_id:
                return
            self._session_id = session_id
            self._entries = entries
            # Missing or truncated referenced chunks invalidate the cache. A
            # renamed-but-unpublished orphan is absent from entries and ignored.
            for entry in entries:
                if (
                    not entry.ranges
                    or Path(entry.name).name != entry.name
                    or not entry.name.startswith("chunk-")
                ):
                    return
                chunk_path = self._chunk_path(entry)
                if (
                    not chunk_path.is_file()
                    or chunk_path.stat().st_size != entry.byte_size
                ):
                    return
            self._rebuild_frame_index()
            self._valid = True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self._valid = False

    def _rebuild_frame_index(self) -> None:
        # Later chunks intentionally win when a resumed full pass recomputes a
        # frame already present in an earlier partial pass.
        self._range_index = sorted(
            (
                (start, end, sequence, entry)
                for sequence, entry in enumerate(self._entries)
                for start, end in entry.ranges
            ),
            key=lambda item: (item[0], item[2]),
        )
        self._range_starts = [item[0] for item in self._range_index]
        running_max = -1
        self._range_prefix_max_end = []
        for _, end, _, _ in self._range_index:
            running_max = max(running_max, end)
            self._range_prefix_max_end.append(running_max)

    def _chunk_path(self, entry: ChunkEntry) -> Path:
        return self.chunk_directory / entry.name

    def written_frames(self) -> set[int]:
        self._load_manifest()
        if not self._valid or self._legacy:
            return set()
        return {
            frame
            for entry in self._entries
            for start, end in entry.ranges
            for frame in range(start, end + 1)
        }

    def covered_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return normalized processed-frame coverage without expanding IDs."""
        self._load_manifest()
        if not self._valid or self._legacy:
            return ()
        source = sorted(bounds for entry in self._entries for bounds in entry.ranges)
        merged: list[tuple[int, int]] = []
        for start, end in source:
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return tuple(merged)

    def append_chunk(
        self, written_frames: list[int] | tuple[int, ...], arrays: dict[str, np.ndarray]
    ) -> None:
        """Publish one payload chunk, then atomically make it visible."""
        frames = tuple(int(frame) for frame in written_frames)
        if not frames:
            return
        self._prepare_for_write()
        sequence = len(self._entries)
        name = f"chunk-{sequence:08d}.npz"
        chunk_path = self.chunk_directory / name
        payload = dict(arrays)
        payload["written_frames"] = np.asarray(frames, dtype=np.int64)
        _atomic_npz_save(chunk_path, **payload)
        self._validate_staged_chunk(chunk_path, frames, set(payload))
        entry = ChunkEntry(
            name=name,
            ranges=_compress_frames(frames),
            byte_size=chunk_path.stat().st_size,
            sha256=_sha256_file(chunk_path),
        )
        new_entries = [*self._entries, entry]
        self._publish_manifest(new_entries)
        self._entries = new_entries
        self._rebuild_frame_index()
        self._valid = True
        self._legacy = False
        self._cached_entry = None
        self._cached_arrays = None

    @staticmethod
    def _validate_staged_chunk(
        chunk_path: Path, frames: tuple[int, ...], expected_fields: set[str]
    ) -> None:
        """Read back one bounded chunk before the manifest can reference it."""
        with np.load(chunk_path, allow_pickle=False) as raw:
            if set(raw.files) != expected_fields:
                raise ValueError(f"cache chunk fields are incomplete: {chunk_path}")
            stored = tuple(int(frame) for frame in raw["written_frames"])
            if stored != frames:
                raise ValueError(f"cache chunk frame index mismatch: {chunk_path}")

    def ensure_manifest(self) -> None:
        """Create a valid empty chunked manifest without migrating on read."""
        self._prepare_for_write()
        if self.path.is_file() and self._valid and not self._legacy:
            return
        self._publish_manifest([])
        self._entries = []
        self._rebuild_frame_index()
        self._valid = True

    def _prepare_for_write(self) -> None:
        self._load_manifest()
        if self._fresh_staged:
            return
        if self._valid and not self._legacy:
            return
        # Invalid, missing, and legacy files start a fresh chunk session on the
        # first actual write. Merely opening a write handle never migrates data.
        self._legacy = False
        self._valid = False
        self._session_id = self._session_prefix() + "-00000000"
        self._entries = []
        self._range_index = []
        self._range_starts = []
        self._range_prefix_max_end = []
        self._fresh_staged = True

    def start_fresh(self) -> None:
        """Stage a new generation while leaving the published cache untouched."""
        self._load_manifest()
        prefix = self._session_prefix()
        generation = 0
        if self._session_id.startswith(prefix + "-"):
            try:
                generation = int(self._session_id.rsplit("-", 1)[1]) + 1
            except ValueError:
                generation = 0
        self._session_id = f"{prefix}-{generation:08d}"
        self._entries = []
        self._range_index = []
        self._range_starts = []
        self._range_prefix_max_end = []
        self._valid = False
        self._legacy = False
        self._cached_entry = None
        self._cached_arrays = None
        self._fresh_staged = True

    def _session_prefix(self) -> str:
        identity = f"{self.kind}\0{self.key.as_string()}".encode("utf-8")
        return hashlib.sha256(identity).hexdigest()[:24]

    def _publish_manifest(self, entries: list[ChunkEntry]) -> None:
        encoded = json.dumps(
            [entry.to_dict() for entry in entries],
            sort_keys=True,
            separators=(",", ":"),
        )
        _atomic_npz_save(
            self.path,
            chunked_format_version=np.asarray([CHUNK_FORMAT_VERSION], dtype=np.int32),
            cache_kind=np.asarray([self.kind]),
            cache_key=np.asarray([self.key.as_string()]),
            session_id=np.asarray([self._session_id]),
            chunks_json=np.asarray([encoded]),
        )

    def read_frame_arrays(self, frame_idx: int) -> dict[str, np.ndarray] | None:
        """Load only the immutable chunk containing *frame_idx*."""
        self._load_manifest()
        if not self._valid or self._legacy:
            return None
        entry = self._entry_for_frame(int(frame_idx))
        if entry is None:
            return None
        if entry != self._cached_entry:
            chunk_path = self._chunk_path(entry)
            try:
                if _sha256_file(chunk_path) != entry.sha256:
                    self._valid = False
                    return None
                with np.load(chunk_path, allow_pickle=False) as raw:
                    arrays = {name: raw[name] for name in raw.files}
                stored_frames = tuple(int(v) for v in arrays["written_frames"])
                if stored_frames != entry.frames:
                    self._valid = False
                    return None
            except (OSError, ValueError, KeyError):
                self._valid = False
                return None
            self._cached_entry = entry
            self._cached_arrays = arrays
        return self._cached_arrays

    def _entry_for_frame(self, frame_idx: int) -> ChunkEntry | None:
        position = bisect_right(self._range_starts, frame_idx) - 1
        best_sequence = -1
        best_entry: ChunkEntry | None = None
        while position >= 0:
            start, end, sequence, entry = self._range_index[position]
            if start > frame_idx:
                position -= 1
                continue
            if end >= frame_idx and sequence > best_sequence:
                best_sequence = sequence
                best_entry = entry
            # Non-overlapping normal caches find the answer immediately. For
            # resumed overlapping chunks, continue only across the same or an
            # earlier start that can still contain this frame.
            position -= 1
            if position < 0 or self._range_prefix_max_end[position] < frame_idx:
                break
        return best_entry

    def iter_chunk_arrays(self) -> Iterator[dict[str, np.ndarray]]:
        """Yield each referenced chunk in manifest order, one at a time."""
        self._load_manifest()
        if not self._valid or self._legacy:
            return
        for entry in self._entries:
            first = entry.first_frame
            arrays = self.read_frame_arrays(first)
            if arrays is None:
                return
            yield arrays

    def load_legacy(self) -> dict[str, np.ndarray] | None:
        """Materialize a legacy NPZ only; new chunked caches never use this."""
        self._load_manifest()
        if not self._valid or not self._legacy:
            return None
        try:
            with np.load(self.path, allow_pickle=False) as raw:
                return {name: raw[name] for name in raw.files}
        except (OSError, ValueError):
            self._valid = False
            return None
