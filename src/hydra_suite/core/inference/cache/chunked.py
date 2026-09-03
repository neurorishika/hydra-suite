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
import re
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
MAX_CHUNK_FRAMES = 1_000_000
MAX_FRAME_INDEX = np.iinfo(np.int32).max
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class ChunkEntry:
    name: str
    ranges: tuple[tuple[int, int], ...]
    byte_size: int
    sha256: str

    @classmethod
    def from_dict(cls, raw: dict) -> "ChunkEntry":
        if not isinstance(raw, dict):
            raise TypeError("chunk entry must be an object")
        name = raw["name"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"chunk-[0-9]{8}\.npz", name) is None
        ):
            raise ValueError("unsafe chunk name")
        byte_size = raw["byte_size"]
        sha256 = raw["sha256"]
        if (
            not isinstance(byte_size, int)
            or isinstance(byte_size, bool)
            or byte_size < 1
        ):
            raise ValueError("invalid chunk byte size")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("invalid chunk checksum")
        ranges_raw = raw.get("ranges")
        if not isinstance(ranges_raw, list) or not ranges_raw:
            raise ValueError("chunk ranges must be a nonempty list")
        ranges: list[tuple[int, int]] = []
        cardinality = 0
        previous_end = -1
        for bounds in ranges_raw:
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError("invalid chunk range")
            start, end = bounds
            if any(not isinstance(v, int) or isinstance(v, bool) for v in bounds):
                raise ValueError("chunk range bounds must be integers")
            if start < 0 or start > end or end > MAX_FRAME_INDEX:
                raise ValueError("chunk range is out of bounds")
            if start <= previous_end:
                raise ValueError("chunk ranges must be ordered and disjoint")
            cardinality += end - start + 1
            if cardinality > MAX_CHUNK_FRAMES:
                raise ValueError("chunk frame cardinality exceeds safety limit")
            ranges.append((start, end))
            previous_end = end
        return cls(
            name=name,
            ranges=tuple(ranges),
            byte_size=byte_size,
            sha256=sha256,
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


def _safe_component(value: str) -> bool:
    return value not in {".", ".."} and _SAFE_COMPONENT.fullmatch(value) is not None


def _scalar(raw: np.lib.npyio.NpzFile, name: str):
    value = raw[name]
    if value.ndim != 1 or value.size != 1:
        raise ValueError(f"manifest field {name!r} must contain exactly one value")
    return value[0]


def _frames_match_ranges(
    frames: np.ndarray, ranges: tuple[tuple[int, int], ...]
) -> bool:
    values = np.asarray(frames)
    if values.ndim != 1 or values.size > MAX_CHUNK_FRAMES:
        return False
    expected_count = sum(end - start + 1 for start, end in ranges)
    if values.size != expected_count:
        return False
    offset = 0
    for start, end in ranges:
        length = end - start + 1
        part = values[offset : offset + length]
        if length and (
            int(part[0]) != start
            or int(part[-1]) != end
            or (length > 1 and not np.all(np.diff(part) == 1))
        ):
            return False
        offset += length
    return True


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
        self._defer_manifest = False
        self._deep_validated = False

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

    def is_reusable(self) -> bool:
        """Validate every referenced payload checksum before replay."""
        self._load_manifest()
        if not self._valid:
            return False
        if self._deep_validated:
            return True
        if self._legacy:
            self._deep_validated = self.load_legacy() is not None
            return self._deep_validated
        for entry in self._entries:
            arrays = self._load_entry_arrays(entry)
            if arrays is None:
                return False
            self._cached_entry = entry
            self._cached_arrays = arrays
        self._deep_validated = True
        return True

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
                        stored_key = str(_scalar(raw, "cache_key"))
                        self._valid = stored_key == self.key.as_string()
                    else:
                        self._valid = True
                    return
                if files != {
                    _FORMAT_FIELD,
                    "cache_kind",
                    "cache_key",
                    "session_id",
                    "chunks_json",
                }:
                    return
                if raw[_FORMAT_FIELD].dtype.kind not in "iu":
                    return
                if any(
                    raw[name].dtype.kind not in "US"
                    for name in ("cache_kind", "cache_key", "session_id", "chunks_json")
                ):
                    return
                version = int(_scalar(raw, _FORMAT_FIELD))
                stored_kind = str(_scalar(raw, "cache_kind"))
                stored_key = str(_scalar(raw, "cache_key"))
                session_id = str(_scalar(raw, "session_id"))
                entries_raw = json.loads(str(_scalar(raw, "chunks_json")))
            if version != CHUNK_FORMAT_VERSION or stored_kind != self.kind:
                return
            if self.require_key and stored_key != self.key.as_string():
                return
            if not isinstance(entries_raw, list) or len(entries_raw) > MAX_CHUNK_FRAMES:
                return
            entries = [ChunkEntry.from_dict(item) for item in entries_raw]
            if not _safe_component(session_id):
                return
            self._session_id = session_id
            self._entries = entries
            # Missing or truncated referenced chunks invalidate the cache. A
            # renamed-but-unpublished orphan is absent from entries and ignored.
            for entry in entries:
                if not entry.ranges or not entry.name.startswith("chunk-"):
                    return
                chunk_path = self._chunk_path(entry)
                if (
                    not chunk_path.is_file()
                    or chunk_path.stat().st_size != entry.byte_size
                ):
                    return
            self._rebuild_frame_index()
            self._valid = True
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            json.JSONDecodeError,
        ):
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

    def contains_frame(self, frame_idx: int) -> bool:
        self._load_manifest()
        return (
            self._valid
            and not self._legacy
            and self._entry_for_frame(int(frame_idx)) is not None
        )

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
        if (
            len(frames) > MAX_CHUNK_FRAMES
            or any(frame < 0 or frame > MAX_FRAME_INDEX for frame in frames)
            or any(right <= left for left, right in zip(frames, frames[1:]))
        ):
            raise ValueError(
                "written frames must be unique, nonnegative, and increasing"
            )
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
        if not self._defer_manifest:
            self._publish_manifest(new_entries)
        self._entries = new_entries
        self._rebuild_frame_index()
        self._valid = True
        self._legacy = False
        self._cached_entry = None
        self._cached_arrays = None
        self._deep_validated = False

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

    def commit_generation(self) -> None:
        """Atomically publish a complete staged replacement generation."""
        if self._defer_manifest:
            self._publish_manifest(self._entries)
            self._defer_manifest = False
            self._valid = True
            self._legacy = False

    def _prepare_for_write(self) -> None:
        self._load_manifest()
        if self._fresh_staged:
            return
        if self._valid and not self._legacy:
            return
        # Invalid, missing, and legacy files start a fresh chunk session on the
        # first actual write. Merely opening a write handle never migrates data.
        preserve_published_cache = self.path.is_file()
        self._legacy = False
        self._valid = False
        self._session_id = self._session_prefix() + "-00000000"
        self._entries = []
        self._range_index = []
        self._range_starts = []
        self._range_prefix_max_end = []
        self._fresh_staged = True
        self._defer_manifest = preserve_published_cache

    def start_fresh(self) -> None:
        """Stage a new generation while leaving the published cache untouched."""
        self._load_manifest()
        preserve_published_cache = self.path.is_file()
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
        self._deep_validated = False
        self._fresh_staged = True
        self._defer_manifest = preserve_published_cache

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
            arrays = self._load_entry_arrays(entry)
            if arrays is None:
                return None
            self._cached_entry = entry
            self._cached_arrays = arrays
        return self._cached_arrays

    def _load_entry_arrays(self, entry: ChunkEntry) -> dict[str, np.ndarray] | None:
        chunk_path = self._chunk_path(entry)
        try:
            if _sha256_file(chunk_path) != entry.sha256:
                self._valid = False
                return None
            with np.load(chunk_path, allow_pickle=False) as raw:
                arrays = {name: raw[name] for name in raw.files}
            if not _frames_match_ranges(arrays["written_frames"], entry.ranges):
                self._valid = False
                return None
            return arrays
        except (OSError, ValueError, KeyError):
            self._valid = False
            return None

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
            arrays = self._load_entry_arrays(entry)
            if arrays is None:
                return
            winning = np.asarray(
                [
                    self._entry_for_frame(int(frame)) == entry
                    for frame in arrays["written_frames"]
                ],
                dtype=bool,
            )
            if not np.any(winning):
                continue
            winning_frames = arrays["written_frames"][winning]
            row_frames = np.asarray(arrays.get("frame_indices", []))
            row_mask = np.isin(row_frames, winning_frames)
            filtered = {}
            metadata_fields = {
                "factor_names_json",
                "class_names_json",
                "class_counts",
            }
            for name, value in arrays.items():
                if name == "written_frames":
                    filtered[name] = winning_frames
                elif (
                    name not in metadata_fields
                    and value.ndim > 0
                    and len(value) == len(row_frames)
                ):
                    filtered[name] = value[row_mask]
                else:
                    filtered[name] = value
            yield filtered

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
