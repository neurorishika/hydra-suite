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
import uuid
import zipfile
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import CacheKey

CHUNK_FORMAT_VERSION = 2
DEFAULT_CHUNK_FRAMES = 64
_FORMAT_FIELD = "chunked_format_version"
MAX_CHUNK_FRAMES = 1_000_000
MAX_FRAME_INDEX = np.iinfo(np.int32).max
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_ARRAY_NAME = re.compile(r"^[A-Za-z0-9_]+\.npy$")
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_CHUNK_BYTES = 256 * 1024 * 1024
MAX_LEGACY_BYTES = 256 * 1024 * 1024
MAX_NPZ_MEMBERS = 64
MAX_STRING_ITEM_BYTES = 1024 * 1024
_CHUNK_META_FIELDS = {
    "_cache_kind",
    "_cache_key",
    "_session_id",
    "_generation_id",
    "_chunk_position",
}


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
            or re.fullmatch(r"chunk-[0-9]{8}-[0-9a-f]{32}\.npz", name) is None
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


def _inspect_npz(
    path: Path,
    *,
    max_uncompressed_bytes: int,
    max_members: int = MAX_NPZ_MEMBERS,
) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Validate ZIP and NPY metadata before NumPy may allocate payload arrays."""
    metadata: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > max_members:
                raise ValueError("unsafe NPZ member count")
            total = 0
            names: set[str] = set()
            for info in infos:
                if (
                    info.is_dir()
                    or _SAFE_ARRAY_NAME.fullmatch(info.filename) is None
                    or info.filename in names
                ):
                    raise ValueError("unsafe or duplicate NPZ member name")
                names.add(info.filename)
                if info.file_size < 0 or info.compress_size < 0:
                    raise ValueError("invalid NPZ member size")
                total += info.file_size
                if (
                    info.file_size > max_uncompressed_bytes
                    or total > max_uncompressed_bytes
                ):
                    raise ValueError("NPZ declared size exceeds safety limit")
                # Production uses ZIP_STORED. Permit ordinary compressed legacy
                # files, but reject extreme expansion before reading a header.
                if (
                    info.file_size > 1024 * 1024
                    and info.file_size > max(1, info.compress_size) * 200
                ):
                    raise ValueError("NPZ compression ratio exceeds safety limit")
                with archive.open(info) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(
                            member
                        )
                    elif version in {(2, 0), (3, 0)}:
                        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(
                            member
                        )
                    else:
                        raise ValueError("unsupported NPY format version")
                    dtype = np.dtype(dtype)
                    if (
                        dtype.hasobject
                        or len(shape) > 4
                        or any(
                            not isinstance(dim, int) or dim < 0 or dim > MAX_FRAME_INDEX
                            for dim in shape
                        )
                    ):
                        raise ValueError("unsafe NPY dtype or shape")
                    if dtype.kind in "US" and dtype.itemsize > MAX_STRING_ITEM_BYTES:
                        raise ValueError("NPY string item exceeds safety limit")
                    count = 1
                    for dim in shape:
                        count *= dim
                        if count * max(1, dtype.itemsize) > max_uncompressed_bytes:
                            raise ValueError("NPY declared array exceeds safety limit")
                    # A truncated member cannot honestly contain its declared
                    # header plus payload. ZIP metadata is checked before load.
                    if member.tell() + count * dtype.itemsize != info.file_size:
                        raise ValueError("NPY member size does not match its header")
                    metadata[info.filename[:-4]] = (tuple(shape), dtype)
    except (
        zipfile.BadZipFile,
        EOFError,
        NotImplementedError,
        OSError,
        OverflowError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise ValueError(f"unsafe NPZ archive: {path}") from exc
    return metadata


def _load_npz_bounded(path: Path, *, max_bytes: int) -> dict[str, np.ndarray]:
    _inspect_npz(path, max_uncompressed_bytes=max_bytes)
    with np.load(path, allow_pickle=False) as raw:
        return {name: raw[name] for name in raw.files}


def _json_no_duplicate_keys(encoded: str):
    def build(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    return json.loads(encoded, object_pairs_hook=build)


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


def _exclusive_npz_save(path: Path, **arrays: np.ndarray) -> None:
    """Create an immutable payload without any overwrite race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        try:
            path.unlink()
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
        generation_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.key = key
        self.kind = str(kind)
        self.require_key = bool(require_key)
        if generation_id is not None and not _safe_component(generation_id):
            raise ValueError("unsafe cache generation id")
        self._expected_generation_id = generation_id
        self._generation_id = generation_id or uuid.uuid4().hex
        self._stored_key = key.as_string()
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
    def generation_id(self) -> str:
        self._load_manifest()
        return self._generation_id

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
            archive_metadata = _inspect_npz(
                self.path, max_uncompressed_bytes=MAX_LEGACY_BYTES
            )
            raw = _load_npz_bounded(
                self.path,
                max_bytes=(
                    MAX_MANIFEST_BYTES
                    if _FORMAT_FIELD in archive_metadata
                    else MAX_LEGACY_BYTES
                ),
            )
            files = set(raw)
            if _FORMAT_FIELD not in files:
                self._legacy = True
                if (
                    raw.get("cache_key") is None
                    or raw["cache_key"].dtype.kind not in "US"
                ):
                    return
                stored_key = str(_scalar(raw, "cache_key"))
                self._stored_key = stored_key
                if self.require_key:
                    self._valid = stored_key == self.key.as_string()
                else:
                    self._valid = True
                return
            if files != {
                _FORMAT_FIELD,
                "cache_kind",
                "cache_key",
                "session_id",
                "generation_id",
                "chunks_json",
            }:
                return
            if raw[_FORMAT_FIELD].dtype.kind not in "iu":
                return
            if any(
                raw[name].dtype.kind not in "US"
                for name in (
                    "cache_kind",
                    "cache_key",
                    "session_id",
                    "generation_id",
                    "chunks_json",
                )
            ):
                return
            version = int(_scalar(raw, _FORMAT_FIELD))
            stored_kind = str(_scalar(raw, "cache_kind"))
            stored_key = str(_scalar(raw, "cache_key"))
            session_id = str(_scalar(raw, "session_id"))
            generation_id = str(_scalar(raw, "generation_id"))
            entries_raw = _json_no_duplicate_keys(str(_scalar(raw, "chunks_json")))
            if version != CHUNK_FORMAT_VERSION or stored_kind != self.kind:
                return
            if self.require_key and stored_key != self.key.as_string():
                return
            if (
                not _safe_component(generation_id)
                or (
                    self._expected_generation_id is not None
                    and generation_id != self._expected_generation_id
                )
                or not isinstance(entries_raw, list)
                or len(entries_raw) > MAX_CHUNK_FRAMES
            ):
                return
            entries = [ChunkEntry.from_dict(item) for item in entries_raw]
            if not _safe_component(session_id):
                return
            if len({entry.name for entry in entries}) != len(entries):
                return
            if any(
                not entry.name.startswith(f"chunk-{position:08d}-")
                for position, entry in enumerate(entries)
            ):
                return
            self._session_id = session_id
            self._generation_id = generation_id
            self._stored_key = stored_key
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
        while True:
            name = f"chunk-{sequence:08d}-{uuid.uuid4().hex}.npz"
            chunk_path = self.chunk_directory / name
            if not chunk_path.exists():
                break
        payload = dict(arrays)
        payload["written_frames"] = np.asarray(frames, dtype=np.int64)
        payload.update(
            {
                "_cache_kind": np.asarray([self.kind]),
                "_cache_key": np.asarray([self.key.as_string()]),
                "_session_id": np.asarray([self._session_id]),
                "_generation_id": np.asarray([self._generation_id]),
                "_chunk_position": np.asarray([sequence], dtype=np.int64),
            }
        )
        while True:
            try:
                _exclusive_npz_save(chunk_path, **payload)
                break
            except FileExistsError:
                name = f"chunk-{sequence:08d}-{uuid.uuid4().hex}.npz"
                chunk_path = self.chunk_directory / name
        self._validate_staged_chunk(chunk_path, frames, set(payload), sequence)
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

    def _validate_staged_chunk(
        self,
        chunk_path: Path,
        frames: tuple[int, ...],
        expected_fields: set[str],
        sequence: int,
    ) -> None:
        """Read back one bounded chunk before the manifest can reference it."""
        raw = _load_npz_bounded(chunk_path, max_bytes=MAX_CHUNK_BYTES)
        if set(raw) != expected_fields:
            raise ValueError(f"cache chunk fields are incomplete: {chunk_path}")
        stored = tuple(int(frame) for frame in raw["written_frames"])
        if stored != frames:
            raise ValueError(f"cache chunk frame index mismatch: {chunk_path}")
        self._validate_chunk_arrays(raw, sequence)

    @staticmethod
    def _require_array(
        arrays: dict[str, np.ndarray],
        name: str,
        *,
        rank: int,
        dtype_kinds: str,
        trailing: tuple[int, ...] = (),
    ) -> np.ndarray:
        value = arrays.get(name)
        if not isinstance(value, np.ndarray):
            raise ValueError(f"cache field {name!r} is missing")
        if value.ndim != rank or value.dtype.kind not in dtype_kinds:
            raise ValueError(f"cache field {name!r} has an invalid dtype or rank")
        if trailing and value.shape[-len(trailing) :] != trailing:
            raise ValueError(f"cache field {name!r} has an invalid shape")
        return value

    def _validate_chunk_arrays(
        self, arrays: dict[str, np.ndarray], sequence: int
    ) -> None:
        """Validate kind-specific schema and row alignment before reuse."""
        expected_payloads = {
            "detection": {
                "frame_indices",
                "centroids",
                "angles",
                "sizes",
                "shapes",
                "confidences",
                "corners",
                "detection_ids",
                "class_ids",
            },
            "headtail": {
                "frame_indices",
                "det_indices",
                "heading_hints",
                "heading_confidences",
                "directed_mask",
            },
            "cnn": {
                "frame_indices",
                "det_indices",
                "factor_names_json",
                "class_names_json",
                "class_counts",
                "probabilities",
            },
            "pose": {"frame_indices", "det_indices", "keypoints", "valid_mask"},
            "apriltag": {
                "frame_indices",
                "tag_ids",
                "det_indices",
                "centers",
                "corners",
            },
        }
        required = expected_payloads.get(self.kind)
        if required is None or set(arrays) != required | _CHUNK_META_FIELDS | {
            "written_frames"
        }:
            raise ValueError(f"invalid {self.kind} cache fields")
        written = self._require_array(
            arrays, "written_frames", rank=1, dtype_kinds="iu"
        )
        frame_indices = self._require_array(
            arrays, "frame_indices", rank=1, dtype_kinds="iu"
        )
        if len(written) > MAX_CHUNK_FRAMES or (
            len(written) > 1 and np.any(np.diff(written) <= 0)
        ):
            raise ValueError("written_frames must be bounded and strictly increasing")
        if len(frame_indices) > MAX_CHUNK_FRAMES * 100_000 or (
            len(frame_indices) > 1 and np.any(np.diff(frame_indices) < 0)
        ):
            raise ValueError("frame_indices must be bounded and ordered")
        if len(frame_indices) and not np.all(np.isin(frame_indices, written)):
            raise ValueError("row frames must belong to written_frames")

        for name, expected in (
            ("_cache_kind", self.kind),
            ("_cache_key", self._stored_key),
            ("_session_id", self._session_id),
            ("_generation_id", self._generation_id),
        ):
            value = self._require_array(arrays, name, rank=1, dtype_kinds="US")
            if value.size != 1 or str(value[0]) != expected:
                raise ValueError(f"cache chunk identity mismatch for {name}")
        position = self._require_array(
            arrays, "_chunk_position", rank=1, dtype_kinds="iu"
        )
        if position.size != 1 or int(position[0]) != sequence:
            raise ValueError("cache chunk position mismatch")

        rows = len(frame_indices)
        one_dimensional = {
            "angles",
            "sizes",
            "confidences",
            "detection_ids",
            "class_ids",
            "det_indices",
            "heading_hints",
            "heading_confidences",
            "directed_mask",
            "valid_mask",
            "tag_ids",
        }
        integer_fields = {
            "detection_ids",
            "class_ids",
            "det_indices",
            "directed_mask",
            "valid_mask",
            "tag_ids",
        }
        for name in required & one_dimensional:
            value = self._require_array(
                arrays,
                name,
                rank=1,
                dtype_kinds="iub" if name in integer_fields else "f",
            )
            if len(value) != rows:
                raise ValueError(f"cache field {name!r} is not row-aligned")

        shape_fields = {
            "centroids": (2,),
            "shapes": (2,),
            "corners": (4, 2),
            "centers": (2,),
        }
        for name, trailing in shape_fields.items():
            if name in required:
                value = self._require_array(
                    arrays,
                    name,
                    rank=len(trailing) + 1,
                    dtype_kinds="f",
                    trailing=trailing,
                )
                if len(value) != rows:
                    raise ValueError(f"cache field {name!r} is not row-aligned")

        if self.kind == "pose":
            keypoints = self._require_array(
                arrays, "keypoints", rank=3, dtype_kinds="f", trailing=(3,)
            )
            if len(keypoints) != rows:
                raise ValueError("pose keypoints are not row-aligned")
        elif self.kind == "cnn":
            probabilities = self._require_array(
                arrays, "probabilities", rank=3, dtype_kinds="f"
            )
            counts = self._require_array(
                arrays, "class_counts", rank=1, dtype_kinds="iu"
            )
            if len(probabilities) != rows or probabilities.shape[1] != len(counts):
                raise ValueError("CNN probabilities are not row/factor aligned")
            if np.any(counts < 0) or (
                len(counts) and int(np.max(counts)) > probabilities.shape[2]
            ):
                raise ValueError("CNN class counts exceed probability shape")
            decoded: list[object] = []
            for name in ("factor_names_json", "class_names_json"):
                encoded = self._require_array(arrays, name, rank=1, dtype_kinds="US")
                if encoded.size != 1 or len(str(encoded[0])) > MAX_STRING_ITEM_BYTES:
                    raise ValueError(f"invalid CNN metadata field {name!r}")
                decoded.append(json.loads(str(encoded[0])))
            factor_names, class_names = decoded
            if (
                not isinstance(factor_names, list)
                or not isinstance(class_names, list)
                or len(factor_names) != len(counts)
                or len(class_names) != len(counts)
                or any(not isinstance(value, str) for value in factor_names)
                or any(not isinstance(values, list) for values in class_names)
                or any(
                    len(values) != int(count)
                    for values, count in zip(class_names, counts)
                )
            ):
                raise ValueError("CNN JSON metadata is inconsistent")

    def _validate_legacy_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        """Apply the same structural contract to monolithic legacy caches."""
        if "cache_key" not in arrays:
            raise ValueError("legacy cache is missing cache_key")
        # Legacy layouts have no chunk identity. Validate through a synthetic
        # chunk while preserving detection's historically optional class_ids.
        payload = {name: value for name, value in arrays.items() if name != "cache_key"}
        if "written_frames" not in payload:
            payload["written_frames"] = np.unique(payload.get("frame_indices", []))
        if self.kind == "detection" and "class_ids" not in payload:
            payload["class_ids"] = np.zeros(
                len(payload.get("frame_indices", [])), np.int64
            )
        payload.update(
            {
                "_cache_kind": np.asarray([self.kind]),
                "_cache_key": np.asarray([self.key.as_string()]),
                "_session_id": np.asarray([self._session_id]),
                "_generation_id": np.asarray([self._generation_id]),
                "_chunk_position": np.asarray([0], np.int64),
            }
        )
        self._validate_chunk_arrays(payload, 0)

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
        self._session_id = self._session_prefix() + "-" + uuid.uuid4().hex
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
        self._session_id = f"{prefix}-{uuid.uuid4().hex}"
        if self._expected_generation_id is None:
            self._generation_id = uuid.uuid4().hex
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
            generation_id=np.asarray([self._generation_id]),
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
            arrays = _load_npz_bounded(chunk_path, max_bytes=MAX_CHUNK_BYTES)
            if not _frames_match_ranges(arrays["written_frames"], entry.ranges):
                self._valid = False
                return None
            sequence = self._entries.index(entry)
            self._validate_chunk_arrays(arrays, sequence)
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
            } | _CHUNK_META_FIELDS
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
            arrays = _load_npz_bounded(self.path, max_bytes=MAX_LEGACY_BYTES)
            self._validate_legacy_arrays(arrays)
            return arrays
        except (OSError, ValueError):
            self._valid = False
            return None
