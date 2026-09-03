"""Bounded, crash-resumable inference cache handles.

New writes use immutable NPZ payload chunks and an atomic manifest. Existing
monolithic NPZ caches remain readable through the same public handle classes.
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNFactorPrediction,
    OBBResult,
)
from .base import CacheKey
from .chunked import DEFAULT_CHUNK_FRAMES, ChunkedArrayStore, _atomic_npz_save


class CacheHandle(ABC):
    @abstractmethod
    def is_valid(self) -> bool:
        """Return True if the cache matches the expected key and is intact."""

    @abstractmethod
    def write_frame(self, frame_idx: int, **kwargs) -> None:
        """Buffer one frame, flushing automatically at the chunk boundary."""

    @abstractmethod
    def read_frame(self, frame_idx: int) -> Any:
        """Return a cached result, or None for invalid/missing frames."""

    @abstractmethod
    def close(self) -> None:
        """Publish the final bounded chunk."""

    @abstractmethod
    def written_frames(self) -> set[int]:
        """Return explicitly processed frame IDs."""

    @abstractmethod
    def coverage_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return normalized processed-frame intervals."""


def _check_key(path: Path, key: CacheKey) -> bool:
    """Legacy helper retained for callers/tests that inspect monolithic NPZ."""
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["cache_key"][0]) == key.as_string()
    except Exception:
        return False


def _npz_save(path: Path, key: CacheKey, **arrays) -> None:
    """Write a legacy monolithic NPZ fixture atomically.

    Production handles no longer call this function. It remains available for
    compatibility tests and third-party code that deliberately creates legacy
    caches.
    """
    payload = {"cache_key": np.asarray([key.as_string()]), **arrays}
    _atomic_npz_save(Path(path), **payload)


def _rows_for_frame(arrays: dict[str, np.ndarray], frame_idx: int) -> np.ndarray:
    return np.asarray(arrays["frame_indices"]) == int(frame_idx)


def _concat(parts: list[np.ndarray], shape: tuple[int, ...], dtype) -> np.ndarray:
    nonempty = [part for part in parts if len(part)]
    return np.concatenate(nonempty) if nonempty else np.zeros(shape, dtype=dtype)


def _buffer_value_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Estimate bytes retained by a handle without double-counting arrays."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        return int(value.nbytes) + sys.getsizeof(value)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(
            _buffer_value_bytes(key, seen) + _buffer_value_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(
            _buffer_value_bytes(item, seen) for item in value
        )
    fields = getattr(value, "__dict__", None)
    if fields is not None:
        return sys.getsizeof(value) + _buffer_value_bytes(fields, seen)
    return sys.getsizeof(value)


class _ChunkedHandleMixin:
    path: Path
    key: CacheKey
    require_key: bool
    chunk_size: int
    _store: ChunkedArrayStore
    _legacy_data: dict[str, np.ndarray] | None
    read_only: bool
    write_mode: str
    max_buffer_bytes: int

    _kind = ""

    def _init_store(self) -> None:
        if int(self.chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1")
        self.path = Path(self.path)
        self._store = ChunkedArrayStore(
            self.path,
            self.key,
            self._kind,
            require_key=self.require_key,
        )
        self._write_started = False
        self._buffer_bytes = 0
        self._last_buffered_frame: int | None = None
        if self.write_mode not in {"auto", "fresh", "resume"}:
            raise ValueError("write_mode must be 'auto', 'fresh', or 'resume'")
        if int(self.max_buffer_bytes) < 1:
            raise ValueError("max_buffer_bytes must be >= 1")

    @property
    def buffered_bytes(self) -> int:
        return self._buffer_bytes

    def set_buffer_limit(self, value: int) -> None:
        if int(value) < 1:
            raise ValueError("cache handle buffer limit must be >= 1")
        self.max_buffer_bytes = int(value)
        if self._buffer_bytes > self.max_buffer_bytes:
            self._flush()

    def _append_buffered(self, frame_idx: int, value: Any) -> None:
        frame = int(frame_idx)
        if frame < 0:
            raise ValueError("frame_idx must be nonnegative")
        if self._last_buffered_frame is not None and frame <= self._last_buffered_frame:
            raise ValueError("cache frame indices must be unique and increasing")
        value_bytes = _buffer_value_bytes(value)
        if value_bytes > self.max_buffer_bytes:
            raise ValueError(
                "cache frame payload exceeds max_buffer_bytes: "
                f"{value_bytes} > {self.max_buffer_bytes}"
            )
        if self._buffer and self._buffer_bytes + value_bytes > self.max_buffer_bytes:
            self._flush()
        self._buffer.append(value)
        self._buffer_bytes += value_bytes
        self._last_buffered_frame = frame
        if (
            len(self._buffer) >= self.chunk_size
            or self._buffer_bytes >= self.max_buffer_bytes
        ):
            self._flush()

    def _clear_buffer(self) -> None:
        self._buffer.clear()
        self._buffer_bytes = 0

    def _finish_close(self, *, commit_generation: bool = True) -> None:
        if self.read_only:
            return
        self._flush()
        if not self.path.exists():
            self._store.ensure_manifest()
        if commit_generation:
            self._store.commit_generation()

    def _prepare_frame_write(self, frame_idx: int) -> None:
        if self.read_only:
            raise RuntimeError("cannot write through a read-only cache handle")
        if self._write_started:
            return
        if self.write_mode == "fresh" or (
            self.write_mode == "auto"
            and self._store.is_valid()
            and (self._store.is_legacy or self._store.contains_frame(int(frame_idx)))
        ):
            # A pass beginning on an already-covered frame is a deliberate
            # recomputation, not a resume. Build a new generation so a crash
            # cannot make the old manifest point at overwritten chunks.
            self._store.start_fresh()
        self._write_started = True

    def is_valid(self) -> bool:
        return self._store.is_valid()

    def is_reusable(self) -> bool:
        return self._store.is_reusable()

    def contains_frame(self, frame_idx: int) -> bool:
        if not self.is_valid():
            return False
        if self._store.is_legacy:
            if self._legacy_data is None:
                self._legacy_data = self._store.load_legacy()
            return int(frame_idx) in self._legacy_written_frames(
                self._legacy_data or {}
            )
        return self._store.contains_frame(int(frame_idx))

    def iter_covered_frames(self, start_frame: int, end_frame: int) -> Iterator[int]:
        start_bound, end_bound = int(start_frame), int(end_frame)
        for start, end in self.coverage_ranges():
            lo, hi = max(start, start_bound), min(end, end_bound)
            if lo <= hi:
                yield from range(lo, hi + 1)

    @property
    def is_legacy(self) -> bool:
        return self._store.is_legacy

    def _arrays_for_frame(self, frame_idx: int) -> dict[str, np.ndarray] | None:
        if not self.is_valid():
            return None
        if self._store.is_legacy:
            if self._legacy_data is None:
                self._legacy_data = self._store.load_legacy()
            if self._legacy_data is None:
                return None
            written = self._legacy_written_frames(self._legacy_data)
            return self._legacy_data if int(frame_idx) in written else None
        return self._store.read_frame_arrays(int(frame_idx))

    def _legacy_written_frames(self, data: dict[str, np.ndarray]) -> set[int]:
        if "written_frames" in data:
            return {int(v) for v in data["written_frames"]}
        return {int(v) for v in data.get("frame_indices", np.zeros(0, np.int32))}

    def written_frames(self) -> set[int]:
        if not self.is_valid():
            return set()
        if self._store.is_legacy:
            if self._legacy_data is None:
                self._legacy_data = self._store.load_legacy()
            return self._legacy_written_frames(self._legacy_data or {})
        return self._store.written_frames()

    def coverage_ranges(self) -> tuple[tuple[int, int], ...]:
        """Normalized coverage intervals without an O(frame-count) set."""
        if not self.is_valid():
            return ()
        if self._store.is_legacy:
            frames = sorted(self.written_frames())
            if not frames:
                return ()
            ranges: list[tuple[int, int]] = []
            start = previous = frames[0]
            for frame in frames[1:]:
                if frame == previous + 1:
                    previous = frame
                    continue
                ranges.append((start, previous))
                start = previous = frame
            ranges.append((start, previous))
            return tuple(ranges)
        return self._store.covered_ranges()

    def covers_frame_range(self, start_frame: int, end_frame: int) -> bool:
        start = int(start_frame)
        end = int(end_frame)
        return self.is_valid() and any(
            range_start <= start and range_end >= end
            for range_start, range_end in self.coverage_ranges()
        )

    def get_missing_frames(
        self, start_frame: int, end_frame: int, max_report: int = 10
    ) -> list[int]:
        ranges = self.coverage_ranges()
        missing: list[int] = []
        for frame in range(int(start_frame), int(end_frame) + 1):
            if not any(start <= frame <= end for start, end in ranges):
                missing.append(frame)
                if len(missing) >= max_report:
                    break
        return missing

    def iter_arrays(self) -> Iterator[dict[str, np.ndarray]]:
        """Yield legacy data once or new payload chunks one at a time."""
        if not self.is_valid():
            return
        if self._store.is_legacy:
            data = self._store.load_legacy()
            if data is not None:
                yield data
            return
        yield from self._store.iter_chunk_arrays()


@dataclass
class DetectionCacheHandle(_ChunkedHandleMixin, CacheHandle):
    path: Path
    key: CacheKey
    require_key: bool = True
    read_only: bool = False
    write_mode: str = "auto"
    chunk_size: int = DEFAULT_CHUNK_FRAMES
    max_buffer_bytes: int = 16 * 1024 * 1024
    _buffer: list[OBBResult] = field(default_factory=list, repr=False)
    _legacy_data: dict[str, np.ndarray] | None = field(default=None, repr=False)
    _store: ChunkedArrayStore = field(init=False, repr=False)

    _kind = "detection"

    def __post_init__(self) -> None:
        self._init_store()

    @property
    def _data(self) -> dict[str, np.ndarray] | None:
        """Compatibility alias for legacy-only callers."""
        return self._legacy_data

    @_data.setter
    def _data(self, value: dict[str, np.ndarray] | None) -> None:
        self._legacy_data = value

    def write_frame(self, frame_idx: int, *, result: OBBResult, **_) -> None:
        if int(frame_idx) != int(result.frame_idx):
            raise ValueError("result.frame_idx must match frame_idx")
        row_lengths = [
            len(result.centroids),
            len(result.angles),
            len(result.sizes),
            len(result.shapes),
            len(result.confidences),
            len(result.corners),
            len(result.detection_ids),
            len(result.class_ids_or_zeros),
        ]
        if len(set(row_lengths)) != 1:
            raise ValueError("detection result arrays must have aligned lengths")
        self._prepare_frame_write(frame_idx)
        self._append_buffered(frame_idx, result)

    def _flush(self) -> None:
        if not self._buffer:
            return
        rows = self._buffer
        frame_indices: list[int] = []
        arrays: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "centroids",
                "angles",
                "sizes",
                "shapes",
                "confidences",
                "corners",
                "detection_ids",
                "class_ids",
            )
        }
        for result in rows:
            count = result.num_detections
            frame_indices.extend([result.frame_idx] * count)
            if count:
                arrays["centroids"].append(result.centroids)
                arrays["angles"].append(result.angles)
                arrays["sizes"].append(result.sizes)
                arrays["shapes"].append(result.shapes)
                arrays["confidences"].append(result.confidences)
                arrays["corners"].append(result.corners)
                arrays["detection_ids"].append(result.detection_ids)
                arrays["class_ids"].append(result.class_ids_or_zeros)
        empty_shapes = {
            "centroids": (0, 2),
            "angles": (0,),
            "sizes": (0,),
            "shapes": (0, 2),
            "confidences": (0,),
            "corners": (0, 4, 2),
            "detection_ids": (0,),
            "class_ids": (0,),
        }
        dtypes = {
            "centroids": np.float32,
            "angles": np.float32,
            "sizes": np.float32,
            "shapes": np.float32,
            "confidences": np.float32,
            "corners": np.float32,
            "detection_ids": np.int64,
            "class_ids": np.int64,
        }
        payload = {
            name: (
                np.concatenate(parts)
                if parts
                else np.zeros(empty_shapes[name], dtype=dtypes[name])
            )
            for name, parts in arrays.items()
        }
        payload["frame_indices"] = np.asarray(frame_indices, dtype=np.int64)
        self._store.append_chunk([result.frame_idx for result in rows], payload)
        self._clear_buffer()

    def read_frame(self, frame_idx: int) -> OBBResult | None:
        arrays = self._arrays_for_frame(frame_idx)
        if arrays is None:
            return None
        mask = _rows_for_frame(arrays, frame_idx)
        class_ids = arrays["class_ids"][mask] if "class_ids" in arrays else None
        return OBBResult(
            frame_idx=int(frame_idx),
            centroids=arrays["centroids"][mask],
            angles=arrays["angles"][mask],
            sizes=arrays["sizes"][mask],
            shapes=arrays["shapes"][mask],
            confidences=arrays["confidences"][mask],
            corners=arrays["corners"][mask],
            detection_ids=arrays["detection_ids"][mask],
            class_ids=class_ids,
        )

    def close(self, *, commit_generation: bool = True) -> None:
        self._finish_close(commit_generation=commit_generation)


@dataclass
class HeadTailCacheHandle(_ChunkedHandleMixin, CacheHandle):
    path: Path
    key: CacheKey
    require_key: bool = True
    read_only: bool = False
    write_mode: str = "auto"
    chunk_size: int = DEFAULT_CHUNK_FRAMES
    max_buffer_bytes: int = 16 * 1024 * 1024
    _buffer: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=list, repr=False
    )
    _legacy_data: dict[str, np.ndarray] | None = field(default=None, repr=False)
    _store: ChunkedArrayStore = field(init=False, repr=False)

    _kind = "headtail"

    def __post_init__(self) -> None:
        self._init_store()

    def write_frame(
        self,
        frame_idx: int,
        *,
        det_indices: np.ndarray,
        heading_hints: np.ndarray,
        heading_confidences: np.ndarray,
        directed_mask: np.ndarray,
        **_,
    ) -> None:
        arrays = (
            int(frame_idx),
            np.asarray(det_indices, dtype=np.int32),
            np.asarray(heading_hints, dtype=np.float32),
            np.asarray(heading_confidences, dtype=np.float32),
            np.asarray(directed_mask, dtype=np.uint8),
        )
        lengths = [len(value) for value in arrays[1:]]
        if len(set(lengths)) != 1:
            raise ValueError("headtail arrays must have aligned lengths")
        self._prepare_frame_write(frame_idx)
        self._append_buffered(frame_idx, arrays)

    def _flush(self) -> None:
        if not self._buffer:
            return
        frames = [row[0] for row in self._buffer]
        counts = [len(row[1]) for row in self._buffer]
        payload = {
            "frame_indices": np.repeat(frames, counts).astype(np.int64),
            "det_indices": _concat([row[1] for row in self._buffer], (0,), np.int32),
            "heading_hints": _concat(
                [row[2] for row in self._buffer], (0,), np.float32
            ),
            "heading_confidences": _concat(
                [row[3] for row in self._buffer], (0,), np.float32
            ),
            "directed_mask": _concat([row[4] for row in self._buffer], (0,), np.uint8),
        }
        self._store.append_chunk(frames, payload)
        self._clear_buffer()

    def read_frame(self, frame_idx: int):
        arrays = self._arrays_for_frame(frame_idx)
        if arrays is None:
            return None
        mask = _rows_for_frame(arrays, frame_idx)
        return (
            arrays["det_indices"][mask].astype(np.int32),
            arrays["heading_hints"][mask],
            arrays["heading_confidences"][mask],
            arrays["directed_mask"][mask],
        )

    def close(self, *, commit_generation: bool = True) -> None:
        self._finish_close(commit_generation=commit_generation)


@dataclass
class CNNCacheHandle(_ChunkedHandleMixin, CacheHandle):
    path: Path
    key: CacheKey
    label: str
    require_key: bool = True
    read_only: bool = False
    write_mode: str = "auto"
    chunk_size: int = DEFAULT_CHUNK_FRAMES
    max_buffer_bytes: int = 16 * 1024 * 1024
    _buffer: list[tuple[int, list[CNNDetectionPrediction]]] = field(
        default_factory=list, repr=False
    )
    _legacy_data: dict[str, np.ndarray] | None = field(default=None, repr=False)
    _store: ChunkedArrayStore = field(init=False, repr=False)

    _kind = "cnn"

    def __post_init__(self) -> None:
        self._init_store()

    def write_frame(
        self, frame_idx: int, *, predictions: list[CNNDetectionPrediction], **_
    ) -> None:
        self._prepare_frame_write(frame_idx)
        self._append_buffered(frame_idx, (int(frame_idx), predictions))

    def _flush(self) -> None:
        if not self._buffer:
            return
        frames = [row[0] for row in self._buffer]
        all_predictions = [
            pred for _, predictions in self._buffer for pred in predictions
        ]
        factor_names: list[str] = []
        class_names: list[list[str]] = []
        for pred in all_predictions:
            if pred.factors:
                factor_names = [factor.factor_name for factor in pred.factors]
                class_names = [factor.class_names for factor in pred.factors]
                break
        class_counts = np.asarray([len(names) for names in class_names], dtype=np.int32)
        class_max = int(class_counts.max()) if class_counts.size else 0
        probabilities = np.full(
            (len(all_predictions), len(factor_names), class_max),
            np.nan,
            dtype=np.float32,
        )
        for row_idx, pred in enumerate(all_predictions):
            for factor_idx, factor in enumerate(pred.factors[: len(factor_names)]):
                count = min(len(factor.raw_probabilities), class_max)
                probabilities[row_idx, factor_idx, :count] = factor.raw_probabilities[
                    :count
                ]
        row_frames = [
            frame
            for frame, predictions in self._buffer
            for _ in range(len(predictions))
        ]
        payload = {
            "frame_indices": np.asarray(row_frames, dtype=np.int64),
            "det_indices": np.asarray(
                [pred.det_index for pred in all_predictions], dtype=np.int32
            ),
            "factor_names_json": np.asarray([json.dumps(factor_names)]),
            "class_names_json": np.asarray([json.dumps(class_names)]),
            "class_counts": class_counts,
            "probabilities": probabilities,
        }
        self._store.append_chunk(frames, payload)
        self._clear_buffer()

    def read_frame(self, frame_idx: int) -> list[CNNDetectionPrediction] | None:
        arrays = self._arrays_for_frame(frame_idx)
        if arrays is None:
            return None
        mask = _rows_for_frame(arrays, frame_idx)
        if not mask.any():
            return []
        factor_names = json.loads(str(arrays["factor_names_json"][0]))
        class_names = json.loads(str(arrays["class_names_json"][0]))
        class_counts = arrays["class_counts"].astype(int)
        det_indices = arrays["det_indices"][mask]
        probabilities = arrays["probabilities"][mask]
        return [
            CNNDetectionPrediction(
                det_index=int(det_idx),
                factors=[
                    CNNFactorPrediction(
                        factor_name=factor_names[factor_idx],
                        class_names=class_names[factor_idx],
                        raw_probabilities=probabilities[
                            row_idx, factor_idx, : class_counts[factor_idx]
                        ].copy(),
                    )
                    for factor_idx in range(len(factor_names))
                ],
            )
            for row_idx, det_idx in enumerate(det_indices)
        ]

    def close(self, *, commit_generation: bool = True) -> None:
        self._finish_close(commit_generation=commit_generation)


@dataclass
class PoseCacheHandle(_ChunkedHandleMixin, CacheHandle):
    path: Path
    key: CacheKey
    require_key: bool = True
    read_only: bool = False
    write_mode: str = "auto"
    chunk_size: int = DEFAULT_CHUNK_FRAMES
    max_buffer_bytes: int = 16 * 1024 * 1024
    _buffer: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=list, repr=False
    )
    _legacy_data: dict[str, np.ndarray] | None = field(default=None, repr=False)
    _store: ChunkedArrayStore = field(init=False, repr=False)

    _kind = "pose"

    def __post_init__(self) -> None:
        self._init_store()

    def write_frame(
        self,
        frame_idx: int,
        *,
        det_indices: np.ndarray,
        keypoints: np.ndarray,
        valid_mask: np.ndarray,
        **_,
    ) -> None:
        arrays = (
            int(frame_idx),
            np.asarray(det_indices, dtype=np.int32),
            np.asarray(keypoints, dtype=np.float32),
            np.asarray(valid_mask, dtype=np.uint8),
        )
        if len(arrays[1]) != len(arrays[2]) or len(arrays[1]) != len(arrays[3]):
            raise ValueError("pose arrays must have aligned lengths")
        self._prepare_frame_write(frame_idx)
        self._append_buffered(frame_idx, arrays)

    def _flush(self) -> None:
        if not self._buffer:
            return
        frames = [row[0] for row in self._buffer]
        counts = [len(row[1]) for row in self._buffer]
        keypoint_shape = next(
            (row[2].shape[1:] for row in self._buffer if row[2].ndim == 3), (0, 3)
        )
        payload = {
            "frame_indices": np.repeat(frames, counts).astype(np.int64),
            "det_indices": _concat([row[1] for row in self._buffer], (0,), np.int32),
            "keypoints": _concat(
                [row[2] for row in self._buffer], (0, *keypoint_shape), np.float32
            ),
            "valid_mask": _concat([row[3] for row in self._buffer], (0,), np.uint8),
        }
        self._store.append_chunk(frames, payload)
        self._clear_buffer()

    def read_frame(self, frame_idx: int):
        arrays = self._arrays_for_frame(frame_idx)
        if arrays is None:
            return None
        mask = _rows_for_frame(arrays, frame_idx)
        return (
            arrays["keypoints"][mask],
            arrays["det_indices"][mask],
            arrays["valid_mask"][mask].astype(bool),
        )

    def close(self, *, commit_generation: bool = True) -> None:
        self._finish_close(commit_generation=commit_generation)


@dataclass
class AprilTagCacheHandle(_ChunkedHandleMixin, CacheHandle):
    path: Path
    key: CacheKey
    require_key: bool = True
    read_only: bool = False
    write_mode: str = "auto"
    chunk_size: int = DEFAULT_CHUNK_FRAMES
    max_buffer_bytes: int = 16 * 1024 * 1024
    _buffer: list[tuple[int, AprilTagResult]] = field(default_factory=list, repr=False)
    _legacy_data: dict[str, np.ndarray] | None = field(default=None, repr=False)
    _store: ChunkedArrayStore = field(init=False, repr=False)

    _kind = "apriltag"

    def __post_init__(self) -> None:
        self._init_store()

    def write_frame(self, frame_idx: int, *, result: AprilTagResult, **_) -> None:
        lengths = [
            len(result.tag_ids),
            len(result.det_indices),
            len(result.centers),
            len(result.corners),
        ]
        if len(set(lengths)) != 1:
            raise ValueError("apriltag arrays must have aligned lengths")
        self._prepare_frame_write(frame_idx)
        self._append_buffered(frame_idx, (int(frame_idx), result))

    def _flush(self) -> None:
        if not self._buffer:
            return
        frames = [row[0] for row in self._buffer]
        tag_ids = [np.asarray(row[1].tag_ids, dtype=np.int32) for row in self._buffer]
        counts = [len(values) for values in tag_ids]
        payload = {
            "frame_indices": np.repeat(frames, counts).astype(np.int64),
            "tag_ids": _concat(tag_ids, (0,), np.int32),
            "det_indices": _concat(
                [
                    np.asarray(row[1].det_indices, dtype=np.int32)
                    for row in self._buffer
                ],
                (0,),
                np.int32,
            ),
            "centers": _concat(
                [np.asarray(row[1].centers, dtype=np.float32) for row in self._buffer],
                (0, 2),
                np.float32,
            ),
            "corners": _concat(
                [np.asarray(row[1].corners, dtype=np.float32) for row in self._buffer],
                (0, 4, 2),
                np.float32,
            ),
        }
        self._store.append_chunk(frames, payload)
        self._clear_buffer()

    def read_frame(self, frame_idx: int) -> AprilTagResult | None:
        arrays = self._arrays_for_frame(frame_idx)
        if arrays is None:
            return None
        mask = _rows_for_frame(arrays, frame_idx)
        return AprilTagResult(
            tag_ids=arrays["tag_ids"][mask],
            det_indices=arrays["det_indices"][mask],
            centers=arrays["centers"][mask],
            corners=arrays["corners"][mask],
        )

    def close(self, *, commit_generation: bool = True) -> None:
        self._finish_close(commit_generation=commit_generation)
