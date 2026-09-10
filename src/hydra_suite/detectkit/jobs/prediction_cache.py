"""Bounded, indexed dataset-prediction storage for the DetectKit GUI."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION, CacheKey
from hydra_suite.core.inference.cache.chunked import (
    DEFAULT_CHUNK_FRAMES,
    ChunkedArrayStore,
)
from hydra_suite.core.inference.content_id import (
    directory_content_id,
    model_content_id,
    video_signature,
)

MAX_FRAME_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_DETECTIONS_PER_FRAME = 1_000
MAX_PREDICTION_CLASSES = 4_096
MAX_PATH_BYTES = 4096
MAX_PATH_INDEX_BYTES = 16 * 1024 * 1024
DEFAULT_LRU_FRAMES = 8


def _source_content_id(source_path: str) -> str:
    """DetectKit prediction-cache sources are DATASET DIRECTORIES
    (dataset_panel.py:394 does Path(source_path)/"images"), not bare video
    files. video_signature() on a directory silently degrades to "" via
    its except OSError, collapsing every source onto one identity. Dispatch
    explicitly instead of guessing from the exception.

    Fix A4 -- hash ONLY the images/ subtree LISTING, not directory_content_id
    of the dataset root. `directory_content_id(source_path)` would hash
    every byte under source_path, including `labels/`
    (operations.py:87 does `images_dir = source_path / "images"`, and
    `labels/` sits as its sibling under the same root, edited by the
    reviewer on every save via dataset_panel.py:394). Two things follow
    from hashing the root wholesale, both real regressions, neither
    exercised by any test in this plan:
      1. `cache_path_for` derives the on-disk cache filename from
         `key.as_string()` (`prediction_cache.py:63-66`), so EVERY label
         save changes the key and orphans the previous prediction cache
         file -- `artifacts/inference_cache/` grows without bound as the
         reviewer works, one dead file per save.
      2. `directory_content_id` reads and sha256's every byte under the
         tree on every request (`dataset_inference.py:64`), which is
         O(dataset bytes) -- and DetectKit datasets are image sets, often
         far larger than a model checkpoint.
    This cache is project-local (it lives beside the dataset, keyed by an
    absolute-ish path already, never shipped by a portable job -- Task 6's
    model-reference machinery does not touch DetectKit at all) and was
    never claimed to be portable, so there is no reason to pay content
    hashing here: identify the SOURCE by what actually invalidates
    predictions -- the image SET, not every byte of every image and
    certainly not the label annotations the reviewer is actively editing.
    List (name, size, mtime_ns) for every file directly under
    `images/` (non-recursive: dataset_panel.py:394 treats images/ as a
    flat pool) and hash that listing; do not recurse into labels/ at all.
    """
    p = Path(source_path)
    if p.is_dir():
        images_dir = p / "images"
        if not images_dir.is_dir():
            return directory_content_id(str(p))
        entries = sorted(
            (child.name, child.stat().st_size, child.stat().st_mtime_ns)
            for child in images_dir.iterdir()
            if child.is_file()
        )
        blob = "\n".join(f"{n}={s}:{m}" for n, s, m in entries).encode("utf-8")
        return f"imgset:{hashlib.sha256(blob).hexdigest()}"
    return video_signature(source_path)


def prediction_cache_key(
    source_path: str | Path,
    model_paths: Sequence[str | Path],
    settings: object,
) -> CacheKey:
    """Build a content-based source/model/settings cache identity."""
    # Fix Z5: expanduser() BEFORE hashing, same as the old implementation did
    # for both model_paths and source_path (prediction_cache.py:37,47) — a
    # raw "~/models/x.pt" path fails Path.is_file()/is_dir() and would
    # otherwise fall through to the missing-model sentinel.
    identities = [
        (model_content_id(str(Path(path).expanduser())), path)
        for path in model_paths
        if path
    ]
    model_id = "|".join(identity for identity, _ in identities)
    encoded = json.dumps(
        {
            # The SOURCE (image/video file OR dataset directory) is identified
            # by content too, so a prediction cache survives the project
            # moving on disk. See _source_content_id above for the dispatch.
            "source": _source_content_id(str(Path(source_path).expanduser())),
            "settings": settings,
        },
        sort_keys=True,
    ).encode("utf-8")
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_id=model_id,
        config_hash=hashlib.sha256(encoded).hexdigest(),
    )


def cache_path_for(project_dir: str | Path, key: CacheKey) -> Path:
    root = Path(project_dir).expanduser().resolve() / "artifacts" / "inference_cache"
    digest = hashlib.sha256(key.as_string().encode("utf-8")).hexdigest()[:24]
    return root / f"detectkit-{digest}.npz"


def remove_prediction_cache(path: str | Path) -> None:
    """Remove one exact prediction cache generation and its bounded indexes."""
    path = Path(path)
    path.unlink(missing_ok=True)
    shutil.rmtree(path.parent / f"{path.name}.chunks", ignore_errors=True)
    path.with_suffix(path.suffix + ".paths").unlink(missing_ok=True)
    path.with_suffix(path.suffix + ".paths.idx").unlink(missing_ok=True)


class DatasetPredictionWriter:
    """Publish prediction frames incrementally through Set 5 chunks."""

    def __init__(
        self,
        path: str | Path,
        key: CacheKey,
        *,
        chunk_size: int = DEFAULT_CHUNK_FRAMES,
        max_buffer_bytes: int = MAX_FRAME_PAYLOAD_BYTES,
        write_mode: str = "fresh",
    ) -> None:
        if chunk_size < 1 or max_buffer_bytes < 1:
            raise ValueError("prediction cache bounds must be positive")
        if write_mode not in {"fresh", "resume"}:
            raise ValueError("write_mode must be fresh or resume")
        self.path = Path(path)
        self.key = key
        self.chunk_size = int(chunk_size)
        self.max_buffer_bytes = int(max_buffer_bytes)
        self._store = ChunkedArrayStore(self.path, key, "detectkit_prediction")
        if write_mode == "fresh":
            self._store.start_fresh()
        self._buffer: list[tuple[int, list[dict]]] = []
        self._buffer_bytes = 0
        self._last_frame = -1

    @staticmethod
    def _normalize(detections: Iterable[dict]) -> tuple[list[dict], int]:
        normalized: list[dict] = []
        byte_count = 0
        for raw in detections:
            if len(normalized) >= MAX_DETECTIONS_PER_FRAME:
                raise ValueError("prediction count exceeds its per-frame cap")
            points = np.asarray(raw.get("polygon_px") or [], dtype=np.float32).reshape(
                -1, 2
            )
            if len(points) < 3:
                continue
            confidence = float(raw.get("confidence", 0.0))
            if not np.isfinite(points).all() or not np.isfinite(confidence):
                raise ValueError("prediction frame contains non-finite values")
            class_id = int(raw.get("class_id", 0))
            if class_id < 0 or class_id >= MAX_PREDICTION_CLASSES:
                raise ValueError("prediction class id exceeds its bounded range")
            normalized.append(
                {
                    "class_id": class_id,
                    "confidence": confidence,
                    "polygon_px": points,
                }
            )
            byte_count += int(points.nbytes) + 32
            if byte_count > MAX_FRAME_PAYLOAD_BYTES:
                raise ValueError("prediction frame exceeds its bounded payload cap")
        return normalized, byte_count

    def write_frame(self, frame_idx: int, detections: Iterable[dict]) -> None:
        frame_idx = int(frame_idx)
        if frame_idx <= self._last_frame:
            raise ValueError("prediction frame indices must be increasing and unique")
        normalized, size = self._normalize(detections)
        if size > self.max_buffer_bytes:
            raise ValueError("prediction frame exceeds configured buffer bound")
        if self._buffer and self._buffer_bytes + size > self.max_buffer_bytes:
            self.flush()
        self._buffer.append((frame_idx, normalized))
        self._buffer_bytes += size
        self._last_frame = frame_idx
        if len(self._buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        frames: list[int] = []
        frame_indices: list[int] = []
        class_ids: list[int] = []
        confidences: list[float] = []
        offsets = [0]
        point_parts: list[np.ndarray] = []
        for frame_idx, detections in self._buffer:
            frames.append(frame_idx)
            for detection in detections:
                points = detection["polygon_px"]
                frame_indices.append(frame_idx)
                class_ids.append(detection["class_id"])
                confidences.append(detection["confidence"])
                point_parts.append(points)
                offsets.append(offsets[-1] + len(points))
        points = (
            np.concatenate(point_parts).astype(np.float32, copy=False)
            if point_parts
            else np.zeros((0, 2), dtype=np.float32)
        )
        self._store.append_chunk(
            frames,
            {
                "frame_indices": np.asarray(frame_indices, dtype=np.int64),
                "class_ids": np.asarray(class_ids, dtype=np.int64),
                "confidences": np.asarray(confidences, dtype=np.float32),
                "polygon_offsets": np.asarray(offsets, dtype=np.int64),
                "polygon_points": points,
            },
        )
        self._buffer.clear()
        self._buffer_bytes = 0

    def close(self) -> None:
        self.flush()
        if not self.path.exists():
            self._store.ensure_manifest()
        self._store.commit_generation()


class DatasetPredictionCache:
    """Indexed reader retaining only a fixed number of decoded frames."""

    def __init__(
        self,
        path: str | Path,
        key: CacheKey,
        *,
        lru_frames: int = DEFAULT_LRU_FRAMES,
    ) -> None:
        if lru_frames < 1:
            raise ValueError("prediction LRU size must be positive")
        self.path = Path(path)
        self.key = key
        self._store = ChunkedArrayStore(
            self.path, key, "detectkit_prediction", require_key=True
        )
        self._lru_frames = int(lru_frames)
        self._lru: OrderedDict[int, list[dict]] = OrderedDict()

    @property
    def retained_frame_count(self) -> int:
        return len(self._lru)

    def is_valid(self) -> bool:
        return self._store.is_valid()

    def coverage_ranges(self) -> tuple[tuple[int, int], ...]:
        return self._store.covered_ranges()

    def read_frame(self, frame_idx: int) -> list[dict] | None:
        frame_idx = int(frame_idx)
        cached = self._lru.pop(frame_idx, None)
        if cached is not None:
            self._lru[frame_idx] = cached
            return cached
        arrays = self._store.read_frame_arrays(frame_idx)
        if arrays is None:
            return None
        rows = np.nonzero(np.asarray(arrays["frame_indices"]) == frame_idx)[0]
        offsets = np.asarray(arrays["polygon_offsets"], dtype=np.int64)
        points = np.asarray(arrays["polygon_points"], dtype=np.float32)
        detections = []
        for row in rows:
            start, end = int(offsets[row]), int(offsets[row + 1])
            detections.append(
                {
                    "class_id": int(arrays["class_ids"][row]),
                    "confidence": float(arrays["confidences"][row]),
                    "polygon_px": [(float(x), float(y)) for x, y in points[start:end]],
                }
            )
        self._lru[frame_idx] = detections
        while len(self._lru) > self._lru_frames:
            self._lru.popitem(last=False)
        return detections

    def iter_frames(self) -> Iterator[tuple[int, list[dict]]]:
        for start, end in self.coverage_ranges():
            for frame_idx in range(start, end + 1):
                yield frame_idx, self.read_frame(frame_idx) or []

    def statistics(self, confidence: float) -> dict:
        image_count = detection_count = 0
        confidence_sum = 0.0
        class_counts: dict[int, int] = {}
        for _frame_idx, detections in self.iter_frames():
            image_count += 1
            for detection in detections:
                value = float(detection["confidence"])
                if value < float(confidence):
                    continue
                detection_count += 1
                confidence_sum += value
                class_id = int(detection["class_id"])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
        return {
            "image_count": image_count,
            "detection_count": detection_count,
            "class_counts": class_counts,
            "mean_confidence": (
                confidence_sum / detection_count if detection_count else 0.0
            ),
        }


def write_path_index(path: str | Path, paths: Iterable[str | Path]) -> None:
    """Stream sorted UTF-8 paths and int64 offsets through atomic files."""
    cache_path = Path(path)
    strings_path = cache_path.with_suffix(cache_path.suffix + ".paths")
    index_path = cache_path.with_suffix(cache_path.suffix + ".paths.idx")
    strings_path.parent.mkdir(parents=True, exist_ok=True)
    strings_descriptor, temporary = tempfile.mkstemp(
        prefix=f".{strings_path.name}.", suffix=".tmp", dir=strings_path.parent
    )
    index_descriptor, index_temporary = tempfile.mkstemp(
        prefix=f".{index_path.name}.", suffix=".tmp", dir=index_path.parent
    )
    try:
        with (
            os.fdopen(strings_descriptor, "wb") as stream,
            os.fdopen(index_descriptor, "wb") as index_stream,
        ):
            previous = ""
            offset = 0
            index_stream.write(struct.pack("<q", offset))
            for raw in paths:
                value = str(Path(raw).expanduser().resolve())
                if previous and value <= previous:
                    raise ValueError("prediction paths must be sorted and unique")
                encoded = value.encode("utf-8")
                if len(encoded) > MAX_PATH_BYTES or b"\n" in encoded:
                    raise ValueError("prediction path exceeds safe index limits")
                stream.write(encoded + b"\n")
                offset += len(encoded) + 1
                if index_stream.tell() + 8 > MAX_PATH_INDEX_BYTES:
                    raise ValueError("prediction path index exceeds safe size")
                index_stream.write(struct.pack("<q", offset))
                previous = value
            stream.flush()
            os.fsync(stream.fileno())
            index_stream.flush()
            os.fsync(index_stream.fileno())
        os.replace(temporary, strings_path)
        os.replace(index_temporary, index_path)
    except BaseException:
        for candidate in (temporary, index_temporary):
            if candidate:
                try:
                    os.unlink(candidate)
                except FileNotFoundError:
                    pass
        raise


class PredictionPathIndex:
    """Binary-search an on-disk path table using bounded record reads."""

    def __init__(self, cache_path: str | Path) -> None:
        cache_path = Path(cache_path)
        self._strings_path = cache_path.with_suffix(cache_path.suffix + ".paths")
        index_path = cache_path.with_suffix(cache_path.suffix + ".paths.idx")
        index_size = index_path.stat().st_size
        if index_size < 8 or index_size > MAX_PATH_INDEX_BYTES or index_size % 8:
            raise ValueError("prediction path index exceeds safe size")
        with index_path.open("rb") as stream:
            encoded = stream.read(MAX_PATH_INDEX_BYTES + 1)
        if len(encoded) != index_size:
            raise ValueError("prediction path index changed while opening")
        offsets = np.frombuffer(encoded, dtype="<i8").astype(np.int64, copy=False)
        if (
            offsets.ndim != 1
            or len(offsets) < 1
            or offsets[0] != 0
            or (len(offsets) > 1 and np.any(np.diff(offsets) <= 0))
            or offsets[-1] != self._strings_path.stat().st_size
        ):
            raise ValueError("prediction path index is invalid")
        self._offsets = offsets

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def path_at(self, frame_idx: int) -> str:
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= len(self):
            raise IndexError(frame_idx)
        size = int(self._offsets[frame_idx + 1] - self._offsets[frame_idx])
        if size < 2 or size > MAX_PATH_BYTES + 1:
            raise ValueError("prediction path record is invalid")
        with self._strings_path.open("rb") as stream:
            stream.seek(int(self._offsets[frame_idx]))
            encoded = stream.read(size)
        if not encoded.endswith(b"\n"):
            raise ValueError("prediction path record is truncated")
        return encoded[:-1].decode("utf-8")

    def index_of(self, path: str | Path) -> int | None:
        wanted = str(Path(path).expanduser().resolve())
        lo, hi = 0, len(self)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.path_at(mid) < wanted:
                lo = mid + 1
            else:
                hi = mid
        return lo if lo < len(self) and self.path_at(lo) == wanted else None
