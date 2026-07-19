from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNFactorPrediction,
    OBBResult,
)
from .base import CacheKey


class CacheHandle(ABC):
    @abstractmethod
    def is_valid(self) -> bool:
        """Return True if the cache file matches the expected key."""

    @abstractmethod
    def write_frame(self, frame_idx: int, **kwargs) -> None:
        """Buffer a frame's result for write on close()."""

    @abstractmethod
    def read_frame(self, frame_idx: int) -> Any:
        """Return the cached result for `frame_idx`, or None if invalid."""

    @abstractmethod
    def close(self) -> None:
        """Flush the buffered writes to disk."""


def _check_key(path: Path, key: CacheKey) -> bool:
    if not path.exists():
        return False
    try:
        data = np.load(path)
        return str(data["cache_key"][0]) == key.as_string()
    except Exception:
        return False


def _npz_save(path: Path, key: CacheKey, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, cache_key=np.array([key.as_string()]), **arrays)


# ---- DetectionCacheHandle ----


@dataclass
class DetectionCacheHandle(CacheHandle):
    path: Path
    key: CacheKey
    _buffer: list[OBBResult] = field(default_factory=list, repr=False)
    _data: dict | None = field(default=None, repr=False)
    _valid: bool | None = field(default=None, repr=False)
    _written: set[int] | None = field(default=None, repr=False)

    def is_valid(self) -> bool:
        if self._valid is None:
            self._valid = _check_key(self.path, self.key)
        return self._valid

    def write_frame(self, frame_idx: int, *, result: OBBResult, **_) -> None:
        self._buffer.append(result)

    def _ensure_data(self) -> None:
        if self._data is None:
            self._data = dict(np.load(self.path))

    def _written_frames(self) -> set[int]:
        """Set of frame indices actually processed into this cache.

        Recorded explicitly so a frame that was processed but had zero
        detections (which contributes no rows to ``frame_indices``) is still
        distinguishable from a frame that was never processed. Falls back to
        the unique ``frame_indices`` for caches written before this field
        existed.
        """
        if self._written is None:
            self._ensure_data()
            d = self._data or {}
            if "written_frames" in d:
                self._written = {int(f) for f in d["written_frames"]}
            else:
                self._written = {int(f) for f in d.get("frame_indices", [])}
        return self._written

    def covers_frame_range(self, start_frame: int, end_frame: int) -> bool:
        """Return True iff every frame in ``[start, end]`` was processed.

        Mirrors legacy ``DetectionCache.covers_frame_range`` so a truncated or
        interrupted forward pass (valid key, but fewer frames) is not silently
        reused for a backward/replay pass over a wider range.
        """
        if not self.is_valid():
            return False
        written = self._written_frames()
        return all(fi in written for fi in range(int(start_frame), int(end_frame) + 1))

    def get_missing_frames(
        self, start_frame: int, end_frame: int, max_report: int = 10
    ) -> list[int]:
        """Return up to ``max_report`` frame indices missing from ``[start, end]``."""
        if not self.is_valid():
            return list(range(int(start_frame), int(end_frame) + 1))[:max_report]
        written = self._written_frames()
        missing = [
            fi
            for fi in range(int(start_frame), int(end_frame) + 1)
            if fi not in written
        ]
        return missing[:max_report]

    def read_frame(self, frame_idx: int) -> OBBResult | None:
        if not self.is_valid():
            return None
        # A frame that was never processed must read back as None (KeyError
        # upstream) rather than a misleading empty "no animals" result.
        if int(frame_idx) not in self._written_frames():
            return None
        self._ensure_data()
        d = self._data
        mask = d["frame_indices"] == frame_idx
        # Backward compatibility: caches written before class_ids existed have
        # no such key. Fall back to None (-> all class 0 via
        # class_ids_or_zeros) rather than a KeyError.
        class_ids = d["class_ids"][mask] if "class_ids" in d else None
        return OBBResult(
            frame_idx=frame_idx,
            centroids=d["centroids"][mask],
            angles=d["angles"][mask],
            sizes=d["sizes"][mask],
            shapes=d["shapes"][mask],
            confidences=d["confidences"][mask],
            corners=d["corners"][mask],
            detection_ids=d["detection_ids"][mask],
            class_ids=class_ids,
        )

    def close(self) -> None:
        if not self._buffer:
            _npz_save(
                self.path,
                self.key,
                frame_count=np.array([0]),
                frame_indices=np.zeros(0, np.int32),
                written_frames=np.zeros(0, np.int32),
                centroids=np.zeros((0, 2), np.float32),
                angles=np.zeros(0, np.float32),
                sizes=np.zeros(0, np.float32),
                shapes=np.zeros((0, 2), np.float32),
                confidences=np.zeros(0, np.float32),
                corners=np.zeros((0, 4, 2), np.float32),
                detection_ids=np.zeros(0, np.int64),
                class_ids=np.zeros(0, np.int64),
            )
            return
        fi_list = []
        cents, angs, szs, shps, confs, corns, dids, clss = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for r in self._buffer:
            n = r.num_detections
            fi_list.extend([r.frame_idx] * n)
            if n > 0:
                cents.append(r.centroids)
                angs.append(r.angles)
                szs.append(r.sizes)
                shps.append(r.shapes)
                confs.append(r.confidences)
                corns.append(r.corners)
                dids.append(r.detection_ids)
                clss.append(r.class_ids_or_zeros)
        _npz_save(
            self.path,
            self.key,
            frame_count=np.array([len(self._buffer)]),
            frame_indices=np.array(fi_list, dtype=np.int32),
            written_frames=np.array(
                [r.frame_idx for r in self._buffer], dtype=np.int32
            ),
            centroids=(
                np.concatenate(cents) if cents else np.zeros((0, 2), np.float32)
            ),
            angles=np.concatenate(angs) if angs else np.zeros(0, np.float32),
            sizes=np.concatenate(szs) if szs else np.zeros(0, np.float32),
            shapes=(np.concatenate(shps) if shps else np.zeros((0, 2), np.float32)),
            confidences=(np.concatenate(confs) if confs else np.zeros(0, np.float32)),
            corners=(
                np.concatenate(corns) if corns else np.zeros((0, 4, 2), np.float32)
            ),
            detection_ids=(np.concatenate(dids) if dids else np.zeros(0, np.int64)),
            class_ids=(np.concatenate(clss) if clss else np.zeros(0, np.int64)),
        )


# ---- HeadTailCacheHandle ----


@dataclass
class HeadTailCacheHandle(CacheHandle):
    path: Path
    key: CacheKey
    _buf_fi: list[int] = field(default_factory=list, repr=False)
    _buf_det: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_hints: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_confs: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_dir: list[np.ndarray] = field(default_factory=list, repr=False)
    _data: dict | None = field(default=None, repr=False)
    _valid: bool | None = field(default=None, repr=False)

    def is_valid(self) -> bool:
        if self._valid is None:
            self._valid = _check_key(self.path, self.key)
        return self._valid

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
        n = len(det_indices)
        self._buf_fi.extend([frame_idx] * n)
        self._buf_det.append(det_indices)
        self._buf_hints.append(heading_hints)
        self._buf_confs.append(heading_confidences)
        self._buf_dir.append(directed_mask)

    def read_frame(self, frame_idx: int):
        if not self.is_valid():
            return None
        if self._data is None:
            self._data = dict(np.load(self.path))
        d = self._data
        mask = d["frame_indices"] == frame_idx
        return (
            d["det_indices"][mask].astype(np.int32),
            d["heading_hints"][mask],
            d["heading_confidences"][mask],
            d["directed_mask"][mask],
        )

    def close(self) -> None:
        _npz_save(
            self.path,
            self.key,
            frame_indices=np.array(self._buf_fi, dtype=np.int32),
            det_indices=(
                np.concatenate(self._buf_det)
                if self._buf_det
                else np.zeros(0, np.int32)
            ),
            heading_hints=(
                np.concatenate(self._buf_hints)
                if self._buf_hints
                else np.zeros(0, np.float32)
            ),
            heading_confidences=(
                np.concatenate(self._buf_confs)
                if self._buf_confs
                else np.zeros(0, np.float32)
            ),
            directed_mask=(
                np.concatenate(self._buf_dir)
                if self._buf_dir
                else np.zeros(0, np.uint8)
            ),
        )


# ---- CNNCacheHandle ----


@dataclass
class CNNCacheHandle(CacheHandle):
    path: Path
    key: CacheKey
    label: str
    _buf_fi: list[int] = field(default_factory=list, repr=False)
    _buf_det: list[int] = field(default_factory=list, repr=False)
    _buf_probs: list[list[np.ndarray]] = field(default_factory=list, repr=False)
    _factor_names: list[str] | None = field(default=None, repr=False)
    _class_names: list[list[str]] | None = field(default=None, repr=False)
    _data: dict | None = field(default=None, repr=False)
    _valid: bool | None = field(default=None, repr=False)

    def is_valid(self) -> bool:
        if self._valid is None:
            self._valid = _check_key(self.path, self.key)
        return self._valid

    def write_frame(
        self,
        frame_idx: int,
        *,
        predictions: list[CNNDetectionPrediction],
        **_,
    ) -> None:
        for pred in predictions:
            if self._factor_names is None and pred.factors:
                self._factor_names = [f.factor_name for f in pred.factors]
                self._class_names = [f.class_names for f in pred.factors]
            self._buf_fi.append(frame_idx)
            self._buf_det.append(pred.det_index)
            # Store one ragged-padded row of shape (F, c_max). c_max is fixed
            # at close() time, so here we just keep the per-factor probability
            # arrays and pad/stack on close.
            if pred.factors:
                self._buf_probs.append([f.raw_probabilities for f in pred.factors])
            else:
                self._buf_probs.append([])

    def read_frame(self, frame_idx: int) -> list[CNNDetectionPrediction] | None:
        if not self.is_valid():
            return None
        if self._data is None:
            self._data = dict(np.load(self.path))
        d = self._data
        fi = d["frame_indices"]
        mask = fi == frame_idx
        if not mask.any():
            return []
        factor_names = json.loads(str(d["factor_names_json"][0]))
        class_names_list = json.loads(str(d["class_names_json"][0]))
        class_counts = d["class_counts"].astype(int)
        probs_all = d["probabilities"]
        det_indices = d["det_indices"][mask]
        probs_frame = probs_all[mask]
        results = []
        for k, det_idx in enumerate(det_indices):
            factors = [
                CNNFactorPrediction(
                    factor_name=factor_names[f],
                    class_names=class_names_list[f],
                    raw_probabilities=probs_frame[k, f, : class_counts[f]].copy(),
                )
                for f in range(len(factor_names))
            ]
            results.append(
                CNNDetectionPrediction(det_index=int(det_idx), factors=factors)
            )
        return results

    def close(self) -> None:
        if not self._buf_probs or self._factor_names is None:
            _npz_save(
                self.path,
                self.key,
                frame_indices=np.zeros(0, np.int32),
                det_indices=np.zeros(0, np.int32),
                factor_names_json=np.array([json.dumps([])]),
                class_names_json=np.array([json.dumps([])]),
                class_counts=np.zeros(0, np.int32),
                probabilities=np.zeros((0, 0, 0), np.float32),
            )
            return
        class_counts = np.array([len(cn) for cn in self._class_names], dtype=np.int32)
        c_max = int(class_counts.max())
        f_count = len(self._factor_names)
        probs_stack = np.full(
            (len(self._buf_probs), f_count, c_max), np.nan, dtype=np.float32
        )
        for m, probs in enumerate(self._buf_probs):
            if not probs:
                continue
            for f_idx in range(min(f_count, len(probs))):
                arr = probs[f_idx]
                n_cls = int(arr.shape[0])
                probs_stack[m, f_idx, :n_cls] = arr[:n_cls]
        _npz_save(
            self.path,
            self.key,
            frame_indices=np.array(self._buf_fi, dtype=np.int32),
            det_indices=np.array(self._buf_det, dtype=np.int32),
            factor_names_json=np.array([json.dumps(self._factor_names)]),
            class_names_json=np.array([json.dumps(self._class_names)]),
            class_counts=class_counts,
            probabilities=probs_stack,
        )


# ---- PoseCacheHandle ----


@dataclass
class PoseCacheHandle(CacheHandle):
    path: Path
    key: CacheKey
    _buf_fi: list[int] = field(default_factory=list, repr=False)
    _buf_det: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_kp: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_valid: list[np.ndarray] = field(default_factory=list, repr=False)
    _data: dict | None = field(default=None, repr=False)
    _valid: bool | None = field(default=None, repr=False)

    def is_valid(self) -> bool:
        if self._valid is None:
            self._valid = _check_key(self.path, self.key)
        return self._valid

    def write_frame(
        self,
        frame_idx: int,
        *,
        det_indices: np.ndarray,
        keypoints: np.ndarray,
        valid_mask: np.ndarray,
        **_,
    ) -> None:
        n = len(det_indices)
        self._buf_fi.extend([frame_idx] * n)
        self._buf_det.append(det_indices)
        self._buf_kp.append(keypoints)
        self._buf_valid.append(valid_mask.astype(np.uint8))

    def read_frame(self, frame_idx: int):
        if not self.is_valid():
            return None
        if self._data is None:
            self._data = dict(np.load(self.path))
        d = self._data
        mask = d["frame_indices"] == frame_idx
        return (
            d["keypoints"][mask],
            d["det_indices"][mask],
            d["valid_mask"][mask].astype(bool),
        )

    def close(self) -> None:
        _npz_save(
            self.path,
            self.key,
            frame_indices=np.array(self._buf_fi, dtype=np.int32),
            det_indices=(
                np.concatenate(self._buf_det)
                if self._buf_det
                else np.zeros(0, np.int32)
            ),
            keypoints=(
                np.concatenate(self._buf_kp)
                if self._buf_kp
                else np.zeros((0, 0, 3), np.float32)
            ),
            valid_mask=(
                np.concatenate(self._buf_valid)
                if self._buf_valid
                else np.zeros(0, np.uint8)
            ),
        )


# ---- AprilTagCacheHandle ----


@dataclass
class AprilTagCacheHandle(CacheHandle):
    path: Path
    key: CacheKey
    _buf_fi: list[int] = field(default_factory=list, repr=False)
    _buf_tag_ids: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_det: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_centers: list[np.ndarray] = field(default_factory=list, repr=False)
    _buf_corners: list[np.ndarray] = field(default_factory=list, repr=False)
    _data: dict | None = field(default=None, repr=False)
    _valid: bool | None = field(default=None, repr=False)

    def is_valid(self) -> bool:
        if self._valid is None:
            self._valid = _check_key(self.path, self.key)
        return self._valid

    def write_frame(self, frame_idx: int, *, result: AprilTagResult, **_) -> None:
        # Convert list inputs to numpy arrays for storage uniformity.
        tag_ids = np.asarray(result.tag_ids, dtype=np.int32)
        det_indices = np.asarray(result.det_indices, dtype=np.int32)
        t = len(tag_ids)
        self._buf_fi.extend([frame_idx] * t)
        self._buf_tag_ids.append(tag_ids)
        self._buf_det.append(det_indices)
        self._buf_centers.append(result.centers)
        self._buf_corners.append(result.corners)

    def read_frame(self, frame_idx: int) -> AprilTagResult | None:
        if not self.is_valid():
            return None
        if self._data is None:
            self._data = dict(np.load(self.path))
        d = self._data
        mask = d["frame_indices"] == frame_idx
        return AprilTagResult(
            tag_ids=d["tag_ids"][mask],
            det_indices=d["det_indices"][mask],
            centers=d["centers"][mask],
            corners=d["corners"][mask],
        )

    def close(self) -> None:
        _npz_save(
            self.path,
            self.key,
            frame_indices=np.array(self._buf_fi, dtype=np.int32),
            tag_ids=(
                np.concatenate(self._buf_tag_ids)
                if self._buf_tag_ids
                else np.zeros(0, np.int32)
            ),
            det_indices=(
                np.concatenate(self._buf_det)
                if self._buf_det
                else np.zeros(0, np.int32)
            ),
            centers=(
                np.concatenate(self._buf_centers)
                if self._buf_centers
                else np.zeros((0, 2), np.float32)
            ),
            corners=(
                np.concatenate(self._buf_corners)
                if self._buf_corners
                else np.zeros((0, 4, 2), np.float32)
            ),
        )
