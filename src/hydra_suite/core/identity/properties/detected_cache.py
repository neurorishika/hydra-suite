"""Persistent cache for detected-frame heading metadata used by rich export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

_STRING_SENTINEL_NAN = "__NaN__"


def _empty_float_array() -> np.ndarray:
    return np.array([], dtype=np.float32)


def _empty_int_array() -> np.ndarray:
    return np.array([], dtype=np.int64)


def _empty_uint8_array() -> np.ndarray:
    return np.array([], dtype=np.uint8)


def _empty_object_array() -> np.ndarray:
    return np.array([], dtype=object)


def _normalize_detection_ids(detection_ids: Any) -> list[int]:
    if detection_ids is None:
        return []
    arr = np.asarray(detection_ids)
    if arr.size == 0:
        return []
    out: list[int] = []
    for raw in arr.reshape(-1).tolist():
        if isinstance(raw, (int, np.integer)):
            out.append(int(raw))
            continue
        value = float(raw)
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError(f"Invalid detection ID value: {raw!r}")
        out.append(int(value))
    return out


def _to_float_array(values: Any) -> np.ndarray:
    if values is None:
        return _empty_float_array()
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return _empty_float_array()
    return arr.reshape(-1)


def _to_uint8_array(values: Any) -> np.ndarray:
    if values is None:
        return _empty_uint8_array()
    arr = np.asarray(values, dtype=np.uint8)
    if arr.size == 0:
        return _empty_uint8_array()
    return arr.reshape(-1)


def _to_object_array(values: Any) -> np.ndarray:
    if values is None:
        return _empty_object_array()
    raw = list(values)
    if not raw:
        return _empty_object_array()
    return np.array(
        [_STRING_SENTINEL_NAN if value is None else str(value) for value in raw],
        dtype=object,
    )


class DetectedPropertiesCache:
    """Persistent frame-by-frame cache of resolved heading metadata."""

    def __init__(self, cache_path: str | Path, mode: str = "w") -> None:
        self._path = Path(cache_path)
        self._mode = str(mode or "w")
        self._data: dict[str, Any] = {}
        self._loaded_data = None
        self._cached_frames: set[int] = set()
        self._compatible = True
        self.metadata: dict[str, Any] = {}

        if self._mode == "r" and self._path.exists():
            self._loaded_data = np.load(str(self._path), allow_pickle=True)
            metadata = self._loaded_data.get("metadata", np.array({})).item()
            version = str(metadata.get("version", ""))
            if version != SCHEMA_VERSION:
                logger.warning(
                    "Incompatible detected-properties cache version '%s' (expected '%s').",
                    version,
                    SCHEMA_VERSION,
                )
                self._compatible = False
                self._loaded_data.close()
                self._loaded_data = None
                return
            self.metadata = metadata
            self._cached_frames = self._extract_cached_frames()

    def _extract_cached_frames(self) -> set[int]:
        if self._loaded_data is None:
            return set()
        cached: set[int] = set()
        for key in self._loaded_data.files:
            if not key.startswith("frame_") or not key.endswith("_detection_ids"):
                continue
            try:
                cached.add(int(key.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return cached

    def is_compatible(self) -> bool:
        return self._compatible

    def get_cached_frames(self) -> list[int]:
        if self._mode == "w":
            cached = []
            for key in self._data:
                if not key.startswith("frame_") or not key.endswith("_detection_ids"):
                    continue
                try:
                    cached.append(int(key.split("_")[1]))
                except (IndexError, ValueError):
                    continue
            return sorted(set(cached))
        return sorted(self._cached_frames)

    def add_frame(
        self,
        frame_idx: int,
        detection_ids: Any,
        theta_raw: Any = None,
        theta_resolved: Any = None,
        heading_source: Any = None,
        heading_directed: Any = None,
        headtail_heading: Any = None,
        headtail_confidence: Any = None,
        headtail_directed: Any = None,
    ) -> None:
        if self._mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot write")

        det_ids_arr = np.asarray(
            _normalize_detection_ids(detection_ids), dtype=np.int64
        )
        frame_key = f"frame_{int(frame_idx):06d}"
        self._data[f"{frame_key}_detection_ids"] = det_ids_arr
        self._data[f"{frame_key}_theta_raw"] = _to_float_array(theta_raw)
        self._data[f"{frame_key}_theta_resolved"] = _to_float_array(theta_resolved)
        self._data[f"{frame_key}_heading_source"] = _to_object_array(heading_source)
        self._data[f"{frame_key}_heading_directed"] = _to_uint8_array(heading_directed)
        self._data[f"{frame_key}_headtail_heading"] = _to_float_array(headtail_heading)
        self._data[f"{frame_key}_headtail_confidence"] = _to_float_array(
            headtail_confidence
        )
        self._data[f"{frame_key}_headtail_directed"] = _to_uint8_array(
            headtail_directed
        )

    def save(self, metadata: dict[str, Any] | None = None) -> None:
        if self._mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot save")
        payload = dict(metadata or {})
        payload["version"] = SCHEMA_VERSION
        self._data["metadata"] = np.array(payload, dtype=object)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self._path), **self._data)

    def get_frame(self, frame_idx: int) -> dict[str, list[Any]]:
        if self._mode != "r":
            raise RuntimeError("Cache opened in write mode, cannot read")
        if self._loaded_data is None:
            return {
                "detection_ids": [],
                "HeadingResolved": [],
                "HeadingMethod": [],
                "HeadingIsDirected": [],
                "HeadTailAngleRad": [],
                "HeadTailClassifierConf": [],
            }

        frame_key = f"frame_{int(frame_idx):06d}"
        det_ids = _normalize_detection_ids(
            self._loaded_data.get(f"{frame_key}_detection_ids", _empty_int_array())
        )
        theta_resolved = self._loaded_data.get(
            f"{frame_key}_theta_resolved", _empty_float_array()
        )
        heading_source_arr = self._loaded_data.get(
            f"{frame_key}_heading_source", _empty_object_array()
        )
        heading_directed = self._loaded_data.get(
            f"{frame_key}_heading_directed", _empty_uint8_array()
        )
        headtail_heading = self._loaded_data.get(
            f"{frame_key}_headtail_heading", _empty_float_array()
        )
        headtail_confidence = self._loaded_data.get(
            f"{frame_key}_headtail_confidence", _empty_float_array()
        )
        headtail_directed = self._loaded_data.get(
            f"{frame_key}_headtail_directed", _empty_uint8_array()
        )

        heading_source = []
        for raw in np.asarray(heading_source_arr, dtype=object).reshape(-1).tolist():
            text = str(raw)
            heading_source.append(np.nan if text == _STRING_SENTINEL_NAN else text)

        heading_directed_arr = np.asarray(heading_directed, dtype=np.uint8).reshape(-1)
        headtail_directed_arr = np.asarray(headtail_directed, dtype=np.uint8).reshape(
            -1
        )
        # Merge both directed flags: heading is directed if either the resolved
        # heading or the head-tail classifier specifically succeeded.
        is_directed = np.logical_or(
            heading_directed_arr.astype(bool),
            headtail_directed_arr.astype(bool),
        ).tolist()

        return {
            "detection_ids": det_ids,
            # HeadingResolved: final disambiguated heading angle (post head-tail).
            # ThetaRaw (= Theta in the tracking CSV) is omitted as redundant.
            "HeadingResolved": np.asarray(theta_resolved, dtype=np.float32)
            .reshape(-1)
            .tolist(),
            # HeadingMethod: which source produced the resolved heading
            # ("headtail", "pose", "velocity", "default", …).
            "HeadingMethod": heading_source,
            # HeadingIsDirected: True when the resolved heading has a known
            # head-vs-tail direction (from any source).
            "HeadingIsDirected": is_directed,
            # HeadTailAngleRad: angle from the head-tail classifier specifically,
            # regardless of whether it was used as the final resolved heading.
            "HeadTailAngleRad": np.asarray(headtail_heading, dtype=np.float32)
            .reshape(-1)
            .tolist(),
            # HeadTailClassifierConf: raw confidence score from the head-tail model.
            "HeadTailClassifierConf": np.asarray(headtail_confidence, dtype=np.float32)
            .reshape(-1)
            .tolist(),
        }

    def close(self) -> None:
        if self._loaded_data is not None:
            self._loaded_data.close()
            self._loaded_data = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
        return False
