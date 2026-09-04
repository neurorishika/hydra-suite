"""Bounded, frame-chunked persistence for SAM3 calibration evidence."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from hydra_suite.core.inference.semantic.calibration import (
    CalibrationGroundTruth,
    CalibrationPreviewFrame,
)
from hydra_suite.core.inference.semantic.tiling import TileCandidate

PREVIEW_ARTIFACT_RELATIVE_PATH = Path(
    "artifacts/detectkit/sam3_calibration_preview.json.gz"
)
PREVIEW_ARTIFACT_ROOT = Path("artifacts/detectkit/sam3_calibration_previews")
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_LEGACY_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 12
MAX_INSTANCES = 1_000
MAX_POLYGON_POINTS = 1_000_000


def _stored_image_path(project_dir: Path, image_path: Path) -> dict[str, str]:
    resolved = Path(image_path).expanduser().resolve()
    try:
        return {"relative": resolved.relative_to(project_dir).as_posix()}
    except ValueError:
        return {"absolute": str(resolved)}


def _restored_image_path(project_dir: Path, stored) -> Path:
    if isinstance(stored, dict) and stored.get("relative"):
        return project_dir / str(stored["relative"])
    if isinstance(stored, dict):
        return Path(str(stored.get("absolute", "")))
    return Path(str(stored))


def _polygon(points) -> list[list[float]]:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not 3 <= len(array) <= MAX_POLYGON_POINTS or not np.isfinite(array).all():
        raise ValueError("calibration preview contains an invalid polygon")
    return array.astype(float).tolist()


def _frame_payload(project_dir: Path, frame: CalibrationPreviewFrame) -> dict:
    if len(frame.ground_truth) > MAX_INSTANCES:
        raise ValueError("calibration preview ground-truth count exceeds its cap")
    candidate_sets = []
    for fraction, candidates in frame.candidates_by_fraction.items():
        if len(candidates) > MAX_INSTANCES:
            raise ValueError("calibration preview candidate count exceeds its cap")
        candidate_sets.append(
            {
                "tile_fraction": fraction,
                "candidates": [
                    {
                        "polygon_px": _polygon(candidate.polygon_px),
                        "confidence": float(candidate.confidence),
                        "tile_index": int(candidate.tile_index),
                    }
                    for candidate in candidates
                ],
            }
        )
    return {
        "version": 2,
        "image_path": _stored_image_path(project_dir, frame.image_path),
        "ground_truth": [
            {"class_id": int(item.class_id), "polygon_px": _polygon(item.polygon_px)}
            for item in frame.ground_truth
        ],
        "candidate_sets": candidate_sets,
    }


def _decode_frame(project_dir: Path, raw: object) -> CalibrationPreviewFrame:
    if not isinstance(raw, dict) or raw.get("version", 2) not in {1, 2}:
        raise ValueError("calibration preview frame is invalid")
    raw_ground_truth = raw.get("ground_truth", [])
    raw_sets = raw.get("candidate_sets", [])
    if (
        not isinstance(raw_ground_truth, list)
        or not isinstance(raw_sets, list)
        or len(raw_ground_truth) > MAX_INSTANCES
        or len(raw_sets) > 32
    ):
        raise ValueError("calibration preview frame exceeds its count cap")
    ground_truth = tuple(
        CalibrationGroundTruth(
            class_id=int(item.get("class_id", 0)),
            polygon_px=np.asarray(_polygon(item["polygon_px"]), dtype=np.float32),
        )
        for item in raw_ground_truth
    )
    candidate_sets: dict[float | None, tuple[TileCandidate, ...]] = {}
    for raw_set in raw_sets:
        raw_candidates = raw_set.get("candidates", [])
        if not isinstance(raw_candidates, list) or len(raw_candidates) > MAX_INSTANCES:
            raise ValueError("calibration preview candidate count exceeds its cap")
        raw_fraction = raw_set.get("tile_fraction")
        fraction = None if raw_fraction is None else float(raw_fraction)
        candidate_sets[fraction] = tuple(
            TileCandidate(
                polygon_px=np.asarray(
                    _polygon(candidate["polygon_px"]), dtype=np.float32
                ),
                confidence=float(candidate.get("confidence", 0.0)),
                tile_index=int(candidate.get("tile_index", 0)),
            )
            for candidate in raw_candidates
        )
    return CalibrationPreviewFrame(
        image_path=_restored_image_path(project_dir, raw["image_path"]),
        ground_truth=ground_truth,
        candidates_by_fraction=candidate_sets,
    )


class CalibrationPreviewStore(Sequence[CalibrationPreviewFrame]):
    """Lazy frame sequence with a fixed two-frame decoded-data footprint."""

    def __init__(self, project_dir: Path, artifact_dir: Path, count: int) -> None:
        self._project_dir = project_dir
        self._artifact_dir = artifact_dir
        self._count = count
        self._lru: OrderedDict[int, CalibrationPreviewFrame] = OrderedDict()

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[ii] for ii in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        if index in self._lru:
            frame = self._lru.pop(index)
            self._lru[index] = frame
            return frame
        path = self._artifact_dir / f"frame-{index:04d}.json"
        with path.open("rb") as stream:
            encoded = stream.read(MAX_FRAME_BYTES + 1)
        if len(encoded) > MAX_FRAME_BYTES:
            raise ValueError("calibration preview frame exceeds its byte cap")
        frame = _decode_frame(self._project_dir, json.loads(encoded))
        self._lru[index] = frame
        while len(self._lru) > 2:
            self._lru.popitem(last=False)
        return frame


def save_calibration_previews(
    project_dir: Path, frames: Iterable[CalibrationPreviewFrame]
) -> str:
    """Write frames incrementally, then atomically publish a complete directory."""
    project_dir = Path(project_dir).expanduser().resolve()
    parent = project_dir / PREVIEW_ARTIFACT_ROOT
    parent.mkdir(parents=True, exist_ok=True)
    artifact_name = uuid.uuid4().hex
    target = parent / artifact_name
    temporary = Path(tempfile.mkdtemp(prefix=f".{artifact_name}.", dir=parent))
    count = 0
    try:
        for frame in frames:
            if count >= MAX_FRAMES:
                raise ValueError("calibration preview frame count exceeds its cap")
            encoded = json.dumps(
                _frame_payload(project_dir, frame), separators=(",", ":")
            ).encode("utf-8")
            if len(encoded) > MAX_FRAME_BYTES:
                raise ValueError("calibration preview frame exceeds its byte cap")
            frame_path = temporary / f"frame-{count:04d}.json"
            with frame_path.open("wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            count += 1
        with (temporary / "manifest.json").open("wb") as stream:
            stream.write(
                json.dumps(
                    {"version": 2, "frames": count}, separators=(",", ":")
                ).encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target.relative_to(project_dir).as_posix()


def _load_legacy(project_dir: Path, target: Path) -> list[CalibrationPreviewFrame]:
    with gzip.open(target, "rb") as handle:
        encoded = handle.read(MAX_LEGACY_BYTES + 1)
    if len(encoded) > MAX_LEGACY_BYTES:
        raise ValueError("legacy calibration preview exceeds its byte cap")
    payload = json.loads(encoded)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("legacy calibration preview is invalid")
    raw_frames = payload.get("frames", [])
    if not isinstance(raw_frames, list) or len(raw_frames) > MAX_FRAMES:
        raise ValueError("legacy calibration preview frame count exceeds its cap")
    return [_decode_frame(project_dir, raw) for raw in raw_frames]


def load_calibration_previews(
    project_dir: Path, relative_path: str
) -> Sequence[CalibrationPreviewFrame]:
    """Load a bounded lazy v2 artifact, with bounded legacy compatibility."""
    root = Path(project_dir).expanduser().resolve()
    target = (root / relative_path).resolve()
    if root not in target.parents:
        return []
    try:
        if target.is_file():
            return _load_legacy(root, target)
        manifest_path = target / "manifest.json"
        with manifest_path.open("rb") as stream:
            encoded = stream.read(4097)
        if len(encoded) > 4096:
            return []
        manifest = json.loads(encoded)
        if not isinstance(manifest, dict) or set(manifest) != {"version", "frames"}:
            return []
        count = manifest["frames"]
        if (
            manifest["version"] != 2
            or not isinstance(count, int)
            or not 0 <= count <= MAX_FRAMES
        ):
            return []
        if any(not (target / f"frame-{ii:04d}.json").is_file() for ii in range(count)):
            return []
        return CalibrationPreviewStore(root, target, count)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return []
