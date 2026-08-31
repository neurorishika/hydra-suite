"""Persistence for the latest SAM3 calibration's visual evidence."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from hydra_suite.core.inference.semantic.calibration import (
    CalibrationGroundTruth,
    CalibrationPreviewFrame,
)
from hydra_suite.core.inference.semantic.tiling import TileCandidate

PREVIEW_ARTIFACT_RELATIVE_PATH = Path(
    "artifacts/detectkit/sam3_calibration_preview.json.gz"
)


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


def save_calibration_previews(
    project_dir: Path, frames: list[CalibrationPreviewFrame]
) -> str:
    """Atomically save preview evidence and return its project-relative path."""
    project_dir = Path(project_dir).expanduser().resolve()
    target = project_dir / PREVIEW_ARTIFACT_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "frames": [
            {
                "image_path": _stored_image_path(project_dir, frame.image_path),
                "ground_truth": [
                    {
                        "class_id": item.class_id,
                        "polygon_px": item.polygon_px.tolist(),
                    }
                    for item in frame.ground_truth
                ],
                "candidate_sets": [
                    {
                        "tile_fraction": fraction,
                        "candidates": [
                            {
                                "polygon_px": candidate.polygon_px.tolist(),
                                "confidence": candidate.confidence,
                                "tile_index": candidate.tile_index,
                            }
                            for candidate in candidates
                        ],
                    }
                    for fraction, candidates in frame.candidates_by_fraction.items()
                ],
            }
            for frame in frames
        ],
    }
    with NamedTemporaryFile(
        mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as raw:
        temporary = Path(raw.name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return PREVIEW_ARTIFACT_RELATIVE_PATH.as_posix()


def load_calibration_previews(
    project_dir: Path, relative_path: str
) -> list[CalibrationPreviewFrame]:
    """Load a project-scoped preview artifact, returning no frames if invalid."""
    root = Path(project_dir).expanduser().resolve()
    target = (root / relative_path).resolve()
    if root not in target.parents or not target.is_file():
        return []
    try:
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("version", 0)) != 1:
            return []
        frames: list[CalibrationPreviewFrame] = []
        for raw_frame in payload.get("frames", []):
            ground_truth = tuple(
                CalibrationGroundTruth(
                    class_id=int(item.get("class_id", 0)),
                    polygon_px=np.asarray(item["polygon_px"], dtype=np.float32).reshape(
                        -1, 2
                    ),
                )
                for item in raw_frame.get("ground_truth", [])
            )
            candidate_sets: dict[float | None, tuple[TileCandidate, ...]] = {}
            for raw_set in raw_frame.get("candidate_sets", []):
                raw_fraction = raw_set.get("tile_fraction")
                fraction = None if raw_fraction is None else float(raw_fraction)
                candidate_sets[fraction] = tuple(
                    TileCandidate(
                        polygon_px=np.asarray(
                            candidate["polygon_px"], dtype=np.float32
                        ).reshape(-1, 2),
                        confidence=float(candidate.get("confidence", 0.0)),
                        tile_index=int(candidate.get("tile_index", 0)),
                    )
                    for candidate in raw_set.get("candidates", [])
                )
            frames.append(
                CalibrationPreviewFrame(
                    image_path=_restored_image_path(root, raw_frame["image_path"]),
                    ground_truth=ground_truth,
                    candidates_by_fraction=candidate_sets,
                )
            )
        return frames
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return []
