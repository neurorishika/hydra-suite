"""Bounded file transport for SAM3 preview results."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from hydra_suite.core.inference.semantic.base import SemanticInstance
from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.utils.geometry_levels import GeometryLevel

MAX_PREVIEW_BYTES = 16 * 1024 * 1024
MAX_INSTANCES = 10_000
MAX_POLYGON_POINTS = 1_000_000


def _polygon(points) -> list[list[float]]:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not 3 <= len(array) <= MAX_POLYGON_POINTS or not np.isfinite(array).all():
        raise ValueError("semantic preview contains an invalid polygon")
    return array.astype(float).tolist()


def write_frame_preview(path: str | Path, result) -> None:
    payload = {
        "version": 1,
        "image_path": str(Path(result.image_path).expanduser().resolve()),
        "source_name": str(result.source_name),
        "seconds": float(result.seconds),
        "tile_px": result.tile_px,
        "tiles_per_frame": int(result.tiles_per_frame),
        "predictions": [
            {
                "polygon_px": _polygon(item.polygon_px),
                "confidence": float(item.confidence),
            }
            for item in result.predictions[:MAX_INSTANCES]
        ],
        "ground_truth": [
            {
                "class_id": int(item.class_id),
                "confidence": float(item.confidence),
                "polygon_px": _polygon(item.points),
            }
            for item in result.ground_truth[:MAX_INSTANCES]
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_PREVIEW_BYTES:
        raise ValueError("semantic preview exceeds its bounded artifact cap")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_frame_preview(path: str | Path):
    from .semantic_escalation import FramePreviewResult

    path = Path(path)
    with path.open("rb") as stream:
        encoded = stream.read(MAX_PREVIEW_BYTES + 1)
    if len(encoded) > MAX_PREVIEW_BYTES:
        raise ValueError("semantic preview exceeds its bounded artifact cap")
    raw = json.loads(encoded)
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("semantic preview artifact is invalid")
    predictions = raw.get("predictions")
    ground_truth = raw.get("ground_truth")
    if (
        not isinstance(predictions, list)
        or not isinstance(ground_truth, list)
        or len(predictions) > MAX_INSTANCES
        or len(ground_truth) > MAX_INSTANCES
    ):
        raise ValueError("semantic preview instance count exceeds its cap")
    return FramePreviewResult(
        image_path=Path(raw["image_path"]),
        source_name=str(raw["source_name"]),
        predictions=[
            SemanticInstance(
                np.asarray(_polygon(item["polygon_px"]), dtype=np.float32),
                float(item["confidence"]),
            )
            for item in predictions
        ],
        ground_truth=[
            LabelRecord(
                class_id=int(item["class_id"]),
                confidence=float(item["confidence"]),
                points=np.asarray(_polygon(item["polygon_px"]), dtype=np.float32),
                level=GeometryLevel.POLYGON,
            )
            for item in ground_truth
        ],
        seconds=float(raw["seconds"]),
        tile_px=None if raw["tile_px"] is None else int(raw["tile_px"]),
        tiles_per_frame=int(raw["tiles_per_frame"]),
    )
