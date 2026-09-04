"""DetectKit-side adapter and worker for direct-detector SAHI calibration.

Core owns the grid, the sweep and the scoring; this module supplies labelled
frames, drives the production runner, and persists project-local evidence.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml as _yaml
from PySide6.QtCore import Signal

_LOGGER = logging.getLogger(__name__)

from hydra_suite.core.inference.direct_calibration import (
    CalibrationDetection,
    CalibrationScore,
    DirectCalibrationPoint,
    score_frames,
)
from hydra_suite.core.inference.direct_calibration_grid import (
    checkpoint_fingerprint,
    estimate_grid_work,
    label_set_fingerprint,
)
from hydra_suite.core.inference.direct_calibration_sweep import (
    config_for_point,
    detections_from_result,
    rescore_parts,
)
from hydra_suite.core.inference.stages.obb import collect_obb_parts_by_frame
from hydra_suite.detectkit.jobs.semantic_escalation import stratified_calibration_frames
from hydra_suite.widgets.workers import BaseWorker

PREVIEW_FRAMES = 8

EXHAUSTIVE_LABEL_WARNING = (
    "Confirm these frames are exhaustively labelled. A real animal missing "
    "from the labels looks like a false positive and biases calibration "
    "toward settings that are too strict."
)
MIN_MATCHED_NOTE = (
    "Too few matched instances for a recommendation. The measurements are "
    "still shown, but label a few more frames before trusting them."
)
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class EvidenceSet:
    frames: list
    split: str
    instances: int
    size_range: tuple
    sampled_from: int
    fingerprint: str


def _recording_key(image_path: Path) -> tuple[str, str]:
    """Group frames by their recording: parent dir + filename stem prefix.

    Neighbouring frames from one video must stay together -- scattering them
    across recordings makes the evidence set look more diverse than it is.
    """
    stem = image_path.stem
    prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return (str(image_path.parent), prefix)


def _labels_dir_for(images_dir: Path) -> Path:
    """Resolve the labels directory that mirrors ``images_dir``.

    Replaces only the LAST ``images`` path segment with ``labels`` -- naive
    string substitution (``str.replace("/images/", "/labels/")``) rewrites
    every occurrence, which corrupts paths where an ancestor directory is
    also named ``images`` (e.g. a dataset rooted at ``/data/images/pilot1``).
    """
    parts = list(images_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def _split_frames(dataset_yaml: Path, split: str) -> list:
    from hydra_suite.data.al.escalation import LabelRecord
    from hydra_suite.detectkit.gui.utils import parse_obb_label
    from hydra_suite.utils.geometry_levels import GeometryLevel

    document = _yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8")) or {}
    root = Path(document.get("path") or Path(dataset_yaml).parent)
    rel = document.get(split)
    images_dir = (root / rel) if rel else (root / "images" / split)
    if not images_dir.is_dir():
        return []
    labels_dir = _labels_dir_for(images_dir)
    if not labels_dir.is_dir():
        _LOGGER.warning(
            "No labels directory found for images dir %s (expected %s); "
            "treating split '%s' as having zero labelled frames.",
            images_dir,
            labels_dir,
            split,
        )
        return []
    out = []
    for image_path in sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS
    ):
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists() or not label_path.read_text().strip():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        parsed = parse_obb_label(label_path, width, height)
        if not parsed:
            continue
        out.append(
            (
                image_path,
                [
                    LabelRecord(
                        class_id=int(d["class_id"]),
                        confidence=1.0,
                        points=np.asarray(d["polygon_px"], dtype=np.float32).reshape(
                            -1, 2
                        ),
                        level=GeometryLevel.POLYGON,
                    )
                    for d in parsed
                ],
            )
        )
    return out


def _bounded_by_recording(frames: list, budget: int) -> list:
    """Take whole recordings until the budget is reached.

    A single recording that alone exceeds the budget is truncated to the
    budget -- the whole-group rule protects against scattering, not against
    an oversized first group swallowing the entire run unbounded.
    """
    if not budget or len(frames) <= budget:
        return frames
    grouped: dict[tuple[str, str], list] = {}
    for item in frames:
        grouped.setdefault(_recording_key(Path(item[0])), []).append(item)
    output: list = []
    for _key, group in sorted(grouped.items()):
        if not output and len(group) > budget:
            return group[:budget]
        if output and len(output) + len(group) > budget:
            break
        output.extend(group)
    return output or frames[:budget]


def resolve_calibration_dataset_yaml(dataset_dir) -> Path | None:
    """The FULL-RESOLUTION ``dataset.yaml`` for a training run's dataset dir.

    A sliced-training run's derived dataset holds TILES, not acquisition
    frames. Calibrating SAHI on tiles measures the wrong thing entirely --
    the whole point of the sweep is how slicing behaves on full frames -- so
    a sliced manifest (``type == "sliced_obb"``) is followed back to the
    unsliced ``source`` dataset it was cut from. The hop is bounded so a
    corrupt manifest chain cannot loop forever.

    Returns ``None`` when no yaml can be resolved; callers must then fall
    back to raw sources and say so (``EvidenceSet.split == "sources"``).
    """
    current = Path(dataset_dir) if dataset_dir else None
    seen: set[str] = set()
    for _hop in range(8):
        if current is None or not current.is_dir() or str(current) in seen:
            return None
        seen.add(str(current))
        manifest_path = current / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
            if isinstance(manifest, dict) and str(manifest.get("type", "")).startswith(
                "sliced"
            ):
                source = str(manifest.get("source", "") or "").strip()
                if not source:
                    _LOGGER.warning(
                        "Sliced dataset %s names no source dataset; refusing to "
                        "calibrate SAHI on tiles.",
                        current,
                    )
                    return None
                current = Path(source)
                continue
        candidate = current / "dataset.yaml"
        return candidate if candidate.is_file() else None
    return None


def collect_evidence(
    *,
    dataset_yaml: Path | None,
    sources: list,
    split: str = "val",
    budget: int = 80,
) -> EvidenceSet:
    """Labelled full-resolution evidence, defaulting to the held-out val split.

    Tuning on frames the model took gradient steps on reports optimistic
    numbers, so ``val`` is the default and any fallback is reported in
    ``EvidenceSet.split`` for the UI to show.
    """
    used_split = split
    frames: list = []
    if dataset_yaml is not None:
        frames = _split_frames(Path(dataset_yaml), split)
        if not frames and split != "train":
            frames = _split_frames(Path(dataset_yaml), "train")
            if frames:
                used_split = "train"
    if not frames and sources:
        frames = stratified_calibration_frames(sources, budget=budget)
        used_split = "sources"
    total = len(frames)
    frames = _bounded_by_recording(frames, budget)
    sizes = []
    for image_path, _labels in frames:
        image = cv2.imread(str(image_path))
        if image is not None:
            sizes.append(tuple(image.shape[:2]))
    size_range = (min(sizes), max(sizes)) if sizes else ((0, 0), (0, 0))
    return EvidenceSet(
        frames=frames,
        split=used_split,
        instances=sum(len(labels) for _p, labels in frames),
        size_range=size_range,
        sampled_from=total,
        fingerprint=label_set_fingerprint(frames),
    )


@dataclass(frozen=True)
class DirectCalibrationRequest:
    model_path: Path
    task: str
    evidence: EvidenceSet
    candidates: list
    confidences: tuple
    merge_settings: tuple
    runtime_tier: str
    max_targets: int
    evidence_dir: Path


@dataclass
class DirectCalibrationOutcome:
    points: list = field(default_factory=list)
    previews: list = field(default_factory=list)
    partial: bool = False
    message: str = ""


@dataclass(frozen=True)
class CalibrationPreview:
    """At most ``PREVIEW_FRAMES`` frames' ground truth vs. post-merge predictions.

    Retains only the image path and polygons -- never a decoded image array,
    which would multiply memory by the frame count for no benefit (the GUI
    re-decodes the image path on demand when the preview is shown).

    One preview exists per ``(candidate_index, merge_threshold, confidence)``
    -- i.e. one per measured ROW. Nothing about a row is reproduced at render
    time: the stored polygons ARE the row's own scored output, produced by
    the same ``rescore_parts`` at the same ``max_targets`` the sweep scores
    with. The previous design stored one permissive preview per
    (geometry, merge) and replayed the confidence gate + size cap in the GUI;
    that was never exact, because ``max_targets`` also derives a RAW cap
    applied BY CONFIDENCE around the merge, so a preview collected with the
    cap lifted was merged from a different candidate set than any row.
    """

    candidate_label: str
    frames: list  # list[tuple[Path, list[np.ndarray], list[np.ndarray]]]
    candidate_index: int = -1
    merge_threshold: float = 0.0
    confidence: float = 0.0


EVIDENCE_FILENAME = "direct_calibration.json"
EVIDENCE_FILENAME_GZ = "direct_calibration.json.gz"
# v3: one preview per MEASURED ROW -- keyed by (candidate_index,
# merge_threshold, confidence) and collected at that row's own max_targets,
# so the stored polygons are the row's output verbatim. v1 and v2 previews
# were collected with the detection cap lifted (v2 also replayed the gate in
# the GUI) and therefore depict a candidate set no row ever emitted, so they
# are DROPPED on load rather than shown as if they described a row (the
# points are still read -- the record's only other job is the
# partial-overwrite guard).
#
# v4: frame identity (path + ground truth) is stored ONCE in a shared
# ``frames`` table and each preview references frames by index -- v3 stored
# the ground truth once PER PREVIEW, and one preview per measured row means
# the same handful of frames' ground truth was duplicated ~600x. Coordinates
# are also rounded to 1 decimal place (a sub-pixel value is meaningless for
# an on-screen overlay) and the payload is gzip-compressed on disk. v3 and
# older previews are DROPPED on load for the same reason v1/v2 previews were
# dropped when v3 shipped -- they lack the frame table v4 depends on.
EVIDENCE_VERSION = 4
_ROUND_NDIGITS = 1


def _polygon_to_list(points) -> list[list[float]]:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2).astype(float)
    return np.round(array, _ROUND_NDIGITS).tolist()


def _point_to_dict(point: DirectCalibrationPoint) -> dict:
    """Flatten a point (with its nested ``score``) into a plain JSON dict."""
    score = point.score
    return {
        "label": point.label,
        "enabled": point.enabled,
        "geometry_mode": point.geometry_mode,
        "tile_width": point.tile_width,
        "tile_height": point.tile_height,
        "overlap": point.overlap,
        "object_tile_fraction": point.object_tile_fraction,
        "max_detections": point.max_detections,
        "tiles_per_frame": point.tiles_per_frame,
        "seconds_per_frame": point.seconds_per_frame,
        "confidence": point.confidence,
        "merge_policy": point.merge_policy,
        "merge_metric": point.merge_metric,
        "merge_threshold": point.merge_threshold,
        "merge_backend": point.merge_backend,
        "failed_reason": point.failed_reason,
        "candidate_index": point.candidate_index,
        "score": {
            "frames": score.frames,
            "matched": score.matched,
            "missed": score.missed,
            "extra": score.extra,
            "duplicate": score.duplicate,
            "precision": score.precision,
            "recall": score.recall,
            "f1": score.f1,
            "mean_iou": score.mean_iou,
        },
    }


def _point_from_dict(raw: dict) -> DirectCalibrationPoint:
    score_raw = raw["score"]
    score = CalibrationScore(
        frames=int(score_raw["frames"]),
        matched=int(score_raw["matched"]),
        missed=int(score_raw["missed"]),
        extra=int(score_raw["extra"]),
        duplicate=int(score_raw["duplicate"]),
        precision=float(score_raw["precision"]),
        recall=float(score_raw["recall"]),
        f1=float(score_raw["f1"]),
        mean_iou=float(score_raw["mean_iou"]),
    )
    return DirectCalibrationPoint(
        label=str(raw["label"]),
        enabled=bool(raw["enabled"]),
        geometry_mode=str(raw["geometry_mode"]),
        tile_width=int(raw["tile_width"]),
        tile_height=int(raw["tile_height"]),
        overlap=float(raw["overlap"]),
        object_tile_fraction=float(raw["object_tile_fraction"]),
        max_detections=int(raw["max_detections"]),
        tiles_per_frame=int(raw["tiles_per_frame"]),
        seconds_per_frame=float(raw["seconds_per_frame"]),
        confidence=float(raw["confidence"]),
        merge_policy=str(raw["merge_policy"]),
        merge_metric=str(raw["merge_metric"]),
        merge_threshold=float(raw["merge_threshold"]),
        merge_backend=str(raw["merge_backend"]),
        score=score,
        failed_reason=str(raw.get("failed_reason", "")),
        candidate_index=int(raw.get("candidate_index", -1)),
    )


def _stored_image_path(evidence_dir: Path, image_path: Path) -> dict:
    try:
        return {"relative": os.path.relpath(image_path, evidence_dir)}
    except ValueError:
        return {"absolute": str(image_path)}


def _load_image_path(evidence_dir: Path, stored) -> Path:
    if isinstance(stored, dict) and stored.get("relative") is not None:
        return (evidence_dir / str(stored["relative"])).resolve()
    if isinstance(stored, dict):
        return Path(str(stored.get("absolute", "")))
    return Path(str(stored))


class _FrameTable:
    """De-duplicates (path, ground truth) across every preview in one save.

    Ground truth is identical across every preview of the same frame -- it
    is the dominant source of duplication in the v3 evidence file, which
    stored it once per PREVIEW rather than once per frame. This assigns each
    distinct frame (keyed by resolved image path) a stable index the first
    time it is seen, so ``_preview_to_dict`` can reference frames instead of
    re-embedding them.
    """

    def __init__(self, evidence_dir: Path) -> None:
        self._evidence_dir = evidence_dir
        self._index_by_key: dict[str, int] = {}
        self.entries: list[dict] = []

    def index_for(self, image_path: Path, gt_polygons) -> int:
        key = str(Path(image_path))
        index = self._index_by_key.get(key)
        if index is not None:
            return index
        index = len(self.entries)
        self._index_by_key[key] = index
        self.entries.append(
            {
                "image_path": _stored_image_path(self._evidence_dir, Path(image_path)),
                "ground_truth": [_polygon_to_list(p) for p in gt_polygons],
            }
        )
        return index


def _preview_to_dict(frame_table: _FrameTable, preview: CalibrationPreview) -> dict:
    frames = []
    for image_path, gt_polygons, pred_polygons in preview.frames:
        frame_index = frame_table.index_for(Path(image_path), gt_polygons)
        frames.append(
            {
                "frame_index": frame_index,
                "predictions": [_polygon_to_list(p) for p in pred_polygons],
            }
        )
    return {
        "candidate_label": preview.candidate_label,
        "candidate_index": int(preview.candidate_index),
        "merge_threshold": float(preview.merge_threshold),
        "confidence": float(preview.confidence),
        "frames": frames,
    }


def _preview_from_dict(
    evidence_dir: Path, frame_entries: list, raw: dict
) -> CalibrationPreview:
    frames = []
    for raw_frame in raw.get("frames", []):
        frame_entry = frame_entries[int(raw_frame["frame_index"])]
        image_path = _load_image_path(evidence_dir, frame_entry["image_path"])
        gt_polygons = [
            np.asarray(p, dtype=np.float32).reshape(-1, 2)
            for p in frame_entry.get("ground_truth", [])
        ]
        pred_polygons = [
            np.asarray(p, dtype=np.float32).reshape(-1, 2)
            for p in raw_frame.get("predictions", [])
        ]
        frames.append((image_path, gt_polygons, pred_polygons))
    return CalibrationPreview(
        candidate_label=str(raw.get("candidate_label", "")),
        frames=frames,
        candidate_index=int(raw.get("candidate_index", -1)),
        merge_threshold=float(raw.get("merge_threshold", 0.0) or 0.0),
        confidence=float(raw.get("confidence", 0.0) or 0.0),
    )


def save_direct_calibration(
    evidence_dir: Path,
    outcome: DirectCalibrationOutcome,
    request: DirectCalibrationRequest,
) -> Path:
    """Persist the frontier. A partial run NEVER replaces complete evidence.

    Writes are atomic: the payload is written to a ``.tmp`` sibling and
    ``os.replace``-d onto the target, so a crash mid-write cannot leave a
    truncated or corrupt record on disk.
    """
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target = evidence_dir / EVIDENCE_FILENAME_GZ
    legacy_target = evidence_dir / EVIDENCE_FILENAME
    existing = load_direct_calibration(evidence_dir)
    if outcome.partial and existing is not None and not existing.partial:
        return target
    frame_table = _FrameTable(evidence_dir)
    previews = [_preview_to_dict(frame_table, preview) for preview in outcome.previews]
    payload = {
        "version": EVIDENCE_VERSION,
        "partial": bool(outcome.partial),
        "message": outcome.message,
        "provenance": {
            "checkpoint_fingerprint": checkpoint_fingerprint(request.model_path),
            "label_set_fingerprint": request.evidence.fingerprint,
            "split": request.evidence.split,
            "runtime_tier": request.runtime_tier,
            "max_targets": request.max_targets,
        },
        "points": [_point_to_dict(point) for point in outcome.points],
        "frames": frame_table.entries,
        "previews": previews,
    }
    fd, tmp_name = tempfile.mkstemp(
        dir=str(evidence_dir), prefix=f".{EVIDENCE_FILENAME_GZ}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as raw_stream:
            with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as gz_stream:
                gz_stream.write(json.dumps(payload).encode("utf-8"))
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    # A stale uncompressed v3-or-earlier file next to a fresh v4 one would
    # otherwise be preferred by a loader that checks the legacy name first;
    # remove it so there is exactly one evidence file per directory.
    legacy_target.unlink(missing_ok=True)
    return target


def load_direct_calibration(evidence_dir: Path) -> DirectCalibrationOutcome | None:
    """Load a persisted frontier, or ``None`` if absent, missing, or corrupt.

    Handles both the current gzip-compressed file and a plain-JSON file left
    by a previous (pre-v4) format -- an older file must degrade cleanly
    (zero previews, since it lacks the v4 frame table) rather than raise.
    """
    evidence_dir = Path(evidence_dir)
    target = evidence_dir / EVIDENCE_FILENAME_GZ
    legacy_target = evidence_dir / EVIDENCE_FILENAME
    try:
        if target.is_file():
            with gzip.open(target, "rt", encoding="utf-8") as stream:
                payload = json.loads(stream.read())
        elif legacy_target.is_file():
            payload = json.loads(legacy_target.read_text(encoding="utf-8"))
        else:
            return None
        if not isinstance(payload, dict):
            return None
        points = [_point_from_dict(raw) for raw in payload.get("points", [])]
        version = int(payload.get("version", 0) or 0)
        frame_entries = payload.get("frames", [])
        previews = (
            [
                _preview_from_dict(evidence_dir, frame_entries, raw)
                for raw in payload.get("previews", [])
            ]
            if version >= EVIDENCE_VERSION
            else []
        )
        return DirectCalibrationOutcome(
            points=points,
            previews=previews,
            partial=bool(payload.get("partial", False)),
            message=str(payload.get("message", "")),
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        json.JSONDecodeError,
        gzip.BadGzipFile,
    ):
        return None


def load_calibration_models(request: DirectCalibrationRequest, candidate):
    """Load the production models/runtime/config for one candidate geometry.

    Returns a 4-tuple ``(models, runtime, config, imgsz)``. Tiny and
    monkeypatchable so the sweep's control flow is testable without weights.
    """
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.stages.obb import _resolve_imgsz, load_obb_models

    config = config_for_point(
        str(request.model_path),
        slice_params=candidate.slice_params(),
        merge=request.merge_settings[0],
        confidence=request.confidences[0],
        max_targets=request.max_targets,
        runtime_tier=request.runtime_tier,
        model_task=request.task,
    )
    runtime = RuntimeContext.from_config(config)
    models = load_obb_models(config.obb, runtime)
    return models, runtime, config, int(_resolve_imgsz(models.direct_model))


def _label_detections(labels) -> list[CalibrationDetection]:
    return [
        CalibrationDetection(
            class_id=int(label.class_id),
            polygon_px=np.asarray(label.points, dtype=np.float32).reshape(-1, 2),
            confidence=1.0,
        )
        for label in labels
    ]


def _zero_score() -> CalibrationScore:
    return CalibrationScore(
        frames=0,
        matched=0,
        missed=0,
        extra=0,
        duplicate=0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        mean_iou=0.0,
    )


def _point_for(
    candidate,
    *,
    request,
    merge,
    confidence,
    tiles,
    seconds,
    score,
    merge_backend="cv2",
    failed_reason="",
    candidate_index=-1,
):
    return DirectCalibrationPoint(
        label=candidate.label,
        candidate_index=int(candidate_index),
        enabled=candidate.enabled,
        geometry_mode=candidate.geometry_mode,
        tile_width=candidate.slice_width,
        tile_height=candidate.slice_height,
        overlap=candidate.overlap,
        object_tile_fraction=candidate.object_tile_fraction,
        max_detections=int(request.max_targets),
        tiles_per_frame=int(tiles),
        seconds_per_frame=float(seconds),
        confidence=float(confidence),
        merge_policy=merge.policy,
        merge_metric=merge.metric,
        merge_threshold=float(merge.threshold),
        merge_backend=merge_backend,
        score=score,
        failed_reason=failed_reason,
    )


def _preview_for(
    request,
    candidate,
    candidate_index,
    merge,
    confidence,
    parts_per_frame,
    source,
    base_config,
    runtime,
):
    """Ground truth vs. the predictions of ONE measured row.

    The preview is keyed by (geometry, merge, confidence) and collected with
    that row's own ``max_targets``, so the stored polygons ARE what the row
    emits -- there is nothing for the GUI to reproduce. Collecting once per
    (geometry, merge) with the cap lifted, and replaying the confidence gate
    and size cap at render time, is NOT equivalent: ``max_targets`` also
    derives a RAW cap applied BY CONFIDENCE around the cross-tile merge
    (config.py:793-795), so a lifted-cap preview is merged from a different
    candidate set than any row -- and a merge that unions members then yields
    different polygons, not a superset.

    Only the image path and polygons are retained -- never a decoded image
    array, per the memory contract that governs the whole sweep.
    """
    frames = request.evidence.frames[: len(parts_per_frame)]
    point_config = config_for_point(
        str(request.model_path),
        slice_params=candidate.slice_params(),
        merge=merge,
        confidence=confidence,
        max_targets=request.max_targets,
        runtime_tier=request.runtime_tier,
        model_task=request.task,
    )
    preview_frames = []
    for i, (image_path, labels) in enumerate(frames):
        result = rescore_parts(
            parts_per_frame[i], source, point_config, runtime, frame_idx=i
        )
        predictions = detections_from_result(result)
        gt_polygons = [
            np.asarray(label.points, dtype=np.float32).reshape(-1, 2)
            for label in labels
        ]
        pred_polygons = [
            np.asarray(pred.polygon_px, dtype=np.float32).reshape(-1, 2)
            for pred in predictions
        ]
        preview_frames.append((Path(image_path), gt_polygons, pred_polygons))
    return CalibrationPreview(
        candidate_label=candidate.label,
        frames=preview_frames,
        candidate_index=int(candidate_index),
        merge_threshold=float(merge.threshold),
        confidence=float(confidence),
    )


def run_direct_calibration(request, *, progress=None, should_stop=None):
    """One model pass per geometry; confidence x merge swept offline.

    Frames are processed one at a time: a calibration run holds each frame's
    pre-merge parts for the whole sweep, so batching them would multiply peak
    memory by the frame count.

    Partial work is returned with ``partial=True``. It is inspectable but the
    caller must never let it replace complete calibration or become a profile.
    """
    outcome = DirectCalibrationOutcome()
    frames = request.evidence.frames
    # Keyed by candidate INDEX, not label: candidate labels are not guaranteed
    # unique (the grid dedups on geometry, not on label), so a label collision
    # would silently attribute the wrong tile count to a measured row.
    estimates = {
        index: estimate
        for index, estimate in enumerate(
            estimate_grid_work(
                request.candidates,
                frame_hw=request.evidence.size_range[1],
                imgsz=0,
                frames=len(frames),
            )
        )
    }
    for index, candidate in enumerate(request.candidates):
        if should_stop is not None and should_stop():
            outcome.partial = True
            outcome.message = "Cancelled; prior complete calibration is untouched."
            return outcome
        if progress is not None:
            progress(index, len(request.candidates), candidate.label)
        try:
            models, runtime, base_config, imgsz = load_calibration_models(
                request, candidate
            )
        except Exception as exc:
            for merge in request.merge_settings:
                for confidence in request.confidences:
                    outcome.points.append(
                        _point_for(
                            candidate,
                            request=request,
                            merge=merge,
                            confidence=confidence,
                            tiles=0,
                            seconds=0.0,
                            score=_zero_score(),
                            failed_reason=str(exc),
                            candidate_index=index,
                        )
                    )
            continue
        # Re-estimate with the model's real imgsz: auto_model/custom geometries
        # depend on it, and the pre-run estimate had to guess.
        measured = estimate_grid_work(
            [candidate],
            frame_hw=request.evidence.size_range[1],
            imgsz=imgsz,
            frames=len(frames),
        )[0]
        tiles = measured.tiles_per_frame or (estimates[index].tiles_per_frame)
        source = None
        elapsed = 0.0
        failure = ""
        parts_per_frame: list = []
        # ONE model call per FRAME (not one call over the whole evidence
        # list): evidence frames are full-resolution acquisition images, so
        # holding every decoded frame in memory at once -- on top of the
        # pre-merge parts already retained for the whole confidence x merge
        # sweep -- would multiply peak memory by the frame count. The
        # decoded ``image`` goes out of scope at the end of each iteration.
        for image_path, _labels in frames:
            if should_stop is not None and should_stop():
                outcome.partial = True
                outcome.message = "Cancelled; prior complete calibration is untouched."
                return outcome
            image = cv2.imread(str(image_path))
            if image is None:
                failure = f"could not read {Path(image_path).name}"
                break
            started = time.perf_counter()
            try:
                parts, source = collect_obb_parts_by_frame(
                    [image], models, base_config.obb, runtime
                )
            except Exception as exc:
                failure = str(exc)
                break
            elapsed += time.perf_counter() - started
            parts_per_frame.append(parts[0])
        if failure or source is None:
            for merge in request.merge_settings:
                for confidence in request.confidences:
                    outcome.points.append(
                        _point_for(
                            candidate,
                            request=request,
                            merge=merge,
                            confidence=confidence,
                            tiles=tiles,
                            seconds=0.0,
                            score=_zero_score(),
                            failed_reason=failure or "no regions produced",
                            candidate_index=index,
                        )
                    )
            continue
        seconds_per_frame = elapsed / max(1, len(parts_per_frame))
        backend = base_config.obb.direct.slice.merge_backend
        for merge in request.merge_settings:
            for confidence in request.confidences:
                point_config = config_for_point(
                    str(request.model_path),
                    slice_params=candidate.slice_params(),
                    merge=merge,
                    confidence=confidence,
                    max_targets=request.max_targets,
                    runtime_tier=request.runtime_tier,
                    model_task=request.task,
                )
                scored = [
                    (
                        detections_from_result(
                            rescore_parts(
                                parts_per_frame[i],
                                source,
                                point_config,
                                runtime,
                                frame_idx=i,
                            )
                        ),
                        _label_detections(frames[i][1]),
                    )
                    for i in range(len(parts_per_frame))
                ]
                outcome.points.append(
                    _point_for(
                        candidate,
                        request=request,
                        merge=merge,
                        confidence=confidence,
                        tiles=tiles,
                        seconds=seconds_per_frame,
                        score=score_frames(scored, task=request.task),
                        merge_backend=backend,
                        candidate_index=index,
                    )
                )
        # ONE preview per measured ROW. The merge threshold changes which
        # polygons exist at all; the confidence gate is post-merge, but
        # recovering a row from a permissive preview needs an UNCAPPED
        # post-filter set, and that cap is clamped at
        # MAX_DOWNSTREAM_CROPS_PER_FRAME -- so no superset exists above it.
        for merge in request.merge_settings:
            for confidence in request.confidences:
                outcome.previews.append(
                    _preview_for(
                        request,
                        candidate,
                        index,
                        merge,
                        confidence,
                        parts_per_frame[:PREVIEW_FRAMES],
                        source,
                        base_config,
                        runtime,
                    )
                )
        del parts_per_frame
    return outcome


class DirectCalibrationWorker(BaseWorker):
    """Runs the sweep off the GUI thread.

    ``BaseWorker`` provides progress/status/error and wraps ``execute()``; it
    has NO stop flag (widgets/workers.py:37-57), so cancellation is this
    worker's own responsibility.
    """

    result_ready = Signal(object)

    def __init__(self, request: DirectCalibrationRequest, parent=None) -> None:
        super().__init__(parent)
        self._request = request
        self._should_stop = False

    def cancel(self) -> None:
        self._should_stop = True

    def execute(self) -> None:
        outcome = run_direct_calibration(
            self._request,
            progress=lambda done, total, label: (
                self.progress.emit(int(100 * done / max(1, total))),
                self.status.emit(f"Measuring {label}"),
            ),
            should_stop=lambda: self._should_stop,
        )
        self.result_ready.emit(outcome)
