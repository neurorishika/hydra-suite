"""DetectKit-side adapter and worker for direct-detector SAHI calibration.

Core owns the grid, the sweep and the scoring; this module supplies labelled
frames, drives the production runner, and persists project-local evidence.
"""

from __future__ import annotations

import logging
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
    """

    candidate_label: str
    frames: list  # list[tuple[Path, list[np.ndarray], list[np.ndarray]]]


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
):
    return DirectCalibrationPoint(
        label=candidate.label,
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


def _preview_for(request, candidate, parts_per_frame, source, base_config, runtime):
    """Ground truth vs. post-merge predictions at the training-geometry point.

    Only the image path and polygons are retained -- never a decoded image
    array, per the memory contract that governs the whole sweep.
    """
    frames = request.evidence.frames[: len(parts_per_frame)]
    point_config = config_for_point(
        str(request.model_path),
        slice_params=candidate.slice_params(),
        merge=request.merge_settings[0],
        confidence=request.confidences[0],
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
    return CalibrationPreview(candidate_label=candidate.label, frames=preview_frames)


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
                        score=score_frames(scored),
                        merge_backend=backend,
                    )
                )
        outcome.previews.append(
            _preview_for(
                request,
                candidate,
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
