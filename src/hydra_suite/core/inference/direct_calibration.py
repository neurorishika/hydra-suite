"""Pure detector-calibration scoring for human-selected SAHI profiles.

This module deliberately scores frame-space, post-merge polygons. DetectKit
adapts direct-executor results to these small records; TrackerKit never imports
the calibration UI. Confidence sweeps are cheap because callers retain the raw
candidate polygons from one fixed geometry pass and call this scorer repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from hydra_suite.utils.polygon_iou import polygon_iou


@dataclass(frozen=True)
class CalibrationDetection:
    """One frame-space prediction or ground-truth instance."""

    class_id: int
    polygon_px: np.ndarray
    confidence: float = 1.0


@dataclass(frozen=True)
class FrameCalibrationScore:
    """One-to-one result for one full-resolution calibration frame."""

    matched: int
    missed: int
    extra: int
    duplicate: int
    mean_iou: float


@dataclass(frozen=True)
class CalibrationScore:
    """Aggregate evidence for one measured SAHI operating point."""

    frames: int
    matched: int
    missed: int
    extra: int
    duplicate: int
    precision: float
    recall: float
    f1: float
    mean_iou: float


def _valid_polygon(value: np.ndarray) -> np.ndarray | None:
    polygon = np.asarray(value, dtype=np.float32).reshape(-1, 2)
    return polygon if polygon.shape[0] >= 3 else None


def _as_task_polygon(polygon: np.ndarray, task: str) -> np.ndarray:
    """Reduce a polygon to the shape its task's model can actually express.

    A ``detect`` model cannot express rotation, so both the prediction and the
    label are reduced to their axis-aligned bounding quad before IoU -- scoring
    a rotated polygon against an axis-aligned one directly would either credit
    the model with overlap it cannot produce, or penalize it below what an
    axis-aligned detector could ever achieve. ``obb`` and ``segment`` keep the
    original polygon and use full polygon IoU.
    """
    if task != "detect":
        return polygon
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def match_frame(
    predictions: Sequence[CalibrationDetection],
    labels: Sequence[CalibrationDetection],
    *,
    iou_threshold: float = 0.5,
    task: str = "obb",
) -> FrameCalibrationScore:
    """Score predictions with class-aware, descending-IoU one-to-one matching.

    ``iou_threshold`` defaults to 0.5, the documented localization-quality
    floor used throughout calibration. ``task`` selects how overlap is
    computed: ``obb``/``segment`` use full polygon IoU; ``detect`` reduces
    both sides to their axis-aligned bounding box first (see
    ``_as_task_polygon``).

    Duplicate counts are predictions that clear the match threshold against an
    already matched label. They remain extras for precision/F1, but exposing
    them separately lets the profile chooser identify bad cross-tile merges.
    """
    valid_predictions = [
        (index, prediction, _valid_polygon(prediction.polygon_px))
        for index, prediction in enumerate(predictions)
    ]
    valid_predictions = [
        (index, prediction, _as_task_polygon(polygon, task))
        for index, prediction, polygon in valid_predictions
        if polygon is not None
    ]
    valid_labels = [
        (index, label, _valid_polygon(label.polygon_px))
        for index, label in enumerate(labels)
    ]
    valid_labels = [
        (index, label, _as_task_polygon(polygon, task))
        for index, label, polygon in valid_labels
        if polygon is not None
    ]
    pairs: list[tuple[float, int, int]] = []
    duplicate_candidates: set[int] = set()
    for pred_index, prediction, pred_polygon in valid_predictions:
        for label_index, label, label_polygon in valid_labels:
            if prediction.class_id != label.class_id:
                continue
            iou = polygon_iou(pred_polygon, label_polygon)
            if iou >= iou_threshold:
                pairs.append((iou, pred_index, label_index))
                duplicate_candidates.add(pred_index)
    pairs.sort(reverse=True)
    matched_predictions: set[int] = set()
    matched_labels: set[int] = set()
    matched_ious: list[float] = []
    for iou, pred_index, label_index in pairs:
        if pred_index in matched_predictions or label_index in matched_labels:
            continue
        matched_predictions.add(pred_index)
        matched_labels.add(label_index)
        matched_ious.append(iou)
    matched = len(matched_predictions)
    extra = max(0, len(valid_predictions) - matched)
    return FrameCalibrationScore(
        matched=matched,
        missed=max(0, len(valid_labels) - len(matched_labels)),
        extra=extra,
        duplicate=max(0, len(duplicate_candidates - matched_predictions)),
        mean_iou=float(np.mean(matched_ious)) if matched_ious else 0.0,
    )


def score_frames(
    frames: Iterable[
        tuple[Sequence[CalibrationDetection], Sequence[CalibrationDetection]]
    ],
    *,
    iou_threshold: float = 0.5,
    task: str = "obb",
) -> CalibrationScore:
    """Aggregate full-frame calibration evidence at one operating point.

    ``iou_threshold`` defaults to the 0.5 localization-quality floor; ``task``
    is forwarded to ``match_frame`` (see its docstring for ``detect`` handling).
    """
    scores = [
        match_frame(predictions, labels, iou_threshold=iou_threshold, task=task)
        for predictions, labels in frames
    ]
    matched = sum(score.matched for score in scores)
    missed = sum(score.missed for score in scores)
    extra = sum(score.extra for score in scores)
    precision = matched / (matched + extra) if matched + extra else 0.0
    recall = matched / (matched + missed) if matched + missed else 0.0
    return CalibrationScore(
        frames=len(scores),
        matched=matched,
        missed=missed,
        extra=extra,
        duplicate=sum(score.duplicate for score in scores),
        precision=precision,
        recall=recall,
        f1=(
            (2 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        ),
        mean_iou=(
            float(np.mean([score.mean_iou for score in scores])) if scores else 0.0
        ),
    )


MIN_MATCHED_INSTANCES = 60
F1_TOLERANCE = 0.01
MIN_LOCALIZATION = 0.5
RECOMMENDATION_RULE = (
    "Balanced rule: drop failed and undersampled points, keep the Pareto "
    "frontier of misses, extras and time, then take the fastest point whose F1 "
    "is within 0.01 of the best and whose localization quality is at least 0.5."
)


@dataclass(frozen=True)
class DirectCalibrationPoint:
    """One fully measured SAHI operating point, with its evidence attached."""

    label: str
    enabled: bool
    geometry_mode: str
    tile_width: int
    tile_height: int
    overlap: float
    object_tile_fraction: float
    max_detections: int
    tiles_per_frame: int
    seconds_per_frame: float
    confidence: float
    merge_policy: str
    merge_metric: str
    merge_threshold: float
    merge_backend: str
    score: CalibrationScore
    failed_reason: str = ""
    # Stable identity for this row's GEOMETRY. Candidate labels are not
    # unique (the grid dedups on geometry, not on label), so nothing may key
    # a row by ``label`` -- overlays and any other per-row lookup use
    # ``(candidate_index, merge_threshold, confidence)`` instead.
    candidate_index: int = -1


def _pareto(points: Sequence[DirectCalibrationPoint]) -> list[DirectCalibrationPoint]:
    """Keep points not dominated on (misses, extras, seconds) simultaneously."""

    def cost(point: DirectCalibrationPoint) -> tuple[float, float, float]:
        return (
            float(point.score.missed),
            float(point.score.extra),
            float(point.seconds_per_frame),
        )

    keep: list[DirectCalibrationPoint] = []
    for candidate in points:
        this = cost(candidate)
        dominated = any(
            all(o <= t for o, t in zip(cost(other), this))
            and any(o < t for o, t in zip(cost(other), this))
            for other in points
            if other is not candidate
        )
        if not dominated:
            keep.append(candidate)
    return keep


def recommend_balanced(
    points: Sequence[DirectCalibrationPoint],
    *,
    min_matched: int = MIN_MATCHED_INSTANCES,
    f1_tolerance: float = F1_TOLERANCE,
    min_iou: float = MIN_LOCALIZATION,
) -> tuple[DirectCalibrationPoint | None, str]:
    """Explain a suggestion, or refuse. It is never applied automatically.

    The floors are ELIGIBILITY filters, not vetoes on the winner: a
    configuration that finds almost nothing would otherwise post a perfect F1
    on a handful of matches and win.
    """
    eligible = [
        point
        for point in points
        if not point.failed_reason
        and point.score.matched >= min_matched
        and point.score.mean_iou >= min_iou
    ]
    if not eligible:
        return None, (
            f"No point cleared the floors: at least {min_matched} matched "
            f"instances and {min_iou:g} localization quality. Label a few more "
            "frames or widen the sweep. " + RECOMMENDATION_RULE
        )
    best_f1 = max(point.score.f1 for point in eligible)
    frontier = _pareto(eligible)
    near_best = [p for p in frontier if p.score.f1 >= best_f1 - f1_tolerance]
    if not near_best:
        # The best-F1 point is always within tolerance of itself, so an empty
        # set here means it was dominated on every cost axis; fall back to it
        # explicitly rather than to an arbitrary frontier member.
        near_best = [p for p in eligible if p.score.f1 >= best_f1 - f1_tolerance]
    chosen = min(near_best, key=lambda p: p.seconds_per_frame)
    return chosen, (
        f"{chosen.label}: F1 {chosen.score.f1:.3f} (best {best_f1:.3f}), "
        f"{chosen.seconds_per_frame:.2f}s/frame on this machine and data. "
        + RECOMMENDATION_RULE
    )
