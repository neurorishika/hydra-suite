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

from hydra_suite.core.inference.match_geometry import match_one_to_one
from hydra_suite.core.inference.shape_prior import (
    MIN_MATCH_QUALITY,
    AreaBand,
    fit_area_band,
    match_quality,
)
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
    mean_quality: float = 0.0


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
    mean_quality: float = 0.0


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


def fit_calibration_area_band(
    label_sets: Iterable[Sequence[CalibrationDetection]],
    *,
    task: str = "obb",
) -> AreaBand | None:
    """Fit ONE size prior over the whole labelled evidence set. D9.

    Mirrors the semantic path's ``calibrate()``: the band is a property of
    the USER'S LABELS, not of the operating point being scored, so it is
    fitted once, pooled over every labelled frame, and threaded unchanged
    through the entire confidence x merge sweep. Fitting per point would
    let each configuration be judged against a prior derived from itself.

    Pooled over the whole label set, NOT per class -- that is the ruling's
    wording, and a per-class band on a sparse class would be fitted from a
    handful of instances.

    Polygons are reduced by ``_as_task_polygon`` first, for the same reason
    the matcher reduces them: under ``detect`` everything is scored as an
    axis-aligned quad, so a band fitted on un-reduced outlines would be
    systematically tighter than the areas it has to admit.

    Returns ``None`` when nothing is fittable; ``in_band`` treats a ``None``
    band as admitting everything, so an unlabelled set degrades to the
    pre-D9 behaviour rather than rejecting all predictions.
    """
    polygons: list[np.ndarray] = []
    for labels in label_sets:
        for label in labels:
            polygon = _valid_polygon(label.polygon_px)
            if polygon is None or not np.isfinite(polygon).all():
                continue
            polygons.append(_as_task_polygon(polygon, task))
    return fit_area_band(polygons)


def match_frame(
    predictions: Sequence[CalibrationDetection],
    labels: Sequence[CalibrationDetection],
    *,
    task: str = "obb",
    area_band: AreaBand | None = None,
    min_quality: float = MIN_MATCH_QUALITY,
) -> FrameCalibrationScore:
    """Score predictions with the SHARED containment matcher, class-aware.

    D7 of ``docs/superpowers/plans/2026-09-06-direct-path-scoring-unification.md``:
    the hard ``IoU >= 0.5`` admissibility gate is GONE. A pair is admissible
    when it clears the area band, is CONTAINED (either representative point
    inside the other polygon) and clears ``min_quality``; ranking is by graded
    ``match_quality``. The gate was removed because a correct silhouette with
    a different extent convention -- masks trace legs and antennae at ~1.7x
    the labelled body-core area -- scored as a MISS and an EXTRA at the same
    time. The matcher itself is ``match_geometry.match_one_to_one``, the very
    function the semantic path runs; it is called, not reimplemented.

    Two things that are ORTHOGONAL to the matcher and therefore survive:

    * ``task``. Both sides are still reduced by ``_as_task_polygon`` FIRST,
      before matching, band-fitting or quality, so a ``detect`` model is
      scored on the axis-aligned shape it can actually express. One reduction
      applied once, up front, keeps every downstream number in the same
      geometry.
    * CLASS-AWARENESS. ``match_one_to_one`` takes flat polygon lists with no
      notion of class, so predictions and labels are grouped by ``class_id``
      and matched WITHIN each class, then indices are mapped back. Grouping
      at the call site rather than teaching the shared function about classes
      keeps the semantic path's behaviour bit-for-bit untouched.

    Duplicates are predictions that were ADMISSIBLE to some label but lost
    the one-to-one race. They remain extras for precision/F1, but exposing
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
    classes = {prediction.class_id for _i, prediction, _p in valid_predictions} & {
        label.class_id for _i, label, _p in valid_labels
    }
    matched_predictions: set[int] = set()
    matched_labels: set[int] = set()
    admissible_predictions: set[int] = set()
    matched_ious: list[float] = []
    matched_qualities: list[float] = []
    for class_id in sorted(classes):
        pred_slice = [
            (index, polygon)
            for index, prediction, polygon in valid_predictions
            if prediction.class_id == class_id
        ]
        label_slice = [
            (index, polygon)
            for index, label, polygon in valid_labels
            if label.class_id == class_id
        ]
        pairs, admissible = match_one_to_one(
            [polygon for _i, polygon in pred_slice],
            [polygon for _i, polygon in label_slice],
            area_band=area_band,
            min_quality=min_quality,
            return_admissible=True,
        )
        for local_pred, _local_label in admissible:
            admissible_predictions.add(pred_slice[local_pred][0])
        for local_pred, local_label in pairs:
            pred_index, pred_polygon = pred_slice[local_pred]
            label_index, label_polygon = label_slice[local_label]
            matched_predictions.add(pred_index)
            matched_labels.add(label_index)
            matched_ious.append(polygon_iou(pred_polygon, label_polygon))
            matched_qualities.append(match_quality(pred_polygon, label_polygon))
    matched = len(matched_predictions)
    extra = max(0, len(valid_predictions) - matched)
    return FrameCalibrationScore(
        matched=matched,
        missed=max(0, len(valid_labels) - len(matched_labels)),
        extra=extra,
        duplicate=max(0, len(admissible_predictions - matched_predictions)),
        # WHAT ``mean_iou`` MEANS NOW: the mean polygon IoU, in task
        # geometry, over the pairs the matcher accepted -- a REPORTED
        # localization-quality diagnostic, no longer an admissibility
        # criterion for anything. Because the 0.5 gate is gone it now
        # includes sub-0.5 matches, so it will read LOWER than it did while
        # measuring strictly better recall; that is the metric becoming
        # honest, not degrading. It is deliberately not the selection
        # objective either: a systematic extent-convention offset shifts
        # every configuration's IoU equally, so it ranks configurations
        # badly while still being the right number to show a human asking
        # "how tightly do the accepted detections sit on the labels?".
        mean_iou=float(np.mean(matched_ious)) if matched_ious else 0.0,
        mean_quality=float(np.mean(matched_qualities)) if matched_qualities else 0.0,
    )


def score_frames(
    frames: Iterable[
        tuple[Sequence[CalibrationDetection], Sequence[CalibrationDetection]]
    ],
    *,
    task: str = "obb",
    area_band: AreaBand | None = None,
    min_quality: float = MIN_MATCH_QUALITY,
) -> CalibrationScore:
    """Aggregate full-frame calibration evidence at one operating point.

    ``task`` is forwarded to ``match_frame`` (see its docstring for
    ``detect`` handling), as is ``area_band`` -- which callers should fit
    ONCE over the whole label set via ``fit_calibration_area_band`` and
    thread through every point of a sweep, so the prior is a property of
    the labels rather than of the operating point being scored.
    """
    scores = [
        match_frame(
            predictions,
            labels,
            task=task,
            area_band=area_band,
            min_quality=min_quality,
        )
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
        mean_quality=(
            float(np.mean([score.mean_quality for score in scores])) if scores else 0.0
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
# Stable machine-readable identifier for the rule ``recommend_balanced``
# currently implements. Bump this (and its effective date) whenever the
# rule's SELECTION LOGIC changes so persisted profiles stay honest about
# which rule produced their "measured best" claim. Never reuse an id for a
# different rule and never back-fill this id onto profiles saved before it
# existed.
RECOMMENDATION_RULE_ID = "balanced-pareto-fastest-v1"
RECOMMENDATION_RULE_EFFECTIVE_DATE = "2026-09-06"


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
