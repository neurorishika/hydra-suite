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


def match_frame(
    predictions: Sequence[CalibrationDetection],
    labels: Sequence[CalibrationDetection],
    *,
    iou_threshold: float = 0.5,
) -> FrameCalibrationScore:
    """Score predictions with class-aware, descending-IoU one-to-one matching.

    Duplicate counts are predictions that clear the match threshold against an
    already matched label. They remain extras for precision/F1, but exposing
    them separately lets the profile chooser identify bad cross-tile merges.
    """
    valid_predictions = [
        (index, prediction, _valid_polygon(prediction.polygon_px))
        for index, prediction in enumerate(predictions)
    ]
    valid_predictions = [item for item in valid_predictions if item[2] is not None]
    valid_labels = [
        (index, label, _valid_polygon(label.polygon_px))
        for index, label in enumerate(labels)
    ]
    valid_labels = [item for item in valid_labels if item[2] is not None]
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
) -> CalibrationScore:
    """Aggregate full-frame calibration evidence at one operating point."""
    scores = [
        match_frame(predictions, labels, iou_threshold=iou_threshold)
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
