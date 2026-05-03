"""Per-frame active-learning signals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Sequence

import numpy as np

from hydra_suite.utils.geometry import (  # noqa: F401
    polygon_overlap_ratio as _polygon_overlap_ratio,
)


@dataclass
class ALSignals:
    """Per-frame signal record consumed by the acquisition selector."""

    frame_id: int
    n_detections: int = 0
    mean_confidence: float = float("nan")
    margin: float = 0.0
    nms_instability: float = 0.0
    count_deviation: float = 0.0
    crowd_score: float = 0.0
    edge_score: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)


def score_uncertainty(
    confidences: Sequence[float],
    conf_floor: float = 0.5,
) -> tuple[float, float]:
    """Return (mean_confidence, margin).

    `margin` is `min(c) - conf_floor` clipped to [0, 1]. A small/zero margin
    indicates at least one detection sits at or below the floor.
    """
    valid = [float(c) for c in confidences if c is not None and not math.isnan(c)]
    if not valid:
        return float("nan"), 0.0
    mean_conf = float(np.mean(valid))
    raw_margin = float(min(valid) - conf_floor)
    margin = float(max(0.0, min(1.0, raw_margin)))
    return mean_conf, margin


def score_count_deviation(n: int, expected: int) -> float:
    """Return |n - expected| / max(expected, 1), clipped to [0, 1]. 0 if expected<=0."""
    if expected <= 0:
        return 0.0
    return float(min(1.0, abs(n - expected) / float(expected)))


def score_crowd(
    obb_corners: Sequence[np.ndarray],
    frame_shape: tuple[int, int],
) -> tuple[float, float]:
    """Return (crowd_score, edge_score).

    crowd_score = max pairwise polygon-overlap ratio across all detection pairs.
    edge_score  = max box-corner proximity to frame border, normalized to [0, 1].
    """
    if len(obb_corners) < 1:
        return 0.0, 0.0
    h, w = int(frame_shape[0]), int(frame_shape[1])

    crowd = 0.0
    if len(obb_corners) >= 2:
        for a, b in combinations(obb_corners, 2):
            crowd = max(crowd, _polygon_overlap_ratio(a, b))

    edge = 0.0
    for box in obb_corners:
        arr = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        if arr.size == 0:
            continue
        dx = np.minimum(arr[:, 0], w - arr[:, 0])
        dy = np.minimum(arr[:, 1], h - arr[:, 1])
        margin_px = float(np.min(np.minimum(dx, dy)))
        ref = max(min(w, h) * 0.10, 1.0)
        edge_norm = max(0.0, 1.0 - margin_px / ref)
        edge = max(edge, edge_norm)

    return float(crowd), float(edge)


Detection = tuple  # (cx, cy, w, h, theta, conf)


def _set_iou_greedy(
    set_a: Sequence[Detection],
    set_b: Sequence[Detection],
    match_distance: float = 12.0,
) -> float:
    """Approximate set IoU via greedy center-distance matching."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    used_b: set[int] = set()
    matched = 0
    for det_a in set_a:
        best_idx, best_dist = -1, math.inf
        for j, det_b in enumerate(set_b):
            if j in used_b:
                continue
            dist = math.hypot(det_a[0] - det_b[0], det_a[1] - det_b[1])
            if dist < best_dist:
                best_dist, best_idx = dist, j
        if best_idx >= 0 and best_dist <= match_distance:
            used_b.add(best_idx)
            matched += 1
    union = len(set_a) + len(set_b) - matched
    return matched / max(union, 1)


def score_nms_instability(
    frame: np.ndarray,
    detector_fn: Callable[[np.ndarray, float, float], Sequence[Detection]],
    base_conf: float,
    base_iou: float,
) -> float:
    """Return 1 - mean(set_IoU) across two (conf, iou) perturbations.

    Higher score = detection set changes meaningfully under small NMS-threshold
    shifts -> model is unstable on this frame -> good AL pick.
    """
    base_set = list(detector_fn(frame, base_conf, base_iou))
    perturbations = [
        (max(base_conf * 0.7, 0.01), base_iou),
        (base_conf, min(base_iou * 1.3, 0.95)),
    ]
    ious = []
    for conf, iou in perturbations:
        ious.append(_set_iou_greedy(base_set, list(detector_fn(frame, conf, iou))))
    if not ious:
        return 0.0
    return float(1.0 - sum(ious) / len(ious))
