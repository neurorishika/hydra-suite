"""Per-frame active-learning signals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import cv2
import numpy as np


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


def _polygon_overlap_ratio(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Intersection area divided by smaller polygon area, clipped to [0, 1]."""
    poly_a = np.asarray(corners_a, dtype=np.float32).reshape(-1, 1, 2)
    poly_b = np.asarray(corners_b, dtype=np.float32).reshape(-1, 1, 2)
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    area_a = abs(float(cv2.contourArea(poly_a)))
    area_b = abs(float(cv2.contourArea(poly_b)))
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    except cv2.error:
        return 0.0
    if inter <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, inter / max(min(area_a, area_b), 1e-6))))


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
