"""Per-frame active-learning signals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Sequence

import numpy as np

from hydra_suite.utils.geometry import clamp01 as _clamp01
from hydra_suite.utils.geometry import (  # noqa: F401
    polygon_overlap_ratio as _polygon_overlap_ratio,
)


@dataclass
class ALSignals:
    """Per-frame signal record consumed by the acquisition selector.

    `uncertainty_score` is NOT derived from `mean_confidence` automatically --
    there is deliberately no `__post_init__` magic here. `mean_confidence`
    alone cannot be turned into an absolute severity without knowing the
    caller's confidence floor, and that floor varies per caller
    (`al_worker.py`'s `base_conf` defaults to 0.25; `dataset_generation.py`'s
    `conf_threshold` defaults to 0.5). A hardcoded fallback floor would
    silently disagree with a live caller-configured floor. Every constructor
    of `ALSignals` MUST compute `uncertainty_score` itself via
    `score_uncertainty(confidences, conf_floor=<its own floor>)` and pass it
    explicitly; the 0.0 default here is a genuine "no uncertainty signal"
    value, not a stand-in for "compute it later".
    """

    frame_id: int
    n_detections: int = 0
    mean_confidence: float = float("nan")
    uncertainty_score: float = 0.0
    nms_instability: float = 0.0
    count_deviation: float = 0.0
    crowd_score: float = 0.0
    fragmentation_score: float = 0.0
    edge_score: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)


def score_uncertainty(
    confidences: Sequence[float],
    conf_floor: float = 0.5,
) -> float:
    """Return absolute detection-uncertainty severity in [0, 1].

    Exactly 0 when the frame's mean confidence sits at or above `conf_floor` --
    a confidently-detected frame is not an active-learning candidate. All-NaN
    confidences (bg-sub, which has no confidence head) also score 0; treating
    "no information" as "maximum uncertainty" would make every bg-sub frame a
    candidate.
    """
    valid = [float(c) for c in confidences if c is not None and not math.isnan(c)]
    if not valid:
        return 0.0
    mean_conf = float(np.mean(valid))
    floor = max(float(conf_floor), 1e-6)
    if mean_conf >= floor:
        return 0.0
    return float(min(1.0, (floor - mean_conf) / floor))


def score_count_deviation(n: int, expected: int) -> float:
    """Return absolute count-mismatch severity in [0, 1]. 0 if expected <= 0.

    Asymmetric by design, preserving the legacy scorer's judgement: a missed
    animal is twice as bad as a spurious box, because a false negative removes
    training signal while a false positive is easy to delete during review.
    """
    if expected <= 0:
        return 0.0
    if n == expected:
        return 0.0
    if n < expected:
        return float(min(1.0, (expected - n) / float(expected)))
    return float(min(1.0, (n - expected) / float(expected)) * 0.5)


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


def score_fragmentation(
    obb_corners: Sequence[np.ndarray],
    reference_major_axis: float | None = None,
) -> float:
    """Return [0, 1] evidence that one object was split into several detections.

    A suspicious pair is close together, overlapping, and *both* smaller than
    the frame's typical detection -- the signature of a single animal broken
    into fragments. This is distinct from `score_crowd`, which measures genuine
    overlap between full-size neighbours.

    Ported from the legacy FrameQualityScorer so the signal keeps its meaning;
    the 0.45 suspicion gate and the pair weights are unchanged.
    """
    boxes = [
        np.asarray(c, dtype=np.float32).reshape(-1, 2)
        for c in obb_corners
        if c is not None
    ]
    boxes = [b for b in boxes if b.shape[0] >= 3]
    if len(boxes) < 2:
        return 0.0

    centers = [b.mean(axis=0) for b in boxes]
    major_axes = [
        float(
            max(
                np.linalg.norm(b[1] - b[0]),
                np.linalg.norm(b[2] - b[1]),
            )
        )
        for b in boxes
    ]
    typical = float(reference_major_axis or np.median(major_axes))
    typical = max(typical, 1.0)

    suspicious = 0
    best = 0.0
    for i, j in combinations(range(len(boxes)), 2):
        distance = float(np.linalg.norm(centers[i] - centers[j]))
        proximity = _clamp01(1.0 - distance / max(typical * 0.65, 1.0))
        overlap = _polygon_overlap_ratio(boxes[i], boxes[j])
        pair_major = (major_axes[i] + major_axes[j]) / 2.0
        smallness = _clamp01(1.0 - pair_major / typical)

        pair_score = _clamp01(0.5 * proximity + 0.3 * overlap + 0.2 * smallness)
        if pair_score >= 0.45:
            suspicious += 1
        best = max(best, pair_score)

    if best < 0.45:
        return 0.0
    return _clamp01(best + min(0.1 * max(suspicious - 1, 0), 0.2))


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
