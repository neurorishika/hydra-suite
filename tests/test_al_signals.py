"""Tests for hydra_suite.data.al.signals."""

from __future__ import annotations

import math

import numpy as np

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.data.al.signals import (
    ALSignals,
    score_count_deviation,
    score_crowd,
    score_nms_instability,
    score_uncertainty,
)
from hydra_suite.utils.geometry import obb_corners_from_dims


def _make_raw_result_with_n_detections(
    centroids: list[tuple[float, float]],
    confidences: list[float],
    w: float = 8.0,
    h: float = 4.0,
    angle: float = 0.0,
) -> OBBResult:
    """Build a raw (pre-filter) OBBResult fixture for `score_nms_instability`.

    `sizes`/`shapes` are populated with plausible placeholder values -- the
    default `OBBConfig` used by these tests leaves min/max_object_size and
    min/max_aspect_ratio at their disabling defaults (0 / inf), so those gates
    are no-ops here. `corners` must be geometrically real, since `_obb_nms`
    (invoked by `filter_with_indices` whenever `iou_threshold < 1.0`) computes
    actual polygon IoU from them.
    """
    n = len(centroids)
    assert len(confidences) == n
    centroids_arr = np.asarray(centroids, dtype=np.float32)
    angles = np.full(n, angle, dtype=np.float32)
    confidences_arr = np.asarray(confidences, dtype=np.float32)
    sizes = np.full(n, float(w * h), dtype=np.float32)
    shapes = np.stack(
        [sizes, np.full(n, float(w / h) if h else 1.0, dtype=np.float32)], axis=1
    )
    corners = np.stack(
        [obb_corners_from_dims(cx, cy, w, h, angle) for cx, cy in centroids]
    ).astype(np.float32)
    return OBBResult(
        frame_idx=0,
        centroids=centroids_arr,
        angles=angles,
        sizes=sizes,
        shapes=shapes,
        confidences=confidences_arr,
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(0, n),
    )


def test_alsignals_defaults():
    s = ALSignals(frame_id=7)
    assert s.frame_id == 7
    assert s.n_detections == 0
    assert math.isnan(s.mean_confidence)
    assert s.extras == {}


def test_score_uncertainty_high_confidence_is_zero_severity():
    severity = score_uncertainty([0.95, 0.92, 0.97], conf_floor=0.5)
    assert severity == 0.0  # well above the floor -- not a candidate


def test_score_uncertainty_low_confidence_yields_high_severity():
    severity = score_uncertainty([0.4, 0.45, 0.55], conf_floor=0.5)
    assert 0.0 < severity <= 1.0


def test_score_uncertainty_empty_returns_zero():
    assert score_uncertainty([], conf_floor=0.5) == 0.0


def test_score_count_deviation():
    assert score_count_deviation(4, expected=4) == 0.0
    assert score_count_deviation(0, expected=4) == 1.0
    assert score_count_deviation(2, expected=4) == 0.5
    assert score_count_deviation(8, expected=4) == 0.5  # overcount halved, clipped
    assert score_count_deviation(100, expected=4) == 0.5  # overcount clip ceiling
    assert score_count_deviation(3, expected=0) == 0.0  # no expected -> no signal


def test_score_crowd_no_overlap():
    boxes = [
        np.array([[50, 50], [60, 50], [60, 60], [50, 60]], dtype=np.float32),
        np.array([[100, 100], [110, 100], [110, 110], [100, 110]], dtype=np.float32),
    ]
    crowd, edge = score_crowd(boxes, frame_shape=(200, 200))
    assert crowd == 0.0
    assert edge == 0.0


def test_score_crowd_full_overlap():
    box = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    crowd, edge = score_crowd([box, box.copy()], frame_shape=(200, 200))
    assert crowd > 0.9


def test_score_nms_instability_uses_raw_result_no_detector_calls():
    """Pins the new contract: `score_nms_instability` takes an already-computed
    raw `OBBResult` + `OBBConfig`, never a `frame`/`detector_fn` pair."""
    raw = _make_raw_result_with_n_detections(
        centroids=[(10, 10), (50, 50), (90, 90), (130, 130), (170, 170)],
        confidences=[0.95, 0.93, 0.97, 0.9, 0.92],
    )
    config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="unused"),
        confidence_threshold=0.25,
        iou_threshold=0.5,
    )
    score = score_nms_instability(raw, config, base_conf=0.25, base_iou=0.5)
    assert 0.0 <= score <= 1.0


def test_nms_instability_stable_detector_returns_low_score():
    # Widely-spaced, uniformly high-confidence detections: neither the
    # confidence*0.7 nor the iou*1.3 perturbation should change the surviving
    # set, so instability should be near zero.
    raw = _make_raw_result_with_n_detections(
        centroids=[(10, 10), (50, 50), (90, 90)],
        confidences=[0.95, 0.93, 0.97],
    )
    config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="unused"),
        confidence_threshold=0.5,
        iou_threshold=0.7,
    )
    score = score_nms_instability(raw, config, base_conf=0.5, base_iou=0.7)
    assert score < 0.05


def test_nms_instability_unstable_detector_returns_high_score():
    # Two low-confidence detections (0.42, 0.45) sit just above base_conf=0.5
    # * 0.7 = 0.35 but below base_conf itself, so the confidence perturbation
    # reveals them while the base pass does not -- a genuinely unstable frame.
    raw = _make_raw_result_with_n_detections(
        centroids=[(10, 10), (30, 30), (60, 60)],
        confidences=[0.45, 0.42, 0.95],
    )
    config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="unused"),
        confidence_threshold=0.5,
        iou_threshold=0.7,
    )
    score = score_nms_instability(raw, config, base_conf=0.5, base_iou=0.7)
    assert score > 0.3
