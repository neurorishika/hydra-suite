from __future__ import annotations

import numpy as np

from hydra_suite.core.inference.direct_calibration import (
    CalibrationDetection,
    match_frame,
    score_frames,
)


def _box(x0, y0, x1, y1, *, class_id=0, confidence=1.0):
    return CalibrationDetection(
        class_id=class_id,
        confidence=confidence,
        polygon_px=np.asarray(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
        ),
    )


def test_score_is_class_aware_and_one_to_one_with_duplicate_reporting():
    labels = [_box(0, 0, 10, 10), _box(20, 0, 30, 10, class_id=1)]
    predictions = [
        _box(0, 0, 10, 10),
        _box(1, 0, 11, 10),  # same class and object: cross-tile duplicate
        _box(20, 0, 30, 10, class_id=0),  # wrong class
    ]
    score = match_frame(predictions, labels, iou_threshold=0.5)
    assert score.matched == 1
    assert score.missed == 1
    assert score.extra == 2
    assert score.duplicate == 1


def test_frame_aggregate_reports_precision_recall_f1_and_localization_quality():
    score = score_frames(
        [
            ([_box(0, 0, 10, 10)], [_box(0, 0, 10, 10)]),
            ([_box(30, 30, 40, 40)], [_box(0, 0, 10, 10)]),
        ]
    )
    assert score.frames == 2
    assert score.matched == score.missed == score.extra == 1
    assert score.precision == score.recall == score.f1 == 0.5
    assert score.mean_iou == 0.5


from hydra_suite.core.inference.direct_calibration import (
    MIN_MATCHED_INSTANCES,
    RECOMMENDATION_RULE,
    CalibrationScore,
    DirectCalibrationPoint,
    recommend_balanced,
)


def _point(
    label, f1, seconds, *, matched=200, missed=10, extra=10, mean_iou=0.8, failed=""
):
    score = CalibrationScore(
        frames=20,
        matched=matched,
        missed=missed,
        extra=extra,
        precision=f1,
        recall=f1,
        f1=f1,
        duplicate=0,
        mean_iou=mean_iou,
    )
    return DirectCalibrationPoint(
        label=label,
        enabled=True,
        geometry_mode="auto_object",
        tile_width=640,
        tile_height=640,
        overlap=0.2,
        object_tile_fraction=0.4,
        max_detections=64,
        tiles_per_frame=9,
        seconds_per_frame=seconds,
        confidence=0.35,
        merge_policy="greedy_nmm",
        merge_metric="ios",
        merge_threshold=0.5,
        merge_backend="cv2",
        score=score,
        failed_reason=failed,
    )


def test_recommendation_prefers_the_cheapest_point_within_f1_tolerance():
    """'fast' is on the frontier (cheapest) and within 0.01 F1 of the best."""
    best, reason = recommend_balanced(
        [
            _point("slow", 0.920, 2.0, missed=8, extra=8),
            _point("fast", 0.915, 0.4, missed=9, extra=9),
            _point("bad", 0.600, 0.1, missed=60, extra=60),
        ]
    )
    assert best.label == "fast"
    assert RECOMMENDATION_RULE in reason


def test_a_dominated_cheap_point_never_wins():
    """'cheap' is worse on misses AND extras AND time is not enough to save it."""
    best, _reason = recommend_balanced(
        [
            _point("good", 0.930, 1.0, missed=5, extra=5),
            _point("cheap", 0.930, 2.0, missed=9, extra=9),
        ]
    )
    assert best.label == "good"


def test_failed_and_undersampled_points_are_never_recommended():
    best, reason = recommend_balanced(
        [
            _point("broken", 0.99, 0.1, failed="tile budget exceeded"),
            _point("thin", 0.99, 0.1, matched=MIN_MATCHED_INSTANCES - 1),
        ]
    )
    assert best is None
    assert "matched instances" in reason


def test_poor_localization_is_excluded_even_at_high_f1():
    best, _reason = recommend_balanced(
        [
            _point("sloppy", 0.99, 0.1, mean_iou=0.2),
            _point("clean", 0.90, 1.0, missed=5, extra=5),
        ]
    )
    assert best.label == "clean"


def test_empty_input_refuses_rather_than_raising():
    best, reason = recommend_balanced([])
    assert best is None and RECOMMENDATION_RULE in reason
