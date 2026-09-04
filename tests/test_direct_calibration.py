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


def test_a_point_dominated_at_equal_speed_is_dropped_by_the_frontier():
    """Removing _pareto's filtering flips this result, so it proves the frontier runs."""
    best, _reason = recommend_balanced(
        [
            _point(
                "dominated", 0.930, 1.0, missed=20, extra=20
            ),  # first in list, same speed
            _point("dominating", 0.935, 1.0, missed=5, extra=5),
        ]
    )
    assert best.label == "dominating"


def test_axis_aligned_matching_counts_crowded_boxes_one_to_one():
    import numpy as np

    from hydra_suite.core.inference.direct_calibration import (
        CalibrationDetection,
        match_frame,
    )

    def box(x, y, w=10, h=10):
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)

    labels = [CalibrationDetection(0, box(0, 0)), CalibrationDetection(0, box(30, 0))]
    predictions = [
        CalibrationDetection(0, box(0, 0)),
        CalibrationDetection(0, box(1, 1)),  # duplicate on label 0
        CalibrationDetection(0, box(100, 100)),  # extra
    ]
    score = match_frame(predictions, labels, task="detect")
    assert score.matched == 1
    assert score.missed == 1
    assert score.extra == 2
    assert score.duplicate == 1


def test_segment_polygons_match_on_mask_overlap():
    import numpy as np

    from hydra_suite.core.inference.direct_calibration import (
        CalibrationDetection,
        match_frame,
    )

    triangle = np.array([[0, 0], [20, 0], [10, 20]], np.float32)
    score = match_frame(
        [CalibrationDetection(0, triangle)],
        [CalibrationDetection(0, triangle)],
        task="segment",
    )
    assert score.matched == 1 and score.mean_iou > 0.99


def test_rotated_prediction_is_scored_as_its_aabb_under_detect():
    """A detect model cannot express rotation; scoring must not credit it."""
    import numpy as np

    from hydra_suite.core.inference.direct_calibration import (
        CalibrationDetection,
        match_frame,
    )

    rotated = np.array([[10, 0], [20, 10], [10, 20], [0, 10]], np.float32)
    aabb = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], np.float32)
    obb_score = match_frame(
        [CalibrationDetection(0, rotated)], [CalibrationDetection(0, aabb)], task="obb"
    )
    detect_score = match_frame(
        [CalibrationDetection(0, rotated)],
        [CalibrationDetection(0, aabb)],
        task="detect",
    )
    assert detect_score.mean_iou > obb_score.mean_iou
