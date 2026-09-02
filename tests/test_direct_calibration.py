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
