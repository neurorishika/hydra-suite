"""Tests for hydra_suite.data.al.signals."""

from __future__ import annotations

import math

import numpy as np

from hydra_suite.data.al.signals import (
    ALSignals,
    score_count_deviation,
    score_crowd,
    score_uncertainty,
)


def test_alsignals_defaults():
    s = ALSignals(frame_id=7)
    assert s.frame_id == 7
    assert s.n_detections == 0
    assert math.isnan(s.mean_confidence)
    assert s.extras == {}


def test_score_uncertainty_high_confidence_yields_high_margin():
    mean_conf, margin = score_uncertainty([0.95, 0.92, 0.97], conf_floor=0.5)
    assert mean_conf > 0.9
    assert margin > 0.4  # well above the floor


def test_score_uncertainty_low_confidence_yields_low_margin():
    mean_conf, margin = score_uncertainty([0.4, 0.45, 0.55], conf_floor=0.5)
    assert mean_conf < 0.55
    assert margin <= 0.05


def test_score_uncertainty_empty_returns_nan_zero():
    mean_conf, margin = score_uncertainty([], conf_floor=0.5)
    assert math.isnan(mean_conf)
    assert margin == 0.0


def test_score_count_deviation():
    assert score_count_deviation(4, expected=4) == 0.0
    assert score_count_deviation(0, expected=4) == 1.0
    assert score_count_deviation(2, expected=4) == 0.5
    assert score_count_deviation(8, expected=4) == 1.0  # clipped
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


def test_nms_instability_stable_detector_returns_low_score():
    from hydra_suite.data.al.signals import score_nms_instability

    base = [
        (10, 10, 8, 4, 0.0, 0.95),
        (50, 50, 8, 4, 0.0, 0.93),
        (90, 90, 8, 4, 0.0, 0.97),
    ]

    def detector(_frame, conf, iou):
        return [d for d in base if d[5] >= conf]

    score = score_nms_instability(
        frame=np.zeros((100, 100, 3), np.uint8),
        detector_fn=detector,
        base_conf=0.5,
        base_iou=0.7,
    )
    assert score < 0.05


def test_nms_instability_unstable_detector_returns_high_score():
    from hydra_suite.data.al.signals import score_nms_instability

    def detector(_frame, conf, iou):
        if conf < 0.4:
            return [
                (10, 10, 8, 4, 0.0, 0.45),
                (30, 30, 8, 4, 0.0, 0.42),
                (60, 60, 8, 4, 0.0, 0.95),
            ]
        return [(60, 60, 8, 4, 0.0, 0.95)]

    score = score_nms_instability(
        frame=np.zeros((100, 100, 3), np.uint8),
        detector_fn=detector,
        base_conf=0.5,
        base_iou=0.7,
    )
    assert score > 0.3
