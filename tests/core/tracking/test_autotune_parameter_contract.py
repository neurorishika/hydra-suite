from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.tracking.optimization.parameter_contract import (
    canonical_evaluation_params,
    quantize_tracking_autotune_params,
    quantize_tracking_autotune_value,
    tracking_autotune_widget_value,
)


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("YOLO_CONFIDENCE_THRESHOLD", 0.125, 0.13),
        ("YOLO_IOU_THRESHOLD", 0.995, 0.99),
        ("KALMAN_NOISE_COVARIANCE", 0.01234567, 0.0123),
        ("W_AREA", 0.12345678, 0.1235),
        ("KALMAN_DAMPING", 0.91234, 0.912),
        ("KALMAN_LONGITUDINAL_NOISE_MULTIPLIER", 5.44, 5.4),
    ],
)
def test_quantization_matches_qt_fixed_decimal_representation(key, raw, expected):
    assert quantize_tracking_autotune_value(key, raw) == expected


def test_candidate_quantization_preserves_integer_frame_storage() -> None:
    candidate = quantize_tracking_autotune_params(
        {
            "W_POSITION": 1.2345,
            "KALMAN_MATURITY_AGE": 7,
            "LOST_THRESHOLD_FRAMES": 24,
        }
    )

    assert candidate == {
        "W_POSITION": 1.23,
        "KALMAN_MATURITY_AGE": 7,
        "LOST_THRESHOLD_FRAMES": 24,
    }
    # Four seconds decimals are the actual TrackerKit widgets' precision. At
    # the maximum UI FPS, round-tripping still restores the original frame.
    seconds = tracking_autotune_widget_value(
        "KALMAN_MATURITY_AGE", candidate["KALMAN_MATURITY_AGE"], fps=240.0
    )
    assert seconds == 0.0292
    assert round(seconds * 240.0) == candidate["KALMAN_MATURITY_AGE"]


def test_fractional_frame_candidate_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer frame count"):
        quantize_tracking_autotune_value("LOST_THRESHOLD_FRAMES", 3.5)


def test_canonical_evaluation_params_hashes_ndarray_content() -> None:
    params = {
        "MAX_TARGETS": 2,
        "ROI_MASK": np.array([[0, 1], [1, 0]], dtype=np.uint8),
        "ARENA_ROIS": [np.array([1.0, 2.0], dtype=np.float32)],
    }
    first = canonical_evaluation_params(params)
    params["ROI_MASK"][0, 0] = 1
    second = canonical_evaluation_params(params)

    assert first != second
    assert first["ROI_MASK"]["sha256"] != second["ROI_MASK"]["sha256"]
