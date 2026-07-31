"""Tests for the density-aware distance gate helper function."""

import numpy as np
import pytest

from hydra_suite.core.tracking.confidence.confidence_density import DensityRegion
from hydra_suite.core.tracking.confidence.density import get_density_region_flags


def _make_region(frame_start=0, frame_end=100, bbox=(10, 10, 50, 50)):
    return DensityRegion(
        label="region-1",
        frame_start=frame_start,
        frame_end=frame_end,
        pixel_bbox=bbox,
    )


def test_no_regions_returns_all_false():
    """With no regions, all flags are False."""
    meas = [np.array([30.0, 30.0, 0.0])]
    result = get_density_region_flags(meas, [], frame_idx=5)
    assert result.shape == (1,)
    assert not result[0]


def test_detection_inside_region_is_flagged():
    """Detection inside a region gets flagged True."""
    region = _make_region()
    meas = [np.array([30.0, 30.0, 0.0])]  # inside (10,10,50,50)
    result = get_density_region_flags(meas, [region], frame_idx=5)
    assert result[0]


def test_detection_outside_region_not_flagged():
    """Detection outside the region is not flagged."""
    region = _make_region()
    meas = [np.array([200.0, 200.0, 0.0])]  # outside (10,10,50,50)
    result = get_density_region_flags(meas, [region], frame_idx=5)
    assert not result[0]


def test_mixed_detections():
    """Only detections inside the region are flagged."""
    region = _make_region()
    meas = [
        np.array([30.0, 30.0, 0.0]),  # inside
        np.array([200.0, 200.0, 0.0]),  # outside
        np.array([40.0, 40.0, 0.0]),  # inside
    ]
    result = get_density_region_flags(meas, [region], frame_idx=5)
    assert result.shape == (3,)
    assert result[0]
    assert not result[1]
    assert result[2]


def test_wrong_frame_not_flagged():
    """Detection in the right place but wrong frame is not flagged."""
    region = _make_region(frame_start=10, frame_end=20)
    meas = [np.array([30.0, 30.0, 0.0])]
    result = get_density_region_flags(meas, [region], frame_idx=5)
    assert not result[0]


def test_cost_matrix_gating():
    """Verify that the density gate blocks long-range but allows short-range matches."""
    region = _make_region(bbox=(0, 0, 100, 100))
    meas = [
        np.array([50.0, 50.0, 0.0]),  # inside region
        np.array([200.0, 200.0, 0.0]),  # outside region
    ]
    flags = get_density_region_flags(meas, [region], frame_idx=5)

    # Simulate: 2 tracks, 2 detections
    cost = np.array([[10.0, 80.0], [90.0, 15.0]], dtype=np.float32)

    # Track predicted positions
    pred_xy = np.array([[48.0, 48.0], [198.0, 198.0]], dtype=np.float32)
    meas_xy = np.array([[50.0, 50.0], [200.0, 200.0]], dtype=np.float32)
    raw_dist = np.linalg.norm(pred_xy[:, None, :] - meas_xy[None, :, :], axis=2)

    MAX_DIST = 100.0
    density_factor = 0.7
    density_max_dist = MAX_DIST * density_factor

    # Apply density gate: block long-range matches to flagged detections
    flagged_cols = np.where(flags)[0]
    for c in flagged_cols:
        cost[raw_dist[:, c] >= density_max_dist, c] = 1e9

    # Detection 0 is inside region:
    # - Track 0 → Det 0: raw_dist ≈ 2.83 < 70 → allowed (cost stays 10.0)
    # - Track 1 → Det 0: raw_dist ≈ 212 > 70 → blocked (cost = 1e9)
    assert cost[0, 0] == pytest.approx(10.0)  # short-range: allowed
    assert cost[1, 0] == pytest.approx(1e9)  # long-range into region: blocked

    # Detection 1 is outside region — no gating applied:
    assert cost[0, 1] == pytest.approx(80.0)  # unchanged
    assert cost[1, 1] == pytest.approx(15.0)  # unchanged
