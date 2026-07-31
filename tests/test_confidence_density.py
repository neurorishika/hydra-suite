# tests/test_confidence_density.py
import numpy as np
from scipy.ndimage import gaussian_filter

from hydra_suite.core.tracking.confidence.confidence_density import (
    DensityRegion,
    accumulate_frame,
    find_regions,
    smooth_and_binarize,
    tag_detections,
)


def _make_detections(n, cx, cy, conf, bbox_diag=30.0):
    meas = np.array([[cx, cy, 0.0]] * n, dtype=np.float32)
    confidences = np.array([conf] * n, dtype=np.float32)
    sizes = np.array([bbox_diag**2] * n, dtype=np.float32)
    return meas, confidences, sizes


def test_accumulate_frame_empty():
    grid = np.zeros((64, 64), dtype=np.float32)
    meas = np.zeros((0, 3), dtype=np.float32)
    confs = np.zeros(0, dtype=np.float32)
    sizes = np.zeros(0, dtype=np.float32)
    result = accumulate_frame(grid, meas, confs, sizes, sigma_scale=1.0)
    assert result.max() == 0.0


def test_accumulate_frame_low_confidence_is_strong():
    h, w = 64, 64
    cx, cy = 32, 32
    meas_hi = np.array([[cx, cy, 0.0]], dtype=np.float32)
    meas_lo = np.array([[cx, cy, 0.0]], dtype=np.float32)
    confs_hi = np.array([0.95], dtype=np.float32)
    confs_lo = np.array([0.10], dtype=np.float32)
    sizes = np.array([900.0], dtype=np.float32)
    grid_hi = accumulate_frame(
        np.zeros((h, w), dtype=np.float32), meas_hi, confs_hi, sizes, sigma_scale=1.0
    )
    grid_lo = accumulate_frame(
        np.zeros((h, w), dtype=np.float32), meas_lo, confs_lo, sizes, sigma_scale=1.0
    )
    assert grid_lo[cy, cx] > grid_hi[cy, cx]


def test_accumulate_frame_high_confidence_near_zero():
    h, w = 64, 64
    grid = np.zeros((h, w), dtype=np.float32)
    meas = np.array([[32, 32, 0.0]], dtype=np.float32)
    confs = np.array([0.99], dtype=np.float32)
    sizes = np.array([900.0], dtype=np.float32)
    result = accumulate_frame(grid, meas, confs, sizes, sigma_scale=1.0)
    assert result.max() < 0.05


def test_smooth_and_binarize_returns_binary():
    frames = np.random.rand(10, 32, 32).astype(np.float32)
    binary = smooth_and_binarize(frames, temporal_sigma=1.0, threshold=0.3)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 1})


def test_smooth_and_binarize_shape():
    frames = np.zeros((20, 48, 64), dtype=np.float32)
    binary = smooth_and_binarize(frames, temporal_sigma=2.0, threshold=0.3)
    assert binary.shape == (20, 48, 64)


def test_smooth_and_binarize_matches_full_volume_thresholding():
    frames = np.random.default_rng(0).random((12, 24, 16), dtype=np.float32)
    smoothed = gaussian_filter(frames, sigma=(1.5, 0.0, 0.0))
    raw_threshold = 0.35 * float(smoothed.max())
    expected = (smoothed >= raw_threshold).astype(np.uint8)

    binary = smooth_and_binarize(frames, temporal_sigma=1.5, threshold=0.35)

    assert np.array_equal(binary, expected)


def test_find_regions_empty():
    binary = np.zeros((10, 32, 32), dtype=np.uint8)
    regions = find_regions(binary, frame_h=32, frame_w=32)
    assert regions == []


def test_find_regions_single_blob():
    binary = np.zeros((10, 64, 64), dtype=np.uint8)
    binary[2:5, 10:20, 10:20] = 1
    regions = find_regions(binary, frame_h=64, frame_w=64)
    assert len(regions) == 1
    r = regions[0]
    assert r.label == "region-1"
    assert r.frame_start <= 2
    assert r.frame_end >= 4
    assert isinstance(r.pixel_bbox, tuple) and len(r.pixel_bbox) == 4


def test_find_regions_two_blobs():
    binary = np.zeros((20, 64, 64), dtype=np.uint8)
    binary[0:3, 0:10, 0:10] = 1
    binary[15:18, 50:64, 50:64] = 1
    regions = find_regions(binary, frame_h=64, frame_w=64)
    assert len(regions) == 2


def test_tag_detections_labels_correctly():
    regions = [
        DensityRegion(
            label="region-1",
            frame_start=5,
            frame_end=15,
            pixel_bbox=(10, 10, 50, 50),
        )
    ]
    inside = {"frame": 10, "cx": 30.0, "cy": 30.0}
    outside_frame = {"frame": 20, "cx": 30.0, "cy": 30.0}
    outside_pos = {"frame": 10, "cx": 5.0, "cy": 5.0}
    assert tag_detections([inside], regions)[0]["region_label"] == "region-1"
    assert tag_detections([outside_frame], regions)[0]["region_label"] == "open_field"
    assert tag_detections([outside_pos], regions)[0]["region_label"] == "open_field"


def test_density_region_is_boundary():
    r = DensityRegion(
        label="region-1",
        frame_start=10,
        frame_end=20,
        pixel_bbox=(0, 0, 100, 100),
    )
    assert r.is_boundary_frame(10, margin=2)
    assert r.is_boundary_frame(20, margin=2)
    assert not r.is_boundary_frame(15, margin=2)
