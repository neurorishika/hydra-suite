# tests/test_confidence_density.py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import hydra_suite.core.tracking.confidence.confidence_density as density_module
from hydra_suite.core.tracking.confidence.confidence_density import (
    DEFAULT_AUTOTUNE_DENSITY_MAX_BYTES,
    ConfidenceDensityCancelled,
    DensityRegion,
    DensityReplayBudgetExceeded,
    accumulate_frame,
    admit_density_map_working_set,
    compute_density_map_from_cache,
    estimate_density_map_working_set,
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


def test_density_cache_regions_keep_absolute_frames_and_break_missing_keys():
    """Sparse cache keys must not be compressed into a false temporal bridge."""

    detections = _make_detections(1, 8.0, 8.0, 0.0, bbox_diag=4.0)
    cache = {frame: detections for frame in (100, 101, 102, 104, 105)}

    density_map, raw_grids = compute_density_map_from_cache(
        cache,
        frame_h=16,
        frame_w=16,
        sigma_scale=0.5,
        temporal_sigma=0.0,
        threshold=0.5,
        downsample_factor=1,
        min_frame_duration=2,
        min_area_px=1,
    )

    assert density_map.frame_indices is not None
    assert density_map.frame_indices.tolist() == [100, 101, 102, 104, 105]
    assert raw_grids.shape[0] == 5
    assert [
        (region.frame_start, region.frame_end) for region in density_map.regions
    ] == [
        (100, 102),
        (104, 105),
    ]


def test_density_cache_large_absolute_gap_stays_sparse_in_memory():
    """Absolute cache keys must never create a max-key-sized density volume."""

    detections = _make_detections(1, 8.0, 8.0, 0.0, bbox_diag=4.0)
    density_map, raw_grids = compute_density_map_from_cache(
        {7: detections, 2_000_007: detections},
        frame_h=16,
        frame_w=16,
        sigma_scale=0.5,
        temporal_sigma=0.0,
        threshold=0.5,
        downsample_factor=1,
        min_frame_duration=1,
        min_area_px=1,
    )

    assert raw_grids.shape == (2, 16, 16)
    assert density_map.binary_volume is not None
    assert density_map.binary_volume.shape == (2, 16, 16)
    assert [
        (region.frame_start, region.frame_end) for region in density_map.regions
    ] == [
        (7, 7),
        (2_000_007, 2_000_007),
    ]


def test_density_admission_uses_sparse_key_count_and_allows_exact_boundary():
    """A high absolute cache key never inflates replay-map admission."""

    estimate = admit_density_map_working_set(
        2,
        16,
        16,
        downsample_factor=1,
        temporal_sigma=0.0,
    )
    detections = _make_detections(1, 8.0, 8.0, 0.0, bbox_diag=4.0)
    cache = {7: detections, 2_000_007: detections}

    density_map, _ = compute_density_map_from_cache(
        cache,
        frame_h=16,
        frame_w=16,
        sigma_scale=0.5,
        temporal_sigma=0.0,
        threshold=0.5,
        downsample_factor=1,
        min_frame_duration=1,
        min_area_px=1,
        max_working_bytes=estimate.peak_bytes,
    )

    assert density_map.frame_indices.tolist() == [7, 2_000_007]
    with pytest.raises(DensityReplayBudgetExceeded) as error:
        compute_density_map_from_cache(
            cache,
            frame_h=16,
            frame_w=16,
            sigma_scale=0.5,
            temporal_sigma=0.0,
            threshold=0.5,
            downsample_factor=1,
            min_frame_duration=1,
            min_area_px=1,
            max_working_bytes=estimate.peak_bytes - 1,
        )
    assert error.value.estimate.frame_count == 2
    assert error.value.max_working_bytes == estimate.peak_bytes - 1


@pytest.mark.parametrize("max_working_bytes", [0, -1])
def test_density_empty_cache_rejects_nonpositive_admission_limit(max_working_bytes):
    """A configured replay budget is validated even when no frames are cached."""

    with pytest.raises(ValueError, match="must be positive"):
        compute_density_map_from_cache(
            {},
            frame_h=16,
            frame_w=16,
            sigma_scale=0.5,
            temporal_sigma=0.0,
            threshold=0.5,
            max_working_bytes=max_working_bytes,
        )


def test_density_admission_refuses_4k_five_minute_read_only_replay():
    """Admission rejects a realistic large replay before any map allocation."""

    estimate = estimate_density_map_working_set(
        frame_count=5 * 60 * 30,
        frame_h=2160,
        frame_w=3840,
        downsample_factor=8,
        temporal_sigma=2.0,
    )

    assert (estimate.grid_h, estimate.grid_w) == (270, 480)
    assert estimate.uses_chunked_smoothing
    assert estimate.peak_bytes > 10 * 1024**3
    with pytest.raises(DensityReplayBudgetExceeded, match="AUTOTUNE_DENSITY_MAX_BYTES"):
        admit_density_map_working_set(
            5 * 60 * 30,
            2160,
            3840,
            downsample_factor=8,
            temporal_sigma=2.0,
            max_working_bytes=DEFAULT_AUTOTUNE_DENSITY_MAX_BYTES,
        )


def test_density_estimate_accounts_for_exact_multi_arena_grid_masks():
    """Arena source-block labels and per-arena masks count toward admission."""

    whole_frame = estimate_density_map_working_set(
        frame_count=50,
        frame_h=101,
        frame_w=100,
        downsample_factor=8,
        temporal_sigma=0.0,
    )
    two_arenas = estimate_density_map_working_set(
        frame_count=50,
        frame_h=101,
        frame_w=100,
        downsample_factor=8,
        temporal_sigma=0.0,
        multi_arena=True,
        arena_count=2,
    )
    four_arenas = estimate_density_map_working_set(
        frame_count=50,
        frame_h=101,
        frame_w=100,
        downsample_factor=8,
        temporal_sigma=0.0,
        multi_arena=True,
        arena_count=4,
    )

    assert whole_frame.arena_mask_bytes == 0
    assert two_arenas.arena_count == 2
    assert four_arenas.arena_count == 4
    assert four_arenas.arena_mask_bytes > two_arenas.arena_mask_bytes > 0
    assert four_arenas.peak_bytes > two_arenas.peak_bytes > whole_frame.peak_bytes


def test_density_compute_cancels_during_accumulation(monkeypatch):
    """Cancellation after one frame stops before another frame is accumulated."""

    cancelled = {"value": False}
    calls = 0
    original = density_module.accumulate_frame

    def _accumulate(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        cancelled["value"] = True
        return result

    monkeypatch.setattr(density_module, "accumulate_frame", _accumulate)
    detections = _make_detections(1, 8.0, 8.0, 0.0, bbox_diag=4.0)
    with pytest.raises(ConfidenceDensityCancelled):
        compute_density_map_from_cache(
            {0: detections, 1: detections},
            frame_h=16,
            frame_w=16,
            sigma_scale=0.5,
            temporal_sigma=0.0,
            threshold=0.5,
            downsample_factor=1,
            should_stop=lambda: cancelled["value"],
        )
    assert calls == 1


def test_density_compute_cancels_during_smoothing(monkeypatch):
    """Cancellation raised after SciPy smoothing never falls through to CC."""

    cancelled = {"value": False}
    original = density_module.gaussian_filter

    def _smoothing(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled["value"] = True
        return result

    monkeypatch.setattr(density_module, "gaussian_filter", _smoothing)
    detections = _make_detections(1, 8.0, 8.0, 0.0, bbox_diag=4.0)
    with pytest.raises(ConfidenceDensityCancelled):
        compute_density_map_from_cache(
            {0: detections, 1: detections},
            frame_h=16,
            frame_w=16,
            sigma_scale=0.5,
            temporal_sigma=1.0,
            threshold=0.5,
            downsample_factor=1,
            should_stop=lambda: cancelled["value"],
        )


def test_density_compute_cancels_inside_first_arena_iteration(monkeypatch):
    """A stop between arena phases prevents the second arena from starting."""

    from hydra_suite.core.tracking.arenas import ArenaLayout

    cancelled = {"value": False}
    smooth_calls = 0

    def _smoothing(frames, *_args, **_kwargs):
        nonlocal smooth_calls
        smooth_calls += 1
        cancelled["value"] = True
        return np.zeros(frames.shape, dtype=np.uint8)

    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[:, :8] = 1
    labels[:, 8:] = 2
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=labels)
    cache = {
        0: _make_detections(1, 4.0, 8.0, 0.0, bbox_diag=4.0),
        1: _make_detections(1, 12.0, 8.0, 0.0, bbox_diag=4.0),
    }
    monkeypatch.setattr(density_module, "smooth_and_binarize", _smoothing)

    with pytest.raises(ConfidenceDensityCancelled):
        compute_density_map_from_cache(
            cache,
            frame_h=16,
            frame_w=16,
            sigma_scale=0.5,
            temporal_sigma=1.0,
            threshold=0.5,
            downsample_factor=1,
            arena_layout=layout,
            should_stop=lambda: cancelled["value"],
        )
    assert smooth_calls == 1


def test_find_regions_cancels_before_connected_component_labeling(monkeypatch):
    """The large int32 label volume is never allocated after cancellation."""

    called = False

    def _label(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("connected-component labeling must not run")

    monkeypatch.setattr(density_module, "label", _label)
    with pytest.raises(ConfidenceDensityCancelled):
        find_regions(
            np.ones((2, 4, 4), dtype=np.uint8),
            frame_h=4,
            frame_w=4,
            should_stop=lambda: True,
        )
    assert not called


@pytest.mark.parametrize("multi_arena", [False, True])
@pytest.mark.parametrize(
    ("grid_bbox", "expected_bbox"),
    [
        ((1, 1, 1, 1), (8, 8, 15, 15)),
        ((4, 4, 4, 4), (32, 32, 32, 32)),
    ],
)
def test_density_grid_bboxes_scale_inclusively_and_cover_remainder_edges(
    monkeypatch, multi_arena, grid_bbox, expected_bbox
):
    """Both density paths preserve inclusive cells and final frame remainders."""

    from hydra_suite.core.tracking.arenas import ArenaLayout

    def _regions(*_args, **_kwargs):
        return [DensityRegion("region-1", 0, 0, grid_bbox)]

    monkeypatch.setattr(density_module, "_find_regions_by_frame_runs", _regions)
    detections = _make_detections(1, 4.0, 4.0, 0.0, bbox_diag=4.0)
    kwargs = {}
    expected_count = 1
    if multi_arena:
        labels = np.zeros((33, 33), dtype=np.uint16)
        labels[:, :16] = 1
        labels[:, 17:] = 2
        kwargs["arena_layout"] = ArenaLayout(
            n_arenas=2, animals_per_arena=1, label_image=labels
        )
        expected_count = 2

    density_map, _ = compute_density_map_from_cache(
        {0: detections},
        frame_h=33,
        frame_w=33,
        sigma_scale=0.5,
        temporal_sigma=0.0,
        threshold=0.5,
        downsample_factor=8,
        min_frame_duration=1,
        min_area_px=1,
        **kwargs,
    )

    assert len(density_map.regions) == expected_count
    assert {region.pixel_bbox for region in density_map.regions} == {expected_bbox}


def test_density_actual_nondivisible_grid_keeps_generating_detection_in_region():
    """Actual density components scale back to include a three-frame source point."""

    detections = _make_detections(1, 82.0, 82.0, 0.0, bbox_diag=4.0)
    density_map, _ = compute_density_map_from_cache(
        {frame: detections for frame in range(3)},
        frame_h=100,
        frame_w=100,
        sigma_scale=0.5,
        temporal_sigma=0.0,
        threshold=0.5,
        downsample_factor=8,
        min_frame_duration=3,
        min_area_px=1,
    )

    assert len(density_map.regions) == 1
    assert all(density_map.regions[0].contains(frame, 82.0, 82.0) for frame in range(3))


def test_density_actual_nondivisible_remainder_edge_is_not_clipped_away():
    """A right-edge source point gets comparable density to an interior point."""

    def _map_at(cx: float, cy: float):
        detections = _make_detections(1, cx, cy, 0.0, bbox_diag=4.0)
        return compute_density_map_from_cache(
            {frame: detections for frame in range(3)},
            frame_h=100,
            frame_w=100,
            sigma_scale=0.5,
            temporal_sigma=0.0,
            threshold=0.5,
            downsample_factor=8,
            min_frame_duration=3,
            min_area_px=1,
        )

    edge_map, edge_raw = _map_at(99.0, 50.0)
    # Compare equal sub-cell phases: 99 / 8 and 51 / 8 both have the same
    # fractional x coordinate, so this only measures edge clipping.
    _interior_map, interior_raw = _map_at(51.0, 50.0)

    assert len(edge_map.regions) == 1
    assert all(edge_map.regions[0].contains(frame, 99.0, 50.0) for frame in range(3))
    assert float(edge_raw.max()) >= 0.95 * float(interior_raw.max())


def test_density_arena_mask_keeps_source_block_straddling_boundary():
    """A ceil-grid cell may serve both arenas when its source block straddles."""

    from hydra_suite.core.tracking.arenas import ArenaLayout

    labels = np.ones((100, 100), dtype=np.uint16)
    labels[:, 50:] = 2
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=labels)
    masks = density_module._arena_grid_masks_from_source_blocks(
        layout,
        frame_h=100,
        frame_w=100,
        grid_h=13,
        grid_w=13,
        ds=8,
    )
    # Cell (6, 6) covers source x/y 48..55 and therefore crosses x=50. It
    # must be retained for both independently tagged arena pipelines.
    assert masks[:, 6, 6].tolist() == [True, True]
    x = np.arange(50, 56, dtype=np.float32)
    detections = (
        np.column_stack([x, np.full_like(x, 50.0), np.zeros_like(x)]).astype(
            np.float32
        ),
        np.zeros(len(x), dtype=np.float32),
        np.full(len(x), 16.0, dtype=np.float32),
    )

    density_map, _ = compute_density_map_from_cache(
        {frame: detections for frame in range(3)},
        frame_h=100,
        frame_w=100,
        sigma_scale=0.5,
        temporal_sigma=0.0,
        threshold=0.1,
        downsample_factor=8,
        min_frame_duration=3,
        min_area_px=1,
        arena_layout=layout,
    )

    arena_two = [region for region in density_map.regions if region.arena == 1]
    assert len(arena_two) == 1
    assert all(arena_two[0].contains(frame, 52.0, 50.0) for frame in range(3))


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
