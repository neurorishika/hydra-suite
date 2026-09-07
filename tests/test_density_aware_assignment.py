"""Tests for the density-aware distance gate helper function."""

import numpy as np
import pytest

from hydra_suite.core.assigners.hungarian import HARD_REJECT_COST, TrackAssigner
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


class _RespawnKF:
    X = np.zeros((1, 5), dtype=np.float32)


def test_proximity_respawn_honors_a_density_hard_preblock():
    """Phase 3 must not recompute around a worker-injected density reject.

    The raw distance is 80 px: it is below the ordinary 100-px respawn radius
    but above a density-tightened 70-px radius.  Before the regression fix,
    `_assign_respawn` ignored the worker's density eligibility and accepted
    the detection anyway.
    """

    assigner = TrackAssigner({"MAX_DISTANCE_THRESHOLD": 100.0})
    rows, cols, rejoined = assigner._assign_respawn(
        cost=np.array([[HARD_REJECT_COST]], dtype=np.float32),
        N=1,
        meas=[np.array([80.0, 0.0, 0.0], dtype=np.float32)],
        track_states=["lost"],
        tracking_continuity=[0],
        kf_manager=_RespawnKF(),
        _MAX_DIST=100.0,
        hard_blocked=np.array([[True]], dtype=bool),
    )

    assert rows == []
    assert cols == []
    assert rejoined == []


def test_proximity_respawn_does_not_treat_raw_distance_cost_as_density_block():
    """The shared 1e9 raw-distance value remains distinct from density policy."""

    assigner = TrackAssigner({"MAX_DISTANCE_THRESHOLD": 100.0})
    rows, cols, _ = assigner._assign_respawn(
        cost=np.array([[HARD_REJECT_COST]], dtype=np.float32),
        N=1,
        meas=[np.array([80.0, 0.0, 0.0], dtype=np.float32)],
        track_states=["lost"],
        tracking_continuity=[0],
        kf_manager=_RespawnKF(),
        _MAX_DIST=100.0,
        hard_blocked=np.array([[False]], dtype=bool),
    )

    assert rows == [0]
    assert cols == [0]


def test_worker_density_hard_block_survives_respawn_and_free_bootstrap(
    monkeypatch, tmp_path
):
    """The real worker must not re-create a rejected Phase-3 match.

    A cold lost slot is 47 px from the repeated density-region detection. It
    is inside the ordinary 55-px respawn radius but outside the 38.5-px
    density radius. The first detection frame runs Phase 3; the second would
    show a regular matched CSV row if either Phase 3 *or* the later
    free-detection bootstrap bypassed the hard cost block.
    """

    import hydra_suite.core.tracking.worker as worker_mod
    from tests.test_arena_bootstrap_gate import _SteadyArena1Runner
    from tests.test_arena_worker_wiring import (
        _bgsub_arena_params,
        _CapturingCSVWriter,
        _FakeProfiler,
        _FakeVideoCapture,
        _matched_rows,
    )

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(worker_mod, "InferenceRunner", _SteadyArena1Runner)

    flagged: list[np.ndarray] = []
    real_flags = worker_mod.get_density_region_flags

    def _record_flags(meas, regions, frame_idx, meas_arena=None):
        values = real_flags(meas, regions, frame_idx, meas_arena=meas_arena)
        flagged.append(values.copy())
        return values

    monkeypatch.setattr(worker_mod, "get_density_region_flags", _record_flags)
    rows = _CapturingCSVWriter()
    worker = worker_mod.TrackingEngineCore(
        str(tmp_path / "density-bootstrap.mp4"),
        on_finished=lambda *_args: None,
        preview_mode=True,
        csv_writer_thread=rows,
    )
    worker.set_parameters(
        _bgsub_arena_params(
            single_arena=True,
            MIN_DETECTION_COUNTS=1,
            MAX_DISTANCE_THRESHOLD=55.0,
            DENSITY_CONSERVATIVE_FACTOR=0.7,
            ENABLE_CONFIDENCE_DENSITY_MAP=True,
        )
    )
    worker._density_regions = [
        DensityRegion("all", frame_start=0, frame_end=9, pixel_bbox=(0, 0, 100, 100))
    ]

    worker.run_tracking()

    assert any(values.tolist() == [True] for values in flagged)
    assert _matched_rows(rows.rows) == [], (
        "a density-hard-blocked detection must stay free rather than respawn "
        "or bootstrap a lost slot"
    )


def test_worker_cold_bootstrap_keeps_far_from_origin_raw_distance_behavior(
    monkeypatch, tmp_path
):
    """A first-frame detection can still initialise past the raw-distance gate.

    Lost KFs start at the origin.  Their 47-px first real detection is outside
    the ordinary 12.5-px tracking gate, so it becomes a 1e9 raw-distance cost;
    the final bootstrap is intentionally what creates its first filter state.
    This pins that behavior separately from density's explicit block mask.
    """

    import hydra_suite.core.tracking.worker as worker_mod
    from tests.test_arena_bootstrap_gate import _SteadyArena1Runner
    from tests.test_arena_worker_wiring import (
        _bgsub_arena_params,
        _CapturingCSVWriter,
        _FakeProfiler,
        _FakeVideoCapture,
        _matched_rows,
    )

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(worker_mod, "InferenceRunner", _SteadyArena1Runner)
    rows = _CapturingCSVWriter()
    worker = worker_mod.TrackingEngineCore(
        str(tmp_path / "cold-bootstrap.mp4"),
        on_finished=lambda *_args: None,
        preview_mode=True,
        csv_writer_thread=rows,
    )
    worker.set_parameters(
        _bgsub_arena_params(
            single_arena=True,
            MIN_DETECTION_COUNTS=1,
            MAX_DISTANCE_THRESHOLD=12.5,
            ENABLE_CONFIDENCE_DENSITY_MAP=False,
        )
    )

    worker.run_tracking()

    assert _matched_rows(rows.rows), (
        "ordinary first-frame raw-distance preblocks must not disable the "
        "intentional lost-slot bootstrap"
    )
