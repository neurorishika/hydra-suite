"""On an `interp_lookup` miss, `_build_frame_bundles` must fall back to the
tracking CSV's own X/Y/Theta for that (frame_id, trajectory_id) row instead
of silently dropping the frame -- the sidecar cache missing an entry does not
mean the CSV itself lacks geometry for it."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry, ClippingStats
from hydra_suite.core.individual.dataset.oriented_video import (
    OrientedTrackVideoExporter,
)


def _make_exporter() -> OrientedTrackVideoExporter:
    # Bypass __init__ (heavy: paths, video encoders, etc.) and hand-set only
    # the attributes _build_frame_bundles / _build_task / _canonical_affine_
    # for_task actually touch.
    exporter = OrientedTrackVideoExporter.__new__(OrientedTrackVideoExporter)
    exporter._geometry = CanonicalGeometry.from_reference(
        reference_body_px=20.0,
        aspect_ratio=2.2,
        margin=1.6,
    )
    exporter._clipping_stats = ClippingStats()
    exporter.detection_cache_path = Path("/nonexistent/detection_cache.npz")
    exporter.fix_direction_flips = False
    exporter.enable_affine_stabilization = False
    return exporter


def test_interp_lookup_miss_falls_back_to_csv_geometry_instead_of_dropping():
    exporter = _make_exporter()

    trajectories_df = pd.DataFrame(
        {
            "FrameID": [0],
            "TrajectoryID": [1],
            "DetectionID": [np.nan],  # interpolated row, not an actual detection
            "X": [100.0],
            "Y": [50.0],
            "Theta": [0.25],
        }
    )

    # interp_lookup is empty: the sidecar has no record for (frame_id=0, traj=1),
    # but the CSV row itself carries valid, non-NaN X/Y/Theta.
    frame_bundles, track_sizes, n_missing = exporter._build_frame_bundles(
        trajectories_df, interp_lookup={}
    )

    assert exporter._last_missing_breakdown["missing_interpolated_rows"] == 0
    assert n_missing == 0
    assert (
        0 in frame_bundles
    ), "frame 0 was dropped despite the CSV having valid geometry"
    tasks = frame_bundles[0].tasks
    assert len(tasks) == 1
    task = tasks[0]
    assert task.frame_id == 0
    assert task.trajectory_id == 1
    assert task.center_x == 100.0
    assert task.center_y == 50.0
    assert 1 in track_sizes

    # Width/height regression guard: derived from CanonicalGeometry.from_
    # reference's own identity (canvas_w = major_axis * margin, since
    # major_axis = body * sqrt(ar) is already baked into canvas_w -- see
    # the docstring on CanonicalGeometry and the comment in
    # _fallback_interp_record). Expressed independently of that method's
    # internals here (not just re-deriving the same formula) by checking
    # the geometric mean of width/height recovers the fixture's
    # reference_body_px=20.0 (within the canvas's integer-pixel _even()
    # rounding), and that the major/minor ratio matches aspect_ratio=2.2.
    geometry = exporter._geometry
    assert task.width == pytest.approx(geometry.canvas_w / geometry.margin, rel=1e-9)
    assert task.height == pytest.approx(task.width / geometry.aspect_ratio, rel=1e-9)
    recovered_body = math.sqrt(task.width * task.height)
    assert recovered_body == pytest.approx(20.0, rel=0.02)
    assert task.width / task.height == pytest.approx(geometry.aspect_ratio, rel=1e-9)

    # Corners: a valid (4, 2) OBB centered on (cx, cy).
    assert task.corners.shape == (4, 2)
    assert np.mean(task.corners[:, 0]) == pytest.approx(100.0, abs=1e-3)
    assert np.mean(task.corners[:, 1]) == pytest.approx(50.0, abs=1e-3)


def test_interp_lookup_miss_with_nan_csv_geometry_still_reports_missing():
    exporter = _make_exporter()

    trajectories_df = pd.DataFrame(
        {
            "FrameID": [0],
            "TrajectoryID": [1],
            "DetectionID": [np.nan],
            "X": [np.nan],
            "Y": [np.nan],
            "Theta": [np.nan],
        }
    )

    frame_bundles, _track_sizes, _n = exporter._build_frame_bundles(
        trajectories_df, interp_lookup={}
    )

    # No usable geometry anywhere (neither the sidecar nor the CSV) -- the
    # frame is correctly dropped, same as before this fix.
    assert 0 not in frame_bundles
