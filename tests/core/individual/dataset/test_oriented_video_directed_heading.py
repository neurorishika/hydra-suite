"""A heading whose head-vs-tail direction is known must be used as-is for the
canonical crop. Collapsing it onto the branch nearest the previous frame's
theta is only right for an undirected body axis: on a directed track, one
legitimate >90 degree frame-to-frame turn flips every later crop of that track
head-to-tail."""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry, ClippingStats
from hydra_suite.core.individual.dataset.oriented_video import (
    OrientedTrackVideoExporter,
)


def _make_exporter() -> OrientedTrackVideoExporter:
    # Bypass __init__ (paths, encoders) like test_oriented_video_interp_fallback.
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


def _thetas_for_track(rows: dict) -> list[float]:
    """Run two rows of one track through _build_frame_bundles; return task thetas."""
    exporter = _make_exporter()
    df = pd.DataFrame(
        {
            "FrameID": [0, 1],
            "TrajectoryID": [1, 1],
            "DetectionID": [np.nan, np.nan],
            "X": [100.0, 102.0],
            "Y": [50.0, 50.0],
            **rows,
        }
    )
    frame_bundles, _, _ = exporter._build_frame_bundles(df, interp_lookup={})
    return [frame_bundles[f].tasks[0].theta for f in (0, 1)]


def _angle_gap(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def test_directed_row_keeps_its_heading_across_a_sharp_turn():
    thetas = _thetas_for_track({"Theta": [0.0, 2.0], "HeadingIsDirected": [True, True]})

    assert _angle_gap(thetas[1], 2.0) < 1e-6


def test_undirected_row_still_collapses_to_the_previous_branch():
    thetas = _thetas_for_track(
        {"Theta": [0.0, 2.0], "HeadingIsDirected": [False, False]}
    )

    assert _angle_gap(thetas[1], 2.0 - math.pi) < 1e-6


def test_rows_without_a_direction_column_keep_the_branch_collapse():
    thetas = _thetas_for_track({"Theta": [0.0, 2.0]})

    assert _angle_gap(thetas[1], 2.0 - math.pi) < 1e-6


def test_direction_flag_is_read_from_the_rich_csv_beside_the_final_csv(
    tmp_path: Path,
):
    # In a real session the base final CSV has no HeadingIsDirected column;
    # only the rich `_with_individual` export beside it does.
    keys = {"TrajectoryID": [1, 1], "FrameID": [0, 1]}
    pd.DataFrame(
        {**keys, "X": [100.0, 102.0], "Y": [50.0, 50.0], "Theta": [0.0, 2.0]}
    ).to_csv(tmp_path / "tracks_final.csv", index=False)
    pd.DataFrame({**keys, "HeadingIsDirected": [True, True]}).to_csv(
        tmp_path / "tracks_final_with_individual.csv", index=False
    )
    exporter = OrientedTrackVideoExporter(
        tmp_path / "individual_crops" / "run",
        tmp_path / "tracks_final.csv",
        video_path=tmp_path / "source.mp4",
        detection_cache_path=tmp_path / "detection.npz",
        fps=5.0,
    )

    df = exporter._load_final_dataframe()
    frame_bundles, _, _ = exporter._build_frame_bundles(df, interp_lookup={})

    assert _angle_gap(frame_bundles[1].tasks[0].theta, 2.0) < 1e-6


def test_directed_cache_heading_is_not_collapsed_to_the_reference():
    row = next(pd.DataFrame([{"Theta": 0.0}]).itertuples(index=False))

    theta = OrientedTrackVideoExporter._resolve_task_theta(
        row,
        0.0,
        reference_theta=0.0,
        directed_heading=2.0,
        directed=True,
    )

    assert _angle_gap(theta, 2.0) < 1e-6
