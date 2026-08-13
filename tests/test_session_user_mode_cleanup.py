import os

import pandas as pd

from hydra_suite.core.tracking.session import (
    TrackingSessionCore,
    _user_mode_intermediate_paths,
)


def test_intermediate_paths_enumerated():
    paths = _user_mode_intermediate_paths(base="/out/clip", ext=".csv")
    assert "/out/clip_final.csv" in paths
    assert "/out/clip_forward.csv" in paths
    assert "/out/clip_backward.csv" in paths
    assert "/out/clip_forward_processed.csv" in paths
    assert "/out/clip_final_with_individual.csv" in paths
    # the clean deliverable must NOT be in the delete set
    assert "/out/clip_tracks.csv" not in paths


def _write_raw_csv(path):
    pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 0],
            "X": [10.0, 11.0, 12.0],
            "Y": [5.0, 5.0, 5.0],
            "Theta": [0.0, 0.0, 0.0],
            "FrameID": [0, 1, 2],
            "State": ["tracked"] * 3,
        }
    ).to_csv(path, index=False)


def _config():
    return {
        "enable_postprocessing": False,
        "interpolation_method": "none",
        "interpolation_max_gap_seconds": 1.0,
        "heading_flip_max_burst": 5,
        "enable_backward_tracking": False,
        "individual_interpolate_occlusions": False,
    }


def test_user_mode_cleanup_skipped_when_tracks_csv_missing(tmp_path, monkeypatch):
    """If the clean tracks.csv was never produced (e.g. rich export returned
    None on an empty frame), the User-mode cleanup must NOT delete the
    intermediates -- otherwise the run leaves zero deliverables behind.
    """
    raw = tmp_path / "clip.csv"
    _write_raw_csv(str(raw))

    # Simulate export_rich_csv failing to produce a tracks.csv (returns None,
    # writes nothing) -- this is the failure mode the guard protects against.
    monkeypatch.setattr(
        "hydra_suite.core.tracking.session.export_rich_csv",
        lambda *a, **k: None,
    )

    core = TrackingSessionCore(
        video_path=str(tmp_path / "clip.mp4"),
        config=_config(),
        params={
            "FPS": 30.0,
            "RESIZE_FACTOR": 1.0,
            "MIN_TRAJECTORY_LENGTH": 1,
            "DEBUG_MODE": False,
        },
        paths={
            "raw_csv_path": str(raw),
            "detection_cache_path": str(tmp_path / "d.npz"),
        },
    )
    result = core.run_post_tracking(pd.read_csv(str(raw)))

    assert result.success is True
    final_csv = result.final_csv_path
    assert final_csv is not None
    # The guard should have skipped deletion: the base-final intermediate
    # (which run_post_tracking wrote and would otherwise be removed) is
    # still present because the expected clean tracks.csv never landed.
    assert os.path.exists(final_csv)
    assert not os.path.exists(
        str(tmp_path / "clip_tracks.csv")
    ), "tracks.csv should not exist in this simulated failure"
