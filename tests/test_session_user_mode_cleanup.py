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


def test_user_mode_run_leaves_only_the_clean_tracks_csv(tmp_path, monkeypatch):
    """End-to-end invariant: the rich CSV is an intermediate, not a deliverable.

    User mode writes `_final_with_individual.csv` so the annotated-video
    exporter can read identity + pose out of it, then deletes it during
    cleanup. This asserts what actually SURVIVES the run -- the guarantee
    users see -- rather than what the low-level writer touches.
    """
    raw = tmp_path / "clip.csv"
    _write_raw_csv(str(raw))

    # The bare fixture has no analysis source, so rich export short-circuits
    # before the writer. Supply a resolved rich frame so the REAL
    # write_final_trajectories + User-mode cleanup path is what gets tested.
    def _fake_rich(final_csv_path, *a, **k):
        base = pd.read_csv(final_csv_path)
        base["IdentityFinalLabel"] = "green_yellow"
        return base

    monkeypatch.setattr(
        "hydra_suite.core.post.rich_export.build_rich_export_dataframe", _fake_rich
    )

    # Spy on the rich writer so this test proves the file was CREATED and then
    # removed -- without it the assertions below would also pass in the broken
    # world where User mode never writes the rich CSV at all, and the
    # annotated video silently loses identity + pose.
    import hydra_suite.core.post.rich_export as _rx

    _real_write_rich = _rx.write_rich_export_csv
    rich_writes = []

    def _spy_write_rich(df, final_csv_path):
        out = _real_write_rich(df, final_csv_path)
        rich_writes.append(out)
        return out

    monkeypatch.setattr(_rx, "write_rich_export_csv", _spy_write_rich)

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

    assert rich_writes, (
        "User mode never wrote the rich CSV -- the annotated video would fall "
        "back to the base final CSV and lose identity labels and pose"
    )
    assert (
        os.path.exists(str(rich_writes[0])) is False
    ), "the rich CSV must not survive cleanup"

    tracks = tmp_path / "clip_tracks.csv"
    assert os.path.exists(tracks), "the clean deliverable must survive"
    assert not os.path.exists(
        tmp_path / "clip_final_with_individual.csv"
    ), "the rich CSV is an intermediate and must be cleaned up"
    assert not os.path.exists(
        tmp_path / "clip_final.csv"
    ), "the base final CSV is an intermediate and must be cleaned up"
