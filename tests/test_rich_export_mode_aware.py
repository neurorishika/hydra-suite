import os

import pandas as pd

from hydra_suite.core.post import rich_export
from hydra_suite.core.post.rich_export import relink_and_export_rich_csv
from hydra_suite.core.post.trajectory_writer import (
    user_tracks_path,
    write_final_trajectories,
)


def _rich_df():
    return pd.DataFrame(
        {
            "TrajectoryID": [1, 1],
            "FrameID": [0, 1],
            "X": [1.0, 2.0],
            "Y": [3.0, 4.0],
            "Theta": [0.0, 0.0],
            "State": ["active", "active"],
            "DetectionConfidence": [0.9, 0.8],
        }
    )


def test_user_mode_returns_the_clean_tracks_csv(tmp_path):
    """User mode's deliverable is `<stem>_tracks.csv`, in the clean schema.

    The rich `_with_individual.csv` is ALSO written here, but as a short-lived
    intermediate: `media_export.load_video_trajectories` is the only reader
    that can supply the annotated video with identity labels, identity-stable
    colours and the pose overlay, and it needs those columns on disk. The
    session deletes it during User-mode cleanup once the dataset / media /
    annotated-video stages have run (see `_user_mode_intermediate_paths`,
    which already enumerates it), exactly as it does for the base
    `_final.csv`. So the invariant this test guards is what SURVIVES a run --
    asserted end-to-end in test_session_user_mode_cleanup.py -- not what the
    low-level writer touches.
    """
    final_csv = str(tmp_path / "clip_final.csv")
    path = write_final_trajectories(_rich_df(), final_csv, debug_mode=False, fps=10.0)
    assert path.endswith("_tracks.csv")
    assert os.path.exists(path)
    cols = pd.read_csv(path).columns.tolist()
    assert cols == [
        "id",
        "frame",
        "time_s",
        "x",
        "y",
        "heading_deg",
        "state",
        "detection_confidence",
    ]


def test_debug_mode_writes_with_individual(tmp_path):
    final_csv = str(tmp_path / "clip_final.csv")
    path = write_final_trajectories(_rich_df(), final_csv, debug_mode=True, fps=10.0)
    assert path.endswith("_with_individual.csv")
    assert os.path.exists(path)
    assert not os.path.exists(str(tmp_path / "clip_tracks.csv"))


def _base_final_df():
    return pd.DataFrame(
        {
            "TrajectoryID": [1, 1, 2, 2],
            "FrameID": [0, 1, 0, 1],
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [5.0, 6.0, 7.0, 8.0],
            "Theta": [0.0, 0.0, 0.0, 0.0],
            "State": ["active"] * 4,
        }
    )


def test_relink_without_pose_refreshes_user_mode_tracks_csv(tmp_path, monkeypatch):
    """Finding #3: when relinking runs but produces no pose-augmented frame,
    the else-branch of relink_and_export_rich_csv must still refresh the
    clean User-mode tracks.csv from the relinked (post-relink ID) base CSV,
    not leave the stale pre-relink tracks.csv from the earlier export_rich
    call in place.
    """
    final_csv = str(tmp_path / "clip_final.csv")
    _base_final_df().to_csv(final_csv, index=False)

    # Simulate the earlier `_export_rich` call having already written a
    # tracks.csv with the PRE-relink TrajectoryIDs (1, 2).
    stale_tracks = user_tracks_path(final_csv)
    pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "frame": [0, 1, 0, 1],
            "time_s": [0.0, 0.1, 0.0, 0.1],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 6.0, 7.0, 8.0],
            "heading_deg": [0.0, 0.0, 0.0, 0.0],
            "state": ["active"] * 4,
            "detection_confidence": [None] * 4,
        }
    ).to_csv(stale_tracks, index=False)

    # No pose-augmented frame available -> with_pose_df is None.
    monkeypatch.setattr(
        rich_export, "build_rich_export_dataframe", lambda *a, **k: None
    )

    # Simulate relinking assigning NEW TrajectoryIDs (100, 200) to the base df.
    def _fake_relink(df, params):
        relinked = df.copy()
        relinked["TrajectoryID"] = relinked["TrajectoryID"].map({1: 100, 2: 200})
        return relinked

    monkeypatch.setattr(
        "hydra_suite.core.post.processing.relink_trajectories_with_pose",
        _fake_relink,
    )

    result = relink_and_export_rich_csv(
        final_csv,
        state=object(),
        params={"FINAL_INTERPOLATION_MAX_GAP": 10},
        min_valid_conf=0.2,
        ignore_keypoints=None,
        debug_mode=False,
        fps=10.0,
        identity_ran=False,
    )

    # else-branch: no pose-augmented rich export happened, so the function
    # falls back to returning the (relinked) final CSV path itself.
    assert result == final_csv

    # The final CSV itself was rewritten with the relinked IDs. (Task 6:
    # identity resolution -- including sort_trajectories_by_identity's
    # renumbering to sequential ids -- now always runs once after relinking,
    # even on this no-pose-data path, so the relinked (100, 200) fragment
    # ids are further canonicalized to sequential ids by the time they are
    # written. The important invariant this test guards is not the exact id
    # VALUES but that the two trajectories survive and the final CSV and
    # tracks.csv agree on whatever ids were assigned.)
    rewritten = pd.read_csv(final_csv)
    rewritten_ids = set(rewritten["TrajectoryID"].unique())
    assert len(rewritten_ids) == 2

    # The clean tracks.csv must now agree -- refreshed from the relinked base,
    # not left stale with the pre-relink IDs (1, 2).
    tracks = pd.read_csv(stale_tracks)
    assert set(tracks["id"].unique()) == rewritten_ids
