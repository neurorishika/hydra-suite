"""User-mode annotated videos must label tracks by identity, not TrajectoryID.

``write_final_trajectories`` is an either/or: Debug writes
``<stem>_final_with_individual.csv``, User writes only ``<stem>_tracks.csv``.
But ``media_export.load_video_trajectories`` looks for the rich CSV first and
otherwise falls back to the *base* final CSV -- which carries neither the
identity columns nor the ``PoseKpt_*`` columns. So in User mode the annotated
video silently lost identity labels, identity-stable colors and the pose
overlay, and every track was labelled with its TrajectoryID.

The rich CSV is already enumerated as a User-mode intermediate to delete
(``_user_mode_intermediate_paths``), and cleanup runs strictly *after* the
media/video stages -- so writing it in User mode too is what the surrounding
design already assumes.
"""

import pandas as pd

import hydra_suite.core.individual.identity.columns as C
from hydra_suite.core.post.media_export import (
    build_video_track_color_key_array,
    build_video_track_label_array,
    load_video_trajectories,
)
from hydra_suite.core.post.rich_export import rich_export_path
from hydra_suite.core.post.trajectory_writer import (
    user_tracks_path,
    write_final_trajectories,
)


def _rich_df():
    """A resolved rich frame: two tracks identified, one genuinely unknown."""
    return pd.DataFrame(
        {
            "TrajectoryID": [0, 1, 2],
            "X": [10.0, 20.0, 30.0],
            "Y": [5.0, 6.0, 7.0],
            "Theta": [0.0, 1.0, 2.0],
            "FrameID": [0, 0, 0],
            "State": ["tracked"] * 3,
            C.FINAL_LABEL: ["green_yellow", "blue_orange", "unknown"],
            C.FINAL_ID: [1, 2, 0],
            C.FINAL_CONFIDENCE: [0.9, 0.8, 0.0],
            C.FINAL_SOURCE: ["cnn", "cnn", ""],
            "PoseKpt_head_X": [11.0, 21.0, 31.0],
            "PoseKpt_head_Y": [5.5, 6.5, 7.5],
            "PoseKpt_head_Conf": [0.9, 0.9, 0.9],
        }
    )


def test_user_mode_writes_rich_csv_for_the_video(tmp_path):
    final_csv = str(tmp_path / "clip_final.csv")
    out = write_final_trajectories(
        _rich_df(), final_csv, debug_mode=False, fps=25.0, identity_ran=True
    )

    # The clean deliverable is still the contract's return value.
    assert out == user_tracks_path(final_csv)
    import os

    assert os.path.exists(out), "User-mode clean tracks.csv must still be written"
    assert os.path.exists(
        rich_export_path(final_csv)
    ), "rich CSV must exist so the annotated video can read identity + pose"


def test_video_loader_finds_identity_in_user_mode(tmp_path):
    final_csv = str(tmp_path / "clip_final.csv")
    # The base final CSV the renderer would otherwise fall back to: no
    # identity columns, no pose columns.
    pd.DataFrame(
        {
            "TrajectoryID": [0, 1, 2],
            "X": [10.0, 20.0, 30.0],
            "Y": [5.0, 6.0, 7.0],
            "Theta": [0.0, 1.0, 2.0],
            "FrameID": [0, 0, 0],
        }
    ).to_csv(final_csv, index=False)

    write_final_trajectories(
        _rich_df(), final_csv, debug_mode=False, fps=25.0, identity_ran=True
    )

    df, chosen = load_video_trajectories(final_csv)
    assert chosen == rich_export_path(final_csv), (
        "load_video_trajectories fell back to the base final CSV, which has no "
        "identity columns -> every label degrades to the TrajectoryID"
    )

    labels = build_video_track_label_array(df)
    assert "green_yellow" in str(labels[0])
    assert "blue_orange" in str(labels[1])
    # An 'unknown' identity must fall back to the TrajectoryID, per the
    # requested behavior.
    assert str(labels[2]).strip() == "ID2"


def test_user_mode_colors_are_identity_stable(tmp_path):
    final_csv = str(tmp_path / "clip_final.csv")
    write_final_trajectories(
        _rich_df(), final_csv, debug_mode=False, fps=25.0, identity_ran=True
    )
    df, _ = load_video_trajectories(final_csv)

    keys = build_video_track_color_key_array(df)
    assert keys[0].startswith("identity:")
    assert keys[1].startswith("identity:")
    assert keys[2] == "trajectory:2"


def test_debug_mode_is_unchanged(tmp_path):
    """Debug must keep writing exactly the rich CSV and no tracks.csv."""
    import os

    final_csv = str(tmp_path / "clip_final.csv")
    out = write_final_trajectories(
        _rich_df(), final_csv, debug_mode=True, fps=25.0, identity_ran=True
    )
    assert out == rich_export_path(final_csv)
    assert os.path.exists(out)
    assert not os.path.exists(
        user_tracks_path(final_csv)
    ), "Debug mode must not emit the User-mode clean CSV"
