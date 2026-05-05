# tests/test_correction_writer.py
import tempfile
from pathlib import Path

import pandas as pd

from hydra_suite.refinekit.core.correction_writer import (
    CorrectionWriter,
    apply_split_and_swap,
)
from hydra_suite.refinekit.core.track_editor_model import TrackEditorModel


def _make_two_track_df():
    frames = list(range(1, 21))
    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * 20 + [2] * 20,
            "FrameID": frames + frames,
            "X": list(range(20)) + list(range(100, 120)),
            "Y": [0.0] * 40,
            "Theta": [0.0] * 40,
            "State": ["active"] * 40,
        }
    )
    return df


def test_apply_split_creates_new_segments():
    """Split at frame 10 creates 4 trajectory segments from 2."""
    df = _make_two_track_df()
    result = apply_split_and_swap(
        df=df,
        track_a=1,
        track_b=2,
        split_frame=10,
        swap_post=True,
    )
    tids = sorted(result["TrajectoryID"].unique())
    assert len(tids) == 4


def test_apply_split_no_row_loss():
    """All rows are preserved after split."""
    df = _make_two_track_df()
    result = apply_split_and_swap(df, 1, 2, split_frame=10, swap_post=True)
    assert len(result) == len(df)


def test_apply_split_no_swap():
    """With swap_post=False identities are not exchanged."""
    df = _make_two_track_df()
    result = apply_split_and_swap(df, 1, 2, split_frame=10, swap_post=False)
    # Track 1's pre-split rows stay as track 1
    pre_1 = result[(result["FrameID"] < 10) & (result["TrajectoryID"] == 1)]
    assert len(pre_1) == 9  # frames 1-9


def test_apply_split_swap_exchanges_post_ids():
    """With swap_post=True, post-split segments get swapped IDs."""
    df = _make_two_track_df()
    result = apply_split_and_swap(df, 1, 2, split_frame=10, swap_post=True)
    # Post-split segment that was track 1 should now be labeled with track 2's ID family
    # Post-split segment that was track 2 should now be labeled with track 1's ID family
    post = result[result["FrameID"] >= 10]
    post_ids = sorted(post["TrajectoryID"].unique())
    pre = result[result["FrameID"] < 10]
    pre_ids = sorted(pre["TrajectoryID"].unique())
    # Pre-split keeps original IDs (1, 2), post-split gets new IDs
    assert set(pre_ids) == {1, 2}
    assert len(post_ids) == 2
    assert set(post_ids) != {1, 2}  # new IDs assigned


def test_correction_writer_creates_proofread_copy():
    """CorrectionWriter.open creates _proofread.csv if it doesn't exist."""
    df = _make_two_track_df()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "test_tracked.csv"
        df.to_csv(src, index=False)
        writer = CorrectionWriter(src)
        writer.open()
        assert writer.proofread_path.exists()
        writer.close()


def test_correction_writer_does_not_overwrite_existing():
    """If _proofread.csv already exists it is NOT overwritten on open."""
    df = _make_two_track_df()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "test_tracked.csv"
        df.to_csv(src, index=False)
        proofread = Path(tmp) / "test_tracked_proofread.csv"
        proofread.write_text("existing content")
        writer = CorrectionWriter(src)
        writer.open()
        assert proofread.read_text() == "existing content"
        writer.close()


def test_correction_writer_apply_correction():
    """apply_correction writes updated CSV with new trajectory IDs."""
    df = _make_two_track_df()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "test_tracked.csv"
        df.to_csv(src, index=False)
        writer = CorrectionWriter(src)
        writer.open()
        writer.apply_correction(track_a=1, track_b=2, split_frame=10, swap_post=True)
        result = pd.read_csv(writer.proofread_path)
        assert len(result["TrajectoryID"].unique()) == 4
        writer.close()


def test_correction_writer_proofread_path_naming():
    """Proofread path follows <stem>_proofread.csv convention."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "fly_run_01_with_pose.csv"
        pd.DataFrame({"TrajectoryID": [1], "FrameID": [1], "X": [0], "Y": [0]}).to_csv(
            src, index=False
        )
        writer = CorrectionWriter(src)
        assert writer.proofread_path.name == "fly_run_01_with_pose_proofread.csv"


def test_track_editor_delete_applies_to_full_contiguous_segment():
    """Deleting a fragment from a clipped editor window removes the full segment."""
    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * 21 + [2] * 21,
            "FrameID": list(range(21)) + list(range(21)),
            "X": [0.0] * 42,
            "Y": [0.0] * 42,
        }
    )

    model = TrackEditorModel(df, [1, 2], (5, 10))
    frag = next(frag for frag in model.fragments if frag.track_id == 1)
    assert (frag.frame_start, frag.frame_end) == (0, 20)
    assert model.delete(frag.frag_id)
    ops = model.compute_ops()

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "test_tracked.csv"
        df.to_csv(src, index=False)
        writer = CorrectionWriter(src)
        writer.open()
        writer.apply_edit_ops(ops)
        result = writer.df

    assert 1 not in result["TrajectoryID"].unique()
    assert set(result["TrajectoryID"].unique()) == {2}


def test_track_editor_reassign_applies_to_full_contiguous_segment():
    """Reassigning from a clipped editor window relabels the full segment."""
    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * 21 + [2] * 5,
            "FrameID": list(range(21)) + list(range(30, 35)),
            "X": [0.0] * 26,
            "Y": [0.0] * 26,
        }
    )

    model = TrackEditorModel(df, [1, 2], (5, 10))
    frag = next(frag for frag in model.fragments if frag.track_id == 1)
    assert model.reassign(frag.frag_id, 2)
    ops = model.compute_ops()

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "test_tracked.csv"
        df.to_csv(src, index=False)
        writer = CorrectionWriter(src)
        writer.open()
        writer.apply_edit_ops(ops)
        result = writer.df

    reassigned = result[result["FrameID"].between(0, 20)].sort_values("FrameID")
    assert set(reassigned["TrajectoryID"].unique()) == {2}
