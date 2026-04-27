"""Tests for identity-aware trajectory split/join post-processing."""

from __future__ import annotations

import pandas as pd

from tests.helpers.module_loader import load_src_module

mod = load_src_module(
    "hydra_suite/core/post/identity_postprocess.py",
    "identity_postprocess_under_test",
)

apply_identity_postprocessing = mod.apply_identity_postprocessing
augment_trajectories_with_detected_apriltags = (
    mod.augment_trajectories_with_detected_apriltags
)


class FakeTagCache:
    """Minimal stand-in for detected AprilTag cache reads."""

    def __init__(self, data):
        self._data = data

    def get_frame(self, frame_idx):
        return self._data.get(
            frame_idx,
            {
                "tag_ids": [],
                "centers_xy": [],
                "hammings": [],
            },
        )


def _cnn_params(max_gap: int = 2, scoring_mode: str = "atomic") -> dict:
    return {
        "USE_APRILTAGS": False,
        "IDENTITY_INTERPOLATION_MAX_GAP": max_gap,
        "AGREEMENT_DISTANCE": 3.0,
        "MAX_VELOCITY_BREAK": 5.0,
        "CNN_CLASSIFIERS": [
            {
                "label": "uid",
                "confidence": 0.5,
                "scoring_mode": scoring_mode,
                "unique_identifier": True,
            }
        ],
    }


def test_augment_detected_apriltags_assigns_nearest_tag_per_row() -> None:
    df = pd.DataFrame(
        [
            {"FrameID": 1, "TrajectoryID": 0, "X": 10.0, "Y": 20.0},
            {"FrameID": 1, "TrajectoryID": 1, "X": 100.0, "Y": 100.0},
        ]
    )
    cache = FakeTagCache(
        {
            1: {
                "tag_ids": [5, 7],
                "centers_xy": [[10.5, 19.5], [101.0, 99.0]],
                "hammings": [0, 1],
            }
        }
    )

    out = augment_trajectories_with_detected_apriltags(
        df,
        cache,
        {"TAG_ASSOCIATION_RADIUS": 10.0},
    )

    assert int(out.iloc[0]["DetectedTagID"]) == 5
    assert int(out.iloc[1]["DetectedTagID"]) == 7
    assert float(out.iloc[0]["DetectedTagConf"]) == 1.0
    assert float(out.iloc[1]["DetectedTagConf"]) == 0.5


def test_identity_postprocess_splits_on_stable_identity_switch_and_rejoins() -> None:
    df = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "alpha",
                "CNN_uid_Conf": 0.95,
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 1.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "alpha",
                "CNN_uid_Conf": 0.95,
            },
            {
                "TrajectoryID": 0,
                "FrameID": 2,
                "X": 2.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "beta",
                "CNN_uid_Conf": 0.96,
            },
            {
                "TrajectoryID": 0,
                "FrameID": 3,
                "X": 3.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "beta",
                "CNN_uid_Conf": 0.96,
            },
            {
                "TrajectoryID": 1,
                "FrameID": 4,
                "X": 4.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "beta",
                "CNN_uid_Conf": 0.94,
            },
            {
                "TrajectoryID": 1,
                "FrameID": 5,
                "X": 5.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "beta",
                "CNN_uid_Conf": 0.94,
            },
        ]
    )

    out = apply_identity_postprocessing(df, _cnn_params(max_gap=1))

    early_ids = set(out.loc[out["FrameID"].isin([0, 1]), "TrajectoryID"].unique())
    late_ids = set(out.loc[out["FrameID"].isin([2, 3, 4, 5]), "TrajectoryID"].unique())

    assert len(early_ids) == 1
    assert len(late_ids) == 1
    assert early_ids != late_ids
    assert out.loc[out["FrameID"] == 0, "UniqueIdentityKey"].iloc[0] == "cnn:uid=alpha"
    assert out.loc[out["FrameID"] == 4, "UniqueIdentityKey"].iloc[0] == "cnn:uid=beta"
    assert set(out["OriginalTrajectoryID"].dropna().astype(int).unique()) == {0, 1}


def test_identity_postprocess_interpolates_small_identity_gap_only() -> None:
    df = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "alpha",
                "CNN_uid_Conf": 0.95,
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 1.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "alpha",
                "CNN_uid_Conf": 0.95,
            },
            {
                "TrajectoryID": 1,
                "FrameID": 4,
                "X": 4.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "alpha",
                "CNN_uid_Conf": 0.96,
            },
            {
                "TrajectoryID": 1,
                "FrameID": 5,
                "X": 5.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "alpha",
                "CNN_uid_Conf": 0.96,
            },
        ]
    )

    filled = apply_identity_postprocessing(df, _cnn_params(max_gap=2))
    assert set(filled["FrameID"].tolist()) == {0, 1, 2, 3, 4, 5}
    assert filled[filled["IdentityInterpolated"]]["FrameID"].tolist() == [2, 3]
    assert filled["TrajectoryID"].nunique() == 1

    unfilled = apply_identity_postprocessing(df, _cnn_params(max_gap=1))
    assert set(unfilled["FrameID"].tolist()) == {0, 1, 4, 5}
    assert unfilled["TrajectoryID"].nunique() == 2
    assert set(unfilled["UniqueIdentityKey"].dropna().tolist()) == {"cnn:uid=alpha"}


def test_identity_postprocess_keeps_impossible_motion_fragments_separate() -> None:
    df = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "alpha",
                "CNN_uid_Conf": 0.95,
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 1.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "alpha",
                "CNN_uid_Conf": 0.95,
            },
            {
                "TrajectoryID": 1,
                "FrameID": 2,
                "X": 500.0,
                "Y": 500.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "alpha",
                "CNN_uid_Conf": 0.96,
            },
            {
                "TrajectoryID": 1,
                "FrameID": 3,
                "X": 501.0,
                "Y": 500.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_Class": "alpha",
                "CNN_uid_Conf": 0.96,
            },
        ]
    )

    out = apply_identity_postprocessing(df, _cnn_params(max_gap=5))

    assert out["TrajectoryID"].nunique() == 2
    assert set(out["UniqueIdentityKey"].dropna().tolist()) == {"cnn:uid=alpha"}


def test_identity_postprocess_multihead_per_head_average_rejoins_partial_match() -> (
    None
):
    df = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_color_Class": "red",
                "CNN_uid_color_Conf": 0.95,
                "CNN_uid_shape_Class": "circle",
                "CNN_uid_shape_Conf": 0.91,
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 1.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_color_Class": "red",
                "CNN_uid_color_Conf": 0.95,
                "CNN_uid_shape_Class": "circle",
                "CNN_uid_shape_Conf": 0.91,
            },
            {
                "TrajectoryID": 1,
                "FrameID": 4,
                "X": 4.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_color_Class": "red",
                "CNN_uid_color_Conf": 0.94,
                "CNN_uid_shape_Class": "",
                "CNN_uid_shape_Conf": 0.10,
            },
            {
                "TrajectoryID": 1,
                "FrameID": 5,
                "X": 5.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_color_Class": "red",
                "CNN_uid_color_Conf": 0.94,
                "CNN_uid_shape_Class": "",
                "CNN_uid_shape_Conf": 0.10,
            },
        ]
    )

    out = apply_identity_postprocessing(
        df,
        _cnn_params(max_gap=2, scoring_mode="per_head_average"),
    )

    assert out["TrajectoryID"].nunique() == 1
    assert (
        "cnn:uid:color=red" in out.loc[out["FrameID"] == 0, "UniqueIdentityKey"].iloc[0]
    )


def test_identity_postprocess_multihead_per_head_average_avoids_split_on_mixed_heads() -> (
    None
):
    df = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_color_Class": "red",
                "CNN_uid_color_Conf": 0.95,
                "CNN_uid_shape_Class": "circle",
                "CNN_uid_shape_Conf": 0.91,
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 1.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_color_Class": "red",
                "CNN_uid_color_Conf": 0.95,
                "CNN_uid_shape_Class": "circle",
                "CNN_uid_shape_Conf": 0.91,
            },
            {
                "TrajectoryID": 0,
                "FrameID": 2,
                "X": 2.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_color_Class": "red",
                "CNN_uid_color_Conf": 0.94,
                "CNN_uid_shape_Class": "square",
                "CNN_uid_shape_Conf": 0.92,
            },
            {
                "TrajectoryID": 0,
                "FrameID": 3,
                "X": 3.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "CNN_uid_color_Class": "red",
                "CNN_uid_color_Conf": 0.94,
                "CNN_uid_shape_Class": "square",
                "CNN_uid_shape_Conf": 0.92,
            },
        ]
    )

    out = apply_identity_postprocessing(
        df,
        _cnn_params(max_gap=1, scoring_mode="per_head_average"),
    )

    assert out["TrajectoryID"].nunique() == 1
    assert set(out["FrameID"].tolist()) == {0, 1, 2, 3}
