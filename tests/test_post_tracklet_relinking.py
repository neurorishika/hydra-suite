from __future__ import annotations

import pandas as pd

from tests.helpers.module_loader import load_src_module

processing_mod = load_src_module(
    "hydra_suite/core/post/processing.py",
    "post_processing_relink_under_test",
)

relink_trajectories_with_pose = processing_mod.relink_trajectories_with_pose


def _params() -> dict:
    return {
        "ENABLE_TRACKLET_RELINKING": True,
        "MAX_OCCLUSION_GAP": 4,
        "MAX_VELOCITY_BREAK": 2.0,
        "AGREEMENT_DISTANCE": 1.0,
        "POSE_MIN_KPT_CONF_VALID": 0.2,
        "RELINK_POSE_MAX_DISTANCE": 0.45,
    }


def _pose_cols(points: list[tuple[float, float, float]]) -> dict[str, float]:
    row = {}
    for idx, (x, y, c) in enumerate(points):
        label = f"kp{idx:03d}"
        row[f"PoseKpt_{label}_X"] = x
        row[f"PoseKpt_{label}_Y"] = y
        row[f"PoseKpt_{label}_Conf"] = c
    return row


def test_relink_connects_short_gap_with_motion_and_pose() -> None:
    params = _params()
    rows = [
        {
            "TrajectoryID": 0,
            "FrameID": 0,
            "X": 0.0,
            "Y": 0.0,
            "Theta": 0.0,
            "State": "active",
            "DetectionID": 10,
            **_pose_cols([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)]),
        },
        {
            "TrajectoryID": 0,
            "FrameID": 1,
            "X": 1.0,
            "Y": 0.0,
            "Theta": 0.0,
            "State": "active",
            "DetectionID": 11,
            **_pose_cols([(1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (3.0, 0.0, 1.0)]),
        },
        {
            "TrajectoryID": 1,
            "FrameID": 4,
            "X": 4.0,
            "Y": 0.0,
            "Theta": 0.0,
            "State": "active",
            "DetectionID": 20,
            **_pose_cols([(4.0, 0.0, 1.0), (5.0, 0.0, 1.0), (6.0, 0.0, 1.0)]),
        },
        {
            "TrajectoryID": 1,
            "FrameID": 5,
            "X": 5.0,
            "Y": 0.0,
            "Theta": 0.0,
            "State": "active",
            "DetectionID": 21,
            **_pose_cols([(5.0, 0.0, 1.0), (6.0, 0.0, 1.0), (7.0, 0.0, 1.0)]),
        },
    ]
    df = pd.DataFrame(rows)

    relinked = relink_trajectories_with_pose(df, params)

    assert relinked["TrajectoryID"].nunique() == 1
    assert relinked["FrameID"].tolist() == [0, 1, 4, 5]
    assert relinked["DetectionID"].tolist() == [10, 11, 20, 21]


def test_relink_refuses_motion_gap_that_exceeds_velocity_limit() -> None:
    params = _params() | {"MAX_VELOCITY_BREAK": 0.2, "AGREEMENT_DISTANCE": 0.1}
    df = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 0.1,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 4,
                "X": 4.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 5,
                "X": 5.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
            },
        ]
    )

    relinked = relink_trajectories_with_pose(df, params)

    assert relinked["TrajectoryID"].nunique() == 2


def test_relink_refuses_pose_mismatch_even_when_motion_matches() -> None:
    params = _params() | {"RELINK_POSE_MAX_DISTANCE": 0.2}
    df = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                **_pose_cols([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)]),
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 1.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                **_pose_cols([(1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (3.0, 0.0, 1.0)]),
            },
            {
                "TrajectoryID": 1,
                "FrameID": 4,
                "X": 4.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                **_pose_cols([(0.0, 4.0, 1.0), (0.0, 5.0, 1.0), (0.0, 6.0, 1.0)]),
            },
            {
                "TrajectoryID": 1,
                "FrameID": 5,
                "X": 5.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                **_pose_cols([(0.0, 5.0, 1.0), (0.0, 6.0, 1.0), (0.0, 7.0, 1.0)]),
            },
        ]
    )

    relinked = relink_trajectories_with_pose(df, params)

    assert relinked["TrajectoryID"].nunique() == 2


def test_relink_refuses_unique_identity_conflict_even_when_motion_matches() -> None:
    params = _params()
    df = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "UniqueIdentityKey": "cnn:uid=alpha",
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 1.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "UniqueIdentityKey": "cnn:uid=alpha",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 4,
                "X": 4.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "UniqueIdentityKey": "cnn:uid=beta",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 5,
                "X": 5.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "UniqueIdentityKey": "cnn:uid=beta",
            },
        ]
    )

    relinked = relink_trajectories_with_pose(df, params)

    assert relinked["TrajectoryID"].nunique() == 2


def test_relink_falls_back_to_motion_only_when_pose_missing() -> None:
    params = _params()
    df = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 1.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 4,
                "X": 4.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 5,
                "X": 5.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
            },
        ]
    )

    relinked = relink_trajectories_with_pose(df, params)

    assert relinked["TrajectoryID"].nunique() == 1


def test_relink_collapses_chain_and_preserves_rows() -> None:
    params = _params()
    df = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 1,
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 1.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 2,
            },
            {
                "TrajectoryID": 1,
                "FrameID": 3,
                "X": 3.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 3,
            },
            {
                "TrajectoryID": 1,
                "FrameID": 4,
                "X": 4.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 4,
            },
            {
                "TrajectoryID": 2,
                "FrameID": 6,
                "X": 6.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 5,
            },
            {
                "TrajectoryID": 2,
                "FrameID": 7,
                "X": 7.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 6,
            },
        ]
    )

    relinked = relink_trajectories_with_pose(df, params)

    assert relinked["TrajectoryID"].nunique() == 1
    assert len(relinked) == len(df)
    assert relinked["FrameID"].tolist() == [0, 1, 3, 4, 6, 7]
    assert relinked["DetectionID"].tolist() == [1, 2, 3, 4, 5, 6]


def test_relink_refuses_long_gap_jump_even_when_velocity_extrapolates() -> None:
    params = _params() | {
        "MAX_OCCLUSION_GAP": 100,
        "MAX_VELOCITY_BREAK": 15.0,
        "AGREEMENT_DISTANCE": 20.0,
    }
    df = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 13.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 60,
                "X": 780.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 61,
                "X": 793.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
            },
        ]
    )

    relinked = relink_trajectories_with_pose(df, params)

    assert relinked["TrajectoryID"].nunique() == 2
