def test_fragment_solver_imports():
    from hydra_suite.core.identity.fragment_solver import run_fragment_solver

    assert callable(run_fragment_solver)


import pandas as pd

from hydra_suite.core.identity.catalog import IdentityCatalog
from hydra_suite.core.identity.fragment_solver import detect_identity_changepoints


def _make_catalog():
    return IdentityCatalog.from_labels(["blue", "green"])


def _make_df_with_prob_cols(n_frames=60, swap_at=30):
    """60-frame single-trajectory DataFrame with a clear CNN swap at frame 30."""
    frames = list(range(n_frames))
    traj_ids = [1] * n_frames
    blue_probs = [0.9] * swap_at + [0.1] * (n_frames - swap_at)
    green_probs = [0.1] * swap_at + [0.9] * (n_frames - swap_at)
    return pd.DataFrame(
        {
            "TrajectoryID": traj_ids,
            "FrameID": frames,
            "X": [float(i) for i in range(n_frames)],
            "Y": [0.0] * n_frames,
            "IdentityAssignedLabel": ["blue"] * n_frames,
            "IdentityAssignedConfidence": [0.8] * n_frames,
            "CNN_test_blue_Prob": blue_probs,
            "CNN_test_green_Prob": green_probs,
        }
    )


def test_changepoint_detects_clear_swap():
    df = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    catalog = _make_catalog()
    result = detect_identity_changepoints(
        df, catalog, {"CHANGEPOINT_PENALTY": 2.0, "MIN_FRAGMENT_FRAMES": 5}
    )
    # Trajectory 1 should have exactly one split near frame 30.
    splits = result.get(1, [])
    assert len(splits) == 1
    assert 27 <= splits[0] <= 32, f"expected split near 29-30, got {splits[0]}"


def test_changepoint_no_split_when_stable():
    df = _make_df_with_prob_cols(n_frames=60, swap_at=60)  # no swap
    catalog = _make_catalog()
    result = detect_identity_changepoints(
        df, catalog, {"CHANGEPOINT_PENALTY": 2.0, "MIN_FRAGMENT_FRAMES": 5}
    )
    assert result.get(1, []) == [], "stable trajectory should have no splits"


def test_changepoint_no_cnn_columns_returns_empty():
    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * 10,
            "FrameID": list(range(10)),
            "IdentityAssignedLabel": ["blue"] * 10,
            "IdentityAssignedConfidence": [0.8] * 10,
        }
    )
    catalog = _make_catalog()
    result = detect_identity_changepoints(df, catalog, {})
    assert result == {}, "no CNN columns should produce empty changepoint dict"


from hydra_suite.core.identity.fragment_solver import build_fragments


def test_build_fragments_splits_at_changepoint():
    df = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    catalog = _make_catalog()
    # Provide changepoint at frame 29 (inclusive end of first segment).
    changepoints = {1: [29]}
    frags = build_fragments(df, changepoints, catalog, {})
    assert len(frags) == 2
    row0 = frags[frags["FragmentID"] == 0].iloc[0]
    row1 = frags[frags["FragmentID"] == 1].iloc[0]
    assert int(row0["StartFrame"]) == 0
    assert int(row0["EndFrame"]) == 29
    assert int(row1["StartFrame"]) == 30
    assert int(row1["EndFrame"]) == 59


def test_build_fragments_no_changepoints_one_fragment():
    df = _make_df_with_prob_cols(n_frames=40, swap_at=40)
    catalog = _make_catalog()
    frags = build_fragments(df, {}, catalog, {})
    assert len(frags) == 1
    assert int(frags.iloc[0]["StartFrame"]) == 0
    assert int(frags.iloc[0]["EndFrame"]) == 39


def test_build_fragments_has_required_columns():
    df = _make_df_with_prob_cols(n_frames=20, swap_at=20)
    catalog = _make_catalog()
    frags = build_fragments(df, {}, catalog, {})
    required = {
        "TrajectoryID",
        "FragmentID",
        "StartFrame",
        "EndFrame",
        "StartX",
        "StartY",
        "EndX",
        "EndY",
        "MeanCNNProbs",
        "OnlineLabel",
        "OnlineConfidence",
    }
    assert required.issubset(
        set(frags.columns)
    ), f"missing: {required - set(frags.columns)}"


from hydra_suite.core.identity.fragment_solver import (
    build_fragments,
    solve_global_assignment,
)


def _make_two_swapped_fragments():
    """Two fragments: frag0 has strong green CNN evidence but online label 'blue';
    frag1 has strong blue CNN evidence but online label 'green'.
    Non-overlapping. Expect solver to swap them."""
    frags = pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FragmentID": 0,
                "StartFrame": 0,
                "EndFrame": 29,
                "StartX": 0.0,
                "StartY": 0.0,
                "EndX": 5.0,
                "EndY": 0.0,
                "MeanCNNProbs": {"blue": 0.1, "green": 0.9},
                "OnlineLabel": "blue",
                "OnlineConfidence": 0.3,
            },
            {
                "TrajectoryID": 2,
                "FragmentID": 1,
                "StartFrame": 30,
                "EndFrame": 59,
                "StartX": 5.0,
                "StartY": 0.0,
                "EndX": 10.0,
                "EndY": 0.0,
                "MeanCNNProbs": {"blue": 0.9, "green": 0.1},
                "OnlineLabel": "green",
                "OnlineConfidence": 0.3,
            },
        ]
    )
    return frags


def test_solve_global_assignment_corrects_swap():
    catalog = _make_catalog()
    frags = _make_two_swapped_fragments()
    params = {
        "ONLINE_PRIOR_WEIGHT": 0.1,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.05,
        "FRAGMENT_CNN_WEIGHT": 0.7,
        "FRAGMENT_SPATIAL_WEIGHT": 0.2,
        "MAX_VELOCITY_BREAK": 50.0,
        "TAG_IDENTITY_LABELS": [],
    }
    result = solve_global_assignment(frags, catalog, params)
    assert "AssignedLabel" in result.columns
    frag0_label = result[result["FragmentID"] == 0].iloc[0]["AssignedLabel"]
    frag1_label = result[result["FragmentID"] == 1].iloc[0]["AssignedLabel"]
    assert frag0_label == "green", f"expected green for frag0, got {frag0_label}"
    assert frag1_label == "blue", f"expected blue for frag1, got {frag1_label}"


def test_solve_global_assignment_keeps_online_label_when_margin_too_small():
    catalog = _make_catalog()
    frags = pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FragmentID": 0,
                "StartFrame": 0,
                "EndFrame": 29,
                "StartX": 0.0,
                "StartY": 0.0,
                "EndX": 5.0,
                "EndY": 0.0,
                "MeanCNNProbs": {"blue": 0.52, "green": 0.48},  # near-tie
                "OnlineLabel": "blue",
                "OnlineConfidence": 0.9,
            }
        ]
    )
    params = {
        "ONLINE_PRIOR_WEIGHT": 0.25,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.20,
        "FRAGMENT_CNN_WEIGHT": 0.7,
        "FRAGMENT_SPATIAL_WEIGHT": 0.3,
        "MAX_VELOCITY_BREAK": 50.0,
        "TAG_IDENTITY_LABELS": [],
    }
    result = solve_global_assignment(frags, catalog, params)
    assert result.iloc[0]["AssignedLabel"] == "blue"


from hydra_suite.core.identity.fragment_solver import (
    apply_fragment_labels,
    run_fragment_solver,
)


def test_apply_fragment_labels_writes_assigned_label():
    df = _make_df_with_prob_cols(n_frames=10, swap_at=10)
    frags = pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FragmentID": 0,
                "StartFrame": 0,
                "EndFrame": 9,
                "StartX": 0.0,
                "StartY": 0.0,
                "EndX": 1.0,
                "EndY": 0.0,
                "MeanCNNProbs": {"blue": 0.9},
                "OnlineLabel": "blue",
                "OnlineConfidence": 0.9,
                "AssignedLabel": "green",
            }
        ]
    )
    result = apply_fragment_labels(df, frags)
    assert (result["IdentityAssignedLabel"] == "green").all()
    assert result["IdentityCommitted"].all()


def test_run_fragment_solver_returns_dataframe():
    df = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    catalog = _make_catalog()
    result = run_fragment_solver(df, catalog, {"CHANGEPOINT_PENALTY": 2.0})
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)
    assert "IdentityAssignedLabel" in result.columns


def test_run_fragment_solver_empty_df():
    catalog = _make_catalog()
    result = run_fragment_solver(pd.DataFrame(), catalog, {})
    assert isinstance(result, pd.DataFrame)
