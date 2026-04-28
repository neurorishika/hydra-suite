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


from hydra_suite.core.identity.fragment_solver import solve_global_assignment


def _make_two_trajectory_df():
    """Two trajectories: traj 1 has strong green CNN but online label 'blue';
    traj 2 has strong blue CNN but online label 'green'. Non-overlapping in time."""
    n = 30
    frames_t1 = list(range(0, n))
    frames_t2 = list(range(n, 2 * n))
    return pd.DataFrame(
        {
            "TrajectoryID": [1] * n + [2] * n,
            "FrameID": frames_t1 + frames_t2,
            "X": [float(i) for i in range(2 * n)],
            "Y": [0.0] * (2 * n),
            "IdentityAssignedLabel": ["blue"] * n + ["green"] * n,
            "IdentityAssignedConfidence": [0.3] * (2 * n),
            "CNN_test_blue_Prob": [0.1] * n + [0.9] * n,
            "CNN_test_green_Prob": [0.9] * n + [0.1] * n,
        }
    )


def test_solve_global_assignment_corrects_swap():
    catalog = _make_catalog()
    df = _make_two_trajectory_df()
    params = {
        "ONLINE_PRIOR_WEIGHT": 0.1,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.05,
        "FRAGMENT_CNN_WEIGHT": 0.7,
        "FRAGMENT_SPATIAL_WEIGHT": 0.2,
        "MAX_VELOCITY_BREAK": 50.0,
    }
    result = solve_global_assignment(df, catalog, params)
    assert "IdentityAssignedLabel" in result.columns
    label_t1 = result[result["TrajectoryID"] == 1]["IdentityAssignedLabel"].iloc[0]
    label_t2 = result[result["TrajectoryID"] == 2]["IdentityAssignedLabel"].iloc[0]
    assert label_t1 == "green", f"expected green for traj 1, got {label_t1}"
    assert label_t2 == "blue", f"expected blue for traj 2, got {label_t2}"


def test_solve_global_assignment_uniform_labels_per_trajectory():
    """Every row within a trajectory must get the same assigned label."""
    catalog = _make_catalog()
    df = _make_two_trajectory_df()
    params = {
        "ONLINE_PRIOR_WEIGHT": 0.1,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.05,
        "FRAGMENT_CNN_WEIGHT": 0.7,
        "FRAGMENT_SPATIAL_WEIGHT": 0.2,
        "MAX_VELOCITY_BREAK": 50.0,
    }
    result = solve_global_assignment(df, catalog, params)
    for tid in result["TrajectoryID"].unique():
        labels = result[result["TrajectoryID"] == tid]["IdentityAssignedLabel"].unique()
        assert len(labels) == 1, f"trajectory {tid} has mixed labels: {labels}"


def test_solve_global_assignment_keeps_online_label_when_margin_too_small():
    catalog = _make_catalog()
    n = 30
    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * n,
            "FrameID": list(range(n)),
            "X": [float(i) for i in range(n)],
            "Y": [0.0] * n,
            "IdentityAssignedLabel": ["blue"] * n,
            "IdentityAssignedConfidence": [0.9] * n,
            "CNN_test_blue_Prob": [0.52] * n,
            "CNN_test_green_Prob": [0.48] * n,
        }
    )
    params = {
        "ONLINE_PRIOR_WEIGHT": 0.25,
        "ASSIGNMENT_MARGIN_THRESHOLD": 0.20,
        "FRAGMENT_CNN_WEIGHT": 0.7,
        "FRAGMENT_SPATIAL_WEIGHT": 0.3,
        "MAX_VELOCITY_BREAK": 50.0,
    }
    result = solve_global_assignment(df, catalog, params)
    assert result.iloc[0]["IdentityAssignedLabel"] == "blue"


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


from hydra_suite.core.identity.fragment_solver import split_trajectories_at_changepoints


def test_split_trajectories_produces_two_ids_on_changepoint():
    """A 60-frame trajectory split at frame 29 becomes two trajectories."""
    df = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    changepoints = {1: [29]}
    result = split_trajectories_at_changepoints(df, changepoints, {})
    ids = result["TrajectoryID"].unique()
    assert len(ids) == 2, f"expected 2 trajectories, got {len(ids)}"
    for tid in ids:
        seg = result[result["TrajectoryID"] == tid]
        frames = sorted(seg["FrameID"].values)
        assert frames == list(
            range(frames[0], frames[-1] + 1)
        ), f"trajectory {tid} has non-contiguous frames: {frames}"


def test_split_trajectories_preserves_original_id():
    df = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    changepoints = {1: [29]}
    result = split_trajectories_at_changepoints(df, changepoints, {})
    assert "OriginalTrajectoryID" in result.columns
    assert (
        result["OriginalTrajectoryID"] == 1
    ).all(), "all rows should reference original trajectory 1"


def test_split_trajectories_no_changepoints_unchanged():
    df = _make_df_with_prob_cols(n_frames=40, swap_at=40)
    result = split_trajectories_at_changepoints(df, {}, {})
    assert sorted(result["TrajectoryID"].unique()) == sorted(
        df["TrajectoryID"].unique()
    )
    assert len(result) == len(df)


def test_split_trajectories_drops_short_segments():
    """A split that would produce a segment shorter than MIN_FRAGMENT_FRAMES is dropped."""
    df = _make_df_with_prob_cols(n_frames=20, swap_at=20)
    changepoints = {1: [2]}
    result = split_trajectories_at_changepoints(
        df, changepoints, {"MIN_FRAGMENT_FRAMES": 5}
    )
    assert len(result["TrajectoryID"].unique()) == 1
    assert len(result) == 17


def test_split_trajectories_preserves_all_columns():
    df = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    changepoints = {1: [29]}
    result = split_trajectories_at_changepoints(df, changepoints, {})
    original_cols = set(df.columns)
    assert original_cols.issubset(
        set(result.columns)
    ), f"missing columns: {original_cols - set(result.columns)}"


def test_split_trajectories_multiple_trajectories_independent():
    """Splitting traj 1 does not affect traj 2."""
    df1 = _make_df_with_prob_cols(n_frames=60, swap_at=30)
    df2 = _make_df_with_prob_cols(n_frames=40, swap_at=40)
    df2 = df2.copy()
    df2["TrajectoryID"] = 2
    df2["FrameID"] = list(range(40))
    combined = pd.concat([df1, df2], ignore_index=True)
    changepoints = {1: [29]}
    result = split_trajectories_at_changepoints(combined, changepoints, {})
    traj2_rows = result[result["OriginalTrajectoryID"] == 2]
    assert len(traj2_rows) == 40
