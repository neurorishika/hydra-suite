"""Tests for the should_stop cancellation hook in postprocess + merge stages.

Covers:
- Output-neutrality: should_stop omitted / None / an always-False callable
  must all produce byte-identical results (the load-bearing invariant for
  the byte-identity re-gate).
- Early-exit: an always-True should_stop must short-circuit the heavy loops
  promptly, proven either by a reduced output (process_trajectories /
  process_trajectories_from_csv) or by a should_stop() call-count spy
  (resolve_trajectories, where the early loops break before any observable
  merge would have happened anyway on well-separated synthetic input).
"""

import os

import pandas as pd

from hydra_suite.core.post.processing import (
    process_trajectories,
    process_trajectories_from_csv,
    resolve_trajectories,
)


def _make_trajectories_full(n_traj=5, n_points=15):
    trajectories_full = []
    for tid in range(n_traj):
        traj = [(float(tid * 100 + i), 0.0, 0.0, float(i)) for i in range(n_points)]
        trajectories_full.append(traj)
    return trajectories_full


def _make_raw_csv(path, n_traj=5, n_points=15):
    rows = []
    for tid in range(n_traj):
        for i in range(n_points):
            rows.append(
                {
                    "TrajectoryID": tid,
                    "X": tid * 100 + i,
                    "Y": 0.0,
                    "Theta": 0.0,
                    "FrameID": i,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_traj_df(tid, x0, n=20, start=0):
    rows = [
        {"TrajectoryID": tid, "FrameID": start + i, "X": x0, "Y": 0.0, "Theta": 0.0}
        for i in range(n)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# process_trajectories
# ---------------------------------------------------------------------------


def test_process_trajectories_output_neutral_none_vs_false():
    trajectories_full = _make_trajectories_full()
    params = {"MIN_TRAJECTORY_LENGTH": 5, "MAX_VELOCITY_BREAK": 1000.0}

    result_omitted, stats_omitted = process_trajectories(trajectories_full, params)
    result_none, stats_none = process_trajectories(
        trajectories_full, params, should_stop=None
    )
    result_false, stats_false = process_trajectories(
        trajectories_full, params, should_stop=lambda: False
    )

    assert result_omitted == result_none == result_false
    assert stats_omitted == stats_none == stats_false
    assert len(result_omitted) == 5


def test_process_trajectories_early_exit_on_should_stop_true():
    trajectories_full = _make_trajectories_full()
    params = {"MIN_TRAJECTORY_LENGTH": 5, "MAX_VELOCITY_BREAK": 1000.0}

    baseline_result, _ = process_trajectories(trajectories_full, params)
    assert len(baseline_result) == 5  # sanity: normal run processes all 5

    stopped_result, stopped_stats = process_trajectories(
        trajectories_full, params, should_stop=lambda: True
    )
    assert stopped_result == []
    assert stopped_stats["final_count"] == 0


# ---------------------------------------------------------------------------
# process_trajectories_from_csv
# ---------------------------------------------------------------------------


def test_process_trajectories_from_csv_output_neutral_none_vs_false(tmp_path):
    raw_path = os.path.join(tmp_path, "raw.csv")
    _make_raw_csv(raw_path)
    params = {
        "MIN_TRAJECTORY_LENGTH": 5,
        "MAX_VELOCITY_BREAK": 1000.0,
        "MAX_OCCLUSION_GAP": 0,
    }

    df_omitted, stats_omitted = process_trajectories_from_csv(raw_path, params)
    df_none, stats_none = process_trajectories_from_csv(
        raw_path, params, should_stop=None
    )
    df_false, stats_false = process_trajectories_from_csv(
        raw_path, params, should_stop=lambda: False
    )

    assert df_omitted.equals(df_none)
    assert df_omitted.equals(df_false)
    assert stats_omitted == stats_none == stats_false
    assert len(df_omitted) == 75


def test_process_trajectories_from_csv_early_exit_on_should_stop_true(tmp_path):
    raw_path = os.path.join(tmp_path, "raw.csv")
    _make_raw_csv(raw_path)
    params = {
        "MIN_TRAJECTORY_LENGTH": 5,
        "MAX_VELOCITY_BREAK": 1000.0,
        "MAX_OCCLUSION_GAP": 0,
    }

    baseline_df, _ = process_trajectories_from_csv(raw_path, params)
    assert len(baseline_df) == 75  # sanity: normal run processes all 5 trajectories

    stopped_df, stopped_stats = process_trajectories_from_csv(
        raw_path, params, should_stop=lambda: True
    )
    assert stopped_df.empty
    assert stopped_stats["final_count"] == 0


# ---------------------------------------------------------------------------
# resolve_trajectories
# ---------------------------------------------------------------------------


def _spaced_forward_backward(n=10):
    """Ten forward + ten backward trajectories, spatially far apart so no
    merges/redundancy/stitching actually change anything -- isolates the
    should_stop polling behaviour from unrelated merge logic."""
    forward = [_make_traj_df(i, i * 1000, 20, 0) for i in range(n)]
    backward = [_make_traj_df(100 + i, 50000 + i * 1000, 20, 0) for i in range(n)]
    return forward, backward


def test_resolve_trajectories_output_neutral_none_vs_false():
    forward, backward = _spaced_forward_backward()
    params = {
        "AGREEMENT_DISTANCE": 15.0,
        "MIN_OVERLAP_FRAMES": 5,
        "MIN_TRAJECTORY_LENGTH": 5,
    }

    result_omitted = resolve_trajectories(forward, backward, params=params)
    result_none = resolve_trajectories(
        forward, backward, params=params, should_stop=None
    )
    result_false = resolve_trajectories(
        forward, backward, params=params, should_stop=lambda: False
    )

    assert len(result_omitted) == len(result_none) == len(result_false)
    for a, b in zip(result_omitted, result_none):
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True)
        )
    for a, b in zip(result_omitted, result_false):
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True)
        )


def test_resolve_trajectories_should_stop_polled_and_short_circuits():
    """An always-False should_stop must be polled many times (proving the
    poll sits inside the heavy loops and they actually iterate over the
    input). An always-True should_stop must short-circuit after only a
    handful of polls (one per should_stop-gated loop entry), proving the
    early exit actually breaks out instead of running to completion."""
    forward, backward = _spaced_forward_backward()
    params = {
        "AGREEMENT_DISTANCE": 15.0,
        "MIN_OVERLAP_FRAMES": 5,
        "MIN_TRAJECTORY_LENGTH": 5,
    }

    false_calls = {"n": 0}

    def counting_false():
        false_calls["n"] += 1
        return False

    resolve_trajectories(forward, backward, params=params, should_stop=counting_false)
    assert false_calls["n"] >= len(forward) + len(backward)

    true_calls = {"n": 0}

    def counting_true():
        true_calls["n"] += 1
        return True

    resolve_trajectories(forward, backward, params=params, should_stop=counting_true)
    # Should be gated at the top of only a handful of loops (redundancy x2,
    # overlap-merge while loop, stitch loop) -- nowhere near a full pass
    # over every trajectory pair.
    assert true_calls["n"] <= 10
    assert true_calls["n"] < false_calls["n"]
