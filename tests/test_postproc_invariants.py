from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.helpers import postproc_runner
from tests.helpers.postproc_runner import (
    available_cases,
    load_resolve_fixture,
    normalize_output,
    run_resolve_trajectories,
)

RESOLVE_CASES = available_cases("resolve")


@pytest.mark.parametrize("case_name", RESOLVE_CASES)
def test_frame_bounds_invariant(case_name: str) -> None:
    forward, backward, params, _ = load_resolve_fixture(case_name)
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    if actual.empty:
        return

    input_df = pd.concat([forward, backward], ignore_index=True, sort=False)
    if input_df.empty:
        return

    min_frame = int(input_df["FrameID"].min())
    max_frame = int(input_df["FrameID"].max())
    assert int(actual["FrameID"].min()) >= min_frame
    assert int(actual["FrameID"].max()) <= max_frame


@pytest.mark.parametrize("case_name", RESOLVE_CASES)
def test_no_duplicate_trajectory_frame_pairs(case_name: str) -> None:
    forward, backward, params, _ = load_resolve_fixture(case_name)
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    if actual.empty:
        return

    dup_mask = actual.duplicated(subset=["TrajectoryID", "FrameID"], keep=False)
    assert not dup_mask.any(), (
        f"Found duplicate (TrajectoryID, FrameID) rows in case {case_name}: "
        f"{actual.loc[dup_mask, ['TrajectoryID', 'FrameID']].to_dict(orient='records')}"
    )


@pytest.mark.parametrize("case_name", RESOLVE_CASES)
def test_row_and_trajectory_counts_match_baseline(case_name: str) -> None:
    forward, backward, params, expected = load_resolve_fixture(case_name)
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    actual_n = normalize_output(actual)
    expected_n = normalize_output(expected)

    assert len(actual_n) == len(expected_n)
    assert actual_n["TrajectoryID"].nunique(dropna=True) == expected_n[
        "TrajectoryID"
    ].nunique(dropna=True)


@pytest.mark.parametrize("case_name", RESOLVE_CASES)
def test_normalization_is_deterministic(case_name: str) -> None:
    forward, backward, params, _ = load_resolve_fixture(case_name)
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    first = normalize_output(actual)
    second = normalize_output(actual)
    assert first.equals(second)


def test_merge_theta_avoids_orthogonal_artifact_for_pi_ambiguous_pairs() -> None:
    forward = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 10.0,
                "Y": 10.0,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 12.0,
                "Y": 10.0,
                "Theta": 0.0,
                "State": "active",
            },
        ]
    )
    backward = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 10.1,
                "Y": 10.0,
                "Theta": float(np.pi),
                "State": "active",
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 12.1,
                "Y": 10.0,
                "Theta": float(np.pi),
                "State": "active",
            },
        ]
    )
    params = {
        "AGREEMENT_DISTANCE": 5.0,
        "MIN_OVERLAP_FRAMES": 1,
        "MIN_TRAJECTORY_LENGTH": 1,
    }

    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    assert not actual.empty

    theta = actual["Theta"].dropna().to_numpy(dtype=float)
    assert theta.size > 0

    # Treat theta as axis-valued for this invariant: result should align to 0/pi,
    # not collapse to orthogonal pi/2.
    dist_to_axis = np.minimum(
        np.abs(np.arctan2(np.sin(theta), np.cos(theta))),
        np.abs(np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))),
    )
    assert float(np.max(dist_to_axis)) < 1e-6


def test_find_merge_candidates_uses_detection_id_before_spatial_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processing = postproc_runner._processing

    def _unexpected_python_fallback(*args, **kwargs):
        raise AssertionError(
            "DetectionID-aware candidate search should stay on the Numba path"
        )

    monkeypatch.setattr(processing, "NUMBA_AVAILABLE", True)
    monkeypatch.setattr(
        processing,
        "_compute_pairwise_distances_numba",
        getattr(
            processing._compute_pairwise_distances_numba,
            "py_func",
            processing._compute_pairwise_distances_numba,
        ),
    )
    monkeypatch.setattr(
        processing,
        "_compute_all_merge_candidates_numba",
        getattr(
            processing._compute_all_merge_candidates_numba,
            "py_func",
            processing._compute_all_merge_candidates_numba,
        ),
    )
    monkeypatch.setattr(
        processing, "_find_merge_candidates_python", _unexpected_python_fallback
    )

    forward_primary = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 10,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 100001,
            },
            {
                "TrajectoryID": 0,
                "FrameID": 11,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 110001,
            },
        ]
    )
    backward_primary = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 10,
                "X": 100.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 100001,
            },
            {
                "TrajectoryID": 0,
                "FrameID": 11,
                "X": 100.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 110001,
            },
        ]
    )
    forward_other = pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FrameID": 20,
                "X": 5.0,
                "Y": 5.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 200001,
            },
            {
                "TrajectoryID": 1,
                "FrameID": 21,
                "X": 6.0,
                "Y": 5.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 210001,
            },
        ]
    )
    backward_other = pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FrameID": 20,
                "X": 50.0,
                "Y": 50.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 200999,
            },
            {
                "TrajectoryID": 1,
                "FrameID": 21,
                "X": 51.0,
                "Y": 50.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 210999,
            },
        ]
    )

    candidates = processing._find_merge_candidates(
        [forward_primary, forward_other],
        [backward_primary, backward_other],
        agreement_distance=5.0,
        min_overlap=2,
    )

    assert any(
        fi == 0 and bi == 0 and agreeing == 2 and total_common == 2
        for fi, bi, agreeing, total_common, _identity_agreeing in candidates
    )


def test_find_merge_candidates_python_detection_id_identity_branch_does_not_crash() -> (
    None
):
    processing = postproc_runner._processing

    forward = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 10,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 100001,
                "IdentityCommitted": 1,
                "IdentityAssignedLabel": "animal_a",
            },
            {
                "TrajectoryID": 0,
                "FrameID": 11,
                "X": 0.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 110001,
                "IdentityCommitted": 1,
                "IdentityAssignedLabel": "animal_a",
            },
        ]
    )
    backward = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 10,
                "X": 100.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 100001,
                "IdentityCommitted": 1,
                "IdentityAssignedLabel": "animal_a",
            },
            {
                "TrajectoryID": 0,
                "FrameID": 11,
                "X": 100.0,
                "Y": 0.0,
                "Theta": 0.0,
                "State": "active",
                "DetectionID": 110001,
                "IdentityCommitted": 1,
                "IdentityAssignedLabel": "animal_a",
            },
        ]
    )

    candidates = processing._find_merge_candidates_python(
        [forward], [backward], agreement_distance=5.0, min_overlap=2
    )

    assert candidates == [(0, 0, 2, 2, 0)]


def test_different_animals_crossing_are_not_merged_into_one_trajectory() -> None:
    """
    Two animals crossing paths should remain as separate trajectories after merging.

    Bug: _merge_overlapping_agreeing_trajectories only required `agreeing >= min_overlap`
    (count), not a ratio. Two different animals that are briefly close (e.g. at a
    crossing for 5+ frames) would be merged, producing a single trajectory that jumps
    between both animals' positions.
    """
    n = 100
    # Animal A moves right: x = 1..100, y = 0
    fwd_a = pd.DataFrame(
        {
            "TrajectoryID": 0,
            "FrameID": list(range(1, n + 1)),
            "X": [float(f) for f in range(1, n + 1)],
            "Y": [0.0] * n,
            "Theta": [0.0] * n,
            "State": ["active"] * n,
        }
    )
    bwd_a = fwd_a.copy()  # forward and backward agree perfectly for animal A

    # Animal B moves left: x = 100..1, y = 0
    fwd_b = pd.DataFrame(
        {
            "TrajectoryID": 1,
            "FrameID": list(range(1, n + 1)),
            "X": [float(n + 1 - f) for f in range(1, n + 1)],
            "Y": [0.0] * n,
            "Theta": [0.0] * n,
            "State": ["active"] * n,
        }
    )
    bwd_b = fwd_b.copy()

    forward = pd.concat([fwd_a, fwd_b], ignore_index=True)
    backward = pd.concat([bwd_a, bwd_b], ignore_index=True)

    params = {
        "AGREEMENT_DISTANCE": 5.0,
        "MIN_OVERLAP_FRAMES": 5,
        "MIN_TRAJECTORY_LENGTH": 5,
    }
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )

    assert not actual.empty
    n_traj = actual["TrajectoryID"].nunique()
    assert n_traj == 2, (
        f"Expected 2 trajectories (one per animal), got {n_traj}. "
        "Different animals crossing paths should not be merged into one."
    )

    # Each trajectory should span all frames without large position jumps
    for traj_id, group in actual.groupby("TrajectoryID"):
        group = group.sort_values("FrameID")
        x = group["X"].dropna().to_numpy(dtype=float)
        if len(x) < 2:
            continue
        max_jump = float(np.max(np.abs(np.diff(x))))
        assert max_jump <= 5.0, (
            f"Trajectory {traj_id} has a jump of {max_jump:.1f} px — "
            "indicates two animals' data were incorrectly merged."
        )


def test_merged_trajectory_absorbs_surviving_unmerged_duplicate() -> None:
    """
    After _merge_overlapping_agreeing_trajectories extends a fragment into a longer
    trajectory, that longer trajectory may become ≥70% spatially redundant with an
    unmerged fragment that survived the first (pre-merge) redundancy-removal pass.
    A second redundancy-removal pass at the end of the pipeline must clean this up.

    Setup
    -----
    Three trajectory fragments, all for the same animal moving rightward (x = frame):
      F1 : frames  1-40  (forward fragment)
      F2 : frames 35-80  (forward fragment, overlaps F1)
      B  : frames  1-80  (unmerged backward, slightly noisy)

    First redundancy pass (before merge):
      B has 80 frames.  It agrees with F1 on ~32 frames (40% of B) → NOT removed.
      It agrees with F2 on ~36 frames (45% of B) → NOT removed.

    _merge_overlapping_agreeing_trajectories extends F1+F2 → M (frames 1-80).
    Now B agrees with M on ~72 frames (90% of B) → should be removed by a final pass.
    Without the final pass, B survives as a duplicate trajectory.
    """
    n = 80
    xs = list(range(1, n + 1))  # x = frame index (linear motion)

    def make_traj(tid, frames, noise=0.0):
        rng = np.random.default_rng(tid)
        ns = rng.uniform(-noise, noise, len(frames))
        return pd.DataFrame(
            {
                "TrajectoryID": tid,
                "FrameID": list(frames),
                "X": [float(xs[f - 1]) + float(ns[i]) for i, f in enumerate(frames)],
                "Y": [0.0] * len(frames),
                "Theta": [0.0] * len(frames),
                "State": ["active"] * len(frames),
            }
        )

    # Forward fragments (clean, no noise)
    f1 = make_traj(0, range(1, 41))  # frames 1-40
    f2 = make_traj(1, range(35, 81))  # frames 35-80 (overlaps F1)
    # Backward: same animal, slight noise (all within agreement_distance=5)
    b = make_traj(2, range(1, 81), noise=1.5)  # frames 1-80

    forward = pd.concat([f1, f2], ignore_index=True)
    backward = b.copy()
    backward["TrajectoryID"] = 0

    params = {
        "AGREEMENT_DISTANCE": 5.0,
        "MIN_OVERLAP_FRAMES": 5,
        "MIN_TRAJECTORY_LENGTH": 5,
    }
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )

    assert not actual.empty
    n_traj = actual["TrajectoryID"].nunique()
    assert n_traj == 1, (
        f"Expected 1 trajectory after merge absorbs the unmerged duplicate, got {n_traj}. "
        "A final redundancy-removal pass is needed after merging/stitching."
    )


def test_same_animal_not_double_represented_after_merge() -> None:
    """
    An animal tracked by both forward and backward passes should appear as exactly
    ONE trajectory after merging, not two overlapping copies.

    This can happen when an unmerged backward trajectory survives redundancy removal
    (because it only agrees with the merged trajectory on ~60-65% of frames) and then
    escapes _merge_overlapping_agreeing_trajectories too (because the passes noise makes
    the ratio look borderline). A second redundancy-removal pass after all merging and
    stitching catches these late-stage duplicates.
    """
    n = 60
    # One animal moving right, tracked identically by both passes
    # Add mild noise to backward so the two passes don't perfectly agree
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1.5, 1.5, n)  # noise within agreement_distance=5

    fwd = pd.DataFrame(
        {
            "TrajectoryID": 0,
            "FrameID": list(range(1, n + 1)),
            "X": [float(f) for f in range(1, n + 1)],
            "Y": [0.0] * n,
            "Theta": [0.0] * n,
            "State": ["active"] * n,
        }
    )
    # Backward trajectory: same animal but slightly offset (< agreement_distance)
    bwd = pd.DataFrame(
        {
            "TrajectoryID": 0,
            "FrameID": list(range(1, n + 1)),
            "X": [float(f) + float(noise[f - 1]) for f in range(1, n + 1)],
            "Y": [0.0] * n,
            "Theta": [0.0] * n,
            "State": ["active"] * n,
        }
    )

    params = {
        "AGREEMENT_DISTANCE": 5.0,
        "MIN_OVERLAP_FRAMES": 5,
        "MIN_TRAJECTORY_LENGTH": 5,
    }
    actual = run_resolve_trajectories({"forward": fwd, "backward": bwd}, params)

    assert not actual.empty
    n_traj = actual["TrajectoryID"].nunique()
    assert n_traj == 1, (
        f"Expected 1 trajectory (one animal, two consistent passes), got {n_traj}. "
        "The same animal should not appear as multiple trajectories after merging."
    )


# ---------------------------------------------------------------------------
# Forward-backward merge audit: segment preservation tests
# ---------------------------------------------------------------------------


def test_short_disagreement_does_not_lose_frames() -> None:
    """
    A brief 3-frame disagreement between forward and backward should NOT cause
    those frames to vanish from output.  The old code silently dropped segments
    shorter than MIN_TRAJECTORY_LENGTH at every state-machine transition in
    _conservative_merge, meaning detections in short agree/disagree bursts
    were permanently lost.
    """
    n = 50
    # Forward: animal A moving right, constant
    xs_fwd = [float(f) for f in range(1, n + 1)]
    fwd = pd.DataFrame(
        {
            "TrajectoryID": 0,
            "FrameID": list(range(1, n + 1)),
            "X": xs_fwd,
            "Y": [0.0] * n,
            "Theta": [0.0] * n,
            "State": ["active"] * n,
        }
    )

    # Backward: same animal, but at frames 24-26 there's a brief tracking glitch
    # (backward jumps to a different position for 3 frames)
    xs_bwd = list(xs_fwd)
    for f in [23, 24, 25]:  # 0-indexed → frames 24, 25, 26
        xs_bwd[f] = xs_fwd[f] + 100.0  # large offset → disagree
    bwd = pd.DataFrame(
        {
            "TrajectoryID": 0,
            "FrameID": list(range(1, n + 1)),
            "X": xs_bwd,
            "Y": [0.0] * n,
            "Theta": [0.0] * n,
            "State": ["active"] * n,
        }
    )

    params = {
        "AGREEMENT_DISTANCE": 5.0,
        "MIN_OVERLAP_FRAMES": 5,
        "MIN_TRAJECTORY_LENGTH": 5,
    }
    actual = run_resolve_trajectories({"forward": fwd, "backward": bwd}, params)

    assert not actual.empty
    all_output_frames = set(actual["FrameID"].unique())
    all_input_frames = set(range(1, n + 1))

    # The disagreement frames (24-26) must still appear in the output
    # from at least one source (forward or backward).
    missing = all_input_frames - all_output_frames
    assert not missing, (
        f"Frames {sorted(missing)} were tracked by both passes but vanished "
        f"from the merged output.  Short segments must be preserved."
    )


def test_unique_frames_preserved_when_partially_redundant() -> None:
    """
    If trajectory B is 70-94% covered by trajectory A, B's unique frames
    (those NOT in A) must survive in the output.  The old code removed B
    entirely, losing those unique detections.
    """
    # A: frames 20-100 (81 frames)
    a_frames = list(range(20, 101))
    a = pd.DataFrame(
        {
            "TrajectoryID": 0,
            "FrameID": a_frames,
            "X": [float(f) for f in a_frames],
            "Y": [0.0] * len(a_frames),
            "Theta": [0.0] * len(a_frames),
            "State": ["active"] * len(a_frames),
        }
    )

    # B: frames 1-100 (100 frames) — overlaps A on 81 frames (81%)
    # B's frames 1-19 are unique
    b_frames = list(range(1, 101))
    b = pd.DataFrame(
        {
            "TrajectoryID": 1,
            "FrameID": b_frames,
            "X": [float(f) for f in b_frames],
            "Y": [0.0] * len(b_frames),
            "Theta": [0.0] * len(b_frames),
            "State": ["active"] * len(b_frames),
        }
    )

    # Forward = A only; Backward = B only.  A is unused forward, B is unused backward.
    # Both go into the pipeline where redundancy removal processes them.
    params = {
        "AGREEMENT_DISTANCE": 5.0,
        "MIN_OVERLAP_FRAMES": 5,
        "MIN_TRAJECTORY_LENGTH": 5,
    }
    actual = run_resolve_trajectories({"forward": a, "backward": b}, params)

    assert not actual.empty
    output_frames = set(actual["FrameID"].unique())

    # B's unique frames 1-19 must still appear in the output
    unique_b_frames = set(range(1, 20))
    missing_unique = unique_b_frames - output_frames
    assert not missing_unique, (
        f"Unique frames {sorted(missing_unique)} from the partially-redundant "
        f"trajectory were lost.  Trimming should preserve non-overlapping frames."
    )


def test_no_detection_double_assigned_after_merge() -> None:
    """
    Core safety invariant: after forward-backward merging, no single
    (FrameID, X, Y) detection should appear in two different trajectories.
    """
    n = 80
    rng = np.random.default_rng(99)

    # Two animals moving in parallel
    fwd_a = pd.DataFrame(
        {
            "TrajectoryID": 0,
            "FrameID": list(range(1, n + 1)),
            "X": [float(f) for f in range(1, n + 1)],
            "Y": [10.0] * n,
            "Theta": [0.0] * n,
            "State": ["active"] * n,
        }
    )
    fwd_b = pd.DataFrame(
        {
            "TrajectoryID": 1,
            "FrameID": list(range(1, n + 1)),
            "X": [float(f) for f in range(1, n + 1)],
            "Y": [50.0] * n,
            "Theta": [0.0] * n,
            "State": ["active"] * n,
        }
    )
    # Backward with slight noise
    bwd_a = fwd_a.copy()
    bwd_a["X"] = bwd_a["X"] + rng.uniform(-1, 1, n)
    bwd_b = fwd_b.copy()
    bwd_b["X"] = bwd_b["X"] + rng.uniform(-1, 1, n)

    forward = pd.concat([fwd_a, fwd_b], ignore_index=True)
    backward = pd.concat([bwd_a, bwd_b], ignore_index=True)

    params = {
        "AGREEMENT_DISTANCE": 5.0,
        "MIN_OVERLAP_FRAMES": 5,
        "MIN_TRAJECTORY_LENGTH": 5,
    }
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )

    assert not actual.empty

    # Check: no frame should appear in more than one trajectory with the same position
    for frame_id, group in actual.groupby("FrameID"):
        if len(group) <= 1:
            continue
        # Multiple trajectories at same frame — positions must differ
        positions = group[["X", "Y"]].dropna().values
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.sqrt(
                    (positions[i][0] - positions[j][0]) ** 2
                    + (positions[i][1] - positions[j][1]) ** 2
                )
                assert dist > 2.0, (
                    f"Frame {frame_id}: two trajectories share nearly identical "
                    f"position ({positions[i]} vs {positions[j]}, dist={dist:.2f}). "
                    f"A detection appears to be double-assigned."
                )


def test_full_coverage_after_merge_simple() -> None:
    """
    For a simple case where both forward and backward perfectly track the same
    animal, every input frame must appear in the output.
    """
    n = 60
    fwd = pd.DataFrame(
        {
            "TrajectoryID": 0,
            "FrameID": list(range(1, n + 1)),
            "X": [float(f) for f in range(1, n + 1)],
            "Y": [0.0] * n,
            "Theta": [0.0] * n,
            "State": ["active"] * n,
        }
    )
    bwd = pd.DataFrame(
        {
            "TrajectoryID": 0,
            "FrameID": list(range(1, n + 1)),
            "X": [float(f) + 0.5 for f in range(1, n + 1)],
            "Y": [0.0] * n,
            "Theta": [0.0] * n,
            "State": ["active"] * n,
        }
    )

    params = {
        "AGREEMENT_DISTANCE": 5.0,
        "MIN_OVERLAP_FRAMES": 5,
        "MIN_TRAJECTORY_LENGTH": 5,
    }
    actual = run_resolve_trajectories({"forward": fwd, "backward": bwd}, params)

    assert not actual.empty
    output_frames = set(actual["FrameID"].unique())
    expected_frames = set(range(1, n + 1))
    missing = expected_frames - output_frames
    assert not missing, (
        f"Frames {sorted(missing)} were tracked by both passes but are "
        f"missing from the merged output."
    )
