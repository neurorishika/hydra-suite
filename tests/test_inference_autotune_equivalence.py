"""Correctness-gate holes the autotuner spec did not intend (Task 5 / B4).

Two gates are exercised here:

* track identity (``TrackID``/``TrajectoryID``/``State``/``ArenaID``) must be
  compared **exactly**, against the positionally matched row -- a Hungarian ID
  swap keeps row counts, keys and XY identical and must still be rejected;
* the angular test must be a per-row **max**, not a mean, so a single pi-flip
  cannot be averaged away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.inference.autotune.equivalence import (
    EquivalencePolicy,
    compare_outputs,
)

from .autotune_helpers import equivalence_frame, equivalence_outputs


def test_a_track_identity_swap_fails_the_gate():
    """Same positions, swapped TrackIDs, must NOT be judged equivalent."""

    base = equivalence_frame(
        rows=2,
        frame_ids=[10, 10],
        detection_ids=[0, 1],
        track_ids=[1, 2],
        x=[5.0, 90.0],
        y=[5.0, 90.0],
    )
    swapped = base.copy()
    swapped["TrackID"] = [2, 1]

    verdict = compare_outputs(equivalence_outputs(base), equivalence_outputs(swapped))

    assert not verdict.passed
    assert any("TrackID" in detail and "10" in detail for detail in verdict.details)
    # Counted once, by the mandatory path only -- not double-counted by the
    # heuristic categorical loop.
    assert verdict.categorical_mismatches == 4  # 2 rows x forward + final


def test_a_track_identity_swap_fails_when_trackid_is_the_row_key():
    """Without DetectionID the aligner keys on TrackID -- the swap must still fail.

    Sorting both sides by ``[FrameID, TrackID]`` makes TrackID equal by
    construction on every aligned row, so the mandatory-exact comparison has to
    run against the positional (Hungarian XY) pairing instead.
    """

    base = equivalence_frame(
        rows=2,
        frame_ids=[10, 10],
        track_ids=[1, 2],
        x=[5.0, 90.0],
        y=[5.0, 90.0],
        drop=("DetectionID",),
    )
    swapped = base.copy()
    swapped["TrackID"] = [2, 1]

    verdict = compare_outputs(equivalence_outputs(base), equivalence_outputs(swapped))

    assert not verdict.passed
    assert any("TrackID" in detail for detail in verdict.details)


def test_a_state_change_fails_the_gate():
    """``State`` is mandatory-exact even though no token heuristic matches it."""

    base = equivalence_frame(rows=1, frame_ids=[7], state="confirmed")
    changed = base.copy()
    changed["State"] = ["tentative"]

    verdict = compare_outputs(equivalence_outputs(base), equivalence_outputs(changed))

    assert not verdict.passed
    assert any("State" in detail and "7" in detail for detail in verdict.details)


def test_mandatory_exact_columns_survive_exact_categorical_disabled():
    """A construction site must not be able to switch identity checking off."""

    base = equivalence_frame(rows=1, track_ids=1)
    changed = base.copy()
    changed["TrackID"] = [2]

    verdict = compare_outputs(
        equivalence_outputs(base),
        equivalence_outputs(changed),
        policy=EquivalencePolicy(exact_categorical=False),
    )

    assert not verdict.passed


def test_a_single_pi_flip_fails_even_though_the_mean_is_small():
    """One flipped row in 100 averages to 0.031 rad; a mean test lets it pass."""

    n = 100
    base = equivalence_frame(
        rows=n,
        detection_ids=[0] * n,
        x=np.zeros(n).tolist(),
        y=np.zeros(n).tolist(),
        theta=np.zeros(n).tolist(),
    )
    flipped = base.copy()
    flipped.loc[0, "Theta"] = np.pi

    verdict = compare_outputs(equivalence_outputs(base), equivalence_outputs(flipped))

    assert not verdict.passed
    assert verdict.angle_max == pytest.approx(np.pi)


def test_zero_versus_two_pi_is_not_a_false_failure():
    """The angular difference wraps, so 0 rad and 2*pi rad agree."""

    base = equivalence_frame(rows=1, theta=0.0)
    wrapped = base.copy()
    wrapped["Theta"] = [2 * np.pi]

    verdict = compare_outputs(equivalence_outputs(base), equivalence_outputs(wrapped))

    assert verdict.angle_max == pytest.approx(0.0, abs=1e-9)
    assert verdict.passed


def test_determinism_floor_is_not_disarmed_by_a_bistable_pi_flip():
    """A head/tail pi-flip in the A-vs-A run must not raise the floor to pi.

    Under a per-row max a single bistable flip would otherwise set the measured
    determinism floor to pi and the angle gate would then admit everything.
    Rows whose head/tail column differs are excluded from the floor's angular
    statistic -- they are already rejected by the exact categorical check, so no
    coverage is lost.
    """

    n = 100
    base = equivalence_frame(
        rows=n,
        detection_ids=[0] * n,
        x=np.zeros(n).tolist(),
        y=np.zeros(n).tolist(),
        theta=np.zeros(n).tolist(),
        extra={"HeadTailAngleRad": np.zeros(n).tolist()},
    )
    repeat = base.copy()
    repeat.loc[0, "Theta"] = np.pi
    repeat.loc[0, "HeadTailAngleRad"] = np.pi

    floor = compare_outputs(
        equivalence_outputs(base),
        equivalence_outputs(repeat),
        for_determinism_floor=True,
    )

    assert floor.angle_max < 0.05
    # The flipped row is still rejected -- exclusion moves it, it does not hide it.
    assert not floor.passed
    assert floor.categorical_mismatches > 0


def test_pi_flip_without_a_headtail_marker_still_raises_the_floor():
    """Exclusion is scoped to head/tail rows only; other flips are not hidden."""

    n = 100
    base = equivalence_frame(
        rows=n,
        detection_ids=[0] * n,
        x=np.zeros(n).tolist(),
        y=np.zeros(n).tolist(),
        theta=np.zeros(n).tolist(),
    )
    repeat = base.copy()
    repeat.loc[0, "Theta"] = np.pi

    floor = compare_outputs(
        equivalence_outputs(base),
        equivalence_outputs(repeat),
        for_determinism_floor=True,
    )

    assert floor.angle_max == pytest.approx(np.pi)
    assert not floor.passed


# ---------------------------------------------------------------------------
# Task 10b: NaN-position rows (lost/coasting tracks) are ordinary output.
#
# Two defects made byte-identical files compare as different:
#   * ``_positional`` matches by nearest (X, Y) and cannot match a NaN-position
#     row, so every such row was counted as *unmatched* on BOTH sides;
#   * ``_aligned`` compared key columns with ``==``, which is False for a NaN
#     ``DetectionID`` even against itself, so keyed alignment returned ``None``
#     -- silently disabling the NaN-pattern and heuristic-categorical checks on
#     every real tracking output.
# ---------------------------------------------------------------------------


def _frame_with_lost_rows():
    """A frame shaped like a real forward CSV: 2 tracked rows + 3 lost rows.

    Mirrors ``fly_obb``'s forward output exactly -- lost tracks carry NaN
    ``X``/``Y``/``Theta`` *and* a NaN ``DetectionID`` (the row key), three of
    them sharing one FrameID.
    """

    return equivalence_frame(
        rows=5,
        frame_ids=[2, 2, 2, 2, 2],
        detection_ids=[np.nan, np.nan, np.nan, 30001, 30002],
        track_ids=[0, 1, 2, 3, 4],
        x=[np.nan, np.nan, np.nan, 498.8, 120.0],
        y=[np.nan, np.nan, np.nan, 302.9, 88.0],
        theta=[np.nan, np.nan, np.nan, 5.98, 1.25],
        state=["lost", "lost", "lost", "active", "active"],
    )


def test_identical_outputs_with_nan_position_rows_compare_equal():
    """RED before Task 10b: two byte-identical outputs must pass the gate.

    This is the determinism-floor case. ``fly_obb``'s A-vs-A repeat produced
    ``passed=False, unmatched=6`` on files ``filecmp`` reported as identical,
    which aborted every calibration with
    ``baseline_nondeterministic_beyond_contract``.
    """

    frame = _frame_with_lost_rows()

    verdict = compare_outputs(
        equivalence_outputs(frame),
        equivalence_outputs(frame.copy()),
        for_determinism_floor=True,
    )

    assert verdict.unmatched_rows == 0
    assert verdict.nan_pattern_mismatches == 0
    assert verdict.categorical_mismatches == 0
    # No "keyed rows are not aligned" either -- the aligner must survive a NaN
    # row key, or the NaN-pattern and categorical checks never run at all.
    assert verdict.details == ()
    assert verdict.passed


def test_nan_position_row_with_changed_trackid_still_fails():
    """Excluding NaN rows from the *distance* population must not excuse them."""

    base = _frame_with_lost_rows()
    changed = base.copy()
    changed.loc[1, "TrackID"] = 99

    verdict = compare_outputs(equivalence_outputs(base), equivalence_outputs(changed))

    assert not verdict.passed
    assert any("TrackID" in detail for detail in verdict.details)


def test_nan_position_row_with_changed_state_still_fails():
    """``State`` is mandatory-exact on NaN-position rows too."""

    base = _frame_with_lost_rows()
    changed = base.copy()
    changed.loc[0, "State"] = "confirmed"

    verdict = compare_outputs(equivalence_outputs(base), equivalence_outputs(changed))

    assert not verdict.passed
    assert any("State" in detail for detail in verdict.details)


def test_row_nan_on_one_side_only_is_a_genuine_difference():
    """A track lost on one run and tracked on the other must fail the gate."""

    base = _frame_with_lost_rows()
    changed = base.copy()
    changed.loc[2, ["X", "Y", "Theta"]] = [400.0, 400.0, 0.5]

    verdict = compare_outputs(equivalence_outputs(base), equivalence_outputs(changed))

    assert not verdict.passed
    assert verdict.unmatched_rows > 0 or verdict.nan_pattern_mismatches > 0


def test_nan_rows_do_not_perturb_the_position_p99():
    """The p99 population must be exactly the non-NaN rows -- no more, no less.

    The gate is percentile-based, so anything entering or leaving the distance
    population moves p99. NaN rows must leave it entirely (they already did,
    via ``dropna``); this pins that they also do not sneak back in as
    zero-distance pairs, which would drag the percentile down.
    """

    real = equivalence_frame(
        rows=4,
        frame_ids=[9, 9, 9, 9],
        detection_ids=[0, 1, 2, 3],
        track_ids=[1, 2, 3, 4],
        x=[0.0, 10.0, 20.0, 30.0],
        y=[0.0, 0.0, 0.0, 0.0],
    )
    shifted = real.copy()
    shifted["X"] = [0.0, 10.0, 20.0, 30.3]

    without_nan = compare_outputs(
        equivalence_outputs(real), equivalence_outputs(shifted)
    )

    # Pad with NaN-position rows ONLY -- adding the real rows of
    # ``_frame_with_lost_rows`` would legitimately enlarge the population.
    lost = _frame_with_lost_rows().iloc[:3].copy()
    padded_real = pd.concat([real, lost], ignore_index=True)
    padded_shifted = pd.concat([shifted, lost.copy()], ignore_index=True)
    with_nan = compare_outputs(
        equivalence_outputs(padded_real), equivalence_outputs(padded_shifted)
    )

    assert without_nan.position_p99 > 0.0
    assert with_nan.position_p99 == without_nan.position_p99
