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
