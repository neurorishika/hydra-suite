"""The pose/confidence product must be gated, not just the body pose (B1).

``pose_batch_size`` is a tuned coordinate, so every column a pose batch can
move has to be reachable by the correctness gate. Before this file the gate
compared only X/Y (Hungarian p99), ``Theta`` (per-row wrapped max), NaN
patterns and a token-matched set of exact-string categoricals. The final
adversarial review demonstrated three mutations of the real
``ant_pose_headtail`` final CSV that the gate admitted:

* every ``PoseKpt_*_X/Y`` shifted by +25 px,
* every ``PoseKpt_*_Conf`` driven to 0.01,
* ``HeadingResolved`` shifted by +0.3 rad.

Each of those is reproduced here against the real exported column names and
must now be rejected, with a legible detail naming the offending column.

Tolerances are anchored on the MEASURED determinism floor, not on taste: an
A-vs-A repeat of ``ant_pose_headtail`` on this box (12500 forward rows, 9096
final rows, MPS, ``/tmp/equiv_autotune/mps``) has ``max|delta| == 0.0``
EXACTLY on every numeric column, keypoints included. The generic numeric
tolerance is therefore a CSV-round-trip guard (1e-6), and keypoints inherit
the body-position budget because they are positions in the same pixel units.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.inference.autotune.equivalence import (
    EquivalencePolicy,
    compare_outputs,
)

from .autotune_helpers import equivalence_frame, equivalence_outputs

_KEYPOINTS = ("clypeus", "neck", "tip_of_gaster")


def _pose_frame(rows: int = 6):
    """A frame carrying the real pose/identity export column names."""

    extra: dict[str, object] = {
        "DetectionConfidence": [0.90 + 0.01 * i for i in range(rows)],
        "HeadingResolved": [0.10 * i for i in range(rows)],
        "PoseMeanConf": [0.80 + 0.01 * i for i in range(rows)],
        "PoseQualityScore": [0.70 + 0.01 * i for i in range(rows)],
        "IdentityFinalConfidence": [0.60 + 0.01 * i for i in range(rows)],
    }
    for index, name in enumerate(_KEYPOINTS):
        extra[f"PoseKpt_{name}_X"] = [10.0 + index + i for i in range(rows)]
        extra[f"PoseKpt_{name}_Y"] = [20.0 + index + i for i in range(rows)]
        extra[f"PoseKpt_{name}_Conf"] = [0.5 + 0.05 * index for _ in range(rows)]
    return equivalence_frame(
        rows=rows,
        frame_ids=list(range(rows)),
        detection_ids=list(range(rows)),
        track_ids=[1] * rows,
        x=[100.0 + i for i in range(rows)],
        y=[200.0 + i for i in range(rows)],
        extra=extra,
    )


def _verdict(mutate=None):
    reference = _pose_frame()
    candidate = reference.copy()
    if mutate is not None:
        mutate(candidate)
    return compare_outputs(
        equivalence_outputs(reference), equivalence_outputs(candidate)
    )


def test_identical_pose_output_still_passes():
    """The widened gate must not reject a run against itself."""

    assert _verdict().passed


# --- the three mutations the review admitted -------------------------------


def test_shifted_keypoints_are_rejected():
    """+25 px on every keypoint: the review's mutation 1."""

    def mutate(frame):
        for name in _KEYPOINTS:
            frame[f"PoseKpt_{name}_X"] += 25.0
            frame[f"PoseKpt_{name}_Y"] += 25.0

    verdict = _verdict(mutate)
    assert not verdict.passed
    assert verdict.keypoint_p99 > 0.5
    assert any("keypoint" in detail.lower() for detail in verdict.details)


def test_keypoint_confidences_driven_to_a_floor_are_rejected():
    """Every ``PoseKpt_*_Conf`` -> 0.01: the review's mutation 2."""

    def mutate(frame):
        for name in _KEYPOINTS:
            frame[f"PoseKpt_{name}_Conf"] = 0.01

    verdict = _verdict(mutate)
    assert not verdict.passed
    assert verdict.numeric_max > 0.0
    assert any("Conf" in detail for detail in verdict.details)


def test_heading_resolved_shift_is_rejected():
    """``HeadingResolved`` +0.3 rad: the review's mutation 3."""

    def mutate(frame):
        frame["HeadingResolved"] = frame["HeadingResolved"] + 0.3

    verdict = _verdict(mutate)
    assert not verdict.passed
    assert any("HeadingResolved" in detail for detail in verdict.details)


# --- the other columns the review showed were free to move -----------------


@pytest.mark.parametrize(
    "column, delta",
    [
        ("DetectionConfidence", -0.45),
        ("PoseMeanConf", -0.30),
        ("PoseQualityScore", -0.70),
        ("IdentityFinalConfidence", -0.60),
    ],
)
def test_confidence_columns_are_rejected_when_they_move(column, delta):
    verdict = _verdict(lambda frame: frame.__setitem__(column, frame[column] + delta))
    assert not verdict.passed
    assert any(column in detail for detail in verdict.details)


def test_heading_wraps_at_pi_instead_of_reading_as_a_two_pi_divergence():
    """``HeadingResolved`` is angular: 0 and 2*pi are the same heading."""

    verdict = _verdict(
        lambda frame: frame.__setitem__(
            "HeadingResolved", frame["HeadingResolved"] + 2 * np.pi
        )
    )
    assert verdict.passed


def test_a_pi_flip_in_heading_is_rejected():
    verdict = _verdict(
        lambda frame: frame.__setitem__(
            "HeadingResolved", frame["HeadingResolved"] + np.pi
        )
    )
    assert not verdict.passed


# --- the measured floor governs, as it does for positions ------------------


def test_a_measured_floor_widens_the_new_budgets_like_the_old_ones():
    """A platform whose A-vs-A repeat is noisy must not reject itself.

    ``compare_outputs`` already raises the position/angle limits to the
    measured determinism floor. The keypoint and numeric budgets have to do
    the same, or a nondeterministic accelerator would fail its own repeat.
    """

    def mutate(frame):
        for name in _KEYPOINTS:
            frame[f"PoseKpt_{name}_X"] += 1.0
        frame["PoseMeanConf"] = frame["PoseMeanConf"] + 0.01

    strict = _verdict(mutate)
    assert not strict.passed

    reference = _pose_frame()
    candidate = reference.copy()
    mutate(candidate)
    lenient = compare_outputs(
        equivalence_outputs(reference),
        equivalence_outputs(candidate),
        determinism_floor=strict,
    )
    assert lenient.passed


def test_a_keypoint_beyond_the_match_gate_is_a_hard_failure():
    """p99 forgives 1% of samples; the match gate must not.

    A body position further than ``match_gate`` from its partner is counted
    unmatched and fails outright. A keypoint gets the same treatment, so a
    single wildly-moved keypoint cannot hide under the percentile.
    """

    def mutate(frame):
        frame.loc[0, "PoseKpt_neck_X"] = frame.loc[0, "PoseKpt_neck_X"] + 500.0

    verdict = _verdict(mutate)
    assert not verdict.passed
    assert any("gate" in detail.lower() for detail in verdict.details)


def test_policy_tolerances_are_honoured_for_the_new_families():
    reference = _pose_frame()
    candidate = reference.copy()
    candidate["PoseMeanConf"] = candidate["PoseMeanConf"] + 0.001
    tight = compare_outputs(
        equivalence_outputs(reference), equivalence_outputs(candidate)
    )
    assert not tight.passed
    loose = compare_outputs(
        equivalence_outputs(reference),
        equivalence_outputs(candidate),
        policy=EquivalencePolicy(numeric_max_tolerance=0.01),
    )
    assert loose.passed


# --- the string product must be covered too ---------------------------------


@pytest.mark.parametrize(
    "column, value",
    [
        ("PoseQualityState", "bad"),
        ("PoseQualityFlags", "clipped"),
        ("PoseSource", "interpolated"),
        ("HeadingMethod", "fallback"),
    ],
)
def test_non_numeric_pose_columns_are_compared_exactly(column, value):
    """No tolerance is meaningful for a label, so it is compared exactly.

    These four were the residual hole after the numeric families landed:
    string columns a pose batch can move, reachable by no heuristic token.
    """

    rows = 6
    reference = _pose_frame(rows)
    reference[column] = ["good"] * rows
    candidate = reference.copy()
    candidate.loc[0, column] = value

    verdict = compare_outputs(
        equivalence_outputs(reference), equivalence_outputs(candidate)
    )
    assert not verdict.passed
    assert verdict.categorical_mismatches > 0
