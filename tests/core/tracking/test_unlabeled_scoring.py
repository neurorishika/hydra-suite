"""Behavioural tests for conservative unlabeled tracking-score primitives."""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.tracking.optimization.unlabeled_scoring import (
    CandidateEvaluation,
    CandidateScore,
    MetricDirection,
    MetricEstimate,
    MetricSpec,
    TemporalSegment,
    aggregate_candidate_evaluations,
    build_train_validation_split,
    decide_heldout_baseline_protection,
    dominates,
    forward_backward_cycle_consistency,
    global_slot_alignment,
    pareto_frontier,
    pareto_ranks,
    partition_temporal_segments,
    recommend_dominating_candidate,
    trajectory_quality_metrics,
)

LOSS_SPECS = (
    MetricSpec("cycle_loss", MetricDirection.MINIMIZE),
    MetricSpec("fragment_loss", MetricDirection.MINIMIZE),
)


def _score(
    candidate_id: str, cycle_loss: float, fragment_loss: float
) -> CandidateScore:
    return CandidateScore(
        candidate_id,
        {
            "cycle_loss": MetricEstimate(cycle_loss, 0.0, 1),
            "fragment_loss": MetricEstimate(fragment_loss, 0.0, 1),
        },
        0.0,
    )


def test_cycle_consistency_is_zero_for_matching_reverse_chronological_pass() -> None:
    forward = np.array(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[1.0, 0.0], [9.0, 0.0]],
            [[2.0, 0.0], [8.0, 0.0]],
        ]
    )

    result = forward_backward_cycle_consistency(forward, forward[::-1])

    assert result.mean_normalized_error == 0.0
    assert result.valid_observations == 6
    assert result.spatial_scale == 8.0


def test_cycle_consistency_aligns_a_global_backward_slot_permutation() -> None:
    forward = np.array(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[1.0, 0.0], [9.0, 0.0]],
            [[2.0, 0.0], [8.0, 0.0]],
        ]
    )

    result = forward_backward_cycle_consistency(forward, forward[::-1, [1, 0]])

    assert result.mean_normalized_error == 0.0
    assert result.valid_observations == 6


def test_cycle_consistency_detects_a_smooth_identity_swap_after_a_crossing() -> None:
    # Both tracks meet at frame five, so switching their labels there creates
    # no discontinuity.  The forward/backward cycle still exposes the swap.
    time = np.arange(11, dtype=float)
    forward = np.stack(
        [
            np.column_stack([time, np.zeros_like(time)]),
            np.column_stack([10 - time, np.zeros_like(time)]),
        ],
        axis=1,
    )
    swapped = forward.copy()
    swapped[6:, [0, 1]] = swapped[6:, [1, 0]]

    consistent = forward_backward_cycle_consistency(forward, forward[::-1])
    inconsistent = forward_backward_cycle_consistency(forward, swapped[::-1])

    assert inconsistent.mean_normalized_error > consistent.mean_normalized_error + 0.2
    assert inconsistent.p95_normalized_error > 0.5


def test_cycle_consistency_ignores_missing_tracks_and_uses_given_scale() -> None:
    forward = np.array([[[0.0, 0.0]], [[np.nan, np.nan]], [[4.0, 0.0]]])
    backward = np.array([[[5.0, 0.0]], [[0.0, 0.0]], [[2.0, 0.0]]])

    result = forward_backward_cycle_consistency(
        forward,
        backward,
        backward_is_reverse_chronological=False,
        spatial_scale=2.0,
    )

    assert result.valid_observations == 2
    assert result.mean_normalized_error == pytest.approx(1.75)


def test_cycle_assignment_prioritizes_sufficient_overlap_before_error() -> None:
    """One-frame accidental matches must not hide longer matched evidence."""

    forward = np.full((12, 2, 2), np.nan)
    backward = np.full_like(forward, np.nan)

    # Slot zero exists in the first half and slot one in the second.  Each
    # backwards slot has a long, slightly displaced compatible interval plus
    # one exact accidental overlap with the other forward slot.  Mean-only
    # assignment chooses the two one-frame matches and reports zero loss.
    forward[:6, 0] = np.column_stack([np.arange(6, dtype=float), np.zeros(6)])
    forward[6:, 1] = np.column_stack([100.0 + np.arange(6, dtype=float), np.zeros(6)])
    backward[:6, 0] = forward[:6, 0] + np.array([1.0, 0.0])
    backward[6, 0] = forward[6, 1]
    backward[5, 1] = forward[5, 0]
    backward[6:, 1] = forward[6:, 1] + np.array([1.0, 0.0])

    alignment = global_slot_alignment(
        forward,
        backward,
        backward_is_reverse_chronological=False,
        spatial_scale=1.0,
    )
    result = forward_backward_cycle_consistency(
        forward,
        backward,
        backward_is_reverse_chronological=False,
        spatial_scale=1.0,
        slot_alignment=alignment,
    )

    assert alignment.backward_for_forward == (0, 1)
    assert alignment.shared_observations == 12
    assert result.valid_observations == 12
    assert result.mean_normalized_error == pytest.approx(1.0)


def test_cycle_assignment_can_leave_a_slot_unmatched_for_far_more_overlap() -> None:
    """An unavailable edge must not force two marginal matches over one strong one."""

    forward = np.full((103, 2, 2), np.nan)
    backward = np.full_like(forward, np.nan)
    # Eligible-overlap matrix with a three-observation floor:
    # [[100, 3], [3, 0]].  A square-only assignment incorrectly chooses the
    # cross-pairs (six observations) to avoid an unavailable real edge.
    forward[:100, 0] = 0.0
    forward[100:, 1] = 0.0
    backward[:, 0] = 0.0
    backward[:3, 1] = 0.0

    alignment = global_slot_alignment(
        forward,
        backward,
        backward_is_reverse_chronological=False,
        spatial_scale=1.0,
        minimum_shared_observations=3,
    )

    assert alignment.backward_for_forward == (0, None)
    assert alignment.shared_observations == 100


def test_temporal_region_metrics_keep_transition_evidence_at_block_boundary() -> None:
    positions = np.zeros((6, 1, 2), dtype=float)
    positions[3:] = np.nan

    second_region = trajectory_quality_metrics(
        positions,
        spatial_scale=1.0,
        segment=TemporalSegment(3, 6),
    )

    # The only observed/present transition occurs from frame 2 to frame 3,
    # exactly at the region boundary.  Scoring a bare 3:6 slice loses it.
    assert second_region.fragmentation_loss == pytest.approx(1.0 / 3.0)
    assert second_region.temporal_transition_count == 3


def test_trajectory_quality_uses_only_observed_positions_and_penalizes_missingness() -> (
    None
):
    constant_velocity = np.array(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[2.0, 0.0]],
            [[3.0, 0.0]],
        ]
    )
    with_gap = constant_velocity.copy()
    with_gap[2] = np.nan

    smooth = trajectory_quality_metrics(constant_velocity, spatial_scale=2.0)
    gappy = trajectory_quality_metrics(with_gap, spatial_scale=2.0)
    all_missing = trajectory_quality_metrics(
        np.full((4, 1, 2), np.nan), spatial_scale=2.0
    )

    assert smooth.coverage_loss == 0.0
    assert smooth.fragmentation_loss == 0.0
    assert smooth.motion_roughness_loss == 0.0
    assert gappy.coverage_loss > smooth.coverage_loss
    assert gappy.fragmentation_loss > smooth.fragmentation_loss
    assert all_missing.coverage_loss == 1.0
    assert all_missing.observed_positions == 0


def test_trajectory_roughness_is_scale_normalized_and_monotonic_with_jitter() -> None:
    smooth = np.array(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[2.0, 0.0]],
            [[3.0, 0.0]],
        ]
    )
    jittery = smooth.copy()
    jittery[2, 0, 1] = 2.0

    smooth_metrics = trajectory_quality_metrics(smooth, spatial_scale=1.0)
    jittery_metrics = trajectory_quality_metrics(jittery, spatial_scale=1.0)
    rescaled_metrics = trajectory_quality_metrics(jittery * 10.0, spatial_scale=10.0)

    assert jittery_metrics.motion_roughness_loss > smooth_metrics.motion_roughness_loss
    assert rescaled_metrics.motion_roughness_loss == pytest.approx(
        jittery_metrics.motion_roughness_loss
    )


def test_temporal_segments_are_balanced_and_split_has_no_temporal_leakage() -> None:
    segments = partition_temporal_segments(10, 3)
    split = build_train_validation_split(20, 0.25, gap_frames=2)

    assert [(segment.start, segment.stop) for segment in segments] == [
        (0, 4),
        (4, 7),
        (7, 10),
    ]
    assert split.train[0].stop == split.gap[0].start
    assert split.gap[0].stop == split.validation[0].start
    assert split.train[0].stop < split.validation[0].start
    assert split.validation[0].stop == 20


def test_aggregation_is_deterministic_and_describes_perturbation_stability() -> None:
    evaluations = (
        CandidateEvaluation("noisy", {"cycle_loss": 1.0, "fragment_loss": 2.0}, "a"),
        CandidateEvaluation("stable", {"cycle_loss": 1.0, "fragment_loss": 2.0}, "a"),
        CandidateEvaluation("noisy", {"cycle_loss": 3.0, "fragment_loss": 6.0}, "b"),
        CandidateEvaluation("stable", {"cycle_loss": 1.1, "fragment_loss": 1.9}, "b"),
    )

    scores = aggregate_candidate_evaluations(evaluations[::-1], LOSS_SPECS)
    noisy, stable = scores

    assert [score.candidate_id for score in scores] == ["noisy", "stable"]
    assert noisy.metrics["cycle_loss"].mean == pytest.approx(2.0)
    assert noisy.metrics["cycle_loss"].std > stable.metrics["cycle_loss"].std
    assert noisy.mean_relative_spread > stable.mean_relative_spread
    assert set(noisy.perturbation_metrics) == {"a", "b"}


def test_pareto_methods_preserve_metric_monotonicity_without_scalar_tradeoffs() -> None:
    baseline = _score("baseline", 5.0, 5.0)
    better = _score("better", 4.0, 4.0)
    tradeoff = _score("tradeoff", 3.0, 6.0)

    assert dominates(better, baseline, LOSS_SPECS)
    assert not dominates(tradeoff, baseline, LOSS_SPECS)
    assert [
        score.candidate_id
        for score in pareto_frontier((baseline, better, tradeoff), LOSS_SPECS)
    ] == [
        "better",
        "tradeoff",
    ]
    ranks = {
        rank.candidate_id: rank.front
        for rank in pareto_ranks((baseline, better, tradeoff), LOSS_SPECS)
    }
    assert ranks == {"baseline": 2, "better": 1, "tradeoff": 1}


def test_recommendation_never_promotes_a_candidate_that_worsens_a_loss() -> None:
    baseline = _score("baseline", 5.0, 5.0)
    worsens_fragmentation = _score("looks_better_by_sum", 1.0, 5.1)
    dominates_baseline = _score("safe", 4.0, 4.0)

    rejected = recommend_dominating_candidate(
        (baseline, worsens_fragmentation), "baseline", LOSS_SPECS
    )
    accepted = recommend_dominating_candidate(
        (baseline, worsens_fragmentation, dominates_baseline), "baseline", LOSS_SPECS
    )

    assert rejected.candidate_id is None
    assert accepted.candidate_id == "safe"
    assert "looks_better_by_sum" not in accepted.eligible_candidate_ids


def test_shared_cycle_coverage_is_a_non_regression_safeguard() -> None:
    specs = (
        MetricSpec("cycle_loss", MetricDirection.MINIMIZE),
        MetricSpec("cycle_observation_coverage", MetricDirection.MAXIMIZE),
    )
    baseline = CandidateScore(
        "baseline",
        {
            "cycle_loss": MetricEstimate(1.0, 0.0, 1),
            "cycle_observation_coverage": MetricEstimate(1.0, 0.0, 1),
        },
        0.0,
    )
    sparse = CandidateScore(
        "sparse",
        {
            "cycle_loss": MetricEstimate(0.0, 0.0, 1),
            "cycle_observation_coverage": MetricEstimate(0.25, 0.0, 1),
        },
        0.0,
    )

    recommendation = recommend_dominating_candidate(
        (baseline, sparse), "baseline", specs
    )

    assert recommendation.candidate_id is None


def test_recommendation_collapses_metric_equivalent_safe_candidates() -> None:
    baseline = _score("baseline", 5.0, 5.0)
    larger_change = _score("larger_change", 4.0, 4.0)
    smaller_change = _score("smaller_change", 4.0, 4.0)

    recommendation = recommend_dominating_candidate(
        (baseline, larger_change, smaller_change),
        "baseline",
        LOSS_SPECS,
        candidate_change_costs={"larger_change": 1.0, "smaller_change": 0.1},
    )

    assert recommendation.candidate_id == "smaller_change"
    assert set(recommendation.eligible_candidate_ids) == {
        "larger_change",
        "smaller_change",
    }


def test_baseline_protection_rejects_uncertain_or_insufficient_heldout_gain() -> None:
    baseline = CandidateScore(
        "baseline",
        {
            "cycle_loss": MetricEstimate(10.0, 2.0, 4),
            "fragment_loss": MetricEstimate(10.0, 0.1, 4),
        },
        0.0,
    )
    uncertain = CandidateScore(
        "uncertain",
        {
            "cycle_loss": MetricEstimate(8.0, 2.0, 4),
            "fragment_loss": MetricEstimate(9.0, 0.1, 4),
        },
        0.0,
    )
    stable = CandidateScore(
        "stable",
        {
            "cycle_loss": MetricEstimate(6.0, 0.1, 4),
            "fragment_loss": MetricEstimate(9.0, 0.1, 4),
        },
        0.0,
    )

    rejected = decide_heldout_baseline_protection(
        baseline,
        uncertain,
        LOSS_SPECS,
        primary_metric="cycle_loss",
        minimum_improvement=0.5,
    )
    accepted = decide_heldout_baseline_protection(
        baseline,
        stable,
        LOSS_SPECS,
        primary_metric="cycle_loss",
        minimum_improvement=0.5,
    )

    assert not rejected.accepted
    assert "cycle_loss" in rejected.reason
    assert accepted.accepted


def test_baseline_protection_uses_paired_regions_and_requires_enough_evidence() -> None:
    evaluations = []
    for index, baseline_cycle in enumerate((0.1, 0.2, 0.3, 0.4), start=1):
        perturbation_id = f"region-{index}"
        evaluations.extend(
            (
                CandidateEvaluation(
                    "baseline",
                    {"cycle_loss": baseline_cycle, "fragment_loss": 1.0},
                    perturbation_id,
                ),
                CandidateEvaluation(
                    "candidate",
                    {
                        "cycle_loss": baseline_cycle - 0.1,
                        "fragment_loss": 1.0,
                    },
                    perturbation_id,
                ),
            )
        )
    baseline, candidate = aggregate_candidate_evaluations(evaluations, LOSS_SPECS)

    paired = decide_heldout_baseline_protection(
        baseline,
        candidate,
        LOSS_SPECS,
        primary_metric="cycle_loss",
        minimum_improvement=0.05,
    )
    insufficient = decide_heldout_baseline_protection(
        aggregate_candidate_evaluations(evaluations[:-2], LOSS_SPECS)[0],
        aggregate_candidate_evaluations(evaluations[:-2], LOSS_SPECS)[1],
        LOSS_SPECS,
        primary_metric="cycle_loss",
        minimum_improvement=0.05,
    )

    # The marginal values vary, but each paired region improves by exactly
    # 0.1.  Treating the two samples as independent spuriously invents error.
    assert paired.accepted
    assert paired.conservative_improvements["cycle_loss"] == pytest.approx(0.1)
    assert not insufficient.accepted
    assert "insufficient paired" in insufficient.reason


def test_inputs_are_validated_instead_of_turning_missing_evidence_into_a_score() -> (
    None
):
    with pytest.raises(ValueError, match="no position"):
        forward_backward_cycle_consistency(
            np.full((2, 1, 2), np.nan), np.full((2, 1, 2), np.nan)
        )
    with pytest.raises(ValueError, match="exactly"):
        aggregate_candidate_evaluations(
            (CandidateEvaluation("x", {"cycle_loss": 1.0}),), LOSS_SPECS
        )
    with pytest.raises(ValueError, match="direction"):
        MetricSpec("cycle_loss", "sideways")  # type: ignore[arg-type]
