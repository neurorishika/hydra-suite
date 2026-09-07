"""Conservative, unlabeled signals for comparing tracking candidates.

These utilities measure internal consistency and repeated-evaluation stability.
They are deliberately *not* a substitute for labelled tracking accuracy: a low
cycle error can still describe a consistently wrong tracker.  Callers should
use them to reject clearly worse candidates and retain a baseline unless there
is held-out, uncertainty-aware evidence for an improvement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, erfc, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t


class MetricDirection(str, Enum):
    """Whether a metric is a loss or a utility."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """A named metric with its explicit optimization direction."""

    name: str
    direction: MetricDirection

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name must not be empty")
        try:
            direction = MetricDirection(self.direction)
        except ValueError as exc:
            raise ValueError("metric direction must be minimize or maximize") from exc
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True, slots=True)
class CycleConsistency:
    """Summary of aligned forward/backward positional disagreement."""

    mean_normalized_error: float
    median_normalized_error: float
    p95_normalized_error: float
    valid_observations: int
    spatial_scale: float
    available_observations: int = 0
    shared_observation_coverage: float = 0.0


@dataclass(frozen=True, slots=True)
class SlotAlignment:
    """A full-window mapping from forward slots to backward slots.

    The mapping is deliberately computed once for a complete evaluation window
    and then reused for temporal-region estimates.  Re-fitting it per region
    would make a time-varying identity swap look locally consistent.
    """

    backward_for_forward: tuple[int | None, ...]
    shared_observations: int
    available_observations: int
    spatial_scale: float

    def __post_init__(self) -> None:
        if not self.backward_for_forward:
            raise ValueError("slot alignment requires at least one forward slot")
        if self.shared_observations < 0 or self.available_observations < 0:
            raise ValueError("slot alignment observation counts must be non-negative")
        if self.shared_observations > self.available_observations:
            raise ValueError(
                "slot alignment shared observations cannot exceed available observations"
            )
        if not np.isfinite(self.spatial_scale) or self.spatial_scale <= 0:
            raise ValueError("slot alignment spatial_scale must be finite and positive")
        seen = [index for index in self.backward_for_forward if index is not None]
        if len(seen) != len(set(seen)) or any(index < 0 for index in seen):
            raise ValueError("slot alignment must map backward slots one-to-one")


@dataclass(frozen=True, slots=True)
class TrajectoryQualityMetrics:
    """Position-only losses derived from emitted track locations.

    These losses intentionally do not inspect Kalman-filter state, assignment
    costs, or any other hidden tracker accumulator.  A low value is therefore
    only evidence of output regularity and coverage, not labelled accuracy.
    """

    coverage_loss: float
    fragmentation_loss: float
    motion_roughness_loss: float
    observed_positions: int
    total_positions: int
    temporal_transition_count: int = 0
    temporal_triplet_count: int = 0
    valid_motion_triplets: int = 0


@dataclass(frozen=True, slots=True)
class OutputSanityMetrics:
    """Unlabelled output safeguards, never a proxy for tracking truth.

    These diagnostics deliberately flag only self-evident output pathologies:
    duplicate/colliding slots, excess source detections, and a starved track
    slot.  They cannot establish that a candidate found the correct animals.
    """

    collision_loss: float
    detection_excess_loss: float
    worst_track_coverage_loss: float


@dataclass(frozen=True, slots=True, order=True)
class TemporalSegment:
    """A half-open contiguous interval of frame indices."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("segments require 0 <= start < stop")

    @property
    def frame_count(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class TemporalTrainValidationSplit:
    """Chronological train/validation segments and an optional excluded gap."""

    train: tuple[TemporalSegment, ...]
    validation: tuple[TemporalSegment, ...]
    gap: tuple[TemporalSegment, ...]


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    """A metric mean and its sample standard deviation across perturbations."""

    mean: float
    std: float
    sample_count: int

    def __post_init__(self) -> None:
        if self.sample_count < 1:
            raise ValueError("sample_count must be at least one")
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std < 0:
            raise ValueError(
                "metric estimates must be finite and have non-negative std"
            )

    @property
    def standard_error(self) -> float:
        return self.std / sqrt(self.sample_count)


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """One candidate evaluated on one deterministic perturbation or held-out run."""

    candidate_id: str
    metrics: Mapping[str, float]
    perturbation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not self.metrics:
            raise ValueError("candidate metrics must not be empty")
        for name, value in self.metrics.items():
            if not name or not _is_finite_scalar(value):
                raise ValueError("candidate metric names and values must be finite")


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Aggregated repeated measurements for one candidate.

    ``mean_relative_spread`` is descriptive stability information, not an
    accuracy score.  It is never used as a surrogate oracle by this module.
    """

    candidate_id: str
    metrics: Mapping[str, MetricEstimate]
    mean_relative_spread: float
    perturbation_metrics: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not self.metrics:
            raise ValueError("candidate metrics must not be empty")
        if any(
            not name or not isinstance(estimate, MetricEstimate)
            for name, estimate in self.metrics.items()
        ):
            raise ValueError("candidate score metrics must map names to MetricEstimate")
        if not np.isfinite(self.mean_relative_spread) or self.mean_relative_spread < 0:
            raise ValueError("mean_relative_spread must be finite and non-negative")
        seen_metric_sets: set[frozenset[str]] = set()
        for perturbation_id, values in self.perturbation_metrics.items():
            if not perturbation_id or not isinstance(values, Mapping):
                raise ValueError("perturbation metrics require named metric mappings")
            if any(
                not name or not _is_finite_scalar(value)
                for name, value in values.items()
            ):
                raise ValueError("perturbation metrics must be finite")
            seen_metric_sets.add(frozenset(values))
        if len(seen_metric_sets) > 1:
            raise ValueError("perturbation metric names must be consistent")
        if seen_metric_sets and next(iter(seen_metric_sets)) != frozenset(self.metrics):
            raise ValueError(
                "perturbation metric names must match aggregated score metrics"
            )


@dataclass(frozen=True, slots=True)
class ParetoRank:
    """One-based non-dominated-front rank (one is the Pareto frontier)."""

    candidate_id: str
    front: int


@dataclass(frozen=True, slots=True)
class CandidateRecommendation:
    """A conservative recommendation, or an explicit reason to keep the baseline."""

    candidate_id: str | None
    eligible_candidate_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class BaselineProtectionDecision:
    """Held-out promotion decision using a lower confidence bound on improvement."""

    accepted: bool
    reason: str
    conservative_improvements: Mapping[str, float]


FloatArray = NDArray[np.float64]


def forward_backward_cycle_consistency(
    forward_positions: NDArray[np.floating],
    backward_positions: NDArray[np.floating],
    *,
    backward_is_reverse_chronological: bool = True,
    spatial_scale: float | None = None,
    slot_alignment: SlotAlignment | None = None,
    minimum_shared_observations: int = 2,
    slot_arena: Sequence[int] | NDArray[np.integer] | None = None,
) -> CycleConsistency:
    """Measure forward/backward disagreement after aligning time direction.

    Arrays have shape ``(frames, tracks, coordinates)`` and use ``NaN`` for a
    missing track.  Before scoring, backward slots are globally aligned to
    forward slots with an overlap-first Hungarian assignment, then the same
    pooled normalized errors are reported.  This removes arbitrary bootstrap
    slot labels without hiding a time-varying identity swap or letting a
    one-frame coincidence replace longer evidence.  When ``slot_arena`` is
    supplied, alignment is additionally restricted to the fixed arena of each
    track slot; a replayed slot must never earn cycle evidence by borrowing a
    complementary slot in another arena. Pairs with insufficient overlap
    cannot contribute evidence.  With no supplied scale, a robust inter-track
    distance (then temporal displacement) is inferred; it falls back to one
    for a stationary singleton track.
    """

    forward = _validated_positions(forward_positions, "forward_positions")
    backward = _validated_positions(backward_positions, "backward_positions")
    if forward.shape != backward.shape:
        raise ValueError(
            "forward_positions and backward_positions must have equal shape"
        )
    if backward_is_reverse_chronological:
        backward = backward[::-1]

    shared_count = _validated_minimum_shared_observations(minimum_shared_observations)
    arena_labels = _validated_slot_arena(slot_arena, forward.shape[1])
    scale = _resolve_spatial_scale(forward, spatial_scale)
    alignment = slot_alignment or global_slot_alignment(
        forward,
        backward,
        backward_is_reverse_chronological=False,
        spatial_scale=scale,
        minimum_shared_observations=shared_count,
        slot_arena=arena_labels,
    )
    if not np.isclose(scale, alignment.spatial_scale, rtol=1e-9, atol=1e-12):
        raise ValueError("slot_alignment spatial_scale does not match this score")
    errors, shared_observations = _errors_for_slot_alignment(
        forward,
        backward,
        alignment,
        scale,
        minimum_shared_observations=shared_count,
        slot_arena=arena_labels,
    )
    available_observations = min(
        int(np.count_nonzero(np.isfinite(forward).all(axis=2))),
        int(np.count_nonzero(np.isfinite(backward).all(axis=2))),
    )
    return CycleConsistency(
        mean_normalized_error=float(np.mean(errors)),
        median_normalized_error=float(np.median(errors)),
        p95_normalized_error=float(np.percentile(errors, 95)),
        valid_observations=int(errors.size),
        spatial_scale=scale,
        available_observations=available_observations,
        shared_observation_coverage=(
            float(shared_observations / available_observations)
            if available_observations
            else 0.0
        ),
    )


def global_slot_alignment(
    forward_positions: NDArray[np.floating],
    backward_positions: NDArray[np.floating],
    *,
    backward_is_reverse_chronological: bool = True,
    spatial_scale: float | None = None,
    minimum_shared_observations: int = 2,
    slot_arena: Sequence[int] | NDArray[np.integer] | None = None,
) -> SlotAlignment:
    """Find one robust, full-window forward/backward slot mapping.

    Assignment first maximises the number of sufficiently shared observations,
    then minimises the same pooled normalized error reported by
    :func:`forward_backward_cycle_consistency`.  This prevents two accidental
    one-frame matches from replacing a longer, informative pairing. Optional
    fixed arena labels only permit mappings within the same arena, matching
    the production assigner's static slot membership.
    """

    forward = _validated_positions(forward_positions, "forward_positions")
    backward = _validated_positions(backward_positions, "backward_positions")
    if forward.shape != backward.shape:
        raise ValueError(
            "forward_positions and backward_positions must have equal shape"
        )
    if backward_is_reverse_chronological:
        backward = backward[::-1]
    shared_count = _validated_minimum_shared_observations(minimum_shared_observations)
    arena_labels = _validated_slot_arena(slot_arena, forward.shape[1])
    scale = _resolve_spatial_scale(forward, spatial_scale)
    mapping, shared_observations = _robust_global_slot_mapping(
        forward,
        backward,
        scale,
        minimum_shared_observations=shared_count,
        slot_arena=arena_labels,
    )
    available_observations = min(
        int(np.count_nonzero(np.isfinite(forward).all(axis=2))),
        int(np.count_nonzero(np.isfinite(backward).all(axis=2))),
    )
    return SlotAlignment(
        backward_for_forward=mapping,
        shared_observations=shared_observations,
        available_observations=available_observations,
        spatial_scale=scale,
    )


def trajectory_quality_metrics(
    positions: NDArray[np.floating],
    *,
    spatial_scale: float,
    segment: TemporalSegment | None = None,
) -> TrajectoryQualityMetrics:
    """Calculate coverage, gap-transition, and acceleration roughness losses.

    ``positions`` has shape ``(frames, tracks, coordinates)`` and uses NaN for
    missing observations.  Coverage loss is the missing-position fraction, so
    an all-missing trajectory is explicitly penalized with a loss of one.
    Fragmentation loss is the fraction of temporal adjacency pairs that change
    between present and missing.  Motion roughness is the mean norm of the
    second finite difference over only three-frame, fully observed windows,
    divided by ``spatial_scale``.  It is zero for constant-velocity motion.
    """

    observed_positions = _validated_positions(positions, "positions")
    if not np.isfinite(spatial_scale) or spatial_scale <= 0:
        raise ValueError("spatial_scale must be finite and positive")
    observed = np.isfinite(observed_positions).all(axis=2)
    selected = _validated_metric_segment(segment, observed.shape[0])
    region_observed = observed[selected.start : selected.stop]
    total = int(region_observed.size)
    observed_count = int(np.count_nonzero(region_observed))
    coverage_loss = 1.0 - observed_count / total

    # Assign a transition/triplet to the region containing its final frame.
    # This yields a non-overlapping partition of full-window temporal evidence
    # and keeps events that cross a temporal-region boundary.
    transition_start = max(1, selected.start)
    transition_stop = selected.stop
    transition_count = max(0, transition_stop - transition_start) * observed.shape[1]
    if transition_start >= transition_stop:
        fragmentation_loss = 0.0
    else:
        transitions = (
            observed[transition_start:transition_stop]
            != observed[transition_start - 1 : transition_stop - 1]
        )
        fragmentation_loss = float(np.count_nonzero(transitions) / transitions.size)

    triplet_start = max(2, selected.start)
    triplet_stop = selected.stop
    triplet_count = max(0, triplet_stop - triplet_start) * observed.shape[1]
    if triplet_start >= triplet_stop:
        motion_roughness_loss = 0.0
        valid_motion_triplets = 0
    else:
        triplets = (
            observed[triplet_start - 2 : triplet_stop - 2]
            & observed[triplet_start - 1 : triplet_stop - 1]
            & observed[triplet_start:triplet_stop]
        )
        accelerations = (
            observed_positions[triplet_start:triplet_stop]
            - 2.0 * observed_positions[triplet_start - 1 : triplet_stop - 1]
            + observed_positions[triplet_start - 2 : triplet_stop - 2]
        )
        roughnesses = np.linalg.norm(accelerations, axis=2)[triplets]
        valid_motion_triplets = int(roughnesses.size)
        motion_roughness_loss = (
            float(np.mean(roughnesses) / spatial_scale) if roughnesses.size else 0.0
        )

    return TrajectoryQualityMetrics(
        coverage_loss=coverage_loss,
        fragmentation_loss=fragmentation_loss,
        motion_roughness_loss=motion_roughness_loss,
        observed_positions=observed_count,
        total_positions=total,
        temporal_transition_count=transition_count,
        temporal_triplet_count=triplet_count,
        valid_motion_triplets=valid_motion_triplets,
    )


def output_sanity_metrics(
    positions: NDArray[np.floating],
    *,
    spatial_scale: float,
    detection_counts: Sequence[int] | NDArray[np.integer] | None = None,
    segment: TemporalSegment | None = None,
    collision_distance_fraction: float = 0.5,
    slot_arena: Sequence[int] | NDArray[np.integer] | None = None,
) -> OutputSanityMetrics:
    """Calculate output-pathology safeguards for one complete or regional slice.

    ``detection_counts`` are source detections after the candidate's production
    filtering contract.  An excess is a useful false-positive *risk signal*,
    not a labelled false-positive rate.  If counts are unavailable, that
    signal is neutral rather than invented from unlabelled tracks. With fixed
    multi-arena slot labels, collision comparisons are restricted to slots in
    the same arena: spatially adjacent animals separated by a real arena
    boundary are not duplicate output evidence.
    """

    observed_positions = _validated_positions(positions, "positions")
    if not np.isfinite(spatial_scale) or spatial_scale <= 0:
        raise ValueError("spatial_scale must be finite and positive")
    if not np.isfinite(collision_distance_fraction) or collision_distance_fraction <= 0:
        raise ValueError("collision_distance_fraction must be finite and positive")
    selected = _validated_metric_segment(segment, observed_positions.shape[0])
    region_positions = observed_positions[selected.start : selected.stop]
    observed = np.isfinite(region_positions).all(axis=2)
    track_count = observed.shape[1]
    arena_labels = _validated_slot_arena(slot_arena, track_count)
    per_track_coverage = np.mean(observed, axis=0)
    worst_track_coverage_loss = float(1.0 - np.min(per_track_coverage))

    collision_count = 0
    comparable_pairs = 0
    threshold = collision_distance_fraction * spatial_scale
    for frame, visible in zip(region_positions, observed, strict=True):
        present = np.flatnonzero(visible)
        for left_offset, left_track in enumerate(present):
            for right_track in present[left_offset + 1 :]:
                if (
                    arena_labels is not None
                    and arena_labels[left_track] != arena_labels[right_track]
                ):
                    continue
                comparable_pairs += 1
                distance = float(np.linalg.norm(frame[left_track] - frame[right_track]))
                collision_count += int(distance < threshold)
    collision_loss = (
        float(collision_count / comparable_pairs) if comparable_pairs else 0.0
    )

    detection_excess_loss = 0.0
    if detection_counts is not None:
        counts = np.asarray(detection_counts, dtype=float)
        if counts.ndim != 1 or counts.shape[0] != observed_positions.shape[0]:
            raise ValueError("detection_counts must contain one value per frame")
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("detection_counts must be finite and non-negative")
        region_counts = counts[selected.start : selected.stop]
        detection_excess_loss = float(
            np.mean(np.clip(region_counts / track_count - 1.0, 0.0, 1.0))
        )

    return OutputSanityMetrics(
        collision_loss=collision_loss,
        detection_excess_loss=detection_excess_loss,
        worst_track_coverage_loss=worst_track_coverage_loss,
    )


def partition_temporal_segments(
    frame_count: int, segment_count: int
) -> tuple[TemporalSegment, ...]:
    """Partition a sequence into balanced, adjacent, non-empty intervals."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if segment_count < 1 or segment_count > frame_count:
        raise ValueError("segment_count must be between one and frame_count")
    base, extra = divmod(frame_count, segment_count)
    start = 0
    result: list[TemporalSegment] = []
    for index in range(segment_count):
        stop = start + base + int(index < extra)
        result.append(TemporalSegment(start, stop))
        start = stop
    return tuple(result)


def build_train_validation_split(
    frame_count: int,
    validation_fraction: float,
    *,
    gap_frames: int = 0,
) -> TemporalTrainValidationSplit:
    """Create a chronological train/held-out-validation split.

    The optional gap is deliberately excluded from both subsets to reduce
    temporal leakage at their boundary.  This function never shuffles frames.
    """

    if frame_count < 2:
        raise ValueError(
            "at least two frames are required for a train/validation split"
        )
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    if gap_frames < 0:
        raise ValueError("gap_frames must be non-negative")
    validation_frames = max(1, ceil(frame_count * validation_fraction))
    train_stop = frame_count - validation_frames - gap_frames
    if train_stop < 1:
        raise ValueError(
            "frame_count leaves no training frames after validation and gap"
        )
    validation_start = train_stop + gap_frames
    gap = (TemporalSegment(train_stop, validation_start),) if gap_frames else ()
    return TemporalTrainValidationSplit(
        train=(TemporalSegment(0, train_stop),),
        validation=(TemporalSegment(validation_start, frame_count),),
        gap=gap,
    )


def aggregate_candidate_evaluations(
    evaluations: Sequence[CandidateEvaluation], metric_specs: Sequence[MetricSpec]
) -> tuple[CandidateScore, ...]:
    """Aggregate repeated perturbation evaluations without collapsing directions.

    Every evaluation must provide exactly the declared metric names.  Results
    are ordered by candidate id, making aggregation deterministic regardless of
    evaluation arrival order.
    """

    specs = _validate_metric_specs(metric_specs)
    if not evaluations:
        raise ValueError("at least one candidate evaluation is required")
    expected_names = {spec.name for spec in specs}
    grouped: dict[str, list[CandidateEvaluation]] = {}
    for evaluation in evaluations:
        if set(evaluation.metrics) != expected_names:
            raise ValueError(
                "each evaluation must provide exactly the declared metrics"
            )
        grouped.setdefault(evaluation.candidate_id, []).append(evaluation)

    scores: list[CandidateScore] = []
    for candidate_id in sorted(grouped):
        samples = grouped[candidate_id]
        estimates: dict[str, MetricEstimate] = {}
        spreads: list[float] = []
        perturbation_metrics: dict[str, Mapping[str, float]] = {}
        for sample in samples:
            if sample.perturbation_id is None:
                continue
            if sample.perturbation_id in perturbation_metrics:
                raise ValueError(
                    "candidate evaluations must not repeat a perturbation_id"
                )
            perturbation_metrics[sample.perturbation_id] = dict(sample.metrics)
        for spec in specs:
            values = np.asarray(
                [sample.metrics[spec.name] for sample in samples], dtype=float
            )
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            estimates[spec.name] = MetricEstimate(mean, std, len(values))
            spreads.append(std / max(abs(mean), 1e-12))
        scores.append(
            CandidateScore(
                candidate_id=candidate_id,
                metrics=estimates,
                mean_relative_spread=float(np.mean(spreads)),
                perturbation_metrics=perturbation_metrics,
            )
        )
    return tuple(scores)


def dominates(
    left: CandidateScore, right: CandidateScore, metric_specs: Sequence[MetricSpec]
) -> bool:
    """Return whether ``left`` is no worse on every metric and better on one."""

    _validate_scores_and_specs((left, right), metric_specs)
    specs = _validate_metric_specs(metric_specs)
    strictly_better = False
    for spec in specs:
        left_value = left.metrics[spec.name].mean
        right_value = right.metrics[spec.name].mean
        if spec.direction is MetricDirection.MINIMIZE:
            if left_value > right_value:
                return False
            strictly_better |= left_value < right_value
        else:
            if left_value < right_value:
                return False
            strictly_better |= left_value > right_value
    return strictly_better


def pareto_frontier(
    scores: Sequence[CandidateScore], metric_specs: Sequence[MetricSpec]
) -> tuple[CandidateScore, ...]:
    """Return the non-dominated candidates in deterministic id order."""

    _validate_scores_and_specs(scores, metric_specs)
    return tuple(
        score
        for score in sorted(scores, key=lambda item: item.candidate_id)
        if not any(
            other.candidate_id != score.candidate_id
            and dominates(other, score, metric_specs)
            for other in scores
        )
    )


def pareto_ranks(
    scores: Sequence[CandidateScore], metric_specs: Sequence[MetricSpec]
) -> tuple[ParetoRank, ...]:
    """Assign deterministic non-dominated-front ranks without scalarizing losses."""

    remaining = list(_validate_scores_and_specs(scores, metric_specs))
    ranks: list[ParetoRank] = []
    front_number = 1
    while remaining:
        front = pareto_frontier(remaining, metric_specs)
        ranks.extend(ParetoRank(score.candidate_id, front_number) for score in front)
        front_ids = {score.candidate_id for score in front}
        remaining = [
            score for score in remaining if score.candidate_id not in front_ids
        ]
        front_number += 1
    return tuple(sorted(ranks, key=lambda item: item.candidate_id))


def recommend_dominating_candidate(
    scores: Sequence[CandidateScore],
    baseline_id: str,
    metric_specs: Sequence[MetricSpec],
    *,
    candidate_change_costs: Mapping[str, float] | None = None,
    equivalence_atol: float = 1e-9,
    equivalence_rtol: float = 1e-6,
) -> CandidateRecommendation:
    """Recommend only an unambiguous candidate that dominates the baseline.

    If multiple baseline-safe alternatives trade off against each other, this
    refuses to invent a weighted winner.  In particular, worsening any loss
    relative to the baseline can never result in a recommendation.
    """

    checked_scores = _validate_scores_and_specs(scores, metric_specs)
    if equivalence_atol < 0 or equivalence_rtol < 0:
        raise ValueError("metric equivalence tolerances must be non-negative")
    change_costs = dict(candidate_change_costs or {})
    if any(
        not _is_finite_scalar(value) or float(value) < 0
        for value in change_costs.values()
    ):
        raise ValueError("candidate change costs must be finite and non-negative")
    baseline = _score_by_id(checked_scores, baseline_id)
    safe = tuple(
        score
        for score in checked_scores
        if score.candidate_id != baseline_id
        and dominates(score, baseline, metric_specs)
    )
    safe_ids = tuple(sorted(score.candidate_id for score in safe))
    if not safe:
        return CandidateRecommendation(
            None, safe_ids, "no candidate dominates the baseline"
        )
    representatives = _metric_equivalence_representatives(
        safe,
        metric_specs,
        change_costs,
        atol=equivalence_atol,
        rtol=equivalence_rtol,
    )
    unambiguous = tuple(
        score
        for score in representatives
        if all(
            other.candidate_id == score.candidate_id
            or dominates(score, other, metric_specs)
            for other in representatives
        )
    )
    if len(unambiguous) != 1:
        return CandidateRecommendation(
            None,
            safe_ids,
            "baseline-safe candidates have unresolved Pareto trade-offs",
        )
    equivalent_count = len(safe) - len(representatives)
    equivalence_note = (
        f" after collapsing {equivalent_count} metric-equivalent candidate(s)"
        if equivalent_count
        else ""
    )
    return CandidateRecommendation(
        unambiguous[0].candidate_id,
        safe_ids,
        "candidate dominates the baseline and every other baseline-safe candidate"
        + equivalence_note,
    )


def decide_heldout_baseline_protection(
    baseline: CandidateScore,
    candidate: CandidateScore,
    metric_specs: Sequence[MetricSpec],
    *,
    primary_metric: str,
    minimum_improvement: float,
    confidence_z: float = 1.96,
    minimum_paired_samples: int = 4,
    selection_family_size: int = 1,
) -> BaselineProtectionDecision:
    """Accept a candidate only with conservative held-out improvement evidence.

    For each metric the lower confidence bound of the direction-aware
    improvement must be non-negative.  The primary metric must additionally
    clear ``minimum_improvement``.  When aggregation preserved matched
    perturbation ids, inference is based on paired region differences and a
    one-sided Student-t lower bound adjusted for the shortlisted candidate
    family.  When legacy pre-aggregated scores have no perturbation evidence,
    the former independent-standard-error behavior remains available for API
    compatibility, but production callers must provide paired evidence.
    """

    _validate_scores_and_specs((baseline, candidate), metric_specs)
    specs = _validate_metric_specs(metric_specs)
    if primary_metric not in {spec.name for spec in specs}:
        raise ValueError("primary_metric must be declared in metric_specs")
    if minimum_improvement < 0 or not np.isfinite(minimum_improvement):
        raise ValueError("minimum_improvement must be finite and non-negative")
    if confidence_z < 0 or not np.isfinite(confidence_z):
        raise ValueError("confidence_z must be finite and non-negative")
    if isinstance(minimum_paired_samples, bool) or minimum_paired_samples < 2:
        raise ValueError("minimum_paired_samples must be at least two")
    if isinstance(selection_family_size, bool) or selection_family_size < 1:
        raise ValueError("selection_family_size must be at least one")

    base_samples = baseline.perturbation_metrics
    candidate_samples = candidate.perturbation_metrics
    if base_samples or candidate_samples:
        if not base_samples or not candidate_samples:
            return BaselineProtectionDecision(
                False,
                "held-out validation has incomplete paired region evidence",
                {},
            )
        base_ids = set(base_samples)
        candidate_ids = set(candidate_samples)
        if base_ids != candidate_ids:
            return BaselineProtectionDecision(
                False,
                "held-out validation has unmatched paired region evidence",
                {},
            )
        if len(base_ids) < minimum_paired_samples:
            return BaselineProtectionDecision(
                False,
                "insufficient paired held-out regions "
                f"(need at least {minimum_paired_samples}, found {len(base_ids)})",
                {},
            )
        conservative = _paired_conservative_improvements(
            base_samples,
            candidate_samples,
            specs,
            confidence_z=confidence_z,
            selection_family_size=int(selection_family_size),
        )
        return _baseline_protection_decision(
            conservative, primary_metric, minimum_improvement
        )

    conservative: dict[str, float] = {}
    for spec in specs:
        base = baseline.metrics[spec.name]
        trial = candidate.metrics[spec.name]
        raw_improvement = (
            base.mean - trial.mean
            if spec.direction is MetricDirection.MINIMIZE
            else trial.mean - base.mean
        )
        uncertainty = sqrt(base.standard_error**2 + trial.standard_error**2)
        conservative[spec.name] = raw_improvement - confidence_z * uncertainty

    return _baseline_protection_decision(
        conservative, primary_metric, minimum_improvement
    )


def _baseline_protection_decision(
    conservative: Mapping[str, float], primary_metric: str, minimum_improvement: float
) -> BaselineProtectionDecision:
    regressions = tuple(
        name for name, improvement in conservative.items() if improvement < 0
    )
    if regressions:
        return BaselineProtectionDecision(
            False,
            "held-out uncertainty does not rule out regression in: "
            + ", ".join(regressions),
            conservative,
        )
    if conservative[primary_metric] < minimum_improvement:
        return BaselineProtectionDecision(
            False,
            "primary held-out improvement does not clear the required margin",
            conservative,
        )
    return BaselineProtectionDecision(
        True,
        "candidate clears held-out improvement margin without conservative regressions",
        conservative,
    )


def _paired_conservative_improvements(
    baseline_samples: Mapping[str, Mapping[str, float]],
    candidate_samples: Mapping[str, Mapping[str, float]],
    metric_specs: Sequence[MetricSpec],
    *,
    confidence_z: float,
    selection_family_size: int,
) -> dict[str, float]:
    """Return family-adjusted one-sided t lower bounds on paired improvements."""

    # ``confidence_z`` is the historical two-sided normal threshold.  Convert
    # its upper-tail probability to a one-sided alpha, then Bonferroni-adjust
    # for both selected candidates and metrics.  This deliberately errs on
    # retaining the current parameters when a short tail is ambiguous.
    one_sided_alpha = 0.5 * erfc(confidence_z / sqrt(2.0))
    comparison_count = max(1, selection_family_size * len(metric_specs))
    adjusted_alpha = one_sided_alpha / comparison_count
    sample_count = len(baseline_samples)
    critical_value = float(student_t.ppf(1.0 - adjusted_alpha, sample_count - 1))

    conservative: dict[str, float] = {}
    for spec in metric_specs:
        differences = np.asarray(
            [
                (
                    baseline_samples[perturbation_id][spec.name]
                    - candidate_samples[perturbation_id][spec.name]
                    if spec.direction is MetricDirection.MINIMIZE
                    else candidate_samples[perturbation_id][spec.name]
                    - baseline_samples[perturbation_id][spec.name]
                )
                for perturbation_id in sorted(baseline_samples)
            ],
            dtype=float,
        )
        standard_error = (
            float(np.std(differences, ddof=1) / sqrt(sample_count))
            if sample_count > 1
            else 0.0
        )
        conservative[spec.name] = float(
            np.mean(differences) - critical_value * standard_error
        )
    return conservative


def _validated_positions(values: NDArray[np.floating], name: str) -> FloatArray:
    positions = np.asarray(values, dtype=np.float64)
    if positions.ndim != 3 or any(dimension < 1 for dimension in positions.shape):
        raise ValueError(f"{name} must have shape (frames, tracks, coordinates)")
    if np.isinf(positions).any():
        raise ValueError(f"{name} may contain NaN for missing data but not infinity")
    return positions


def _validated_slot_arena(
    values: Sequence[int] | NDArray[np.integer] | None,
    track_count: int,
) -> NDArray[np.int64] | None:
    """Validate fixed per-slot arena labels for optional grouped metrics."""
    if values is None:
        return None
    try:
        numeric = np.asarray(values, dtype=np.float64)
        labels = np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "slot_arena must contain finite non-negative integers"
        ) from exc
    if labels.ndim != 1 or labels.shape[0] != track_count:
        raise ValueError("slot_arena must contain one label per track slot")
    if (
        numeric.ndim != 1
        or numeric.shape[0] != track_count
        or not np.isfinite(numeric).all()
        or np.any(numeric != labels)
        or np.any(labels < 0)
    ):
        raise ValueError("slot_arena must contain finite non-negative integers")
    return labels


def _resolve_spatial_scale(
    positions: FloatArray, supplied_scale: float | None
) -> float:
    if supplied_scale is not None:
        if not np.isfinite(supplied_scale) or supplied_scale <= 0:
            raise ValueError("spatial_scale must be finite and positive")
        return float(supplied_scale)

    observed = np.isfinite(positions).all(axis=2)
    inter_track: list[float] = []
    for frame, mask in zip(positions, observed, strict=True):
        points = frame[mask]
        if len(points) > 1:
            distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
            inter_track.extend(distances[np.triu_indices(len(points), k=1)].tolist())
    positive = np.asarray([value for value in inter_track if value > 0], dtype=float)
    if positive.size:
        return float(np.median(positive))

    steps = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    step_valid = observed[1:] & observed[:-1]
    positive_steps = steps[step_valid & (steps > 0)]
    return float(np.median(positive_steps)) if positive_steps.size else 1.0


def _robust_global_slot_mapping(
    forward: FloatArray,
    backward: FloatArray,
    scale: float,
    *,
    minimum_shared_observations: int,
    slot_arena: NDArray[np.int64] | None = None,
) -> tuple[tuple[int | None, ...], int]:
    """Lexicographically maximise overlap, then minimise pooled cycle error."""

    track_count = forward.shape[1]
    overlaps = np.zeros((track_count, track_count), dtype=np.int64)
    total_errors = np.full((track_count, track_count), np.inf, dtype=float)
    for forward_track in range(track_count):
        for backward_track in range(track_count):
            if (
                slot_arena is not None
                and slot_arena[forward_track] != slot_arena[backward_track]
            ):
                continue
            valid = np.isfinite(forward[:, forward_track]).all(axis=1) & np.isfinite(
                backward[:, backward_track]
            ).all(axis=1)
            overlap = int(np.count_nonzero(valid))
            if overlap < minimum_shared_observations:
                continue
            differences = (
                forward[valid, forward_track] - backward[valid, backward_track]
            )
            overlaps[forward_track, backward_track] = overlap
            total_errors[forward_track, backward_track] = float(
                np.sum(np.linalg.norm(differences, axis=1) / scale)
            )

    eligible = np.isfinite(total_errors)
    if not np.any(eligible):
        raise ValueError(
            "no position is observed in a forward/backward slot pair with "
            "sufficient shared observations"
        )

    # With a fixed total overlap, minimising the sum of pair errors is exactly
    # minimising the pooled error later reported to callers.  The multiplier is
    # larger than every possible assignment-error difference, making one extra
    # shared observation dominate all error terms rather than acting as a soft
    # trade-off.
    overlap_priority = float(np.sum(total_errors[eligible]) + 1.0)
    eligible_cost = -overlaps * overlap_priority + total_errors
    # Match real slots in a rectangular-with-dummies problem rather than a
    # forced square real-slot problem.  Every forward slot can choose its own
    # zero-cost dummy column and every backward slot can be consumed by a
    # zero-cost dummy row.  Thus an unavailable real edge is never needed just
    # to complete a permutation, and the negative eligible costs genuinely
    # maximise *total* overlap before pooled error.  The input ordering and
    # SciPy's deterministic assignment tie resolution retain stable results.
    #
    # Layout: real forward rows / dummy-backward rows × real backward columns /
    # dummy-forward columns.  A positive unavailable cost is noncompetitive
    # with either unmatched route (zero) while every eligible edge is negative.
    unavailable_cost = 1.0
    assignment_cost = np.zeros((2 * track_count, 2 * track_count), dtype=float)
    assignment_cost[:track_count, :track_count] = np.where(
        eligible, eligible_cost, unavailable_cost
    )
    rows, columns = linear_sum_assignment(assignment_cost)
    mapping: list[int | None] = [None] * track_count
    shared_observations = 0
    for forward_track, backward_track in zip(rows, columns, strict=True):
        if forward_track >= track_count or backward_track >= track_count:
            continue
        if not eligible[forward_track, backward_track]:
            continue
        mapping[int(forward_track)] = int(backward_track)
        shared_observations += int(overlaps[forward_track, backward_track])
    if not shared_observations:  # Defensive: eligibility above guarantees this.
        raise ValueError("no position is observed in matched forward/backward tracks")
    return tuple(mapping), shared_observations


def _errors_for_slot_alignment(
    forward: FloatArray,
    backward: FloatArray,
    alignment: SlotAlignment,
    scale: float,
    *,
    minimum_shared_observations: int,
    slot_arena: NDArray[np.int64] | None = None,
) -> tuple[FloatArray, int]:
    """Return the pooled errors for a precomputed full-window slot mapping."""

    if len(alignment.backward_for_forward) != forward.shape[1]:
        raise ValueError("slot_alignment does not match the number of track slots")
    errors: list[float] = []
    shared_observations = 0
    for forward_track, backward_track in enumerate(alignment.backward_for_forward):
        if backward_track is None:
            continue
        if backward_track >= backward.shape[1]:
            raise ValueError("slot_alignment references an unavailable backward slot")
        if (
            slot_arena is not None
            and slot_arena[forward_track] != slot_arena[backward_track]
        ):
            raise ValueError("slot_alignment maps slots across fixed arena labels")
        valid = np.isfinite(forward[:, forward_track]).all(axis=1) & np.isfinite(
            backward[:, backward_track]
        ).all(axis=1)
        if np.count_nonzero(valid) < minimum_shared_observations:
            continue
        differences = forward[valid, forward_track] - backward[valid, backward_track]
        normalized = np.linalg.norm(differences, axis=1) / scale
        errors.extend(normalized.tolist())
        shared_observations += int(normalized.size)
    if not errors:
        raise ValueError("no position is observed in matched forward/backward tracks")
    return np.asarray(errors, dtype=np.float64), shared_observations


def _validated_minimum_shared_observations(value: int) -> int:
    """Return an integral overlap floor for alignment and regional reporting."""

    try:
        shared_count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_shared_observations must be at least one") from exc
    if isinstance(value, bool) or shared_count != value or shared_count < 1:
        raise ValueError("minimum_shared_observations must be at least one")
    return shared_count


def _validated_metric_segment(
    segment: TemporalSegment | None, frame_count: int
) -> TemporalSegment:
    if segment is None:
        return TemporalSegment(0, frame_count)
    if segment.stop > frame_count:
        raise ValueError("segment must lie within the supplied positions")
    return segment


def _metric_equivalence_representatives(
    scores: Sequence[CandidateScore],
    metric_specs: Sequence[MetricSpec],
    change_costs: Mapping[str, float],
    *,
    atol: float,
    rtol: float,
) -> tuple[CandidateScore, ...]:
    """Collapse metric-equivalent alternatives using change then id tie-breaks."""

    groups: list[list[CandidateScore]] = []
    for score in sorted(scores, key=lambda item: item.candidate_id):
        for group in groups:
            if _scores_metric_equivalent(
                score, group[0], metric_specs, atol=atol, rtol=rtol
            ):
                group.append(score)
                break
        else:
            groups.append([score])
    return tuple(
        min(
            group,
            key=lambda item: (
                float(change_costs.get(item.candidate_id, 0.0)),
                item.candidate_id,
            ),
        )
        for group in groups
    )


def _scores_metric_equivalent(
    left: CandidateScore,
    right: CandidateScore,
    metric_specs: Sequence[MetricSpec],
    *,
    atol: float,
    rtol: float,
) -> bool:
    return all(
        np.isclose(
            left.metrics[spec.name].mean,
            right.metrics[spec.name].mean,
            atol=atol,
            rtol=rtol,
        )
        for spec in metric_specs
    )


def _validate_metric_specs(
    metric_specs: Sequence[MetricSpec],
) -> tuple[MetricSpec, ...]:
    specs = tuple(metric_specs)
    if not specs:
        raise ValueError("at least one metric specification is required")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("metric specification names must be unique")
    return specs


def _validate_scores_and_specs(
    scores: Sequence[CandidateScore], metric_specs: Sequence[MetricSpec]
) -> tuple[CandidateScore, ...]:
    specs = _validate_metric_specs(metric_specs)
    checked_scores = tuple(scores)
    if not checked_scores:
        raise ValueError("at least one candidate score is required")
    ids = [score.candidate_id for score in checked_scores]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate ids must be unique")
    expected_names = {spec.name for spec in specs}
    for score in checked_scores:
        if set(score.metrics) != expected_names:
            raise ValueError(
                "each candidate score must provide exactly the declared metrics"
            )
    return checked_scores


def _score_by_id(scores: Sequence[CandidateScore], candidate_id: str) -> CandidateScore:
    for score in scores:
        if score.candidate_id == candidate_id:
            return score
    raise ValueError(f"baseline candidate {candidate_id!r} is not present")


def _is_finite_scalar(value: object) -> bool:
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError):
        return False
