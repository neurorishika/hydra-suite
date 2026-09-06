"""Conservative, unlabeled signals for comparing tracking candidates.

These utilities measure internal consistency and repeated-evaluation stability.
They are deliberately *not* a substitute for labelled tracking accuracy: a low
cycle error can still describe a consistently wrong tracker.  Callers should
use them to reject clearly worse candidates and retain a baseline unless there
is held-out, uncertainty-aware evidence for an improvement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import ceil, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


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
) -> CycleConsistency:
    """Measure forward/backward disagreement after aligning time direction.

    Arrays have shape ``(frames, tracks, coordinates)`` and use ``NaN`` for a
    missing track.  Before scoring, backward slots are globally aligned to
    forward slots with a Hungarian assignment over their mean disagreement on
    overlapping observations.  This removes arbitrary bootstrap slot labels
    without hiding a time-varying identity swap.  Pairs with no overlap cannot
    contribute evidence and are left unmatched.  With no supplied scale, a
    robust inter-track distance (then temporal displacement) is inferred; it
    falls back to one for a stationary singleton track.
    """

    forward = _validated_positions(forward_positions, "forward_positions")
    backward = _validated_positions(backward_positions, "backward_positions")
    if forward.shape != backward.shape:
        raise ValueError(
            "forward_positions and backward_positions must have equal shape"
        )
    if backward_is_reverse_chronological:
        backward = backward[::-1]

    scale = _resolve_spatial_scale(forward, spatial_scale)
    errors = _globally_aligned_errors(forward, backward, scale)
    return CycleConsistency(
        mean_normalized_error=float(np.mean(errors)),
        median_normalized_error=float(np.median(errors)),
        p95_normalized_error=float(np.percentile(errors, 95)),
        valid_observations=int(errors.size),
        spatial_scale=scale,
    )


def trajectory_quality_metrics(
    positions: NDArray[np.floating], *, spatial_scale: float
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
    total = int(observed.size)
    observed_count = int(np.count_nonzero(observed))
    coverage_loss = 1.0 - observed_count / total

    if observed.shape[0] < 2:
        fragmentation_loss = 0.0
    else:
        transitions = observed[1:] != observed[:-1]
        fragmentation_loss = float(np.count_nonzero(transitions) / transitions.size)

    if observed.shape[0] < 3:
        motion_roughness_loss = 0.0
    else:
        triplets = observed[:-2] & observed[1:-1] & observed[2:]
        accelerations = (
            observed_positions[2:]
            - 2.0 * observed_positions[1:-1]
            + observed_positions[:-2]
        )
        roughnesses = np.linalg.norm(accelerations, axis=2)[triplets]
        motion_roughness_loss = (
            float(np.mean(roughnesses) / spatial_scale) if roughnesses.size else 0.0
        )

    return TrajectoryQualityMetrics(
        coverage_loss=coverage_loss,
        fragmentation_loss=fragmentation_loss,
        motion_roughness_loss=motion_roughness_loss,
        observed_positions=observed_count,
        total_positions=total,
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
) -> CandidateRecommendation:
    """Recommend only an unambiguous candidate that dominates the baseline.

    If multiple baseline-safe alternatives trade off against each other, this
    refuses to invent a weighted winner.  In particular, worsening any loss
    relative to the baseline can never result in a recommendation.
    """

    checked_scores = _validate_scores_and_specs(scores, metric_specs)
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
    unambiguous = tuple(
        score
        for score in safe
        if all(
            other.candidate_id == score.candidate_id
            or dominates(score, other, metric_specs)
            for other in safe
        )
    )
    if len(unambiguous) != 1:
        return CandidateRecommendation(
            None,
            safe_ids,
            "baseline-safe candidates have unresolved Pareto trade-offs",
        )
    return CandidateRecommendation(
        unambiguous[0].candidate_id,
        safe_ids,
        "candidate dominates the baseline and every other baseline-safe candidate",
    )


def decide_heldout_baseline_protection(
    baseline: CandidateScore,
    candidate: CandidateScore,
    metric_specs: Sequence[MetricSpec],
    *,
    primary_metric: str,
    minimum_improvement: float,
    confidence_z: float = 1.96,
) -> BaselineProtectionDecision:
    """Accept a candidate only with conservative held-out improvement evidence.

    For each metric the lower confidence bound of the direction-aware
    improvement must be non-negative.  The primary metric must additionally
    clear ``minimum_improvement``.  Independent-run standard errors are added
    in quadrature; callers should supply evaluations from their held-out
    perturbations rather than treating this as ground-truth accuracy.
    """

    _validate_scores_and_specs((baseline, candidate), metric_specs)
    specs = _validate_metric_specs(metric_specs)
    if primary_metric not in {spec.name for spec in specs}:
        raise ValueError("primary_metric must be declared in metric_specs")
    if minimum_improvement < 0 or not np.isfinite(minimum_improvement):
        raise ValueError("minimum_improvement must be finite and non-negative")
    if confidence_z < 0 or not np.isfinite(confidence_z):
        raise ValueError("confidence_z must be finite and non-negative")

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


def _validated_positions(values: NDArray[np.floating], name: str) -> FloatArray:
    positions = np.asarray(values, dtype=np.float64)
    if positions.ndim != 3 or any(dimension < 1 for dimension in positions.shape):
        raise ValueError(f"{name} must have shape (frames, tracks, coordinates)")
    if np.isinf(positions).any():
        raise ValueError(f"{name} may contain NaN for missing data but not infinity")
    return positions


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


def _globally_aligned_errors(
    forward: FloatArray, backward: FloatArray, scale: float
) -> FloatArray:
    """Return errors after a global minimum-mean-cost slot assignment."""

    track_count = forward.shape[1]
    costs = np.full((track_count, track_count), np.inf, dtype=float)
    overlaps = np.zeros((track_count, track_count), dtype=bool)
    for forward_track in range(track_count):
        for backward_track in range(track_count):
            valid = np.isfinite(forward[:, forward_track]).all(axis=1) & np.isfinite(
                backward[:, backward_track]
            ).all(axis=1)
            if np.any(valid):
                differences = (
                    forward[valid, forward_track] - backward[valid, backward_track]
                )
                costs[forward_track, backward_track] = float(
                    np.mean(np.linalg.norm(differences, axis=1) / scale)
                )
                overlaps[forward_track, backward_track] = True

    finite_costs = costs[np.isfinite(costs)]
    if not finite_costs.size:
        raise ValueError("no position is observed in both forward and backward passes")
    # scipy's assignment requires a finite matrix.  The sentinel cannot be
    # selected over any compatible pairing and incompatible assignments are
    # filtered below, so no-overlap slots manufacture no score contribution.
    unavailable_cost = float(np.max(finite_costs) * (track_count + 1) + 1.0)
    rows, columns = linear_sum_assignment(np.where(overlaps, costs, unavailable_cost))
    errors: list[float] = []
    for forward_track, backward_track in zip(rows, columns, strict=True):
        if not overlaps[forward_track, backward_track]:
            continue
        valid = np.isfinite(forward[:, forward_track]).all(axis=1) & np.isfinite(
            backward[:, backward_track]
        ).all(axis=1)
        differences = forward[valid, forward_track] - backward[valid, backward_track]
        errors.extend((np.linalg.norm(differences, axis=1) / scale).tolist())
    if not errors:  # Defensive: finite pair costs above guarantee this in practice.
        raise ValueError("no position is observed in matched forward/backward tracks")
    return np.asarray(errors, dtype=np.float64)


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
