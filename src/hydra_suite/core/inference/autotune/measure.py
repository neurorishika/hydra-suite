"""Warmup/measurement contracts and deterministic robust statistics."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar

from .models import CandidateEvidence

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MeasurementProtocol:
    warmup_calls: int = 3
    warmup_frames: int = 8
    minimum_blocks: int = 5
    minimum_stage_seconds: float = 2.0
    maximum_frames: int = 128
    budget_seconds: float = 120.0
    random_seed: int = 0x48594452

    def __post_init__(self) -> None:
        if self.warmup_calls < 3 or self.warmup_frames < 8:
            raise ValueError("inference tuning warmup is below the public minimum")
        if self.minimum_blocks < 5 or self.minimum_stage_seconds < 2.0:
            raise ValueError("inference tuning measurement is below the public minimum")
        if self.maximum_frames < self.warmup_frames or self.budget_seconds <= 0:
            raise ValueError("measurement bounds are invalid")


@dataclass(frozen=True, slots=True)
class RobustSummary:
    median: float
    median_absolute_deviation: float
    confidence_low: float
    confidence_high: float
    count: int


def deterministic_block_order(
    candidates: Sequence[T], blocks: int, *, seed: int
) -> tuple[tuple[T, ...], ...]:
    """Return reproducible per-block shuffles; grouped candidate order is absent."""

    if blocks < 1:
        raise ValueError("blocks must be positive")
    source = tuple(candidates)
    if not source:
        return ()
    generator = random.Random(int(seed))
    output = []
    previous: tuple[T, ...] | None = None
    for _ in range(blocks):
        block = list(source)
        generator.shuffle(block)
        current = tuple(block)
        # With two candidates a seeded shuffle may repeat for many blocks. Rotate
        # deterministically so the protocol is visibly interleaved even then.
        if len(current) > 1 and current == previous:
            current = current[1:] + current[:1]
        output.append(current)
        previous = current
    return tuple(output)


def median_absolute_deviation(values: Iterable[float]) -> float:
    samples = tuple(float(value) for value in values)
    if not samples:
        return 0.0
    median = statistics.median(samples)
    return statistics.median(abs(value - median) for value in samples)


def bootstrap_median_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> tuple[float, float]:
    samples = tuple(float(value) for value in values)
    if not samples:
        return 0.0, 0.0
    if not 0.0 < confidence < 1.0 or resamples < 100:
        raise ValueError("bootstrap parameters are invalid")
    generator = random.Random(int(seed))
    medians = sorted(
        statistics.median(generator.choice(samples) for _ in samples)
        for _ in range(resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    low = medians[max(0, int(alpha * (resamples - 1)))]
    high = medians[min(resamples - 1, int((1.0 - alpha) * (resamples - 1)))]
    return float(low), float(high)


def paired_gain_interval(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap the paired fractional throughput gain ``candidate/base - 1``."""

    count = min(len(baseline), len(candidate))
    if count < 2:
        return float("-inf"), float("inf")
    gains = []
    for old, new in zip(baseline[:count], candidate[:count]):
        old, new = float(old), float(new)
        if not math.isfinite(old) or not math.isfinite(new) or old <= 0 or new <= 0:
            raise ValueError("paired throughput samples must be finite and positive")
        gains.append(new / old - 1.0)
    return bootstrap_median_interval(gains, seed=seed)


def robust_summary(values: Sequence[float], *, seed: int = 0) -> RobustSummary:
    samples = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("measurements must be finite and positive")
    low, high = bootstrap_median_interval(samples, seed=seed)
    return RobustSummary(
        median=statistics.median(samples) if samples else 0.0,
        median_absolute_deviation=median_absolute_deviation(samples),
        confidence_low=low,
        confidence_high=high,
        count=len(samples),
    )


def measurement_complete(
    evidence: CandidateEvidence, protocol: MeasurementProtocol
) -> bool:
    measured_seconds = (
        sum(evidence.stage_seconds_samples)
        if evidence.stage_seconds_samples
        else evidence.steady_state_seconds
    )
    return bool(
        evidence.warmup_calls >= protocol.warmup_calls
        and evidence.warmup_frames >= protocol.warmup_frames
        and len(evidence.throughput_samples) >= protocol.minimum_blocks
        and (
            measured_seconds >= protocol.minimum_stage_seconds
            or evidence.measured_frames >= protocol.maximum_frames
        )
    )
