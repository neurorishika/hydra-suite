"""Bounded coordinate search with full-pipeline correctness/performance gates."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

from .candidates import CandidatePlanner
from .equivalence import CalibrationOutputs, EquivalencePolicy, compare_outputs
from .measure import (
    MeasurementProtocol,
    deterministic_block_order,
    measurement_complete,
    paired_gain_interval,
    robust_summary,
)
from .models import CandidateEvidence, EquivalenceVerdict, InferenceTuningSettings


@dataclass(frozen=True, slots=True)
class TrialObservation:
    settings: InferenceTuningSettings
    throughput: float
    stage_seconds: float
    outputs: CalibrationOutputs | None
    host_peak_bytes: int = 0
    accelerator_peak_bytes: int = 0
    queue_high_water_bytes: int = 0
    frame_buffer_high_water_bytes: int = 0
    thermal_c: float | None = None
    warmup_calls: int = 3
    warmup_frames: int = 8
    prepare_seconds: float = 0.0
    measured_frames: int = 0
    artifact_ids: tuple[str, ...] = ()
    stage_shares: tuple[tuple[str, float], ...] = ()
    failure_class: str | None = None

    def __post_init__(self) -> None:
        if self.failure_class is None and (
            not math.isfinite(self.throughput)
            or self.throughput <= 0
            or not math.isfinite(self.stage_seconds)
            or self.stage_seconds <= 0
        ):
            raise ValueError("successful trial observations require positive timing")


class TrialExecutor(Protocol):
    """Each call executes one warmed measurement block in a fresh sidecar."""

    def run(
        self,
        settings: InferenceTuningSettings,
        *,
        phase: str,
        field_name: str | None,
        block_index: int,
        should_cancel: Callable[[], bool],
    ) -> TrialObservation:
        """Run one fresh, contained, warmed measurement block."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    selected: InferenceTuningSettings
    evidence: tuple[CandidateEvidence, ...]
    rejected: tuple[tuple[str, str], ...]
    completed: bool
    reason: str


class CoordinateSearch:
    """At most two coordinate passes; stage screens never authorize a winner."""

    def __init__(
        self,
        planner: CandidatePlanner,
        executor: TrialExecutor,
        *,
        protocol: MeasurementProtocol | None = None,
        equivalence_policy: EquivalencePolicy | None = None,
        minimum_gain: float = 0.02,
        near_optimal_fraction: float = 0.02,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.protocol = protocol or MeasurementProtocol()
        self.equivalence_policy = equivalence_policy or EquivalencePolicy()
        self.minimum_gain = float(minimum_gain)
        self.near_optimal_fraction = float(near_optimal_fraction)
        self.monotonic = monotonic

    def run(
        self,
        baseline: InferenceTuningSettings,
        *,
        manual_fields: frozenset[str] = frozenset(),
        stage_shares: Mapping[str, float] | None = None,
        should_cancel: Callable[[], bool] = lambda: False,
        status_callback: Callable[[str], None] = lambda _message: None,
    ) -> SearchResult:
        started = self.monotonic()
        deadline = started + self.protocol.budget_seconds
        evidence: list[CandidateEvidence] = []
        rejected: list[tuple[str, str]] = []

        self._status(status_callback, started, "baseline", baseline)

        baseline_measurements = self._measure(
            (baseline,),
            phase="baseline",
            field_name=None,
            reference=None,
            determinism_floor=None,
            deadline=deadline,
            should_cancel=should_cancel,
            rejected=rejected,
        )
        baseline_evidence, baseline_outputs = baseline_measurements.get(
            baseline, (None, ())
        )
        if baseline_evidence is None or len(baseline_outputs) < 2:
            return SearchResult(
                baseline,
                tuple(evidence),
                tuple(rejected),
                False,
                "baseline_measurement_incomplete",
            )
        evidence.append(baseline_evidence)
        determinism_floor = compare_outputs(
            baseline_outputs[0],
            baseline_outputs[1],
            policy=self.equivalence_policy,
            for_determinism_floor=True,
        )
        if not determinism_floor.passed:
            return SearchResult(
                baseline,
                tuple(evidence),
                tuple(rejected),
                False,
                "baseline_nondeterministic_beyond_contract",
            )
        reference = baseline_outputs[0]
        incumbent = baseline
        incumbent_evidence = baseline_evidence
        field_names = [
            field for field in baseline.field_names() if field not in manual_fields
        ]
        shares = dict(stage_shares or baseline_evidence.stage_shares)
        field_names.sort(key=lambda field: (-float(shares.get(field, 0.0)), field))
        accepted_fields: list[str] = []

        for pass_index in range(2):
            changed_this_pass = False
            fields = field_names if pass_index == 0 else accepted_fields
            for field_index, field_name in enumerate(tuple(fields)):
                if self._expired(deadline, should_cancel):
                    return SearchResult(
                        baseline,
                        tuple(evidence),
                        tuple(rejected),
                        False,
                        "cancelled" if should_cancel() else "budget_expired",
                    )
                values = self.planner.values_for(field_name, incumbent)
                mutations = tuple(
                    incumbent.with_value(field_name, value)
                    for value in values
                    if value != incumbent.value_for(field_name)
                )
                if pass_index == 1:
                    mutations = self._nearest_mutations(
                        field_name, incumbent, mutations
                    )
                if not mutations:
                    continue
                self._status(
                    status_callback,
                    started,
                    f"field {field_name}",
                    incumbent,
                )
                screened = self._measure(
                    mutations,
                    phase="stage",
                    field_name=field_name,
                    reference=reference,
                    determinism_floor=determinism_floor,
                    deadline=deadline,
                    should_cancel=should_cancel,
                    seed_offset=pass_index * 10_000 + field_index * 100,
                    rejected=rejected,
                )
                screen_evidence = [
                    item[0]
                    for item in screened.values()
                    if item[0].equivalence is not None and item[0].equivalence.passed
                ]
                evidence.extend(item[0] for item in screened.values())
                rejected.extend(
                    (self._label(settings), self._rejection_reason(item[0]))
                    for settings, item in screened.items()
                    if item[0].equivalence is None or not item[0].equivalence.passed
                )
                fastest_screened = sorted(
                    screen_evidence,
                    key=lambda item: item.median_throughput,
                    reverse=True,
                )[:2]
                if not fastest_screened:
                    continue
                finalists = tuple(item.settings for item in fastest_screened)
                self._status(
                    status_callback,
                    started,
                    f"full-pipeline confirmation for {field_name}",
                    incumbent,
                )
                full = self._measure(
                    (incumbent, *finalists),
                    phase="full",
                    field_name=field_name,
                    reference=reference,
                    determinism_floor=determinism_floor,
                    deadline=deadline,
                    should_cancel=should_cancel,
                    seed_offset=50_000 + pass_index * 10_000 + field_index * 100,
                    rejected=rejected,
                )
                evidence.extend(item[0] for item in full.values())
                candidates = []
                full_incumbent = full.get(incumbent, (incumbent_evidence, ()))[0]
                for settings in finalists:
                    candidate = full.get(settings)
                    if candidate is None:
                        rejected.append(
                            (self._label(settings), "incomplete_full_pipeline")
                        )
                        continue
                    candidate_evidence = candidate[0]
                    if (
                        candidate_evidence.equivalence is None
                        or not candidate_evidence.equivalence.passed
                    ):
                        rejected.append(
                            (
                                self._label(settings),
                                self._rejection_reason(candidate_evidence),
                            )
                        )
                        continue
                    low, _ = paired_gain_interval(
                        full_incumbent.throughput_samples,
                        candidate_evidence.throughput_samples,
                        seed=self.protocol.random_seed + field_index + pass_index,
                    )
                    gain = (
                        candidate_evidence.median_throughput
                        / full_incumbent.median_throughput
                        - 1.0
                    )
                    if gain < self.minimum_gain or low <= 0.0:
                        rejected.append(
                            (self._label(settings), "gain_or_confidence_gate")
                        )
                        continue
                    candidates.append(candidate_evidence)
                if not candidates:
                    incumbent_evidence = full_incumbent
                    continue
                winner = self._choose_near_optimal(candidates)
                incumbent = winner.settings
                incumbent_evidence = winner
                changed_this_pass = True
                if field_name not in accepted_fields:
                    accepted_fields.append(field_name)
            if pass_index == 0 and not accepted_fields:
                break
            if pass_index == 1 or not changed_this_pass:
                break

        if self._expired(deadline, should_cancel):
            return SearchResult(
                baseline,
                tuple(evidence),
                tuple(rejected),
                False,
                "cancelled" if should_cancel() else "budget_expired",
            )
        self._status(status_callback, started, "final confirmation", incumbent)
        final = self._measure(
            (incumbent,),
            phase="final_validation",
            field_name=None,
            reference=reference,
            determinism_floor=determinism_floor,
            deadline=deadline,
            should_cancel=should_cancel,
            seed_offset=90_000,
            rejected=rejected,
        ).get(incumbent)
        if (
            final is None
            or final[0].equivalence is None
            or not final[0].equivalence.passed
        ):
            return SearchResult(
                baseline,
                tuple(evidence),
                tuple(rejected),
                False,
                "final_validation_failed",
            )
        if incumbent != baseline:
            final_evidence = final[0]
            low, _high = paired_gain_interval(
                baseline_evidence.throughput_samples,
                final_evidence.throughput_samples,
                seed=self.protocol.random_seed + 90_000,
            )
            gain = (
                final_evidence.median_throughput / baseline_evidence.median_throughput
                - 1.0
            )
            if gain < self.minimum_gain or low <= 0.0:
                evidence.append(final_evidence)
                rejected.append(
                    (self._label(incumbent), "final_performance_gate_failed")
                )
                return SearchResult(
                    baseline,
                    tuple(evidence),
                    tuple(rejected),
                    False,
                    "final_performance_gate_failed",
                )
        evidence.append(final[0])
        reason = (
            "kept_current_settings"
            if incumbent == baseline
            else "validated_throughput_gain"
        )
        return SearchResult(incumbent, tuple(evidence), tuple(rejected), True, reason)

    def _measure(
        self,
        settings: Sequence[InferenceTuningSettings],
        *,
        phase: str,
        field_name: str | None,
        reference: CalibrationOutputs | None,
        determinism_floor: EquivalenceVerdict | None,
        deadline: float,
        should_cancel: Callable[[], bool],
        seed_offset: int = 0,
        rejected: list[tuple[str, str]] | None = None,
    ) -> dict[
        InferenceTuningSettings,
        tuple[CandidateEvidence, tuple[CalibrationOutputs, ...]],
    ]:
        settings = tuple(dict.fromkeys(settings))
        observations: dict[InferenceTuningSettings, list[TrialObservation]] = {
            item: [] for item in settings
        }
        orders = deterministic_block_order(
            settings,
            self.protocol.minimum_blocks,
            seed=self.protocol.random_seed + seed_offset,
        )
        for block_index, order in enumerate(orders):
            for candidate in order:
                if self._expired(deadline, should_cancel):
                    return {}
                observation = self.executor.run(
                    candidate,
                    phase=phase,
                    field_name=field_name,
                    block_index=block_index,
                    should_cancel=should_cancel,
                )
                if observation.settings != candidate:
                    raise ValueError(
                        "trial executor returned evidence for another candidate"
                    )
                observations[candidate].append(observation)
        output = {}
        for candidate, samples in observations.items():
            successful = [item for item in samples if item.failure_class is None]
            outputs = tuple(
                item.outputs for item in successful if item.outputs is not None
            )
            if len(successful) < self.protocol.minimum_blocks:
                # Never drop a candidate silently: an unrecorded `continue`
                # here is indistinguishable from a candidate that was never
                # proposed, which is how the detector search space quietly
                # collapsed to its small batches. This is also the arrival
                # point for every executor-side failure -- a per-trial
                # `timeout`, an `accelerator-oom`, a crashed child -- so the
                # observed failure classes are named in the reason.
                if rejected is not None:
                    failures = sorted(
                        {
                            str(item.failure_class)
                            for item in samples
                            if item.failure_class
                        }
                    )
                    rejected.append(
                        (
                            self._label(candidate),
                            "measurement_incomplete: "
                            f"blocks={len(successful)}/"
                            f"{self.protocol.minimum_blocks}"
                            + (f" failures={','.join(failures)}" if failures else ""),
                        )
                    )
                continue
            verdict = None
            if reference is not None:
                verdicts = tuple(
                    compare_outputs(
                        reference,
                        item,
                        determinism_floor=determinism_floor,
                        policy=self.equivalence_policy,
                    )
                    for item in outputs
                )
                verdict = self._combine_verdicts(verdicts)
            thermal = [
                item.thermal_c for item in successful if item.thermal_c is not None
            ]
            throughputs = tuple(item.throughput for item in successful)
            throughput_summary = robust_summary(
                throughputs, seed=self.protocol.random_seed + seed_offset
            )
            evidence = CandidateEvidence(
                settings=candidate,
                throughput_samples=throughputs,
                stage_seconds_samples=tuple(item.stage_seconds for item in successful),
                measured_frames=sum(item.measured_frames for item in successful),
                host_peak_bytes=max(item.host_peak_bytes for item in successful),
                accelerator_peak_bytes=max(
                    item.accelerator_peak_bytes for item in successful
                ),
                queue_high_water_bytes=max(
                    item.queue_high_water_bytes for item in successful
                ),
                frame_buffer_high_water_bytes=max(
                    item.frame_buffer_high_water_bytes for item in successful
                ),
                thermal_c_range=(min(thermal), max(thermal)) if thermal else None,
                warmup_calls=min(item.warmup_calls for item in successful),
                warmup_frames=min(item.warmup_frames for item in successful),
                prepare_seconds=sum(item.prepare_seconds for item in successful),
                steady_state_seconds=sum(item.stage_seconds for item in successful),
                artifact_ids=tuple(
                    sorted(
                        {
                            artifact
                            for item in successful
                            for artifact in item.artifact_ids
                        }
                    )
                ),
                equivalence=verdict,
                phase=phase,
                throughput_confidence_95=(
                    throughput_summary.confidence_low,
                    throughput_summary.confidence_high,
                ),
                stage_shares=tuple(
                    sorted(
                        {
                            name: sum(
                                dict(item.stage_shares).get(name, 0.0)
                                for item in successful
                            )
                            / len(successful)
                            for item in successful
                            for name, _value in item.stage_shares
                        }.items()
                    )
                ),
            )
            if not measurement_complete(evidence, self.protocol):
                if rejected is not None:
                    rejected.append(
                        (
                            self._label(candidate),
                            "measurement_incomplete: "
                            f"warmup_calls={evidence.warmup_calls}/"
                            f"{self.protocol.warmup_calls} "
                            f"warmup_frames={evidence.warmup_frames}/"
                            f"{self.protocol.warmup_frames} "
                            f"blocks={len(evidence.throughput_samples)}/"
                            f"{self.protocol.minimum_blocks} "
                            f"measured_frames={evidence.measured_frames}",
                        )
                    )
                continue
            output[candidate] = (evidence, outputs)
        return output

    @staticmethod
    def _combine_verdicts(verdicts: Sequence[EquivalenceVerdict]) -> EquivalenceVerdict:
        if not verdicts:
            return EquivalenceVerdict(False, details=("candidate produced no outputs",))
        return EquivalenceVerdict(
            passed=all(item.passed for item in verdicts),
            nonzero_rows=all(item.nonzero_rows for item in verdicts),
            row_counts_match=all(item.row_counts_match for item in verdicts),
            unmatched_rows=max(item.unmatched_rows for item in verdicts),
            position_p99=max(item.position_p99 for item in verdicts),
            angle_max=max(item.angle_max for item in verdicts),
            nan_pattern_mismatches=sum(
                item.nan_pattern_mismatches for item in verdicts
            ),
            categorical_mismatches=sum(
                item.categorical_mismatches for item in verdicts
            ),
            details=tuple(detail for item in verdicts for detail in item.details),
        )

    def _choose_near_optimal(
        self, candidates: Sequence[CandidateEvidence]
    ) -> CandidateEvidence:
        fastest = max(item.median_throughput for item in candidates)
        band = [
            item
            for item in candidates
            if item.median_throughput >= fastest * (1.0 - self.near_optimal_fraction)
        ]
        return min(
            band,
            key=lambda item: (
                item.accelerator_peak_bytes,
                item.host_peak_bytes,
                sum(
                    item.settings.value_for(name) or 0
                    for name in item.settings.field_names()
                ),
                item.prepare_seconds,
            ),
        )

    @staticmethod
    def _nearest_mutations(
        field_name: str,
        incumbent: InferenceTuningSettings,
        mutations: Sequence[InferenceTuningSettings],
    ) -> tuple[InferenceTuningSettings, ...]:
        incumbent_value = incumbent.value_for(field_name) or 1
        return tuple(
            sorted(
                mutations,
                key=lambda item: abs(
                    (item.value_for(field_name) or 1) - incumbent_value
                ),
            )[:2]
        )

    def _expired(self, deadline: float, should_cancel: Callable[[], bool]) -> bool:
        return should_cancel() or self.monotonic() >= deadline

    def _status(
        self,
        callback: Callable[[str], None],
        started: float,
        phase: str,
        incumbent: InferenceTuningSettings,
    ) -> None:
        elapsed = max(0.0, self.monotonic() - started)
        callback(
            f"Optimizing inference — {phase}; incumbent "
            f"{self._label(incumbent)}; {elapsed:.1f}/{self.protocol.budget_seconds:g}s"
        )

    @staticmethod
    def _label(settings: InferenceTuningSettings) -> str:
        return ",".join(
            f"{name}={settings.value_for(name)}" for name in settings.field_names()
        )

    @staticmethod
    def _rejection_reason(evidence: CandidateEvidence) -> str:
        if evidence.failure_class:
            return evidence.failure_class
        if evidence.equivalence is None:
            return "missing_equivalence_evidence"
        if evidence.equivalence.details:
            return "; ".join(evidence.equivalence.details)[:1024]
        return "correctness_gate"
