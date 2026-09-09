"""Bounded candidate generation and pre-load analytical/measured admission."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable

from hydra_suite.core.inference.pipeline import MAX_PIPELINE_BUFFER_BYTES
from hydra_suite.runtime.memory_profiles import (
    MemoryMeasurement,
    measured_envelope_bytes,
)
from hydra_suite.runtime.resource_budget import ResourceObservation, ResourcePolicy

from .models import InferenceTuningSettings


@dataclass(frozen=True, slots=True)
class MemoryCostModel:
    """Conservative bytes retained by the full active model combination."""

    fixed_host_bytes: int = 0
    fixed_accelerator_bytes: int = 0
    detector_frame_host_bytes: int = 0
    detector_frame_accelerator_bytes: int = 0
    tile_accelerator_bytes: int = 0
    crop_accelerator_bytes: int = 0
    queue_host_bytes: int = 0

    def __post_init__(self) -> None:
        if any(getattr(self, item.name) < 0 for item in fields(self)):
            raise ValueError("memory costs must be non-negative")


@dataclass(frozen=True, slots=True)
class AdmissionContext:
    observation: ResourceObservation
    frame_bytes: int
    crop_count_p95: int
    hard_maxima: tuple[tuple[str, int], ...]
    cost: MemoryCostModel = MemoryCostModel()
    policy: ResourcePolicy = ResourcePolicy()
    frame_buffer_budget_bytes: int = MAX_PIPELINE_BUFFER_BYTES
    realtime: bool = False
    coreml_obb: bool = False
    cached_fields: frozenset[str] = frozenset()
    measured_records: tuple[tuple[str, tuple[MemoryMeasurement, ...]], ...] = ()

    def __post_init__(self) -> None:
        if self.frame_bytes < 1 or self.frame_buffer_budget_bytes < 1:
            raise ValueError("frame and buffer sizes must be positive")
        if self.crop_count_p95 < 0:
            raise ValueError("crop_count_p95 must be non-negative")

    @property
    def maxima(self) -> dict[str, int]:
        return dict(self.hard_maxima)

    @property
    def measured(self) -> dict[str, tuple[MemoryMeasurement, ...]]:
        return dict(self.measured_records)


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    settings: InferenceTuningSettings
    host_required_bytes: int
    accelerator_required_bytes: int
    frame_buffer_bytes: int
    reason: str | None = None


class CandidatePlanner:
    """Coordinate planner; it never chooses independent largest-fit winners."""

    def __init__(self, context: AdmissionContext) -> None:
        self.context = context

    @staticmethod
    def geometric_values(current: int, maximum: int) -> tuple[int, ...]:
        maximum = max(1, int(maximum))
        values = {1, min(maximum, max(1, int(current))), maximum}
        value = 2
        while value < maximum:
            values.add(value)
            value *= 2
        return tuple(sorted(values))

    def candidate_space(
        self, field_name: str, incumbent: InferenceTuningSettings
    ) -> tuple[int, ...]:
        """Unfiltered candidate values for *field_name*, before admission.

        This enumerates the same values ``values_for`` considers, but does
        not filter them through ``admit()`` (which reads live resource
        observation). Returns ``()`` under the same early-exit conditions as
        ``values_for`` (no current value, or a cached field) so the two stay
        in lockstep.
        """
        current = incumbent.value_for(field_name)
        if current is None or field_name in self.context.cached_fields:
            return ()
        if field_name == "pipeline_depth":
            values = tuple(range(1, min(4, self.context.maxima.get(field_name, 4)) + 1))
        else:
            maximum = self.context.maxima.get(field_name, current)
            if field_name in {
                "pose_batch_size",
                "headtail_batch_size",
            } or field_name.startswith("identity_batch_size:"):
                ceiling = max(1, self.context.crop_count_p95)
                maximum = min(maximum, ceiling)
            values = self.geometric_values(current, maximum)
            if self.context.crop_count_p95 > 0 and (
                field_name in {"pose_batch_size", "headtail_batch_size"}
                or field_name.startswith("identity_batch_size:")
            ):
                values = tuple(
                    sorted({*values, min(maximum, self.context.crop_count_p95)})
                )
        return values

    def values_for(
        self, field_name: str, incumbent: InferenceTuningSettings
    ) -> tuple[int, ...]:
        output = []
        for value in self.candidate_space(field_name, incumbent):
            candidate = incumbent.with_value(field_name, value)
            if self.admit(candidate).admitted:
                output.append(value)
        return tuple(output)

    def static_max_for(
        self, field: str, baseline: InferenceTuningSettings
    ) -> int | None:
        """Largest candidate value for *field*, ignoring live memory admission.

        ``values_for`` filters through ``admit()``, which reads the live
        resource observation. Anything that feeds the PROFILE KEY must not,
        or the key drifts with free memory and a stored profile becomes
        unfindable on a busier machine.

        When ``candidate_space`` returns empty (no current value, or a
        cached field), falls back to the baseline's current value for
        *field* so a cached field cannot contribute a bogus artifact batch
        size to the key. That fallback is ``None`` for a field the project
        does not have at all (e.g. ``slice_tile_batch_size`` on any
        non-SAHI project), hence the ``int | None`` return -- callers that
        aggregate several fields MUST drop ``None`` before comparing.
        """
        return max(
            self.candidate_space(field, baseline),
            default=baseline.value_for(field),
        )

    def canonicalize(
        self, settings: InferenceTuningSettings
    ) -> InferenceTuningSettings:
        result = settings
        ceiling = max(1, self.context.crop_count_p95)
        for field_name in settings.field_names():
            if field_name in {
                "pose_batch_size",
                "headtail_batch_size",
            } or field_name.startswith("identity_batch_size:"):
                value = settings.value_for(field_name)
                if value is not None and value > ceiling:
                    result = result.with_value(field_name, ceiling)
        if self.context.realtime or self.context.coreml_obb:
            result = result.with_value("detection_batch_size", 1)
        return result

    def admit(self, settings: InferenceTuningSettings) -> AdmissionDecision:
        settings = self.canonicalize(settings)
        retained_windows = (
            settings.pipeline_depth + 2 if settings.pipeline_depth >= 2 else 1
        )
        frame_bytes = (
            self.context.frame_bytes * settings.detection_batch_size * retained_windows
        )
        if frame_bytes > self.context.frame_buffer_budget_bytes:
            return AdmissionDecision(
                False,
                settings,
                frame_bytes,
                0,
                frame_bytes,
                "frame_buffer_budget",
            )

        cost = self.context.cost
        crop_batches = [
            value
            for value in (
                settings.pose_batch_size,
                settings.headtail_batch_size,
                *(value for _, value in settings.identity_batch_sizes),
            )
            if value is not None
        ]
        max_crop_batch = max(crop_batches, default=1)
        tile_batch = settings.slice_tile_batch_size or 1
        host_required = (
            cost.fixed_host_bytes
            + frame_bytes
            + cost.detector_frame_host_bytes * settings.detection_batch_size
            + cost.queue_host_bytes * max(0, settings.pipeline_depth - 1)
        )
        accelerator_required = (
            cost.fixed_accelerator_bytes
            + cost.detector_frame_accelerator_bytes * settings.detection_batch_size
            + cost.tile_accelerator_bytes * tile_batch
            + cost.crop_accelerator_bytes * max_crop_batch
        )

        for field_name, records in self.context.measured.items():
            value = settings.value_for(field_name)
            if value is None or not records:
                continue
            largest_success = max(record.settings.batch_size for record in records)
            if value > largest_success:
                return AdmissionDecision(
                    False,
                    settings,
                    host_required,
                    accelerator_required,
                    frame_bytes,
                    f"above_largest_measured_success:{field_name}",
                )
            accelerator_required = max(
                accelerator_required, measured_envelope_bytes(records, value)
            )

        obs = self.context.observation
        reserve = max(
            self.context.policy.reserve_host_bytes,
            int(obs.total_host_bytes * self.context.policy.reserve_host_fraction),
        )
        usable_host = max(0, obs.available_host_bytes - reserve)
        if host_required > usable_host:
            return AdmissionDecision(
                False,
                settings,
                host_required,
                accelerator_required,
                frame_bytes,
                "host_memory",
            )
        if obs.available_accelerator_bytes is not None:
            usable_accelerator = int(
                obs.available_accelerator_bytes
                * self.context.policy.accelerator_safety_fraction
            )
            if accelerator_required > usable_accelerator:
                return AdmissionDecision(
                    False,
                    settings,
                    host_required,
                    accelerator_required,
                    frame_bytes,
                    "accelerator_memory",
                )
        return AdmissionDecision(
            True,
            settings,
            host_required,
            accelerator_required,
            frame_bytes,
        )

    def down_admit(
        self,
        selected: InferenceTuningSettings,
        successful_equivalent: Iterable[InferenceTuningSettings],
        baseline: InferenceTuningSettings,
    ) -> AdmissionDecision:
        """Choose only among settings already proven successful/equivalent."""

        candidates = {selected, baseline, *successful_equivalent}
        ordered = sorted(
            candidates,
            key=lambda item: (
                item == selected,
                sum(item.value_for(name) or 0 for name in item.field_names()),
            ),
            reverse=True,
        )
        for candidate in ordered:
            decision = self.admit(candidate)
            if decision.admitted:
                return decision
        decision = self.admit(baseline)
        return AdmissionDecision(
            decision.admitted,
            baseline,
            decision.host_required_bytes,
            decision.accelerator_required_bytes,
            decision.frame_buffer_bytes,
            decision.reason or "baseline_not_admitted",
        )
