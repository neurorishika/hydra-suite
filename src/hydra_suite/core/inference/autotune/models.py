"""Immutable public contracts for inference throughput tuning.

The tuner never mutates a saved project configuration.  It produces an
``InferenceRuntimeOverlay`` whose requested, admitted, and effective values
remain independently observable for the lifetime of the run.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:
    from hydra_suite.core.inference.config import InferenceConfig


SETTING_FIELDS = (
    "detection_batch_size",
    "slice_tile_batch_size",
    "pose_batch_size",
    "headtail_batch_size",
    "pipeline_depth",
)
IDENTITY_FIELD_PREFIX = "identity_batch_size:"


def _positive_optional(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive when enabled")
    return value


@dataclass(frozen=True, slots=True)
class InferenceTuningSettings:
    """One complete, jointly evaluated inference execution strategy."""

    detection_batch_size: int = 1
    slice_tile_batch_size: int | None = None
    pose_batch_size: int | None = None
    headtail_batch_size: int | None = None
    identity_batch_sizes: tuple[tuple[str, int], ...] = ()
    pipeline_depth: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "detection_batch_size",
            int(_positive_optional(self.detection_batch_size, "detection_batch_size")),
        )
        for name in ("slice_tile_batch_size", "pose_batch_size", "headtail_batch_size"):
            object.__setattr__(
                self, name, _positive_optional(getattr(self, name), name)
            )
        depth = int(self.pipeline_depth)
        if not 1 <= depth <= 4:
            raise ValueError("pipeline_depth must be between one and four")
        object.__setattr__(self, "pipeline_depth", depth)
        normalized = tuple(
            sorted(
                (str(label), int(value)) for label, value in self.identity_batch_sizes
            )
        )
        if any(not label or value < 1 for label, value in normalized):
            raise ValueError(
                "identity batch settings require a label and positive value"
            )
        if len({label for label, _ in normalized}) != len(normalized):
            raise ValueError("identity classifier labels must be unique")
        object.__setattr__(self, "identity_batch_sizes", normalized)

    @classmethod
    def from_config(cls, config: "InferenceConfig") -> "InferenceTuningSettings":
        slice_config = None
        if config.obb is not None:
            if config.obb.mode == "direct" and config.obb.direct is not None:
                slice_config = config.obb.direct.slice
            elif config.obb.sequential is not None:
                slice_config = config.obb.sequential.stage1_slice
        pose_batch = None
        if config.pose is not None:
            backend = getattr(config.pose, config.pose.backend, None)
            pose_batch = getattr(backend, "batch_size", None)
        return cls(
            detection_batch_size=config.detection_batch_size,
            slice_tile_batch_size=(
                slice_config.tile_batch_size
                if slice_config is not None and slice_config.enabled
                else None
            ),
            pose_batch_size=pose_batch,
            headtail_batch_size=(
                config.headtail.batch_size if config.headtail is not None else None
            ),
            identity_batch_sizes=tuple(
                (phase.label, phase.batch_size) for phase in config.cnn_phases
            ),
            pipeline_depth=config.pipeline_depth,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["identity_batch_sizes"] = {
            label: value for label, value in self.identity_batch_sizes
        }
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InferenceTuningSettings":
        identities = value.get("identity_batch_sizes", {})
        if isinstance(identities, Mapping):
            identity_items = tuple((str(k), int(v)) for k, v in identities.items())
        else:
            identity_items = tuple((str(k), int(v)) for k, v in identities)
        return cls(
            detection_batch_size=int(value.get("detection_batch_size", 1)),
            slice_tile_batch_size=value.get("slice_tile_batch_size"),
            pose_batch_size=value.get("pose_batch_size"),
            headtail_batch_size=value.get("headtail_batch_size"),
            identity_batch_sizes=identity_items,
            pipeline_depth=int(value.get("pipeline_depth", 2)),
        )

    def field_names(self) -> tuple[str, ...]:
        names = [name for name in SETTING_FIELDS if getattr(self, name) is not None]
        names.extend(
            f"{IDENTITY_FIELD_PREFIX}{label}" for label, _ in self.identity_batch_sizes
        )
        return tuple(names)

    def value_for(self, field_name: str) -> int | None:
        if field_name.startswith(IDENTITY_FIELD_PREFIX):
            label = field_name[len(IDENTITY_FIELD_PREFIX) :]
            return dict(self.identity_batch_sizes).get(label)
        if field_name not in SETTING_FIELDS:
            raise KeyError(field_name)
        return getattr(self, field_name)

    def with_value(self, field_name: str, value: int) -> "InferenceTuningSettings":
        value = int(value)
        if field_name.startswith(IDENTITY_FIELD_PREFIX):
            label = field_name[len(IDENTITY_FIELD_PREFIX) :]
            identities = dict(self.identity_batch_sizes)
            if label not in identities:
                raise KeyError(field_name)
            identities[label] = value
            return replace(self, identity_batch_sizes=tuple(identities.items()))
        if field_name not in SETTING_FIELDS:
            raise KeyError(field_name)
        if getattr(self, field_name) is None:
            raise KeyError(f"inactive tuning field: {field_name}")
        return replace(self, **{field_name: value})

    def apply(
        self, config: "InferenceConfig", *, disable_tile_autotune: bool = True
    ) -> "InferenceConfig":
        """Return a detached config carrying these execution-only values.

        ``disable_tile_autotune`` must only be true when this call is actually
        overriding ``slice_tile_batch_size`` with a coordinated-tuner value
        (i.e. overlays with status ``calibrated``/``cache_hit``/
        ``cache_hit_after_wait``). No-op overlays (record, kept-current,
        fallback, disabled, ...) reuse the configured baseline value and must
        not silently disable the process-local SAHI tile-batch tuner.
        """

        output = deepcopy(config)
        output.detection_batch_size = self.detection_batch_size
        output.pipeline_depth = self.pipeline_depth
        if output.obb is not None and self.slice_tile_batch_size is not None:
            if output.obb.mode == "direct" and output.obb.direct is not None:
                output.obb.direct.slice.tile_batch_size = self.slice_tile_batch_size
                if disable_tile_autotune:
                    # The coordinated tuner supersedes the old process-local
                    # SAHI tuner.
                    output.obb.direct.slice.tile_batch_autotune = False
            elif output.obb.sequential is not None:
                output.obb.sequential.stage1_slice.tile_batch_size = (
                    self.slice_tile_batch_size
                )
                if disable_tile_autotune:
                    output.obb.sequential.stage1_slice.tile_batch_autotune = False
        if output.headtail is not None and self.headtail_batch_size is not None:
            output.headtail.batch_size = self.headtail_batch_size
        if output.pose is not None and self.pose_batch_size is not None:
            backend = getattr(output.pose, output.pose.backend, None)
            if backend is not None:
                backend.batch_size = self.pose_batch_size
        identities = dict(self.identity_batch_sizes)
        for phase in output.cnn_phases:
            if phase.label in identities:
                phase.batch_size = identities[phase.label]
        return output


class ProfileState(str, Enum):
    """Whether a stored profile may currently affect a production run."""

    PROVISIONAL = "provisional"
    VALIDATED = "validated"
    # S5: a negative-cache marker. Written when calibration could not
    # complete (budget_expired/timeout/baseline_measurement_incomplete) so a
    # project that cannot finish calibration doesn't re-burn the entire
    # tuning budget on every run. Never applies settings to a production
    # run; the coordinator short-circuits to a baseline "fallback"-style
    # overlay (status="deferred_due_to_prior_failure") while
    # ``now - last_validation_unix_ns < INCOMPLETE_RETRY_SECONDS``, then
    # retries calibration as normal. Never written for a run where
    # ``contention_detected`` was true -- a transient GPU neighbour must not
    # buy a 24-hour lockout.
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class EquivalenceVerdict:
    """Bounded correctness metrics for both forward and final outputs."""

    passed: bool
    nonzero_rows: bool = True
    row_counts_match: bool = True
    unmatched_rows: int = 0
    position_p99: float = 0.0
    angle_max: float = 0.0
    nan_pattern_mismatches: int = 0
    categorical_mismatches: int = 0
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Measured performance, memory, artifact, and correctness evidence."""

    settings: InferenceTuningSettings
    throughput_samples: tuple[float, ...]
    stage_seconds_samples: tuple[float, ...] = ()
    measured_frames: int = 0
    host_peak_bytes: int = 0
    accelerator_peak_bytes: int = 0
    queue_high_water_bytes: int = 0
    frame_buffer_high_water_bytes: int = 0
    thermal_c_range: tuple[float, float] | None = None
    warmup_calls: int = 0
    warmup_frames: int = 0
    prepare_seconds: float = 0.0
    steady_state_seconds: float = 0.0
    artifact_ids: tuple[str, ...] = ()
    equivalence: EquivalenceVerdict | None = None
    failure_class: str | None = None
    phase: str = "unknown"
    throughput_confidence_95: tuple[float, float] | None = None
    stage_shares: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in self.throughput_samples):
            raise ValueError("throughput samples must be finite and positive")
        if any(not math.isfinite(v) or v <= 0 for v in self.stage_seconds_samples):
            raise ValueError("stage samples must be finite and positive")
        if (
            min(
                self.host_peak_bytes,
                self.accelerator_peak_bytes,
                self.queue_high_water_bytes,
                self.frame_buffer_high_water_bytes,
                self.warmup_calls,
                self.warmup_frames,
                self.measured_frames,
            )
            < 0
        ):
            raise ValueError(
                "candidate counters and memory observations must be non-negative"
            )
        if self.phase not in {
            "unknown",
            "baseline",
            "stage",
            "full",
            "final_validation",
        }:
            raise ValueError("unknown inference tuning evidence phase")
        if self.throughput_confidence_95 is not None:
            low, high = self.throughput_confidence_95
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError("invalid throughput confidence interval")
        if any(
            not name or not math.isfinite(value) or value < 0
            for name, value in self.stage_shares
        ):
            raise ValueError("invalid inference stage share")

    @property
    def median_throughput(self) -> float:
        """Return the candidate's robust central throughput."""
        import statistics

        return (
            statistics.median(self.throughput_samples)
            if self.throughput_samples
            else 0.0
        )

    @property
    def median_absolute_deviation(self) -> float:
        """Return the candidate's within-run throughput dispersion."""
        import statistics

        if not self.throughput_samples:
            return 0.0
        median = self.median_throughput
        return statistics.median(abs(v - median) for v in self.throughput_samples)


@dataclass(frozen=True, slots=True)
class InferenceTuningProfile:
    """Persisted evidence for one exact system/model/workload key."""

    profile_id: str
    key: Any
    baseline: InferenceTuningSettings
    requested: InferenceTuningSettings
    admitted: InferenceTuningSettings
    selected: InferenceTuningSettings
    candidates: tuple[CandidateEvidence, ...]
    state: ProfileState
    selection_reason: str
    rejected: tuple[tuple[str, str], ...] = ()
    calibration_summary: tuple[tuple[str, Any], ...] = ()
    created_at_unix_ns: int = 0
    last_validation_unix_ns: int = 0
    observed_production_throughput: tuple[float, ...] = ()
    invalidation_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.profile_id or len(self.profile_id) > 128:
            raise ValueError("profile_id is invalid")
        if not self.selection_reason or len(self.selection_reason) > 1024:
            raise ValueError("selection_reason is invalid")
        if len(self.candidates) > 256 or len(self.rejected) > 256:
            raise ValueError("profile evidence exceeds its bounded record cap")
        if len(self.observed_production_throughput) > 64:
            raise ValueError("production throughput history exceeds its bounded cap")
        if self.invalidation_reason is not None and (
            not self.invalidation_reason or len(self.invalidation_reason) > 1024
        ):
            raise ValueError("invalidation_reason is invalid")


@dataclass(frozen=True, slots=True)
class InferenceRuntimeOverlay:
    """The only mechanism by which a tuner changes one production run."""

    requested: InferenceTuningSettings
    admitted: InferenceTuningSettings
    effective: InferenceTuningSettings
    field_sources: tuple[tuple[str, str], ...]
    status: str
    reason: str
    profile_id: str | None = None

    #: Statuses where ``effective`` actually carries a coordinated-tuner
    #: value distinct from the configured baseline. Every other status
    #: reuses the configured settings verbatim and must not disable the
    #: process-local SAHI tile-batch tuner as a side effect.
    _ACTIVE_OVERRIDE_STATUSES = frozenset(
        {"calibrated", "cache_hit", "cache_hit_after_wait"}
    )

    def apply(self, config: "InferenceConfig") -> "InferenceConfig":
        """Apply only the effective values to a detached inference config."""
        return self.effective.apply(
            config,
            disable_tile_autotune=self.status in self._ACTIVE_OVERRIDE_STATUSES,
        )

    @classmethod
    def baseline(
        cls, settings: InferenceTuningSettings, *, status: str, reason: str
    ) -> "InferenceRuntimeOverlay":
        """Construct a no-override decision using configured settings."""
        return cls(
            requested=settings,
            admitted=settings,
            effective=settings,
            field_sources=tuple(
                (field, "configured") for field in settings.field_names()
            ),
            status=status,
            reason=reason,
        )


def settings_from_values(
    baseline: InferenceTuningSettings, values: Mapping[str, int]
) -> InferenceTuningSettings:
    """Return an immutable settings vector with named coordinates replaced."""
    result = baseline
    for field_name, value in values.items():
        result = result.with_value(field_name, value)
    return result


def bounded_evidence(
    values: Iterable[CandidateEvidence], maximum: int = 256
) -> tuple[CandidateEvidence, ...]:
    """Materialize evidence while enforcing the persistent record cap."""
    result = tuple(values)
    if len(result) > maximum:
        raise ValueError("candidate evidence exceeds its cap")
    return result
