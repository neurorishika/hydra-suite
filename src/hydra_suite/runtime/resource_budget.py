"""Typed memory estimates and resource admission decisions.

Estimates are an admission aid, not a containment boundary.  Callers must run
high-memory work in a separately supervised process even after admission.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional

GiB = 1024**3
ESTIMATOR_VERSION = "resource-budget-v1"


class AcceleratorKind(str, Enum):
    """Memory topology relevant to admission."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass(frozen=True)
class ResourceObservation:
    """Free resources observed at one point in time."""

    total_host_bytes: int
    available_host_bytes: int
    accelerator_kind: AcceleratorKind = AcceleratorKind.CPU
    accelerator_name: str = "CPU"
    total_accelerator_bytes: Optional[int] = None
    available_accelerator_bytes: Optional[int] = None
    observed_at_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        for name in ("total_host_bytes", "available_host_bytes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.available_host_bytes > self.total_host_bytes:
            raise ValueError("available_host_bytes cannot exceed total_host_bytes")
        device_values = (
            self.total_accelerator_bytes,
            self.available_accelerator_bytes,
        )
        if any(value is not None and value < 0 for value in device_values):
            raise ValueError("accelerator memory observations must be non-negative")
        if (self.total_accelerator_bytes is None) != (
            self.available_accelerator_bytes is None
        ):
            raise ValueError("accelerator total and available bytes must be paired")
        if (
            self.total_accelerator_bytes is not None
            and self.available_accelerator_bytes is not None
            and self.available_accelerator_bytes > self.total_accelerator_bytes
        ):
            raise ValueError(
                "available_accelerator_bytes cannot exceed total_accelerator_bytes"
            )
        if self.accelerator_kind is AcceleratorKind.MPS and (
            self.total_accelerator_bytes is not None
            or self.available_accelerator_bytes is not None
        ):
            raise ValueError(
                "MPS uses unified host memory; do not provide a separate device pool"
            )
        if self.accelerator_kind is AcceleratorKind.CPU and (
            self.total_accelerator_bytes is not None
            or self.available_accelerator_bytes is not None
        ):
            raise ValueError(
                "CPU observations cannot claim a separate accelerator memory pool"
            )


@dataclass(frozen=True)
class PhaseEstimate:
    """Steady and peak allocations for one phase of a job."""

    name: str
    host_steady_bytes: int = 0
    host_peak_bytes: int = 0
    accelerator_steady_bytes: int = 0
    accelerator_peak_bytes: int = 0
    disk_transient_bytes: int = 0
    dominant_allocations: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        values = (
            self.host_steady_bytes,
            self.host_peak_bytes,
            self.accelerator_steady_bytes,
            self.accelerator_peak_bytes,
            self.disk_transient_bytes,
        )
        if not self.name:
            raise ValueError("phase name must not be empty")
        if any(value < 0 for value in values):
            raise ValueError("phase byte estimates must be non-negative")
        if self.host_peak_bytes < self.host_steady_bytes:
            raise ValueError("host peak must be at least host steady bytes")
        if self.accelerator_peak_bytes < self.accelerator_steady_bytes:
            raise ValueError("accelerator peak must be at least accelerator steady")
        if any(size < 0 for _, size in self.dominant_allocations):
            raise ValueError("dominant allocation sizes must be non-negative")


@dataclass(frozen=True)
class WorkLimits:
    """Effective in-flight limits attached to an estimate."""

    batch_size: int = 1
    workers: int = 0
    prefetch_batches: int = 0
    tiles: Optional[int] = None
    candidates: Optional[int] = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least one")
        if self.workers < 0 or self.prefetch_batches < 0:
            raise ValueError("worker and prefetch limits must be non-negative")
        if self.tiles is not None and self.tiles < 1:
            raise ValueError("tiles must be positive when specified")
        if self.candidates is not None and self.candidates < 1:
            raise ValueError("candidates must be positive when specified")


@dataclass(frozen=True)
class ResourceRequest:
    """A job's phase estimates and effective execution limits."""

    job_name: str
    phases: tuple[PhaseEstimate, ...]
    limits: WorkLimits = WorkLimits()
    estimator_version: str = ESTIMATOR_VERSION

    def __post_init__(self) -> None:
        if not self.job_name:
            raise ValueError("job_name must not be empty")
        if not self.phases:
            raise ValueError("at least one phase estimate is required")
        phase_names = [phase.name for phase in self.phases]
        if len(phase_names) != len(set(phase_names)):
            raise ValueError("phase names must be unique")


@dataclass(frozen=True)
class ResourcePolicy:
    """Configurable safety margins used during admission."""

    reserve_host_bytes: int = 8 * GiB
    reserve_host_fraction: float = 0.15
    accelerator_safety_fraction: float = 0.85
    warning_fraction: float = 0.80

    def __post_init__(self) -> None:
        if self.reserve_host_bytes < 0:
            raise ValueError("reserve_host_bytes must be non-negative")
        for name in (
            "reserve_host_fraction",
            "accelerator_safety_fraction",
            "warning_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True)
class ResourceBudget:
    """Auditable output from :func:`evaluate_resource_request`."""

    admitted: bool
    host_steady_bytes: int
    host_peak_bytes: int
    accelerator_steady_bytes: int
    accelerator_peak_bytes: int
    disk_transient_bytes: int
    available_host_bytes: int
    usable_host_bytes: int
    reserved_host_bytes: int
    available_accelerator_bytes: Optional[int]
    usable_accelerator_bytes: Optional[int]
    dominant_phase: str
    dominant_host_phase: str
    dominant_accelerator_phase: Optional[str]
    dominant_allocations: tuple[tuple[str, int], ...]
    dominant_host_allocations: tuple[tuple[str, int], ...]
    dominant_accelerator_allocations: tuple[tuple[str, int], ...]
    limits: WorkLimits
    refusals: tuple[str, ...]
    warnings: tuple[str, ...]
    estimator_version: str


def probe_resources(
    accelerator_kind: AcceleratorKind = AcceleratorKind.CPU,
    *,
    accelerator_name: Optional[str] = None,
    accelerator_probe: Optional[Callable[[], tuple[int, int]]] = None,
) -> ResourceObservation:
    """Observe host memory, plus an optional discrete-accelerator pool.

    ``accelerator_probe`` returns ``(free_bytes, total_bytes)``.  It is
    deliberately injected so this lower-level module never imports torch.
    MPS has no separate pool because its allocations consume host memory.
    """
    import psutil

    host = psutil.virtual_memory()
    if accelerator_kind is AcceleratorKind.MPS:
        return ResourceObservation(
            total_host_bytes=int(host.total),
            available_host_bytes=int(host.available),
            accelerator_kind=accelerator_kind,
            accelerator_name=accelerator_name or "Apple Metal (unified memory)",
        )
    if accelerator_probe is None:
        return ResourceObservation(
            total_host_bytes=int(host.total),
            available_host_bytes=int(host.available),
            accelerator_kind=accelerator_kind,
            accelerator_name=accelerator_name or accelerator_kind.value.upper(),
        )
    free, total = accelerator_probe()
    return ResourceObservation(
        total_host_bytes=int(host.total),
        available_host_bytes=int(host.available),
        accelerator_kind=accelerator_kind,
        accelerator_name=accelerator_name or accelerator_kind.value.upper(),
        total_accelerator_bytes=int(total),
        available_accelerator_bytes=int(free),
    )


def evaluate_resource_request(
    request: ResourceRequest,
    observation: ResourceObservation,
    policy: Optional[ResourcePolicy] = None,
) -> ResourceBudget:
    """Evaluate a typed request against a live observation.

    Phases are alternatives over time, so their peaks are compared rather
    than summed.  Within an MPS phase, host and accelerator estimates are
    summed against the single unified host pool.
    """
    policy = policy or ResourcePolicy()
    phase_host_peaks = {
        phase.name: phase.host_peak_bytes
        + (
            phase.accelerator_peak_bytes
            if observation.accelerator_kind is AcceleratorKind.MPS
            else 0
        )
        for phase in request.phases
    }
    dominant_host_phase = max(phase_host_peaks, key=phase_host_peaks.__getitem__)
    phase_accelerator_peaks = {
        phase.name: phase.accelerator_peak_bytes for phase in request.phases
    }
    accelerator_peak = max(phase_accelerator_peaks.values(), default=0)
    dominant_accelerator_phase = (
        max(phase_accelerator_peaks, key=phase_accelerator_peaks.__getitem__)
        if accelerator_peak
        else None
    )
    host_steady = max(
        phase.host_steady_bytes
        + (
            phase.accelerator_steady_bytes
            if observation.accelerator_kind is AcceleratorKind.MPS
            else 0
        )
        for phase in request.phases
    )
    host_peak = phase_host_peaks[dominant_host_phase]
    accelerator_steady = max(
        (phase.accelerator_steady_bytes for phase in request.phases), default=0
    )
    disk_peak = max((phase.disk_transient_bytes for phase in request.phases), default=0)
    dominant = next(
        phase for phase in request.phases if phase.name == dominant_host_phase
    )
    accelerator_dominant = (
        next(
            phase
            for phase in request.phases
            if phase.name == dominant_accelerator_phase
        )
        if dominant_accelerator_phase is not None
        else None
    )
    dominant_host_allocations = _sorted_allocations(dominant.dominant_allocations)
    dominant_accelerator_allocations = _sorted_allocations(
        accelerator_dominant.dominant_allocations
        if accelerator_dominant is not None
        else ()
    )

    reserve = max(
        policy.reserve_host_bytes,
        int(observation.total_host_bytes * policy.reserve_host_fraction),
    )
    usable_host = max(0, observation.available_host_bytes - reserve)
    refusals: list[str] = []
    warnings: list[str] = []
    if host_peak > usable_host:
        refusals.append(
            f"{request.job_name} needs an estimated peak of "
            f"{_format_bytes(host_peak)} host memory during {dominant_host_phase}, "
            f"but only {_format_bytes(usable_host)} is usable after reserving "
            f"{_format_bytes(reserve)}"
        )
    elif usable_host and host_peak >= int(usable_host * policy.warning_fraction):
        warnings.append(
            f"Estimated host peak {_format_bytes(host_peak)} uses at least "
            f"{policy.warning_fraction:.0%} of the admitted host-memory budget"
        )

    usable_accelerator: Optional[int] = None
    if observation.accelerator_kind is AcceleratorKind.CPU and accelerator_peak:
        refusals.append(
            f"{request.job_name} estimates {_format_bytes(accelerator_peak)} of "
            "accelerator memory, but the CPU observation has no accelerator pool"
        )
    if observation.accelerator_kind is AcceleratorKind.CUDA:
        if observation.available_accelerator_bytes is None:
            refusals.append("CUDA memory availability could not be measured")
        else:
            usable_accelerator = int(
                observation.available_accelerator_bytes
                * policy.accelerator_safety_fraction
            )
            if accelerator_peak > usable_accelerator:
                refusals.append(
                    f"{request.job_name} needs an estimated accelerator peak of "
                    f"{_format_bytes(accelerator_peak)}, but only "
                    f"{_format_bytes(usable_accelerator)} is usable"
                )
            elif usable_accelerator and accelerator_peak >= int(
                usable_accelerator * policy.warning_fraction
            ):
                warnings.append(
                    f"Estimated accelerator peak {_format_bytes(accelerator_peak)} "
                    "is close to the admitted device-memory budget"
                )

    return ResourceBudget(
        admitted=not refusals,
        host_steady_bytes=host_steady,
        host_peak_bytes=host_peak,
        accelerator_steady_bytes=accelerator_steady,
        accelerator_peak_bytes=accelerator_peak,
        disk_transient_bytes=disk_peak,
        available_host_bytes=observation.available_host_bytes,
        usable_host_bytes=usable_host,
        reserved_host_bytes=reserve,
        available_accelerator_bytes=observation.available_accelerator_bytes,
        usable_accelerator_bytes=usable_accelerator,
        dominant_phase=dominant_host_phase,
        dominant_host_phase=dominant_host_phase,
        dominant_accelerator_phase=dominant_accelerator_phase,
        dominant_allocations=dominant_host_allocations,
        dominant_host_allocations=dominant_host_allocations,
        dominant_accelerator_allocations=dominant_accelerator_allocations,
        limits=request.limits,
        refusals=tuple(refusals),
        warnings=tuple(warnings),
        estimator_version=request.estimator_version,
    )


def _sorted_allocations(
    allocations: Iterable[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(allocations, key=lambda item: item[1], reverse=True))


def _format_bytes(value: int) -> str:
    if value >= GiB:
        return f"{value / GiB:.1f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MiB"
    return f"{value} bytes"
