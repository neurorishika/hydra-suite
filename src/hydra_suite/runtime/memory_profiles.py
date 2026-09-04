"""Versioned, device-specific memory profiles and bounded OOM adaptation.

Profiles are advisory inputs to admission. They never replace process
containment, live reserve checks, or canonical heavy-job leases.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .process_supervisor import ExitKind
from .resource_budget import ESTIMATOR_VERSION, AcceleratorKind

PROFILE_SCHEMA_VERSION = 1
MAX_PROFILE_BYTES = 4 * 1024 * 1024
MAX_PROFILE_RECORDS = 2_048
MAX_RETRIES = 2


def _bounded_text(value: object, name: str) -> str:
    text = str(value)
    if not text or len(text) > 256 or any(ord(char) < 0x20 for char in text):
        raise ValueError(f"{name} is invalid")
    return text


@dataclass(frozen=True, slots=True)
class ProfileIdentity:
    """Stable workload/device identity; use hashes, not filesystem paths."""

    operation: str
    model_identity: str
    backend: str
    device_identity: str
    precision: str
    task: str
    tiling_mode: str = "none"
    adapter_scope: str = "none"
    adapter_rank: int = 0

    def __post_init__(self) -> None:
        for name in (
            "operation",
            "model_identity",
            "backend",
            "device_identity",
            "precision",
            "task",
            "tiling_mode",
            "adapter_scope",
        ):
            object.__setattr__(self, name, _bounded_text(getattr(self, name), name))
        if self.adapter_rank < 0:
            raise ValueError("adapter_rank must be non-negative")


@dataclass(frozen=True, slots=True)
class PressureSettings:
    """Every independently reducible source of in-flight memory pressure."""

    input_width: int
    input_height: int
    batch_size: int = 1
    pipeline_depth: int = 1
    workers: int = 0
    prefetch_batches: int = 0
    tile_chunk: int = 1
    crop_batch: int = 1
    cache_chunk: int = 1

    def __post_init__(self) -> None:
        positive = (
            "input_width",
            "input_height",
            "batch_size",
            "pipeline_depth",
            "tile_chunk",
            "crop_batch",
            "cache_chunk",
        )
        if any(int(getattr(self, name)) < 1 for name in positive):
            raise ValueError("positive pressure settings must be at least one")
        if self.workers < 0 or self.prefetch_batches < 0:
            raise ValueError("workers and prefetch_batches must be non-negative")


@dataclass(frozen=True, slots=True)
class MemoryMeasurement:
    identity: ProfileIdentity
    settings: PressureSettings
    accelerator_kind: AcceleratorKind
    host_peak_bytes: int
    accelerator_allocated_peak_bytes: int = 0
    accelerator_reserved_peak_bytes: int = 0
    queue_high_water_bytes: int = 0
    observed_at_unix_ns: int = 0
    estimator_version: str = ESTIMATOR_VERSION

    def __post_init__(self) -> None:
        for name in (
            "host_peak_bytes",
            "accelerator_allocated_peak_bytes",
            "accelerator_reserved_peak_bytes",
            "queue_high_water_bytes",
            "observed_at_unix_ns",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.accelerator_allocated_peak_bytes > self.accelerator_reserved_peak_bytes:
            raise ValueError("allocated accelerator peak cannot exceed reserved peak")
        if not self.estimator_version:
            raise ValueError("estimator_version must not be empty")


@dataclass(frozen=True, slots=True)
class ProbePlan:
    """An unknown-profile probe that must be launched by a contained sidecar."""

    identity: ProfileIdentity
    settings: PressureSettings
    hard_host_bytes: int
    usable_accelerator_bytes: int | None

    def __post_init__(self) -> None:
        if self.settings.batch_size != 1:
            raise ValueError("an unknown-profile probe must begin at batch size one")
        if self.hard_host_bytes < 1:
            raise ValueError("probe hard_host_bytes must be positive")
        if (
            self.usable_accelerator_bytes is not None
            and self.usable_accelerator_bytes < 1
        ):
            raise ValueError("probe accelerator budget must be positive")


def recommend_batch_size(
    measurement: MemoryMeasurement,
    *,
    available_host_bytes: int,
    available_accelerator_bytes: int | None,
    input_width: int,
    input_height: int,
    maximum: int,
    safety_fraction: float = 0.8,
) -> int:
    """Return a conservative monotonic recommendation from one measured point."""

    if maximum < 1 or input_width < 1 or input_height < 1:
        raise ValueError("recommendation bounds and input dimensions must be positive")
    if not 0.0 < safety_fraction <= 1.0:
        raise ValueError("safety_fraction must be in (0, 1]")
    sample_pixels = measurement.settings.input_width * measurement.settings.input_height
    target_pixels = input_width * input_height
    scale = max(1.0, target_pixels / sample_pixels)
    sample_batch = measurement.settings.batch_size

    if measurement.accelerator_kind is AcceleratorKind.MPS:
        # MPS is one unified pool. Never add or fraction host and device peaks.
        measured_peak = max(
            measurement.host_peak_bytes,
            measurement.accelerator_reserved_peak_bytes,
        )
        usable = available_host_bytes
    elif measurement.accelerator_kind is AcceleratorKind.CUDA:
        measured_peak = max(
            measurement.accelerator_reserved_peak_bytes,
            measurement.accelerator_allocated_peak_bytes,
        )
        usable = int(available_accelerator_bytes or 0)
    else:
        measured_peak = measurement.host_peak_bytes
        usable = available_host_bytes
    if measured_peak <= 0 or usable <= 0:
        return 1
    per_item = math.ceil((measured_peak / sample_batch) * scale)
    return max(1, min(int(usable * safety_fraction) // max(1, per_item), maximum))


class PressureField(str, Enum):
    BATCH_SIZE = "batch_size"
    TILE_CHUNK = "tile_chunk"
    CROP_BATCH = "crop_batch"
    PIPELINE_DEPTH = "pipeline_depth"
    WORKERS = "workers"
    PREFETCH_BATCHES = "prefetch_batches"
    CACHE_CHUNK = "cache_chunk"


@dataclass(frozen=True, slots=True)
class AttemptTelemetry:
    attempt: int
    exit_kind: ExitKind
    settings: PressureSettings
    hard_host_bytes: int
    peak_tree_rss_bytes: int = 0
    minimum_system_available_bytes: int | None = None
    peak_accelerator_bytes: int | None = None
    queue_high_water_bytes: int = 0


@dataclass(frozen=True, slots=True)
class AdaptiveAttemptResult:
    success: bool
    exit_kind: ExitKind
    telemetry: AttemptTelemetry


@dataclass(frozen=True, slots=True)
class AdaptiveRunResult:
    result: AdaptiveAttemptResult
    attempts: tuple[AttemptTelemetry, ...]
    adjustments: tuple[Mapping[str, int | str], ...]


def _reduce_pressure(
    settings: PressureSettings, fields: Iterable[PressureField]
) -> tuple[PressureSettings, Mapping[str, int | str]] | None:
    for field in fields:
        old = int(getattr(settings, field.value))
        minimum = (
            0 if field in {PressureField.WORKERS, PressureField.PREFETCH_BATCHES} else 1
        )
        if old <= minimum:
            continue
        new = max(minimum, old // 2)
        return replace(settings, **{field.value: new}), {
            "field": field.value,
            "from": old,
            "to": new,
        }
    return None


def run_with_bounded_oom_retries(
    initial: PressureSettings,
    launch_fresh: Callable[[PressureSettings, int], AdaptiveAttemptResult],
    *,
    pressure_order: Iterable[PressureField],
    max_retries: int = MAX_RETRIES,
) -> AdaptiveRunResult:
    """Retry only recognized recoverable pressure exits, always in a fresh child."""

    if max_retries < 0 or max_retries > MAX_RETRIES:
        raise ValueError(f"max_retries must be between zero and {MAX_RETRIES}")
    settings = initial
    attempts: list[AttemptTelemetry] = []
    adjustments: list[Mapping[str, int | str]] = []
    recoverable = {ExitKind.ACCELERATOR_OOM, ExitKind.HOST_SOFT_LIMIT}
    for attempt in range(max_retries + 1):
        result = launch_fresh(settings, attempt)
        attempts.append(result.telemetry)
        if (
            result.success
            or result.exit_kind not in recoverable
            or attempt >= max_retries
        ):
            return AdaptiveRunResult(result, tuple(attempts), tuple(adjustments))
        reduced = _reduce_pressure(settings, pressure_order)
        if reduced is None:
            return AdaptiveRunResult(result, tuple(attempts), tuple(adjustments))
        settings, adjustment = reduced
        adjustments.append({"attempt": attempt + 1, **adjustment})
    raise AssertionError("bounded retry loop did not return")


class MemoryProfileStore:
    """Small atomic local profile store with strict schema invalidation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> tuple[MemoryMeasurement, ...]:
        if not self.path.exists():
            return ()
        with self.path.open("rb") as stream:
            encoded = stream.read(MAX_PROFILE_BYTES + 1)
        if len(encoded) > MAX_PROFILE_BYTES:
            raise ValueError("memory profile store exceeds its size cap")
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "records"}:
            raise ValueError("memory profile store has invalid fields")
        if raw["schema_version"] != PROFILE_SCHEMA_VERSION:
            return ()
        records = raw["records"]
        if not isinstance(records, list) or len(records) > MAX_PROFILE_RECORDS:
            raise ValueError("memory profile record count exceeds its cap")
        output = []
        for record in records:
            identity = ProfileIdentity(**record.pop("identity"))
            settings = PressureSettings(**record.pop("settings"))
            record["accelerator_kind"] = AcceleratorKind(record["accelerator_kind"])
            measurement = MemoryMeasurement(identity, settings, **record)
            if measurement.estimator_version == ESTIMATOR_VERSION:
                output.append(measurement)
        return tuple(output)

    def save(self, records: Iterable[MemoryMeasurement]) -> None:
        bounded = tuple(records)
        if len(bounded) > MAX_PROFILE_RECORDS:
            raise ValueError("memory profile record count exceeds its cap")
        serialized = []
        for measurement in bounded:
            raw = asdict(measurement)
            raw["accelerator_kind"] = measurement.accelerator_kind.value
            serialized.append(raw)
        encoded = json.dumps(
            {"schema_version": PROFILE_SCHEMA_VERSION, "records": serialized},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MAX_PROFILE_BYTES:
            raise ValueError("memory profile store exceeds its size cap")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
