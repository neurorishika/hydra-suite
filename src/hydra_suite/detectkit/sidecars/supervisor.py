"""Parent-only launcher for protected DetectKit model operations."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from hydra_suite.runtime.memory_profiles import resource_telemetry
from hydra_suite.runtime.process_supervisor import (
    ContainmentPlan,
    ExitKind,
    SupervisedSidecar,
    WorkloadStillOwnedError,
)
from hydra_suite.runtime.resource_budget import (
    AcceleratorKind,
    GiB,
    PhaseEstimate,
    ResourcePolicy,
    ResourceRequest,
    WorkLimits,
    evaluate_resource_request,
    probe_resources,
)
from hydra_suite.runtime.resource_limits import (
    ProcessMemoryLimits,
    build_limited_launch,
)
from hydra_suite.runtime.safe_text import bounded_terminal_text

from .protocol import (
    Operation,
    SidecarRequest,
    SidecarStatus,
    read_result,
    write_request,
)

OUTPUT_MAX_LINES = 256
OUTPUT_MAX_CHARS = 128 * 1024
POLL_SECONDS = 0.1
MAX_PROCESSES = 256


@dataclass(frozen=True, slots=True)
class ProtectedOutcome:
    success: bool
    canceled: bool
    failure_kind: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    hard_host_bytes: int = 0
    peak_tree_rss_bytes: int = 0
    peak_accelerator_bytes: int | None = None
    dropped_output_lines: int = 0
    telemetry: dict[str, Any] = field(default_factory=dict)


def _operation_estimate(operation: Operation, input_paths: Iterable[Path]) -> int:
    model_bytes = 0
    for path in input_paths:
        try:
            model_bytes += Path(path).stat().st_size
        except OSError:
            pass
    base = {
        Operation.ACTIVE_LEARNING: 4 * GiB,
        Operation.DATASET_INFERENCE: 1536 * 1024**2,
        Operation.EVALUATION: 2 * GiB,
        Operation.SEMANTIC_ESCALATION: 7 * GiB,
        Operation.SEMANTIC_CALIBRATION: 7 * GiB,
        Operation.SEMANTIC_PREVIEW: 7 * GiB,
    }[operation]
    return base + min(model_bytes * 6, 12 * GiB)


def _accelerator_for(device: str):
    value = str(device or "auto").strip().lower()
    if value.startswith("cuda") or (value == "auto" and sys.platform != "darwin"):
        from hydra_suite.training.sam3_lora.preflight import _probe_cuda_device

        observed = _probe_cuda_device(value)
        if observed is not None:
            return AcceleratorKind.CUDA, observed.uuid, None, observed
        if value.startswith("cuda"):
            raise RuntimeError("the requested CUDA device is unavailable")
    if value == "mps" or (value == "auto" and sys.platform == "darwin"):
        return AcceleratorKind.MPS, None, None, None
    return AcceleratorKind.CPU, None, None, None


def _identities(paths: Iterable[Path]) -> tuple[tuple[str, int, int], ...]:
    values = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        stat = path.stat()
        values.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(values)


def _parse_progress(line: str) -> tuple[int, str] | None:
    if len(line) > 16_384 or not line.startswith("{"):
        return None
    try:
        raw = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("detectkit_sidecar") != 1:
        return None
    if set(raw) != {"detectkit_sidecar", "type", "percent", "message"}:
        return None
    if raw["type"] != "progress" or not isinstance(raw["percent"], int):
        return None
    if not isinstance(raw["message"], str) or len(raw["message"]) > 4096:
        return None
    return max(0, min(100, raw["percent"])), raw["message"]


def _failed_outcome(
    supervised, hard: int, telemetry: dict[str, Any] | None = None
) -> ProtectedOutcome:
    peak = supervised.peak_tree_rss_bytes
    return ProtectedOutcome(
        False,
        supervised.classified_exit.kind is ExitKind.CANCELED,
        supervised.classified_exit.kind.value,
        (
            f"{supervised.classified_exit.message}. Memory cap: {hard / GiB:.1f} GiB; "
            f"observed peak: {peak / GiB:.1f} GiB. Reduce image/tile size, "
            "batch size, or candidates."
        ),
        hard_host_bytes=hard,
        peak_tree_rss_bytes=peak,
        peak_accelerator_bytes=supervised.peak_accelerator_bytes,
        dropped_output_lines=supervised.dropped_output_lines,
        telemetry=dict(telemetry or {}),
    )


def _reported_failure_outcome(
    supervised, result, hard: int, telemetry: dict[str, Any] | None = None
) -> ProtectedOutcome:
    """Prefer a valid failure report written by a normally exiting child.

    The entrypoint catches operational exceptions, writes a bounded result, and
    exits with code 1.  That is an ordinary failure, not evidence of a memory
    limit, so surfacing the report is essential for actionable GUI feedback.
    A child killed by a containment limit cannot reliably write this result;
    callers only reach here after protocol validation succeeds.
    """
    fallback = _failed_outcome(supervised, hard, telemetry)
    message = str(getattr(result, "message", "") or "").strip()
    if not message:
        return fallback
    canceled = result.status is SidecarStatus.CANCELED
    return ProtectedOutcome(
        False,
        canceled,
        ExitKind.CANCELED.value if canceled else fallback.failure_kind,
        message,
        hard_host_bytes=fallback.hard_host_bytes,
        peak_tree_rss_bytes=fallback.peak_tree_rss_bytes,
        peak_accelerator_bytes=fallback.peak_accelerator_bytes,
        dropped_output_lines=fallback.dropped_output_lines,
        telemetry=fallback.telemetry,
    )


class ProtectedOperation:
    """One synchronously executed sidecar with thread-safe group cancellation."""

    def __init__(
        self,
        request: SidecarRequest,
        *,
        device: str,
        input_paths: Iterable[str | Path] = (),
        cleanup_paths: Iterable[str | Path] = (),
    ) -> None:
        self.request = request
        self.device = str(device or "cpu")
        self.input_paths = tuple(Path(path) for path in input_paths if path)
        self.cleanup_paths = tuple(Path(path) for path in cleanup_paths if path)
        self._lock = threading.Lock()
        self._sidecar: SupervisedSidecar | None = None
        self._cancel_requested = False

    def cancel(self) -> None:
        with self._lock:
            self._cancel_requested = True

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def _cleanup(self) -> None:
        for path in self.cleanup_paths:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
                    chunks = path.parent / f"{path.name}.chunks"
                    if chunks.is_dir():
                        shutil.rmtree(chunks)
                    path.with_suffix(path.suffix + ".paths").unlink(missing_ok=True)
                    path.with_suffix(path.suffix + ".paths.idx").unlink(missing_ok=True)
            except OSError:
                pass

    def discard_artifacts(self) -> None:
        """Remove only the private outputs declared by this operation."""
        self._cleanup()

    def run(
        self,
        *,
        progress: Callable[[int, str], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> ProtectedOutcome:
        progress = progress or (lambda _pct, _message: None)
        log = log or (lambda _line: None)
        control_dir = Path(tempfile.mkdtemp(prefix="hydra-detectkit-sidecar-"))
        request_path = control_dir / "request.json"
        result_path = control_dir / "result.json"
        control_dir_owned = True

        def retain_recovery(error: WorkloadStillOwnedError) -> None:
            """Keep child inputs and private outputs until teardown is proven."""

            nonlocal control_dir_owned
            control_dir_owned = False

            def cleanup_after_recovery() -> None:
                with self._lock:
                    self._sidecar = None
                self._cleanup()
                shutil.rmtree(control_dir, ignore_errors=True)

            error.recovery_cleanup = cleanup_after_recovery

        try:
            write_request(request_path, self.request)
            accelerator, cuda_uuid, cuda_pci, cuda = _accelerator_for(self.device)
            estimate = _operation_estimate(self.request.operation, self.input_paths)
            observation = probe_resources(
                accelerator,
                accelerator_name=getattr(cuda, "name", None),
                accelerator_probe=(
                    (lambda: (cuda.free_bytes, cuda.total_bytes))
                    if cuda is not None
                    else None
                ),
            )
            policy = ResourcePolicy()
            budget = evaluate_resource_request(
                ResourceRequest(
                    job_name=f"DetectKit {self.request.operation.value}",
                    phases=(
                        PhaseEstimate(
                            "model-operation",
                            host_peak_bytes=estimate,
                            accelerator_peak_bytes=(
                                estimate if cuda is not None else 0
                            ),
                        ),
                    ),
                    limits=WorkLimits(batch_size=1, workers=0, prefetch_batches=0),
                ),
                observation,
                policy,
            )
            if not budget.admitted:
                self._cleanup()
                return ProtectedOutcome(
                    False,
                    False,
                    ExitKind.HOST_ADMISSION_REFUSAL.value,
                    "; ".join(budget.refusals),
                    telemetry=resource_telemetry(
                        budget, hard_host_bytes=0, soft_host_bytes=0
                    ),
                )
            hard = min(budget.usable_host_bytes, max(estimate, 2 * GiB))
            soft = max(1, int(hard * 0.9))
            mps_ratio = (
                min(0.9, hard / max(1, observation.total_host_bytes))
                if accelerator is AcceleratorKind.MPS
                else None
            )
            environment = dict(os.environ)
            if cuda_uuid:
                environment["CUDA_VISIBLE_DEVICES"] = cuda_uuid
            command = (
                sys.executable,
                "-m",
                "hydra_suite.detectkit.sidecars.entrypoint",
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            )
            launch = build_limited_launch(
                command,
                ProcessMemoryLimits(
                    soft_host_bytes=soft,
                    hard_host_bytes=hard,
                    mps_high_watermark_ratio=mps_ratio,
                    max_processes=MAX_PROCESSES,
                ),
                environment=environment,
                accelerator_kind=accelerator,
                accelerator_device_uuid=cuda_uuid,
                accelerator_pci_bus_id=cuda_pci,
            )
            plan = ContainmentPlan(
                launch,
                f"DetectKit {self.request.operation.value}",
                budget.reserved_host_bytes,
                poll_interval_seconds=POLL_SECONDS,
                terminate_grace_seconds=2.0,
            )
            initial_identities = _identities(self.input_paths)

            def prelaunch_check() -> None:
                live = probe_resources(accelerator)
                reserve = max(
                    policy.reserve_host_bytes,
                    int(live.total_host_bytes * policy.reserve_host_fraction),
                )
                if hard > max(0, live.available_host_bytes - reserve):
                    raise RuntimeError(
                        "available host memory fell before launch; the immutable cap would expose the reserve"
                    )
                if cuda_uuid is not None:
                    from hydra_suite.training.sam3_lora.preflight import (
                        _probe_cuda_device,
                    )

                    current = _probe_cuda_device(cuda_uuid)
                    if current is None or current.uuid != cuda_uuid:
                        raise RuntimeError("the selected physical CUDA device changed")
                    usable_vram = int(
                        current.free_bytes * policy.accelerator_safety_fraction
                    )
                    if budget.accelerator_peak_bytes > usable_vram:
                        raise RuntimeError(
                            "available accelerator memory fell before launch"
                        )
                if _identities(self.input_paths) != initial_identities:
                    raise RuntimeError(
                        "a model or input artifact changed after admission"
                    )

            def accelerator_probe() -> int:
                if cuda_uuid is None:
                    return 0
                from hydra_suite.training.sam3_lora.preflight import _probe_cuda_device

                current = _probe_cuda_device(cuda_uuid)
                if current is None or current.uuid != cuda_uuid:
                    raise RuntimeError(
                        "selected CUDA device telemetry became unavailable"
                    )
                return max(0, current.total_bytes - current.free_bytes)

            if self.cancelled:
                self._cleanup()
                return ProtectedOutcome(
                    False,
                    True,
                    ExitKind.CANCELED.value,
                    "Operation canceled before launch.",
                )
            sidecar = SupervisedSidecar(
                plan,
                prelaunch_check=prelaunch_check,
                accelerator_probe=(
                    accelerator_probe if accelerator is AcceleratorKind.CUDA else None
                ),
                output_max_lines=OUTPUT_MAX_LINES,
                output_max_chars=OUTPUT_MAX_CHARS,
            )
            with self._lock:
                self._sidecar = sidecar
                cancel_now = self._cancel_requested
            if cancel_now:
                sidecar.cancel(2.0)
                self._cleanup()
                return ProtectedOutcome(
                    False, True, ExitKind.CANCELED.value, "Operation canceled."
                )
            while sidecar.process is not None and sidecar.process.poll() is None:
                lines, _eof, output_error = sidecar.output.drain(POLL_SECONDS)
                if output_error is not None:
                    raise output_error
                for raw_line in lines:
                    parsed = _parse_progress(raw_line.rstrip("\r\n"))
                    if parsed is None:
                        log(raw_line.rstrip("\r\n"))
                    else:
                        progress(*parsed)
                if self.cancelled:
                    sidecar.cancel(2.0)
                    self._cleanup()
                    return ProtectedOutcome(
                        False, True, ExitKind.CANCELED.value, "Operation canceled."
                    )
            supervised = sidecar.wait()
            with self._lock:
                self._sidecar = None
            if supervised.classified_exit.kind is not ExitKind.SUCCESS:
                telemetry = resource_telemetry(
                    budget,
                    hard_host_bytes=hard,
                    soft_host_bytes=soft,
                    result=supervised,
                    effective_parameters={
                        "operation": self.request.operation.value,
                        "chunk_frames": int(
                            self.request.payload.get("chunk_frames", 1)
                        ),
                        "max_targets": int(self.request.payload.get("max_targets", 1)),
                    },
                    cache_chunk_size=(
                        int(self.request.payload.get("chunk_frames", 1))
                        if self.request.operation is Operation.DATASET_INFERENCE
                        else None
                    ),
                )
                try:
                    reported_result = read_result(result_path, expected=self.request)
                except (OSError, ValueError):
                    reported_result = None
                self._cleanup()
                if (
                    reported_result is not None
                    and reported_result.status is not SidecarStatus.SUCCESS
                ):
                    return _reported_failure_outcome(
                        supervised, reported_result, hard, telemetry
                    )
                return _failed_outcome(
                    supervised,
                    hard,
                    telemetry,
                )
            result = read_result(result_path, expected=self.request)
            if result.status is not SidecarStatus.SUCCESS:
                self._cleanup()
                return ProtectedOutcome(
                    False,
                    result.status is SidecarStatus.CANCELED,
                    ExitKind.ORDINARY_FAILURE.value,
                    result.message or "DetectKit sidecar failed without a diagnostic.",
                    hard_host_bytes=hard,
                    peak_tree_rss_bytes=supervised.peak_tree_rss_bytes,
                    telemetry=resource_telemetry(
                        budget,
                        hard_host_bytes=hard,
                        soft_host_bytes=soft,
                        result=supervised,
                    ),
                )
            return ProtectedOutcome(
                True,
                False,
                ExitKind.SUCCESS.value,
                result.message,
                dict(result.payload),
                hard_host_bytes=hard,
                peak_tree_rss_bytes=supervised.peak_tree_rss_bytes,
                peak_accelerator_bytes=supervised.peak_accelerator_bytes,
                dropped_output_lines=supervised.dropped_output_lines,
                telemetry=resource_telemetry(
                    budget,
                    hard_host_bytes=hard,
                    soft_host_bytes=soft,
                    result=supervised,
                    effective_parameters={
                        "operation": self.request.operation.value,
                        "chunk_frames": int(
                            self.request.payload.get("chunk_frames", 1)
                        ),
                        "max_targets": int(self.request.payload.get("max_targets", 1)),
                    },
                    cache_chunk_size=(
                        int(self.request.payload.get("chunk_frames", 1))
                        if self.request.operation is Operation.DATASET_INFERENCE
                        else None
                    ),
                ),
            )
        except WorkloadStillOwnedError as exc:
            retain_recovery(exc)
            raise
        except Exception as exc:
            with self._lock:
                sidecar = self._sidecar
            if sidecar is not None:
                try:
                    sidecar.cancel(2.0)
                except WorkloadStillOwnedError as exc:
                    retain_recovery(exc)
                    raise
            self._cleanup()
            return ProtectedOutcome(
                False,
                self.cancelled,
                (
                    ExitKind.CANCELED.value
                    if self.cancelled
                    else ExitKind.HOST_ADMISSION_REFUSAL.value
                ),
                bounded_terminal_text(exc),
            )
        finally:
            if control_dir_owned:
                shutil.rmtree(control_dir, ignore_errors=True)
