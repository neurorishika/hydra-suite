"""Contained SAM3 LoRA sidecar launcher.

The parent never imports torch or SAM3. Metadata-only admission happens
before launch and is repeated from live host/CUDA observations while the
canonical host and physical-GPU leases are held. The child bootstrap applies
the selected host boundary before conda, torch, or SAM3 can be imported.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

from hydra_suite.runtime.process_supervisor import (
    ContainmentPlan,
    ExitKind,
    SupervisedResult,
    SupervisedSidecar,
    WorkloadStillOwnedError,
)
from hydra_suite.runtime.resource_budget import AcceleratorKind
from hydra_suite.runtime.resource_lease import ResourceBusyError
from hydra_suite.runtime.resource_limits import (
    ProcessMemoryLimits,
    build_limited_launch,
)

from . import preflight as preflight_module
from .artifacts import remove_artifact, validate_completion
from .env import resolve_sam3_env, sam3_env_command, sam3_env_environ
from .protocol import dispatch_record, parse_record

OUTPUT_MAX_LINES = 512
OUTPUT_MAX_CHARS = 256 * 1024
MAX_PROCESSES = 512


class _AdmissionRefused(RuntimeError):
    """A lease-held final resource observation no longer fits the budget."""


class _ArtifactInvalid(RuntimeError):
    """A successful child did not publish a usable adapter artifact."""


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _result(
    *,
    success: bool,
    canceled: bool = False,
    message: str = "",
    failure_kind: str = "",
    exit_code: Optional[int] = None,
    artifact_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    command: tuple[str, ...] = (),
    resource_preflight: Optional[str] = None,
    containment: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": success,
        "canceled": canceled,
        "artifact_path": str(artifact_path) if artifact_path else None,
        "metrics_path": str(metrics_path) if metrics_path else None,
        "command": list(command),
        "exit_code": exit_code,
        "failure_kind": failure_kind,
        "resource_preflight": resource_preflight,
        "containment": containment or {},
    }
    if message:
        payload["error_message"] = message
        payload["error"] = message
    return payload


def _memory_limits(spec: Any, host_peak_bytes: int) -> ProcessMemoryLimits:
    params = spec.sam3_params
    soft = max(1, math.ceil(host_peak_bytes * 1.10))
    hard = max(
        soft,
        math.ceil(host_peak_bytes * float(params.host_limit_headroom_fraction)),
    )
    return ProcessMemoryLimits(
        soft_host_bytes=soft,
        hard_host_bytes=hard,
        max_processes=MAX_PROCESSES,
    )


def _containment_diagnostic(
    plan: ContainmentPlan,
    result: Optional[SupervisedResult] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": plan.launch.backend.value,
        "soft_host_bytes": plan.launch.limits.soft_host_bytes,
        "hard_host_bytes": plan.launch.limits.hard_host_bytes,
        "minimum_system_available_bytes": plan.minimum_system_available_bytes,
        "poll_interval_seconds": plan.poll_interval_seconds,
        "resource_keys": list(plan.expected_resource_keys),
        "limitations": list(plan.launch.limitations),
        "cuda_vram_enforcement": (
            "telemetry-and-admission-only; discrete CUDA VRAM is not kernel-capped"
        ),
    }
    if result is not None:
        payload.update(
            {
                "peak_observed_device_used_bytes": result.peak_accelerator_bytes,
                "peak_observed_tree_rss_bytes": result.peak_tree_rss_bytes,
                "minimum_observed_system_available_bytes": (
                    result.minimum_system_available_bytes
                ),
                "accelerator_observation_error": result.accelerator_observation_error,
                "dropped_output_lines": result.dropped_output_lines,
                "output_error": result.output_error,
            }
        )
    return payload


def train_sam3_lora(
    spec: Any,
    run_dir: str,
    *,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """Run SAM3 training under immutable limits and canonical leases."""

    log_cb = log_cb or (lambda _message: None)
    progress_cb = progress_cb or (lambda _epoch, _total: None)
    should_cancel = should_cancel or (lambda: False)

    try:
        initial = preflight_module.assess_preflight(spec)
    except (OSError, ValueError) as exc:
        return _result(
            success=False,
            message=f"SAM3 resource preflight could not inspect the run: {exc}",
            failure_kind=ExitKind.HOST_ADMISSION_REFUSAL.value,
        )
    if not initial.admitted:
        return _result(
            success=False,
            message="; ".join(initial.refusals),
            failure_kind=ExitKind.HOST_ADMISSION_REFUSAL.value,
        )
    initial_cuda_device = initial.cuda_device
    assert initial_cuda_device is not None

    run_dir_path = Path(run_dir).expanduser().resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir_path / "spec.json"
    diagnostics_path = run_dir_path / "resource_preflight.json"
    _write_json(spec_path, spec.to_dict())

    artifact_path = run_dir_path / "adapters.pt"
    remove_artifact(artifact_path)
    params = spec.sam3_params
    env_name = resolve_sam3_env(params.env_name)
    command = sam3_env_command(
        env_name,
        [
            "hydra_suite.training.sam3_lora.cli",
            "--spec",
            str(spec_path),
            "--run-dir",
            str(run_dir_path),
        ],
    )
    child_environment = {
        **os.environ,
        **sam3_env_environ(),
        # Bind the child logical cuda:0 to the exact physical GPU admitted,
        # probed, and leased by UUID. Never let the runtime choose another GPU.
        "CUDA_VISIBLE_DEVICES": initial_cuda_device.uuid,
    }
    limits = _memory_limits(spec, initial.budget.host_peak_bytes)
    launch = build_limited_launch(
        command,
        limits,
        environment=child_environment,
        accelerator_kind=AcceleratorKind.CUDA,
        accelerator_device_uuid=initial_cuda_device.uuid,
    )
    plan = ContainmentPlan(
        launch=launch,
        job_name="SAM3 LoRA training",
        minimum_system_available_bytes=initial.budget.reserved_host_bytes,
        poll_interval_seconds=float(params.watchdog_poll_seconds),
    )
    diagnostic: dict[str, Any] = {
        "initial": initial.to_dict(),
        "live": None,
        "containment": _containment_diagnostic(plan),
    }
    _write_json(diagnostics_path, diagnostic)

    def final_live_check() -> None:
        live = preflight_module.assess_preflight(spec, dataset=initial.dataset)
        diagnostic["live"] = live.to_dict()
        _write_json(diagnostics_path, diagnostic)
        if live.cuda_device is None:
            raise _AdmissionRefused("CUDA disappeared before SAM3 launch")
        if live.cuda_device.uuid != initial_cuda_device.uuid:
            raise _AdmissionRefused(
                "The selected physical CUDA device changed between admission "
                "and launch; refusing to use a different GPU."
            )
        if not live.admitted:
            raise _AdmissionRefused("; ".join(live.refusals))

    def accelerator_used_bytes() -> int:
        current = preflight_module._probe_cuda_device(str(spec.device))
        if current is None or current.uuid != initial_cuda_device.uuid:
            raise RuntimeError("selected CUDA device telemetry became unavailable")
        return max(0, current.total_bytes - current.free_bytes)

    try:
        sidecar = SupervisedSidecar(
            plan,
            prelaunch_check=final_live_check,
            accelerator_probe=accelerator_used_bytes,
            output_max_lines=OUTPUT_MAX_LINES,
            output_max_chars=OUTPUT_MAX_CHARS,
        )
    except WorkloadStillOwnedError:
        raise
    except (
        ResourceBusyError,
        _AdmissionRefused,
        FileNotFoundError,
        RuntimeError,
    ) as exc:
        return _result(
            success=False,
            message=f"SAM3 sidecar launch refused: {exc}",
            failure_kind=ExitKind.HOST_ADMISSION_REFUSAL.value,
            command=launch.command,
            resource_preflight=str(diagnostics_path),
            containment=_containment_diagnostic(plan),
        )

    try:
        while True:
            lines, eof, output_error = sidecar.output.drain(
                float(params.watchdog_poll_seconds)
            )
            for raw_line in lines:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                record = parse_record(line)
                if record is None:
                    log_cb(line)
                else:
                    dispatch_record(record, log_cb, progress_cb)
            if output_error is not None:
                raise output_error
            process_returncode = sidecar.process.poll()
            # Root exit transfers control to wait(), which owns final tree
            # quiescence. A descendant may inherit stdout and hold EOF open.
            if process_returncode is not None:
                break
            if should_cancel():
                sidecar.cancel(plan.terminate_grace_seconds)
                remove_artifact(artifact_path)
                return _result(
                    success=False,
                    canceled=True,
                    failure_kind=ExitKind.CANCELED.value,
                    command=launch.command,
                    resource_preflight=str(diagnostics_path),
                    containment=_containment_diagnostic(plan),
                )
            if eof:
                # A workload may deliberately close stdout before it exits.
                # Once the buffer is at EOF, drain() cannot block for us.
                time.sleep(float(params.watchdog_poll_seconds))

        def validate_artifact(result: SupervisedResult) -> None:
            validation_error = validate_completion(artifact_path)
            if result.classified_exit.kind is ExitKind.SUCCESS and validation_error:
                raise _ArtifactInvalid(
                    "SAM3 training subprocess exited successfully but did not "
                    f"produce a validated artifact: {validation_error}; refusing "
                    "to report success for a run that trained nothing."
                )

        supervised = sidecar.wait(post_exit_check=validate_artifact)
    except _ArtifactInvalid as exc:
        remove_artifact(artifact_path)
        return _result(
            success=False,
            message=str(exc),
            failure_kind=ExitKind.ORDINARY_FAILURE.value,
            exit_code=0,
            command=launch.command,
            resource_preflight=str(diagnostics_path),
            containment=_containment_diagnostic(plan),
        )
    except BaseException:
        try:
            sidecar.cancel(plan.terminate_grace_seconds)
        except WorkloadStillOwnedError:
            raise
        remove_artifact(artifact_path)
        raise

    diagnostic["containment"] = _containment_diagnostic(plan, supervised)
    _write_json(diagnostics_path, diagnostic)
    classified = supervised.classified_exit
    if classified.kind is not ExitKind.SUCCESS:
        remove_artifact(artifact_path)
        tail = "".join(supervised.output_tail).strip() or "(no output)"
        return _result(
            success=False,
            canceled=classified.kind is ExitKind.CANCELED,
            message=f"{classified.message}. Child output tail:\n{tail}",
            failure_kind=classified.kind.value,
            exit_code=supervised.returncode,
            command=launch.command,
            resource_preflight=str(diagnostics_path),
            containment=diagnostic["containment"],
        )

    metrics_candidate = run_dir_path / "val_stats.json"
    metrics_path = metrics_candidate if metrics_candidate.exists() else None
    return _result(
        success=True,
        artifact_path=artifact_path,
        metrics_path=metrics_path,
        exit_code=supervised.returncode,
        command=launch.command,
        resource_preflight=str(diagnostics_path),
        containment=diagnostic["containment"],
    )
