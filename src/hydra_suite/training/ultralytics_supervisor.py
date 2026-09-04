"""Contain generic Ultralytics CLI training in the shared sidecar supervisor."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

from hydra_suite.runtime.process_supervisor import (
    ContainmentPlan,
    ExitKind,
    SupervisedSidecar,
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
from hydra_suite.runtime.resource_lease import ResourceBusyError
from hydra_suite.runtime.resource_limits import (
    ProcessMemoryLimits,
    build_limited_launch,
)
from hydra_suite.runtime.safe_text import bounded_terminal_text

OUTPUT_MAX_LINES = 512
OUTPUT_MAX_CHARS = 256 * 1024
POLL_SECONDS = 0.1
MAX_PROCESSES = 512


def _accelerator(device: str):
    value = str(device or "auto").strip().lower()
    if value == "mps" or (value == "auto" and sys.platform == "darwin"):
        return AcceleratorKind.MPS, None
    if value.startswith("cuda") or value == "auto":
        from hydra_suite.training.sam3_lora.preflight import _probe_cuda_device

        observed = _probe_cuda_device(value)
        if observed is not None:
            return AcceleratorKind.CUDA, observed
        if value.startswith("cuda"):
            raise RuntimeError("the requested CUDA device is unavailable")
    return AcceleratorKind.CPU, None


def _estimate_host_bytes(spec) -> int:
    params = spec.hyperparams
    batch = max(1, int(params.batch))
    imgsz = max(32, int(params.imgsz))
    # Activations, augmentation workspace, optimizer/model state, and runtime.
    estimate = 2 * GiB + batch * imgsz * imgsz * 3 * 4 * 10
    model = Path(str(spec.base_model)).expanduser()
    if model.is_file():
        estimate += min(model.stat().st_size * 6, 8 * GiB)
    if bool(params.cache):
        dataset = Path(spec.derived_dataset_dir).expanduser()
        count = sum(
            1
            for path in dataset.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )
        estimate += count * imgsz * imgsz * 3
    return max(4 * GiB, estimate)


def _extract_progress(message: str) -> tuple[int, int] | None:
    patterns = (
        re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE),
        re.compile(r"Epoch\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE),
        re.compile(r"^\s*(\d+)\s*/\s*(\d+)(?:\s|$)"),
        re.compile(
            r"\bepoch\s*[=:]\s*(\d+)\b.*\btotal\s*[=:]\s*(\d+)\b",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        match = pattern.search(message)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def run_ultralytics_supervised(
    command: Sequence[str],
    spec,
    *,
    log_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Run one CLI under immutable limits and return bounded exit evidence."""
    accelerator, cuda = _accelerator(spec.device)
    estimate = _estimate_host_bytes(spec)
    policy = ResourcePolicy()

    def observe():
        if accelerator is AcceleratorKind.CUDA:
            assert cuda is not None
            return probe_resources(
                accelerator,
                accelerator_name=cuda.name,
                accelerator_probe=lambda: (cuda.free_bytes, cuda.total_bytes),
            )
        return probe_resources(accelerator)

    initial = observe()
    budget = evaluate_resource_request(
        ResourceRequest(
            job_name="Ultralytics training",
            phases=(PhaseEstimate("training", host_peak_bytes=estimate),),
            limits=WorkLimits(
                batch_size=max(1, int(spec.hyperparams.batch)),
                workers=max(0, int(spec.hyperparams.workers)),
                prefetch_batches=2,
            ),
        ),
        initial,
        policy,
    )
    if not budget.admitted:
        return {
            "success": False,
            "failure_kind": ExitKind.HOST_ADMISSION_REFUSAL.value,
            "error_message": "; ".join(budget.refusals),
        }
    hard = min(budget.usable_host_bytes, estimate)
    soft = max(1, int(hard * 0.9))
    environment = dict(os.environ)
    cuda_uuid = None
    cuda_pci = None
    accelerator_probe = None
    if cuda is not None:
        cuda_uuid = cuda.uuid
        environment["CUDA_VISIBLE_DEVICES"] = cuda_uuid

        def accelerator_probe() -> int:
            from hydra_suite.training.sam3_lora.preflight import _probe_cuda_device

            current = _probe_cuda_device(cuda_uuid)
            if current is None or current.uuid != cuda_uuid:
                raise RuntimeError("selected CUDA device telemetry became unavailable")
            return max(0, current.total_bytes - current.free_bytes)

    ratio = (
        min(0.9, hard / max(1, initial.total_host_bytes))
        if accelerator is AcceleratorKind.MPS
        else None
    )
    launch = build_limited_launch(
        command,
        ProcessMemoryLimits(soft, hard, ratio, MAX_PROCESSES),
        environment=environment,
        accelerator_kind=accelerator,
        accelerator_device_uuid=cuda_uuid,
        accelerator_pci_bus_id=cuda_pci,
    )
    plan = ContainmentPlan(
        launch,
        "Ultralytics training",
        budget.reserved_host_bytes,
        poll_interval_seconds=POLL_SECONDS,
        terminate_grace_seconds=2.0,
    )

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
        if cuda is not None:
            from hydra_suite.training.sam3_lora.preflight import _probe_cuda_device

            current = _probe_cuda_device(cuda_uuid)
            if current is None or current.uuid != cuda_uuid:
                raise RuntimeError("the selected physical CUDA device changed")

    try:
        sidecar = SupervisedSidecar(
            plan,
            prelaunch_check=prelaunch_check,
            accelerator_probe=accelerator_probe,
            output_max_lines=OUTPUT_MAX_LINES,
            output_max_chars=OUTPUT_MAX_CHARS,
        )
    except (ResourceBusyError, RuntimeError, OSError, ValueError) as exc:
        return {
            "success": False,
            "failure_kind": ExitKind.HOST_ADMISSION_REFUSAL.value,
            "error_message": bounded_terminal_text(exc),
        }
    try:
        while sidecar.process is not None and sidecar.process.poll() is None:
            lines, eof, output_error = sidecar.output.drain(POLL_SECONDS)
            if output_error is not None:
                raise output_error
            for line in lines:
                message = line.rstrip("\r\n")
                if message and log_cb is not None:
                    log_cb(message)
                progress = _extract_progress(message)
                if progress is not None and progress_cb is not None:
                    progress_cb(*progress)
            if should_cancel is not None and should_cancel():
                sidecar.cancel(2.0)
                return {
                    "success": False,
                    "canceled": True,
                    "exit_code": (
                        sidecar.process.returncode if sidecar.process else None
                    ),
                    "failure_kind": ExitKind.CANCELED.value,
                    "error_message": "Ultralytics training canceled.",
                    "hard_host_bytes": hard,
                }
            if eof:
                time.sleep(POLL_SECONDS)
        result = sidecar.wait()
    except BaseException:
        if sidecar.process is not None and sidecar.process.poll() is None:
            sidecar.cancel(2.0)
        raise
    return {
        "success": result.classified_exit.kind is ExitKind.SUCCESS,
        "canceled": result.classified_exit.kind is ExitKind.CANCELED,
        "exit_code": result.returncode,
        "failure_kind": result.classified_exit.kind.value,
        "error_message": result.classified_exit.message,
        "hard_host_bytes": hard,
        "peak_tree_rss_bytes": result.peak_tree_rss_bytes,
        "peak_accelerator_bytes": result.peak_accelerator_bytes,
        "dropped_output_lines": result.dropped_output_lines,
    }
