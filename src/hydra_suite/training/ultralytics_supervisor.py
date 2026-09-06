"""Contain generic Ultralytics CLI training in the shared sidecar supervisor."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from hydra_suite.runtime.memory_profiles import (
    AdaptiveAttemptResult,
    AttemptTelemetry,
    PressureField,
    PressureSettings,
    resource_telemetry,
    run_with_bounded_oom_retries,
)
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
    ResourceObservation,
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
from hydra_suite.training.device_ids import (
    is_bare_ordinal_device,
    normalize_cuda_device,
)
from hydra_suite.training.yolo_autobatch import (
    BATCH_RESOLUTION_FILENAME,
    ResolutionCanceled,
    batch_resolution_block,
    child_degraded_reasons,
    resolve_yolo_batch,
)

#: WARNING PERIOD (temporary). `_accelerator` now recognises Ultralytics' bare
#: ordinal device convention ("0", "0,1") as CUDA. Those runs were classified
#: CPU before, so they never faced the accelerator admission gate. While this
#: flag is True, a bare-ordinal run that the accelerator gate would REFUSE is
#: instead admitted with a loud warning and host-only accounting -- exactly the
#: accounting it had before the widening. Nothing else is downgraded: a run the
#: PRE-CHANGE evaluation would also have refused is still refused.
#:
#: The warning re-emits once per rung of the bounded OOM-retry ladder: that is
#: INTENDED, not a duplicate -- each rung carries a different batch and so a
#: different estimate, and a later rung may clear the gate on its own.
#:
#: TO END THE WARNING PERIOD: set this to False (then delete this constant and
#: the `_bare_ordinal_gate_warning` branch in `_run_ultralytics_once`).
BARE_ORDINAL_ACCELERATOR_GATE_WARNING_PERIOD = True

OUTPUT_MAX_LINES = 512
OUTPUT_MAX_CHARS = 256 * 1024
POLL_SECONDS = 0.1
MAX_PROCESSES = 512


def _accelerator(device: str):
    """Classify a device string for containment purposes.

    Bare ordinals ("0", "0,1") are Ultralytics' OWN convention and are what
    the runbook's example uses; they are CUDA, and are resolved through the
    shared `normalize_cuda_device` (multi-GPU forms resolve against the FIRST
    device, matching what Ultralytics profiles).

    A bare ordinal on a box WITHOUT CUDA still returns CPU rather than raising:
    raising would turn a run that worked yesterday into a hard failure. Only an
    explicit ``cuda...`` request raises when the device is absent.
    """

    value = str(device or "auto").strip().lower()
    if value == "mps" or (value == "auto" and sys.platform == "darwin"):
        return AcceleratorKind.MPS, None
    bare_ordinal = is_bare_ordinal_device(value)
    if value.startswith("cuda") or value == "auto" or bare_ordinal:
        from hydra_suite.training.sam3_lora.preflight import _probe_cuda_device

        observed = _probe_cuda_device(
            normalize_cuda_device(value) if bare_ordinal else value
        )
        if observed is not None:
            return AcceleratorKind.CUDA, observed
        if value.startswith("cuda"):
            raise RuntimeError("the requested CUDA device is unavailable")
    return AcceleratorKind.CPU, None


def _stamp_admission(telemetry: dict, warnings: list[str]) -> dict:
    """Record a warning-period downgrade in the DURABLE telemetry record.

    On a downgrade the persisted budget is the host-only one -- correct, that
    is what containment enforced -- but the run really executed on CUDA with a
    device pin. Without this stamp a later reader concludes it was host-only.
    """

    stamped = dict(telemetry or {})
    stamped["admission_downgraded"] = bool(warnings)
    stamped["admission_warnings"] = list(warnings)
    return stamped


def _gib(value: int) -> str:
    return f"{max(0, int(value)) / GiB:.1f} GiB"


def _bare_ordinal_gate_warning(device, budget) -> str:
    """The warning-period notice; see BARE_ORDINAL_ACCELERATOR_GATE_WARNING_PERIOD."""

    return (
        "WARNING: device=%s is Ultralytics' bare-ordinal convention. It used to "
        "be classified as CPU for containment (host-only accounting, no CUDA "
        "pin); it is now correctly classified as CUDA, and under that "
        "classification the accelerator admission gate WOULD REFUSE this run: "
        "estimated accelerator peak %s against %s usable of %s available "
        "device memory. During the warning period the run is allowed to "
        "proceed with the host-only accounting it had before. THIS "
        "CONFIGURATION WILL BE REFUSED IN A FUTURE RELEASE -- lower the batch "
        "or free device memory. Gate refusals: %s"
        % (
            device,
            _gib(budget.accelerator_peak_bytes),
            _gib(budget.usable_accelerator_bytes or 0),
            _gib(budget.available_accelerator_bytes or 0),
            "; ".join(budget.refusals),
        )
    )


def _estimate_host_bytes(spec) -> int:
    params = spec.hyperparams
    batch = int(params.batch)
    if batch <= 0:
        # Pre-launch resolution runs in front of every estimate, so a
        # non-positive batch here means it was skipped. Silently coercing to 1
        # was the original bug: containment, the host estimate and the
        # accelerator estimate were all admitted for batch 1 while the child
        # went on to choose something much larger.
        raise ValueError(
            "the host estimate needs a resolved positive batch; " f"got batch={batch}"
        )
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


def _run_ultralytics_once(
    command: Sequence[str],
    spec,
    *,
    log_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Run one CLI under immutable limits and return bounded exit evidence."""
    accelerator, cuda = _accelerator(spec.device)
    if (
        accelerator is AcceleratorKind.CUDA
        and is_bare_ordinal_device(spec.device)
        and str(spec.device).count(",") >= 1
        and log_cb is not None
    ):
        log_cb(
            f"containment: {spec.device} names several devices; accounting and "
            f"the device pin resolve against {normalize_cuda_device(spec.device)} "
            "alone rather than overstating the capacity of the set."
        )
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

    def request(accelerator_peak: int):
        return ResourceRequest(
            job_name="Ultralytics training",
            phases=(
                PhaseEstimate(
                    "training",
                    host_peak_bytes=estimate,
                    accelerator_peak_bytes=accelerator_peak,
                ),
            ),
            limits=WorkLimits(
                batch_size=int(spec.hyperparams.batch),
                workers=max(0, int(spec.hyperparams.workers)),
                prefetch_batches=2,
            ),
        )

    initial = observe()
    budget = evaluate_resource_request(
        request(estimate if cuda is not None else 0), initial, policy
    )
    admission_warnings: list[str] = []
    if (
        not budget.admitted
        and BARE_ORDINAL_ACCELERATOR_GATE_WARNING_PERIOD
        and accelerator is AcceleratorKind.CUDA
        and is_bare_ordinal_device(spec.device)
    ):
        # The oracle is the LITERAL pre-change evaluation: this device string
        # used to classify as CPU, so re-run the admission it actually faced
        # yesterday -- a CPU observation with no accelerator estimate. Only if
        # THAT admits is the refusal newly introduced by the widening; any
        # refusal it reproduces (host, and every non-admission refusal further
        # down: lease, prelaunch, dataset) is left to refuse untouched.
        #
        # It is derived from the SAME observation rather than probed again:
        # a second psutil reading milliseconds later could show more free host
        # memory and so admit a genuine HOST refusal, which is exactly the
        # broad suppression this design forbids. Sharing one observation means
        # `legacy` and `budget` can differ ONLY in the accelerator refusal --
        # which is the whole scoping argument in one sentence.
        legacy = evaluate_resource_request(
            request(0),
            ResourceObservation(
                total_host_bytes=initial.total_host_bytes,
                available_host_bytes=initial.available_host_bytes,
            ),
            policy,
        )
        if legacy.admitted:
            warning = _bare_ordinal_gate_warning(spec.device, budget)
            admission_warnings.append(warning)
            if log_cb is not None:
                log_cb(warning)
            budget = legacy
    if not budget.admitted:
        return {
            "success": False,
            "failure_kind": ExitKind.HOST_ADMISSION_REFUSAL.value,
            "error_message": "; ".join(budget.refusals),
            "admission_warnings": admission_warnings,
            "resource_telemetry": _stamp_admission(
                resource_telemetry(budget, hard_host_bytes=0, soft_host_bytes=0),
                admission_warnings,
            ),
        }
    hard = min(budget.usable_host_bytes, estimate)
    soft = max(1, int(hard * 0.9))
    environment = dict(os.environ)
    cuda_uuid = None
    cuda_pci = None
    accelerator_probe = None
    # The PARENT re-probes by DEVICE STRING, never by UUID. `_probe_cuda_device`
    # resolves a UUID only out of CUDA_VISIBLE_DEVICES, which is set on the
    # CHILD's environment below and not on the parent's -- so passing
    # `cuda_uuid` here made every re-probe return None and both call sites
    # raise unconditionally ("telemetry became unavailable" / "the selected
    # physical CUDA device changed"), regardless of memory. Verified against a
    # real device on the CUDA box. SAM3 already had the right shape: probe by
    # the device string, then compare the observed `.uuid` to the pinned one --
    # that equality is what actually detects a device swap, and it is kept.
    cuda_probe_device = normalize_cuda_device(spec.device)
    if cuda is not None:
        cuda_uuid = cuda.uuid
        environment["CUDA_VISIBLE_DEVICES"] = cuda_uuid

        def accelerator_probe() -> int:
            from hydra_suite.training.sam3_lora.preflight import _probe_cuda_device

            current = _probe_cuda_device(cuda_probe_device)
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

            current = _probe_cuda_device(cuda_probe_device)
            if current is None or current.uuid != cuda_uuid:
                raise RuntimeError("the selected physical CUDA device changed")
            if budget.accelerator_peak_bytes > int(
                current.free_bytes * policy.accelerator_safety_fraction
            ):
                raise RuntimeError("available accelerator memory fell before launch")

    try:
        sidecar = SupervisedSidecar(
            plan,
            prelaunch_check=prelaunch_check,
            accelerator_probe=accelerator_probe,
            output_max_lines=OUTPUT_MAX_LINES,
            output_max_chars=OUTPUT_MAX_CHARS,
        )
    except WorkloadStillOwnedError:
        raise
    except (ResourceBusyError, RuntimeError, OSError, ValueError) as exc:
        return {
            "success": False,
            "failure_kind": ExitKind.HOST_ADMISSION_REFUSAL.value,
            "error_message": bounded_terminal_text(exc),
            "admission_warnings": admission_warnings,
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
                    "admission_warnings": admission_warnings,
                }
            if eof:
                time.sleep(POLL_SECONDS)
        result = sidecar.wait()
    except WorkloadStillOwnedError:
        raise
    except BaseException:
        if sidecar.process is not None and sidecar.process.poll() is None:
            sidecar.cancel(2.0)
        raise
    return {
        "admission_warnings": admission_warnings,
        "success": result.classified_exit.kind is ExitKind.SUCCESS,
        "canceled": result.classified_exit.kind is ExitKind.CANCELED,
        "exit_code": result.returncode,
        "failure_kind": result.classified_exit.kind.value,
        "error_message": result.classified_exit.message,
        "hard_host_bytes": hard,
        "peak_tree_rss_bytes": result.peak_tree_rss_bytes,
        "peak_accelerator_bytes": result.peak_accelerator_bytes,
        "dropped_output_lines": result.dropped_output_lines,
        "resource_telemetry": _stamp_admission(
            resource_telemetry(
                budget,
                hard_host_bytes=hard,
                soft_host_bytes=soft,
                result=result,
                effective_parameters={
                    "imgsz": int(spec.hyperparams.imgsz),
                    "cache": bool(spec.hyperparams.cache),
                },
            ),
            admission_warnings,
        ),
    }


def _write_batch_resolution(
    run_dir: str | Path,
    resolution: dict,
    log_cb: Callable[[str], None] | None,
) -> None:
    """Persist ``batch_resolution.json``; never fail a run over provenance."""

    try:
        target = Path(run_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / BATCH_RESOLUTION_FILENAME).write_text(
            json.dumps(resolution, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        if log_cb is not None:
            log_cb(f"auto batch: could not persist the resolution: {exc}")


def _command_with_pressure(
    command: Sequence[str], settings: PressureSettings
) -> tuple[str, ...]:
    replacements = {
        "batch": settings.batch_size,
        "workers": settings.workers,
    }
    output = []
    seen = set()
    for raw in command:
        text = str(raw)
        key, separator, _value = text.partition("=")
        if separator and key in replacements:
            output.append(f"{key}={replacements[key]}")
            seen.add(key)
        else:
            output.append(text)
    for key, value in replacements.items():
        if key not in seen:
            output.append(f"{key}={value}")
    return tuple(output)


def run_ultralytics_supervised(
    command: Sequence[str],
    spec,
    *,
    run_dir: Path | str | None = None,
    log_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Run training with finite, classified OOM retries in fresh sidecars.

    A non-positive ``spec.hyperparams.batch`` means "let Ultralytics size it".
    That is resolved to a positive number BEFORE the first launch, so
    ``PressureSettings`` keeps its ``>= 1`` invariant, containment is sized for
    the batch that will actually run, and the OOM-retry halving ladder starts
    from a real number.
    """

    requested_batch = int(spec.hyperparams.batch)
    resolved_batch, provenance = requested_batch, "explicit"
    resolution: dict | None = None
    if requested_batch <= 0:
        # Ask the classification question about the NORMALISED device.
        # `_accelerator` now normalises bare ordinals itself, so this is
        # belt-and-braces rather than the only path; `spec.device` is still
        # left alone, and the launch command keeps the user's original string.
        try:
            accelerator_kind, _cuda = _accelerator(normalize_cuda_device(spec.device))
        except RuntimeError:
            # The normalised device is not actually present. Resolution has
            # nothing to measure on, but this run would still launch under its
            # original classification, so fall back instead of refusing it.
            accelerator_kind = AcceleratorKind.CPU
        try:
            resolved_batch, provenance = resolve_yolo_batch(
                spec,
                Path(run_dir) if run_dir is not None else None,
                accelerator_kind=accelerator_kind,
                run_child=lambda child_command, child_spec: _run_ultralytics_once(
                    child_command,
                    child_spec,
                    log_cb=log_cb,
                    should_cancel=should_cancel,
                ),
                log_cb=log_cb,
                should_cancel=should_cancel,
            )
        except ResolutionCanceled as exc:
            # Launch nothing and write no resolution: this run never resolved.
            return {
                "success": False,
                "canceled": True,
                "failure_kind": ExitKind.CANCELED.value,
                "error_message": str(exc),
                "effective_command": list(command),
            }
        spec = replace(
            spec, hyperparams=replace(spec.hyperparams, batch=resolved_batch)
        )
    if run_dir is not None:
        # The child flags anything that made its estimate weaker (a missing
        # train-label root hides assigner memory and skews the estimate
        # optimistic). Carry that through rather than writing a clean [].
        degraded = child_degraded_reasons(Path(run_dir)) if requested_batch <= 0 else []
        if degraded and log_cb is not None:
            log_cb("auto batch: DEGRADED estimate -- " + "; ".join(degraded))
        resolution = batch_resolution_block(
            requested_batch, resolved_batch, provenance, degraded
        )
        # Written BEFORE the run so a crash, a kill, or a machine going away
        # mid-training still leaves the provenance of what was chosen. It is
        # rewritten once the OOM-retry ladder settles to fill in
        # `effective_batch`; until then that key is null, which is honest --
        # nothing has trained yet.
        _write_batch_resolution(run_dir, resolution, log_cb)

    initial = PressureSettings(
        input_width=max(1, int(spec.hyperparams.imgsz)),
        input_height=max(1, int(spec.hyperparams.imgsz)),
        batch_size=int(spec.hyperparams.batch),
        workers=max(0, int(spec.hyperparams.workers)),
        prefetch_batches=2,
    )
    results: list[dict] = []
    attempted_settings: list[PressureSettings] = []

    def launch_fresh(settings: PressureSettings, attempt: int) -> AdaptiveAttemptResult:
        attempted_settings.append(settings)
        attempt_spec = replace(
            spec,
            hyperparams=replace(
                spec.hyperparams,
                batch=settings.batch_size,
                workers=settings.workers,
            ),
        )
        attempt_command = _command_with_pressure(command, settings)
        if attempt and log_cb is not None:
            log_cb(
                "Retrying after a classified memory-pressure exit with "
                f"batch={settings.batch_size}, workers={settings.workers}."
            )
        result = _run_ultralytics_once(
            attempt_command,
            attempt_spec,
            log_cb=log_cb,
            progress_cb=progress_cb,
            should_cancel=should_cancel,
        )
        result["effective_command"] = list(attempt_command)
        results.append(result)
        try:
            kind = ExitKind(str(result.get("failure_kind", "ordinary-failure")))
        except ValueError:
            kind = ExitKind.ORDINARY_FAILURE
        telemetry = dict(result.get("resource_telemetry") or {})
        observed = dict(telemetry.get("observed") or {})
        return AdaptiveAttemptResult(
            bool(result.get("success", False)),
            kind,
            AttemptTelemetry(
                attempt,
                kind,
                settings,
                hard_host_bytes=int(result.get("hard_host_bytes", 0)),
                peak_tree_rss_bytes=int(observed.get("peak_tree_rss_bytes", 0) or 0),
                minimum_system_available_bytes=observed.get(
                    "minimum_system_available_bytes"
                ),
                peak_accelerator_bytes=observed.get("peak_accelerator_bytes"),
                queue_high_water_bytes=int(
                    observed.get("queue_high_water_bytes", 0) or 0
                ),
            ),
        )

    adaptive = run_with_bounded_oom_retries(
        initial,
        launch_fresh,
        pressure_order=(PressureField.BATCH_SIZE, PressureField.WORKERS),
    )
    final = results[-1]
    if run_dir is not None and resolution is not None and attempted_settings:
        # `resolved` keeps meaning "what resolution chose"; `effective_batch`
        # records what the last attempt actually trained with. The ladder
        # halves the batch on a classified OOM, so without this the durable
        # artifact could claim a batch 2x or 4x larger than the one that ran.
        # Rewriting (rather than appending a second file) keeps one artifact
        # per run with one meaning per key.
        resolution["effective_batch"] = int(attempted_settings[-1].batch_size)
        _write_batch_resolution(run_dir, resolution, log_cb)
    history = [dict(item) for item in adaptive.adjustments]
    final["retry_history"] = history
    telemetry = dict(final.get("resource_telemetry") or {})
    telemetry["retry_history"] = history
    final["resource_telemetry"] = telemetry
    return final
