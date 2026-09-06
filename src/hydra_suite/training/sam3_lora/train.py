"""Contained SAM3 LoRA sidecar launcher.

The parent never imports torch or SAM3. Metadata-only admission happens
before launch and is repeated from live host/CUDA observations while the
canonical host and physical-GPU leases are held. The child bootstrap applies
the selected host boundary before conda, torch, or SAM3 can be imported.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from hydra_suite.runtime.memory_profiles import (
    MemoryProfileStore,
    fit_batch_curve,
    merge_records,
    profile_store_path,
)
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

from ..model_publish import get_models_root
from . import autobatch
from . import preflight as preflight_module
from .artifacts import remove_artifact, validate_completion
from .env import resolve_sam3_env, sam3_env_command, sam3_env_environ
from .protocol import dispatch_record, parse_record

OUTPUT_MAX_LINES = 512
OUTPUT_MAX_CHARS = 256 * 1024
MAX_PROCESSES = 512


GiB = 1024**3


class _AdmissionRefused(RuntimeError):
    """A lease-held final resource observation no longer fits the budget."""


class _Canceled(RuntimeError):
    """The user cancelled; the child has already been torn down."""


class _BatchResolutionRefused(RuntimeError):
    """Auto batch sizing could not produce a batch size this run may use."""


# CUDA OOM is invisible to the supervisor -- it sees exit codes and cgroup
# kills, not a Python `OutOfMemoryError` -- so the probe child ALSO writes an
# explicit `{"outcome": "oom"}` record; both routes are honoured.
#
# HOST limit kills are deliberately NOT treated as a device verdict. They are
# caused by whatever else is running on the box, so a ladder that ends on one
# is INCOMPLETE: caching its survivors is right, but letting them permanently
# cap the batch size would turn one bad afternoon into a silent forever
# ceiling.
_HOST_LIMIT_KINDS = frozenset({ExitKind.HOST_HARD_LIMIT, ExitKind.HOST_SOFT_LIMIT})


def _store_path() -> Path:
    """Where measured SAM3 memory profiles live (seam for tests)."""

    return profile_store_path(autobatch.PROFILE_SCOPE)


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


def _memory_limits(decision: Any) -> ProcessMemoryLimits:
    return ProcessMemoryLimits(
        soft_host_bytes=decision.containment_soft_host_bytes,
        hard_host_bytes=decision.containment_hard_host_bytes,
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


def _pump_child_output(
    sidecar: Any,
    plan: ContainmentPlan,
    params: Any,
    log_cb: Callable[[str], None],
    progress_cb: Callable[[int, int], None],
    should_cancel: Callable[[], bool],
) -> None:
    """Drain one supervised child until it exits, or cancel it and raise.

    Shared by the memory probe and by training so a probe child is torn down
    through exactly the same supervisor path a training child is. Raises
    `_Canceled` AFTER `sidecar.cancel` has run, so a `WorkloadStillOwnedError`
    from that teardown still propagates to the caller's own handler.
    """

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
            return
        if should_cancel():
            sidecar.cancel(plan.terminate_grace_seconds)
            raise _Canceled()
        if eof:
            # A workload may deliberately close stdout before it exits.
            # Once the buffer is at EOF, drain() cannot block for us.
            time.sleep(float(params.watchdog_poll_seconds))


def _child_environment(cuda_device: Any) -> dict[str, str]:
    """The sidecar environment, with the physical GPU pinned by UUID.

    Bind the child logical cuda:0 to the exact physical GPU admitted, probed,
    and leased by UUID. Never let the runtime choose another GPU -- a probe
    that measured a different card than training will use is worse than no
    measurement at all.
    """

    return {
        **os.environ,
        **sam3_env_environ(),
        "CUDA_VISIBLE_DEVICES": cuda_device.uuid,
    }


def _sam3_command(env_name: str, arguments: list[str]) -> tuple[str, ...]:
    return sam3_env_command(
        env_name, ["hydra_suite.training.sam3_lora.cli"] + arguments
    )


def _run_probe_candidate(
    spec: Any,
    run_dir_path: Path,
    batch: int,
    *,
    spec_path: Path,
    cuda_device: Any,
    env_name: str,
    params: Any,
    models_root: Optional[Path],
    log_cb: Callable[[str], None],
    should_cancel: Callable[[], bool],
) -> dict[str, Any]:
    """Measure ONE candidate in its own freshly contained sidecar.

    Every candidate gets its own `assess_probe_preflight` decision, its own
    `WorkLimits`, and the same `CUDA_VISIBLE_DEVICES` pin training uses.
    Candidates 2-8 must not run under batch-1 containment: a model reload
    costs seconds, wrong containment costs the box.

    The candidate arrives on the command line, never in `spec.json`: the spec
    carries exactly one `batch`, and it is rewritten once, with the resolved
    value, after the ladder.
    """

    decision = preflight_module.assess_probe_preflight(
        spec,
        batch=batch,
        run_dir=run_dir_path,
        models_root=models_root,
    )
    if not decision.admitted:
        raise autobatch.ProbeCandidateRefused("; ".join(decision.refusals))

    record_path = run_dir_path / autobatch.PROBE_RECORDS_DIRNAME / f"batch_{batch}.json"
    if record_path.exists():
        # A stale record from an earlier attempt must never be mistaken for
        # this candidate's measurement.
        record_path.unlink()

    command = _sam3_command(
        env_name,
        [
            "--spec",
            str(spec_path),
            "--run-dir",
            str(run_dir_path),
            "--probe",
            "--probe-batch",
            str(batch),
        ],
    )
    launch = build_limited_launch(
        command,
        _memory_limits(decision),
        environment=_child_environment(cuda_device),
        accelerator_kind=AcceleratorKind.CUDA,
        accelerator_device_uuid=cuda_device.uuid,
    )
    plan = ContainmentPlan(
        launch=launch,
        job_name=f"SAM3 LoRA memory probe (batch {batch})",
        minimum_system_available_bytes=decision.budget.reserved_host_bytes,
        poll_interval_seconds=float(params.watchdog_poll_seconds),
    )
    try:
        sidecar = SupervisedSidecar(
            plan,
            output_max_lines=OUTPUT_MAX_LINES,
            output_max_chars=OUTPUT_MAX_CHARS,
        )
    except WorkloadStillOwnedError:
        raise
    except (ResourceBusyError, FileNotFoundError, RuntimeError) as exc:
        # Mirrors the training constructor guard. A busy lease -- another
        # heavy job on this box, the normal case this machinery exists for --
        # is a structured refusal, not an unhandled traceback out of the
        # probe ladder. `ResourceBusyError` is a RuntimeError, not an
        # OSError, so it would not otherwise be caught upstream.
        #
        # Raised as a CANDIDATE refusal, not a resolution refusal: if earlier
        # rungs already measured, those measurements are facts about this
        # hardware and must still be cached. `run_probe` owns the
        # first-candidate-vs-later distinction.
        raise autobatch.ProbeCandidateRefused(
            f"SAM3 probe sidecar launch refused at batch {batch}: {exc}"
        ) from exc
    try:
        _pump_child_output(
            sidecar, plan, params, log_cb, lambda _e, _t: None, should_cancel
        )
        supervised = sidecar.wait()
    except _Canceled:
        raise autobatch.ProbeCanceled(
            f"cancelled while probing batch {batch}"
        ) from None
    except WorkloadStillOwnedError:
        raise
    except BaseException:
        try:
            sidecar.cancel(plan.terminate_grace_seconds)
        except WorkloadStillOwnedError:
            raise
        raise

    payload: dict[str, Any] = {}
    if record_path.exists():
        try:
            loaded = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            payload = loaded
    outcome = payload.get("outcome")
    kind = supervised.classified_exit.kind
    if outcome == "ok" and kind is ExitKind.SUCCESS:
        return payload
    if kind in _HOST_LIMIT_KINDS:
        # A HOST cgroup/RLIMIT kill says something about what else is running
        # on this box right now, not about what this workload needs on this
        # GPU. Recorded as transient so it can never become a permanent
        # batch ceiling.
        raise autobatch.ProbeHostLimitError(
            f"SAM3 memory probe at batch {batch} was stopped by a host memory "
            f"limit ({kind.value}); this is transient, not a device verdict."
        )
    if outcome == "oom" or kind is ExitKind.ACCELERATOR_OOM:
        raise autobatch.ProbeOutOfMemoryError(
            f"SAM3 memory probe at batch {batch} did not fit "
            f"({outcome or kind.value})."
        )
    tail = "".join(supervised.output_tail).strip() or "(no output)"
    raise autobatch.ProbeCandidateRefused(
        f"SAM3 memory probe at batch {batch} failed without producing a "
        f"measurement ({kind.value}). Child output tail:\n{tail}"
    )


def _resolve_measured_batch(
    spec: Any,
    run_dir_path: Path,
    params: Any,
    *,
    env_name: str,
    models_root: Optional[Path],
    log_cb: Callable[[str], None],
    should_cancel: Callable[[], bool],
) -> tuple[int, dict[str, Any]]:
    """Steps 0-6 of the parent flow: observe, fingerprint, probe, select.

    Returns `(resolved_batch, batch_resolution)`. Raises
    `_BatchResolutionRefused` when the run must not launch and
    `autobatch.ProbeCanceled` when the user cancelled.
    """

    # Step 0: observe the physical device BEFORE any preflight. The
    # fingerprint needs its identity and every launch needs its UUID.
    cuda_device = preflight_module._probe_cuda_device(
        str(getattr(spec, "device", "auto"))
    )
    if cuda_device is None:
        raise _BatchResolutionRefused(
            "No CUDA device is available; SAM3 LoRA training requires CUDA."
        )

    dataset = autobatch.sam3_dataset_density_profile(spec)
    fingerprint = autobatch.sam3_workload_fingerprint(
        spec, cuda_device=cuda_device, dataset=dataset
    )
    if fingerprint.degraded_reasons:
        # As loud as the resolved-batch banner, deliberately. A degraded key
        # still works -- it cannot collide with a healthy one -- but it means
        # the cache will miss more often, and a silent degradation is exactly
        # how this whole class of bug hides.
        log_cb(
            "auto batch: DEGRADED fingerprint -- "
            + ", ".join(fingerprint.degraded_reasons)
            + ". The measurement will still be cached, but this key is weaker "
            "than a full one and will miss more often."
        )

    store = MemoryProfileStore(_store_path())
    stored = store.load()
    cached = autobatch.validate_probe_records(stored, fingerprint.identity)
    forced = os.environ.get(autobatch.FORCE_PROBE_ENV_VAR) == "1"
    key = _fingerprint_key(fingerprint.identity)
    incomplete = _incomplete_ladders()
    was_incomplete = incomplete.get(key)

    if cached and not forced and was_incomplete is None:
        records = cached
        provenance = "cached"
        terminated_by = autobatch.LADDER_COMPLETE
    else:
        if cached and was_incomplete is not None:
            log_cb(
                "auto batch: re-probing -- the last ladder for this workload "
                f"stopped early ({was_incomplete}), so its highest measured "
                "batch is a floor, not a ceiling. Cached records alone would "
                "have capped this run permanently."
            )
        run_dir_path.mkdir(parents=True, exist_ok=True)
        spec_path = run_dir_path / "spec.json"
        # The probe children need a spec to load. This is the REQUESTED spec;
        # the resolved rewrite happens exactly once, later.
        _write_json(spec_path, spec.to_dict())

        def step(batch: int) -> dict[str, Any]:
            return _run_probe_candidate(
                spec,
                run_dir_path,
                batch,
                spec_path=spec_path,
                cuda_device=cuda_device,
                env_name=env_name,
                params=params,
                models_root=models_root,
                log_cb=log_cb,
                should_cancel=should_cancel,
            )

        try:
            measured = autobatch.run_probe(
                spec,
                run_dir_path,
                step_fn=step,
                identity=fingerprint.identity,
                should_cancel=should_cancel,
            )
        except autobatch.ProbeFailedError as exc:
            # Fail-closed: the ONLY case where a completed probe caches
            # nothing at all.
            raise _BatchResolutionRefused(str(exc)) from exc
        terminated_by = measured.terminated_by
        # Validate, then merge, then select -- in that order and on purpose.
        records = autobatch.validate_probe_records(measured, fingerprint.identity)
        if not records:
            raise _BatchResolutionRefused(
                "The SAM3 memory probe produced no record that survived "
                "validation; refusing to size this run from a broken "
                "measurement."
            )
        # A measurement is a fact about this hardware and workload; a
        # selection also depends on how much VRAM happens to be free right
        # now. Cache the fact even if the selection below refuses, so the
        # next attempt on a quieter GPU reuses it rather than re-probing.
        store.save(merge_records(stored, records))
        # A ladder cut short by a TRANSIENT host event has not proved that
        # the untried rungs are unreachable, so it must not be allowed to
        # cap every future run through the cache. Marked here and cleared on
        # any ladder that ended for an authoritative reason.
        _mark_incomplete_ladder(
            key,
            (
                terminated_by
                if terminated_by in autobatch.TRANSIENT_LADDER_TERMINATIONS
                else None
            ),
        )
        provenance = "measured"

    if should_cancel():
        raise autobatch.ProbeCanceled("cancelled before batch selection")

    live = preflight_module._probe_cuda_device(str(getattr(spec, "device", "auto")))
    free_bytes = int(
        live.free_bytes
        if live is not None and live.uuid == cuda_device.uuid
        else cuda_device.free_bytes
    )
    resolved, _selection_provenance = autobatch.resolve_batch(
        spec,
        records,
        usable_bytes=free_bytes,
        maximum=autobatch.MAX_AUTO_BATCH,
    )
    requirement = max(
        (record.accelerator_reserved_peak_bytes for record in records), default=0
    )
    if resolved <= 0:
        raise _BatchResolutionRefused(
            "SAM3 auto batch sizing refuses this run: the smallest measured "
            f"configuration needs {requirement / GiB:.1f} GiB reserved, but "
            f"only {free_bytes / GiB:.1f} GiB is free on the selected GPU "
            f"(usable at {autobatch.MEASURED_SAFETY_FRACTION:.0%} safety). "
            "Free the device and retry; the measurement has been cached."
        )
    # Report the requirement that was ACTUALLY cleared. When `resolved` is
    # itself an observed rung, that is a measurement; when it falls between
    # rungs (records at 1, 2, 4 can resolve to 3), the number is the fitted
    # envelope and must be labelled as extrapolated rather than passed off as
    # something someone measured.
    observed_peaks = {
        record.settings.batch_size: record.accelerator_reserved_peak_bytes
        for record in records
    }
    if resolved in observed_peaks:
        selected_peak = observed_peaks[resolved]
        requirement_basis = "measured"
    else:
        base_bytes, slope_bytes = fit_batch_curve(records)
        selected_peak = max(
            base_bytes + slope_bytes * resolved,
            max(
                (peak for batch, peak in observed_peaks.items() if batch <= resolved),
                default=0,
            ),
        )
        requirement_basis = "extrapolated"
    resolution = {
        "requested": int(params.batch),
        "resolved": int(resolved),
        "provenance": provenance,
        "requirement_basis": requirement_basis,
        "ladder_terminated_by": terminated_by,
        "fingerprint": key,
        "degraded_reasons": list(fingerprint.degraded_reasons),
        "measured_reserved_bytes": int(selected_peak),
        "free_bytes": free_bytes,
        "resolved_at_unix_ns": time.time_ns(),
    }
    log_cb(
        f"auto batch: {resolved} ({requirement_basis} "
        f"{selected_peak / GiB:.1f} GiB reserved at batch {resolved}, "
        f"{selected_peak / max(1, free_bytes):.0%} of "
        f"{free_bytes / GiB:.1f} GiB free; provenance={provenance}, "
        f"ladder={terminated_by})"
    )
    return int(resolved), resolution


def _incomplete_ladders_path() -> Path:
    """Sibling of the profile store recording ladders that ended early.

    A separate small file rather than a new `MemoryMeasurement` field:
    `MemoryMeasurement` is a shared, closed schema, and this is a property of
    a PROBE ATTEMPT, not of any single measurement.
    """

    store = _store_path()
    return store.with_name(store.name + ".incomplete.json")


def _incomplete_ladders() -> dict[str, str]:
    try:
        loaded = json.loads(_incomplete_ladders_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(k): str(v) for k, v in loaded.items() if isinstance(v, str)}


def _mark_incomplete_ladder(key: str, reason: Optional[str]) -> None:
    """Record (or clear) that this workload's ladder ended for a transient reason."""

    marks = _incomplete_ladders()
    if reason is None:
        if marks.pop(key, None) is None:
            return
    else:
        marks[key] = reason
    path = _incomplete_ladders_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, marks)
    except OSError:
        # Best-effort bookkeeping: losing this file costs a re-probe, never
        # correctness.
        pass


def _fingerprint_key(identity: Any) -> str:
    """A short, stable, log-safe digest of a `ProfileIdentity`."""

    payload = json.dumps(asdict(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
    run_dir_path = Path(run_dir).expanduser().resolve()
    try:
        auto_import = bool(getattr(spec.publish_policy, "auto_import", True))
        models_root = get_models_root() if auto_import else None
    except (OSError, ValueError) as exc:
        return _result(
            success=False,
            message=f"SAM3 resource preflight could not inspect the run: {exc}",
            failure_kind=ExitKind.HOST_ADMISSION_REFUSAL.value,
        )

    # --- Batch resolution comes FIRST -------------------------------------
    # The original ordering bug: preflight ran, spec.json was written, and
    # immutable containment limits were built BEFORE the child started, while
    # `build_resource_request` quietly turned `-1` into 1. A child that then
    # chose batch 2-8 ran under limits, a host estimate, and an accelerator
    # estimate all admitted for batch 1. Nothing below may observe a
    # non-positive batch.
    requested_params = getattr(spec, "sam3_params", None)
    requested_batch = (
        int(getattr(requested_params, "batch", 1))
        if requested_params is not None
        else 1
    )
    resolved_spec = spec
    batch_resolution: dict[str, Any] = {
        "requested": requested_batch,
        "resolved": requested_batch,
        "provenance": "explicit",
        "fingerprint": "",
        "degraded_reasons": [],
        "measured_reserved_bytes": 0,
        "free_bytes": 0,
        "resolved_at_unix_ns": time.time_ns(),
    }
    if requested_params is not None and requested_batch <= 0:
        try:
            resolved_batch, batch_resolution = _resolve_measured_batch(
                spec,
                run_dir_path,
                requested_params,
                env_name=resolve_sam3_env(requested_params.env_name),
                models_root=models_root,
                log_cb=log_cb,
                should_cancel=should_cancel,
            )
        except autobatch.ProbeCanceled:
            # Merge nothing, write no batch_resolution, launch nothing.
            return _result(
                success=False,
                canceled=True,
                failure_kind=ExitKind.CANCELED.value,
            )
        except (_BatchResolutionRefused, OSError, ValueError) as exc:
            return _result(
                success=False,
                message=str(exc),
                failure_kind=ExitKind.HOST_ADMISSION_REFUSAL.value,
            )
        resolved_spec = preflight_module.spec_with_batch(spec, resolved_batch)
        if should_cancel():
            return _result(
                success=False,
                canceled=True,
                failure_kind=ExitKind.CANCELED.value,
            )

    spec = resolved_spec

    try:
        initial = preflight_module.assess_preflight(
            spec,
            run_dir=run_dir_path,
            models_root=models_root,
        )
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

    run_dir_path.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir_path / "spec.json"
    diagnostics_path = run_dir_path / "resource_preflight.json"
    # The ONE mutation inside `sam3_params` is the resolved positive batch.
    # Provenance is a top-level, output-only block: `_SidecarSpec` does
    # `Sam3LoraParams(**sam3_data)`, so any extra key inside `sam3_params`
    # would raise `TypeError` in the child, while an unknown TOP-LEVEL key is
    # inert (every field is read by `.get`). Written unconditionally on every
    # run, after resolution, overwriting anything present -- it is not
    # forgeable input.
    spec_payload = spec.to_dict()
    spec_payload["batch_resolution"] = batch_resolution
    _write_json(spec_path, spec_payload)
    _write_json(run_dir_path / "batch_resolution.json", batch_resolution)

    artifact_path = run_dir_path / "adapters.pt"
    remove_artifact(artifact_path, remove_staging=True)
    params = spec.sam3_params
    env_name = resolve_sam3_env(params.env_name)
    command = _sam3_command(
        env_name,
        ["--spec", str(spec_path), "--run-dir", str(run_dir_path)],
    )
    child_environment = _child_environment(initial_cuda_device)
    limits = _memory_limits(initial)
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
        "batch_resolution": batch_resolution,
        "containment": _containment_diagnostic(plan),
    }
    _write_json(diagnostics_path, diagnostic)

    def final_live_check() -> None:
        live = preflight_module.assess_preflight(
            spec,
            run_dir=run_dir_path,
            models_root=models_root,
        )
        diagnostic["live"] = live.to_dict()
        _write_json(diagnostics_path, diagnostic)
        if live.cuda_device is None:
            raise _AdmissionRefused("CUDA disappeared before SAM3 launch")
        if live.cuda_device.uuid != initial_cuda_device.uuid:
            raise _AdmissionRefused(
                "The selected physical CUDA device changed between admission "
                "and launch; refusing to use a different GPU."
            )
        immutable_hard_limit = initial.containment_hard_host_bytes
        if immutable_hard_limit > live.budget.usable_host_bytes:
            raise _AdmissionRefused(
                "Available host memory changed before launch: the immutable "
                "containment limit would expose the reserved host-memory floor."
            )
        if live.containment_hard_host_bytes > immutable_hard_limit:
            raise _AdmissionRefused(
                "The SAM3 workload profile grew after initial admission and "
                "no longer fits the immutable containment limit."
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
    except WorkloadStillOwnedError as owned_error:
        owned_error.recovery_cleanup = lambda: remove_artifact(
            artifact_path, remove_staging=True
        )
        raise
    except (
        ResourceBusyError,
        _AdmissionRefused,
        FileNotFoundError,
        RuntimeError,
    ) as exc:
        # These paths either refused before Popen or completed constructor
        # cleanup and proved quiescence. Any exact private-run staging file is
        # now stale and safe to remove.
        remove_artifact(artifact_path, remove_staging=True)
        return _result(
            success=False,
            message=f"SAM3 sidecar launch refused: {exc}",
            failure_kind=ExitKind.HOST_ADMISSION_REFUSAL.value,
            command=launch.command,
            resource_preflight=str(diagnostics_path),
            containment=_containment_diagnostic(plan),
        )

    try:
        _pump_child_output(sidecar, plan, params, log_cb, progress_cb, should_cancel)

        def validate_artifact(result: SupervisedResult) -> None:
            validation_error = validate_completion(artifact_path)
            if result.classified_exit.kind is ExitKind.SUCCESS and validation_error:
                raise _ArtifactInvalid(
                    "SAM3 training subprocess exited successfully but did not "
                    f"produce a validated artifact: {validation_error}; refusing "
                    "to report success for a run that trained nothing."
                )

        supervised = sidecar.wait(post_exit_check=validate_artifact)
    except _Canceled:
        remove_artifact(artifact_path, remove_staging=True)
        return _result(
            success=False,
            canceled=True,
            failure_kind=ExitKind.CANCELED.value,
            command=launch.command,
            resource_preflight=str(diagnostics_path),
            containment=_containment_diagnostic(plan),
        )
    except _ArtifactInvalid as exc:
        remove_artifact(artifact_path, remove_staging=True)
        return _result(
            success=False,
            message=str(exc),
            failure_kind=ExitKind.ORDINARY_FAILURE.value,
            exit_code=0,
            command=launch.command,
            resource_preflight=str(diagnostics_path),
            containment=_containment_diagnostic(plan),
        )
    except WorkloadStillOwnedError as owned_error:
        owned_error.recovery_cleanup = lambda: remove_artifact(
            artifact_path, remove_staging=True
        )
        raise
    except BaseException:
        try:
            sidecar.cancel(plan.terminate_grace_seconds)
        except WorkloadStillOwnedError as owned_error:
            owned_error.recovery_cleanup = lambda: remove_artifact(
                artifact_path, remove_staging=True
            )
            raise
        remove_artifact(artifact_path, remove_staging=True)
        raise

    classified = supervised.classified_exit
    if classified.kind is not ExitKind.SUCCESS:
        # wait() proved the complete workload quiescent. Remove failed-run
        # staging before any diagnostics I/O can fail and bypass cleanup.
        remove_artifact(artifact_path, remove_staging=True)
    diagnostic["containment"] = _containment_diagnostic(plan, supervised)
    _write_json(diagnostics_path, diagnostic)
    if classified.kind is not ExitKind.SUCCESS:
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
