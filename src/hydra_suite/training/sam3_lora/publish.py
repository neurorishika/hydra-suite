"""Parent-side launcher and registry commit for SAM3 publication.

This module deliberately has no Torch import. Loading, merging, hashing,
serialization, and consumer validation run in ``publish_cli`` behind the shared
Set 2 containment boundary; the parent performs only bounded metadata I/O and
the final atomic registry transaction.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from hydra_suite.runtime.process_supervisor import (
    ContainmentPlan,
    ExitKind,
    SupervisedResult,
    SupervisedSidecar,
    WorkloadStillOwnedError,
)
from hydra_suite.runtime.resource_budget import (
    AcceleratorKind,
    PhaseEstimate,
    ResourceBudget,
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
from hydra_suite.training.contracts import sam3_prompt_text_error

from .env import resolve_sam3_env, sam3_env_command, sam3_env_environ

OUTPUT_MAX_LINES = 512
OUTPUT_MAX_CHARS = 256 * 1024
MAX_PROCESSES = 64
MAX_RESULT_BYTES = 64 * 1024
MAX_SIDECAR_BYTES = 4 * 1024 * 1024
MAX_REGISTRY_BYTES = 16 * 1024 * 1024
MAX_REGISTRY_ENTRIES = 10_000
PUBLISH_RUNTIME_BYTES = 2 * 1024**3
PUBLISH_DISK_MARGIN_BYTES = 64 * 1024**2
PUBLISH_ESTIMATOR_VERSION = "sam3-publish-inplace-v1"
GiB = 1024**3

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_PARAM_FIELDS = (
    "prompt",
    "rank",
    "alpha",
    "dropout",
    "label_quality_acknowledged",
    "adapt_vision_encoder",
    "adapt_text_encoder",
    "adapt_geometry_encoder",
    "adapt_detr_encoder",
    "adapt_detr_decoder",
    "adapt_mask_decoder",
)


class Sam3PublishError(RuntimeError):
    """A classified, fully reaped publish-sidecar failure."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: str,
        canceled: bool = False,
        containment: Optional[dict[str, Any]] = None,
    ) -> None:
        self.failure_kind = failure_kind
        self.canceled = canceled
        self.containment = containment or {}
        super().__init__(message)


@dataclass(frozen=True)
class _PublishDecision:
    admitted: bool
    request: ResourceRequest
    budget: ResourceBudget
    soft_host_bytes: int
    hard_host_bytes: int
    disk_required_bytes: int
    disk_free_bytes: int
    base_identity: tuple[int, int]
    adapters_identity: tuple[int, int]
    refusals: tuple[str, ...]


def stripped_keys(state_dict: dict[str, Any]) -> list[str]:
    """Reproduce ultralytics' substring filter and replacement exactly."""

    return sorted(
        key.replace("detector.", "") for key in state_dict if "detector" in key
    )


def _file_identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _free_disk_bytes(path: Path) -> int:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return int(shutil.disk_usage(target).free)


def _resource_policy(params: Any) -> ResourcePolicy:
    return ResourcePolicy(
        reserve_host_bytes=max(
            8 * GiB,
            int(float(getattr(params, "host_reserve_gb", 8.0)) * GiB),
        ),
        reserve_host_fraction=max(
            0.15, float(getattr(params, "host_reserve_fraction", 0.15))
        ),
        accelerator_safety_fraction=min(
            0.90, float(getattr(params, "cuda_safety_fraction", 0.85))
        ),
        warning_fraction=0.80,
    )


def _containment_host_limits(params: Any, host_peak_bytes: int) -> tuple[int, int]:
    soft = max(1, math.ceil(host_peak_bytes * 1.10))
    hard = max(
        soft,
        math.ceil(
            host_peak_bytes
            * float(getattr(params, "host_limit_headroom_fraction", 1.25))
        ),
    )
    return soft, hard


def _assess_publish(
    *,
    base_checkpoint: Path,
    adapters_path: Path,
    models_root: Path,
    params: Any,
) -> _PublishDecision:
    """Build a host-only publish budget from current file and RAM metadata."""

    base_identity = _file_identity(base_checkpoint)
    adapters_identity = _file_identity(adapters_path)
    base_bytes = base_identity[0]
    adapters_bytes = adapters_identity[0]
    # The largest possible active delta is bounded by the whole checkpoint.
    # In real SAM3 it is much smaller, but this upper bound remains honest
    # without importing Torch merely to inspect tensor metadata in the parent.
    active_tensor_bytes = base_bytes
    host_steady = base_bytes + adapters_bytes + PUBLISH_RUNTIME_BYTES
    host_peak = host_steady + active_tensor_bytes + adapters_bytes
    disk_required = base_bytes + PUBLISH_DISK_MARGIN_BYTES
    request = ResourceRequest(
        job_name="SAM3 checkpoint publish",
        phases=(
            PhaseEstimate(
                "publish",
                host_steady_bytes=host_steady,
                host_peak_bytes=host_peak,
                disk_transient_bytes=disk_required,
                dominant_allocations=(
                    ("base checkpoint", base_bytes),
                    ("largest possible active tensor", active_tensor_bytes),
                    ("LoRA adapter and reload", 2 * adapters_bytes),
                    ("Torch/serialization runtime", PUBLISH_RUNTIME_BYTES),
                ),
            ),
        ),
        limits=WorkLimits(),
        estimator_version=PUBLISH_ESTIMATOR_VERSION,
    )
    policy = _resource_policy(params)
    budget = evaluate_resource_request(request, probe_resources(), policy)
    soft, hard = _containment_host_limits(params, budget.host_peak_bytes)
    disk_free = _free_disk_bytes(models_root)
    refusals = list(budget.refusals)
    if hard > budget.usable_host_bytes:
        refusals.append(
            "SAM3 publish containment headroom would expose the reserved "
            "host-memory floor"
        )
    if disk_free < disk_required:
        refusals.append(
            f"SAM3 publish requires {disk_required} transient bytes but only "
            f"{disk_free} bytes are free"
        )
    return _PublishDecision(
        admitted=not refusals,
        request=request,
        budget=budget,
        soft_host_bytes=soft,
        hard_host_bytes=hard,
        disk_required_bytes=disk_required,
        disk_free_bytes=disk_free,
        base_identity=base_identity,
        adapters_identity=adapters_identity,
        refusals=tuple(refusals),
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_bounded(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise RuntimeError(f"JSON payload {path} exceeds its safe size bound")
    with path.open("rb") as stream:
        raw = stream.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise RuntimeError(f"JSON payload {path} grew beyond its safe size bound")
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON payload {path} is invalid") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"JSON payload {path} is not an object")
    return parsed


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_paths(models_root: Path, run_id: str) -> tuple[Path, Path]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("SAM3 publish run_id is not a safe artifact name")
    artifact = models_root / "sam3_finetuned" / f"{run_id}.pt"
    return artifact, artifact.with_name(artifact.name + ".sam3_meta.json")


def _cleanup_attempt(
    *,
    artifact_path: Path,
    sidecar_path: Path,
    control_dir: Optional[Path],
    attempt_id: str,
) -> None:
    """Remove only files proven to belong to this publish attempt."""

    cleanup_error: Optional[Exception] = None

    def attempt(action: Callable[[], None]) -> None:
        nonlocal cleanup_error
        try:
            action()
        except Exception as exc:  # preserve the first cleanup failure
            if cleanup_error is None:
                cleanup_error = exc

    owned_final_pair = False
    if sidecar_path.is_file():
        try:
            metadata = _read_json_bounded(sidecar_path, maximum_bytes=MAX_SIDECAR_BYTES)
            owned_final_pair = metadata.get("publish_attempt_id") == attempt_id
        except (OSError, RuntimeError):
            # A path that cannot prove ownership must never be deleted. It may
            # have appeared after our initial no-overwrite check.
            owned_final_pair = False
    if owned_final_pair:
        attempt(lambda: artifact_path.unlink(missing_ok=True))
        attempt(lambda: sidecar_path.unlink(missing_ok=True))
    if artifact_path.parent.exists():
        staged_artifact = artifact_path.with_name(
            f".{artifact_path.name}.{attempt_id}.tmp"
        )
        staged_sidecar = sidecar_path.with_name(
            f".{sidecar_path.name}.{attempt_id}.tmp"
        )
        attempt(lambda: staged_artifact.unlink(missing_ok=True))
        attempt(lambda: staged_sidecar.unlink(missing_ok=True))
        attempt(lambda: _fsync_directory(artifact_path.parent))
    if control_dir is not None:
        attempt(lambda: shutil.rmtree(control_dir))
    if cleanup_error is not None:
        raise cleanup_error


def _containment_diagnostic(
    plan: ContainmentPlan, result: Optional[SupervisedResult] = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": plan.launch.backend.value,
        "soft_host_bytes": plan.launch.limits.soft_host_bytes,
        "hard_host_bytes": plan.launch.limits.hard_host_bytes,
        "minimum_system_available_bytes": plan.minimum_system_available_bytes,
        "resource_keys": list(plan.expected_resource_keys),
        "limitations": list(plan.launch.limitations),
    }
    if result is not None:
        payload.update(
            {
                "peak_observed_tree_rss_bytes": result.peak_tree_rss_bytes,
                "minimum_observed_system_available_bytes": (
                    result.minimum_system_available_bytes
                ),
                "dropped_output_lines": result.dropped_output_lines,
                "output_error": result.output_error,
            }
        )
    return payload


def _request_payload(
    *,
    run_id: str,
    adapters_path: Path,
    base_checkpoint: Path,
    build_manifest: dict[str, Any],
    params: Any,
    source_fingerprint: str,
    models_root: Path,
    attempt_id: str,
) -> dict[str, Any]:
    if not isinstance(source_fingerprint, str) or len(source_fingerprint) > 256:
        raise ValueError("SAM3 source fingerprint is invalid")
    prompt_error = sam3_prompt_text_error(getattr(params, "prompt", None))
    if prompt_error is not None:
        raise ValueError(f"SAM3 publish prompt {prompt_error}")

    geometry: dict[str, int | float | None] = {}
    for field in ("tile_px", "reference_body_px", "object_tile_fraction"):
        value = build_manifest.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ValueError(f"SAM3 publish geometry {field!r} must be numeric")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"SAM3 publish geometry {field!r} must be finite")
        geometry[field] = value
    return {
        "run_id": run_id,
        "adapters_path": str(adapters_path),
        "base_checkpoint": str(base_checkpoint),
        "build_manifest": geometry,
        "params": {field: getattr(params, field) for field in _PARAM_FIELDS},
        "source_fingerprint": source_fingerprint,
        "models_root": str(models_root),
        "publish_attempt_id": attempt_id,
    }


def _validate_result(
    result_path: Path,
    artifact_path: Path,
    sidecar_path: Path,
    *,
    attempt_id: str,
) -> dict[str, Any]:
    result = _read_json_bounded(result_path, maximum_bytes=MAX_RESULT_BYTES)
    if result != {
        "artifact_path": str(artifact_path),
        "sidecar_path": str(sidecar_path),
    }:
        raise RuntimeError("SAM3 publish child returned an unexpected artifact receipt")
    if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
        raise RuntimeError("SAM3 publish child did not produce a non-empty checkpoint")
    if not sidecar_path.is_file() or sidecar_path.stat().st_size <= 0:
        raise RuntimeError("SAM3 publish child did not produce its guard sidecar")
    metadata = _read_json_bounded(sidecar_path, maximum_bytes=MAX_SIDECAR_BYTES)
    if metadata.get("publish_attempt_id") != attempt_id:
        raise RuntimeError(
            "SAM3 publish child returned an artifact from another attempt"
        )
    return result


def publish_sam3_model(
    *,
    run_id: str,
    adapters_path: str | Path,
    base_checkpoint: str | Path,
    build_manifest: dict[str, Any],
    params: Any,
    source_fingerprint: str,
    models_root: str | Path | None = None,
    registry_path: str | Path | None = None,
    log_cb: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> tuple[str, str]:
    """Publish under containment, then atomically add the registry entry."""

    if models_root is None:
        from ..model_publish import get_models_root

        models_root = get_models_root()
    models_root = Path(models_root).expanduser().resolve()
    adapters_path = Path(adapters_path).expanduser().resolve()
    base_checkpoint = Path(base_checkpoint).expanduser().resolve()
    registry_path = (
        Path(registry_path).expanduser().resolve()
        if registry_path is not None
        else models_root / "model_registry.json"
    )
    artifact_path, sidecar_path = _artifact_paths(models_root, run_id)
    if artifact_path.exists() or sidecar_path.exists():
        raise FileExistsError(
            f"SAM3 publish target already exists for run {run_id!r}; refusing "
            "to overwrite a previously published artifact"
        )
    log_cb = log_cb or (lambda _message: None)
    should_cancel = should_cancel or (lambda: False)
    initial = _assess_publish(
        base_checkpoint=base_checkpoint,
        adapters_path=adapters_path,
        models_root=models_root,
        params=params,
    )
    if not initial.admitted:
        raise Sam3PublishError(
            "; ".join(initial.refusals),
            failure_kind=ExitKind.HOST_ADMISSION_REFUSAL.value,
        )

    attempt_id = uuid.uuid4().hex
    request_payload = _request_payload(
        run_id=run_id,
        adapters_path=adapters_path,
        base_checkpoint=base_checkpoint,
        build_manifest=build_manifest,
        params=params,
        source_fingerprint=source_fingerprint,
        models_root=models_root,
        attempt_id=attempt_id,
    )
    control_dir = Path(
        tempfile.mkdtemp(prefix=f".sam3_publish_{run_id}.", dir=adapters_path.parent)
    )
    request_path = control_dir / "request.json"
    result_path = control_dir / "result.json"
    try:
        _write_json_atomic(request_path, request_payload)
        env_name = resolve_sam3_env(getattr(params, "env_name", ""))
        command = sam3_env_command(
            env_name,
            [
                "hydra_suite.training.sam3_lora.publish_cli",
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ],
        )
        launch = build_limited_launch(
            command,
            ProcessMemoryLimits(
                soft_host_bytes=initial.soft_host_bytes,
                hard_host_bytes=initial.hard_host_bytes,
                max_processes=MAX_PROCESSES,
            ),
            environment={
                **os.environ,
                **sam3_env_environ(),
                "CUDA_VISIBLE_DEVICES": "",
            },
            accelerator_kind=AcceleratorKind.CPU,
        )
        plan = ContainmentPlan(
            launch=launch,
            job_name="SAM3 checkpoint publish",
            minimum_system_available_bytes=initial.budget.reserved_host_bytes,
            poll_interval_seconds=float(getattr(params, "watchdog_poll_seconds", 1.0)),
        )
    except BaseException:
        shutil.rmtree(control_dir, ignore_errors=True)
        raise

    def final_live_check() -> None:
        live = _assess_publish(
            base_checkpoint=base_checkpoint,
            adapters_path=adapters_path,
            models_root=models_root,
            params=params,
        )
        if live.base_identity != initial.base_identity:
            raise RuntimeError("SAM3 base checkpoint changed after admission")
        if live.adapters_identity != initial.adapters_identity:
            raise RuntimeError("SAM3 adapters changed after admission")
        if initial.hard_host_bytes > live.budget.usable_host_bytes:
            raise RuntimeError(
                "available host memory changed before publish; the immutable "
                "cap would expose the reserved floor"
            )
        if live.hard_host_bytes > initial.hard_host_bytes or not live.admitted:
            raise RuntimeError("SAM3 publish no longer fits its immutable resource cap")

    try:
        sidecar = SupervisedSidecar(
            plan,
            prelaunch_check=final_live_check,
            output_max_lines=OUTPUT_MAX_LINES,
            output_max_chars=OUTPUT_MAX_CHARS,
        )
    except WorkloadStillOwnedError as owned_error:
        owned_error.recovery_cleanup = lambda: _cleanup_attempt(
            artifact_path=artifact_path,
            sidecar_path=sidecar_path,
            control_dir=control_dir,
            attempt_id=attempt_id,
        )
        raise
    except ResourceBusyError as exc:
        # No child was created and the conflicting lease owner may be
        # publishing the same run. Never delete paths that could belong to it.
        shutil.rmtree(control_dir, ignore_errors=True)
        raise Sam3PublishError(
            f"SAM3 publish sidecar launch refused: {bounded_terminal_text(exc)}",
            failure_kind=ExitKind.HOST_ADMISSION_REFUSAL.value,
            containment=_containment_diagnostic(plan),
        ) from exc
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        _cleanup_attempt(
            artifact_path=artifact_path,
            sidecar_path=sidecar_path,
            control_dir=control_dir,
            attempt_id=attempt_id,
        )
        raise Sam3PublishError(
            f"SAM3 publish sidecar launch refused: {bounded_terminal_text(exc)}",
            failure_kind=ExitKind.HOST_ADMISSION_REFUSAL.value,
            containment=_containment_diagnostic(plan),
        ) from exc

    try:
        while True:
            lines, eof, output_error = sidecar.output.drain(
                float(getattr(params, "watchdog_poll_seconds", 1.0))
            )
            for line in lines:
                text = line.rstrip("\r\n")
                if text:
                    log_cb(text)
            if output_error is not None:
                raise output_error
            if sidecar.process.poll() is not None:
                break
            if should_cancel():
                sidecar.cancel(plan.terminate_grace_seconds)
                _cleanup_attempt(
                    artifact_path=artifact_path,
                    sidecar_path=sidecar_path,
                    control_dir=control_dir,
                    attempt_id=attempt_id,
                )
                raise Sam3PublishError(
                    "SAM3 publish canceled",
                    failure_kind=ExitKind.CANCELED.value,
                    canceled=True,
                    containment=_containment_diagnostic(plan),
                )
            if eof:
                time.sleep(plan.poll_interval_seconds)

        supervised = sidecar.wait(
            post_exit_check=lambda result: (
                _validate_result(
                    result_path,
                    artifact_path,
                    sidecar_path,
                    attempt_id=attempt_id,
                )
                if result.classified_exit.kind is ExitKind.SUCCESS
                else None
            )
        )
    except WorkloadStillOwnedError as owned_error:
        owned_error.recovery_cleanup = lambda: _cleanup_attempt(
            artifact_path=artifact_path,
            sidecar_path=sidecar_path,
            control_dir=control_dir,
            attempt_id=attempt_id,
        )
        raise
    except Sam3PublishError:
        raise
    except BaseException:
        try:
            sidecar.cancel(plan.terminate_grace_seconds)
        except WorkloadStillOwnedError as owned_error:
            owned_error.recovery_cleanup = lambda: _cleanup_attempt(
                artifact_path=artifact_path,
                sidecar_path=sidecar_path,
                control_dir=control_dir,
                attempt_id=attempt_id,
            )
            raise
        _cleanup_attempt(
            artifact_path=artifact_path,
            sidecar_path=sidecar_path,
            control_dir=control_dir,
            attempt_id=attempt_id,
        )
        raise

    containment = _containment_diagnostic(plan, supervised)
    if supervised.classified_exit.kind is not ExitKind.SUCCESS:
        _cleanup_attempt(
            artifact_path=artifact_path,
            sidecar_path=sidecar_path,
            control_dir=control_dir,
            attempt_id=attempt_id,
        )
        tail = "".join(supervised.output_tail).strip() or "(no output)"
        raise Sam3PublishError(
            f"{supervised.classified_exit.message}. Child output tail:\n{tail}",
            failure_kind=supervised.classified_exit.kind.value,
            canceled=supervised.classified_exit.kind is ExitKind.CANCELED,
            containment=containment,
        )
    if should_cancel():
        _cleanup_attempt(
            artifact_path=artifact_path,
            sidecar_path=sidecar_path,
            control_dir=control_dir,
            attempt_id=attempt_id,
        )
        raise Sam3PublishError(
            "SAM3 publish canceled before registry commit",
            failure_kind=ExitKind.CANCELED.value,
            canceled=True,
            containment=containment,
        )

    try:
        _register(
            registry_path=registry_path,
            artifact_path=artifact_path,
            sidecar_path=sidecar_path,
            run_id=run_id,
            prompt=str(getattr(params, "prompt", "")),
            source_fingerprint=source_fingerprint,
        )
    except BaseException:
        _cleanup_attempt(
            artifact_path=artifact_path,
            sidecar_path=sidecar_path,
            control_dir=control_dir,
            attempt_id=attempt_id,
        )
        raise
    shutil.rmtree(control_dir, ignore_errors=True)
    return f"sam3_finetuned/{artifact_path.name}", str(artifact_path)


@contextmanager
def _registry_lock(registry_path: Path) -> Iterator[None]:
    lock_path = registry_path.with_name(registry_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        if os.name == "posix":
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


_atomic_replace_registry = os.replace


def _load_registry_bounded(registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        return {"schema_version": 2, "entries": {}}
    data = _read_json_bounded(registry_path, maximum_bytes=MAX_REGISTRY_BYTES)
    entries = data.get("entries")
    if data.get("schema_version") != 2 or not isinstance(entries, dict):
        raise RuntimeError("Published-model registry has an unsupported shape")
    if len(entries) > MAX_REGISTRY_ENTRIES:
        raise RuntimeError("Published-model registry exceeds its safe entry cap")
    return data


def _save_registry_atomic(registry_path: Path, data: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    encoded_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=registry_path.parent,
            prefix=f".{registry_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for chunk in json.JSONEncoder(indent=2).iterencode(data):
                encoded_bytes += len(chunk.encode("utf-8"))
                if encoded_bytes > MAX_REGISTRY_BYTES:
                    raise RuntimeError(
                        "Published-model registry exceeds its safe size cap"
                    )
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_replace_registry(temporary, registry_path)
        _fsync_directory(registry_path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _register(
    *,
    registry_path: Path,
    artifact_path: Path,
    sidecar_path: Path,
    run_id: str,
    prompt: str,
    source_fingerprint: str,
) -> None:
    """Atomically register only an already validated, visible artifact pair."""

    if not artifact_path.is_file() or not sidecar_path.is_file():
        raise RuntimeError("refusing to register an incomplete SAM3 artifact pair")
    key = f"sam3_finetuned/{artifact_path.name}"
    with _registry_lock(registry_path):
        data = _load_registry_bounded(registry_path)
        entries = data["entries"]
        if key in entries:
            raise FileExistsError(f"SAM3 registry key {key!r} already exists")
        if len(entries) >= MAX_REGISTRY_ENTRIES:
            raise RuntimeError("Published-model registry exceeds its safe entry cap")
        entries[key] = {
            "task_family": "semantic",
            "usage_role": "semantic_sam3",
            "stored_filename": artifact_path.name,
            "stored_path": str(artifact_path),
            "sidecar_path": str(sidecar_path),
            "trained_from_run_id": run_id,
            "prompt": prompt,
            "dataset_fingerprint": source_fingerprint,
            "added_at": datetime.now().isoformat(timespec="seconds"),
        }
        _save_registry_atomic(registry_path, data)
