"""Process-isolated DetectKit dataset preparation.

The GUI/CLI process only serializes a bounded request and supervises a CPU
child.  All source discovery, validation, decoding, splitting, and building
happens after the child memory boundary is active.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import psutil

from hydra_suite.runtime.process_supervisor import (
    ContainmentPlan,
    ExitKind,
    SupervisedSidecar,
    WorkloadStillOwnedError,
)
from hydra_suite.runtime.resource_budget import AcceleratorKind, GiB
from hydra_suite.runtime.resource_limits import (
    ProcessMemoryLimits,
    build_limited_launch,
)
from hydra_suite.training.contracts import (
    Sam3LoraParams,
    SourceDataset,
    SplitConfig,
    TrainingRole,
    ValidationIssue,
    ValidationReport,
    sam3_prompt_pool_error,
)
from hydra_suite.training.dataset_io import DatasetLimitError, read_bounded_text

from .training import (
    DatasetPreparationCancelled,
    DatasetPreparationRequest,
    DatasetPreparationResult,
)

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
MAX_SOURCES = 128
MAX_CLASSES = 4096
MAX_ROLES = 32
MAX_SOURCE_FILES = 1_000_000
MAX_SOURCE_BYTES = 2 * 1024**4
OUTPUT_MAX_LINES = 512
OUTPUT_MAX_CHARS = 256 * 1024


class DatasetPreparationSidecarError(RuntimeError):
    """A classified preparation-child failure with bounded diagnostics."""

    def __init__(self, failure_kind: ExitKind, message: str) -> None:
        self.failure_kind = failure_kind
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class DatasetPreparationBudget:
    """Immutable host/disk budget for one preparation child."""

    soft_host_bytes: int
    hard_host_bytes: int
    minimum_system_available_bytes: int
    disk_required_bytes: int
    disk_available_bytes: int
    source_files: int
    source_bytes: int
    source_fingerprint: str

    def __post_init__(self) -> None:
        if self.soft_host_bytes <= 0 or self.hard_host_bytes <= 0:
            raise ValueError("dataset preparation host limits must be positive")
        if self.soft_host_bytes > self.hard_host_bytes:
            raise ValueError("soft host limit cannot exceed hard host limit")
        if self.disk_required_bytes > self.disk_available_bytes:
            raise DatasetLimitError(
                "Dataset preparation needs approximately "
                f"{self.disk_required_bytes / GiB:.1f} GiB transient disk space, "
                f"but only {self.disk_available_bytes / GiB:.1f} GiB is available."
            )


def _bounded_request_payload(request: DatasetPreparationRequest) -> dict[str, Any]:
    if len(request.sources) > MAX_SOURCES:
        raise DatasetLimitError(f"Preparation request exceeds {MAX_SOURCES} sources")
    if len(request.class_names) > MAX_CLASSES:
        raise DatasetLimitError(f"Preparation request exceeds {MAX_CLASSES} classes")
    if len(request.roles) > MAX_ROLES:
        raise DatasetLimitError(f"Preparation request exceeds {MAX_ROLES} roles")
    if len(request.imgsz_by_role) > MAX_ROLES:
        raise DatasetLimitError(
            f"Preparation request exceeds {MAX_ROLES} role image sizes"
        )

    serialized_text_bytes = 0

    def bounded_text(value: object, label: str) -> str:
        nonlocal serialized_text_bytes
        if type(value) is not str:
            raise DatasetLimitError(f"{label} must be a string")
        if len(value) > 16 * 1024:
            raise DatasetLimitError(f"{label} exceeds 16384 characters")
        try:
            encoded_size = len(value.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise DatasetLimitError(f"{label} is not valid UTF-8 text") from exc
        serialized_text_bytes += encoded_size
        if serialized_text_bytes > MAX_REQUEST_BYTES:
            raise DatasetLimitError(
                f"Preparation request text exceeds {MAX_REQUEST_BYTES} bytes"
            )
        return value

    for index, source in enumerate(request.sources):
        bounded_text(source.path, f"sources[{index}].path")
        bounded_text(source.source_type, f"sources[{index}].source_type")
        bounded_text(source.name, f"sources[{index}].name")
        bounded_text(source.level, f"sources[{index}].level")
    for index, class_name in enumerate(request.class_names):
        bounded_text(class_name, f"class_names[{index}]")
    if request.sam3_params is not None:
        prompt_error = sam3_prompt_pool_error(
            request.sam3_params.prompt, request.sam3_params.negative_prompts
        )
        if prompt_error is not None:
            raise DatasetLimitError(
                f"Invalid SAM3 prompt configuration: {prompt_error}"
            )
    slice_settings = request.slice_settings
    for name in ("target_size_fractions", "target_sizes"):
        values = getattr(slice_settings, name, ())
        if len(values) > MAX_CLASSES:
            raise DatasetLimitError(
                f"slice_settings.{name} exceeds {MAX_CLASSES} entries"
            )
    slice_payload = (
        slice_settings.to_dict()
        if hasattr(slice_settings, "to_dict")
        else asdict(slice_settings)
    )
    payload = {
        "sources": [asdict(source) for source in request.sources],
        "roles": [role.value for role in request.roles],
        "class_names": list(request.class_names),
        "split": asdict(request.split),
        "seed": request.seed,
        "dedup": request.dedup,
        "crop_pad_ratio": request.crop_pad_ratio,
        "min_crop_size_px": request.min_crop_size_px,
        "enforce_square": request.enforce_square,
        "imgsz_by_role": [list(value) for value in request.imgsz_by_role],
        "slice_settings": slice_payload,
        "sam3_params": asdict(request.sam3_params) if request.sam3_params else None,
    }
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_REQUEST_BYTES:
        raise DatasetLimitError(
            f"Preparation request exceeds {MAX_REQUEST_BYTES} serialized bytes"
        )
    return payload


def decode_request(payload: object) -> DatasetPreparationRequest:
    """Reconstruct a child request after its file was byte-bounded."""

    if type(payload) is not dict:
        raise ValueError("dataset preparation request must be an object")
    from hydra_suite.detectkit.config.training import SliceTrainingConfig

    raw = payload
    sources_raw = raw.get("sources")
    roles_raw = raw.get("roles")
    classes_raw = raw.get("class_names")
    if type(sources_raw) is not list or len(sources_raw) > MAX_SOURCES:
        raise ValueError("invalid or excessive dataset source list")
    if type(roles_raw) is not list or len(roles_raw) > MAX_ROLES:
        raise ValueError("invalid or excessive dataset role list")
    if type(classes_raw) is not list or len(classes_raw) > MAX_CLASSES:
        raise ValueError("invalid or excessive dataset class list")
    sam3_raw = raw.get("sam3_params")
    return DatasetPreparationRequest(
        sources=tuple(SourceDataset(**entry) for entry in sources_raw),
        roles=tuple(TrainingRole(value) for value in roles_raw),
        class_names=tuple(str(value) for value in classes_raw),
        split=SplitConfig(**raw["split"]),
        seed=int(raw["seed"]),
        dedup=bool(raw["dedup"]),
        crop_pad_ratio=float(raw["crop_pad_ratio"]),
        min_crop_size_px=int(raw["min_crop_size_px"]),
        enforce_square=bool(raw["enforce_square"]),
        imgsz_by_role=tuple((str(k), int(v)) for k, v in raw["imgsz_by_role"]),
        slice_settings=SliceTrainingConfig.from_dict(raw.get("slice_settings")),
        sam3_params=Sam3LoraParams(**sam3_raw) if type(sam3_raw) is dict else None,
    )


def _scan_source_footprint(
    sources: tuple[SourceDataset, ...],
) -> tuple[int, int, str]:
    count = 0
    total = 0
    fingerprint = 0

    def visit(directory: Path, root: Path, depth: int):
        if depth > 32:
            raise DatasetLimitError(f"Dataset directory depth exceeds 32: {directory}")
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if len(os.fsencode(path.relative_to(root).as_posix())) > 16 * 1024:
                    raise DatasetLimitError(f"Dataset path exceeds 16384 bytes: {path}")
                if entry.is_dir(follow_symlinks=False):
                    yield from visit(path, root, depth + 1)
                elif entry.is_file(follow_symlinks=False):
                    yield path

    for source in sources:
        root = Path(source.path).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"Dataset source not found: {root}")
        for path in visit(root, root, 0):
            count += 1
            if count > MAX_SOURCE_FILES:
                raise DatasetLimitError(
                    f"Dataset sources exceed {MAX_SOURCE_FILES} files"
                )
            try:
                metadata = path.stat()
            except OSError as exc:
                raise RuntimeError(
                    f"Could not inspect source file {path}: {exc}"
                ) from exc
            total += metadata.st_size
            record = (
                f"{root}\0{path.relative_to(root).as_posix()}\0"
                f"{metadata.st_size}\0{metadata.st_mtime_ns}"
            ).encode("utf-8")
            fingerprint ^= int.from_bytes(
                hashlib.blake2b(record, digest_size=16).digest(), "big"
            )
            if total > MAX_SOURCE_BYTES:
                raise DatasetLimitError(
                    f"Dataset sources exceed {MAX_SOURCE_BYTES} bytes"
                )
    return count, total, f"{fingerprint:032x}"


def assess_preparation_budget(
    workspace: Path, request: DatasetPreparationRequest
) -> DatasetPreparationBudget:
    """Perform streaming file/disk admission without retaining source paths."""

    source_files, source_bytes, source_fingerprint = _scan_source_footprint(
        request.sources
    )
    memory = psutil.virtual_memory()
    reserve = max(4 * GiB, int(memory.total * 0.15))
    usable = int(memory.available) - reserve
    hard = min(8 * GiB, usable)
    if hard < GiB:
        raise DatasetLimitError(
            "Dataset preparation refused: less than 1 GiB remains after the "
            "host-memory safety reserve."
        )
    soft = max(512 * 1024**2, int(hard * 0.85))
    workspace.mkdir(parents=True, exist_ok=True)
    disk_available = shutil.disk_usage(workspace).free
    # Copy-based builders can coexist with their staging tree.  Polygon COCO
    # JSON and JPEG re-encoding add headroom; the estimate is an admission aid
    # while the filesystem check in the child remains authoritative.
    disk_required = source_bytes * 3 + 512 * 1024**2
    return DatasetPreparationBudget(
        soft_host_bytes=soft,
        hard_host_bytes=hard,
        minimum_system_available_bytes=reserve,
        disk_required_bytes=disk_required,
        disk_available_bytes=disk_available,
        source_files=source_files,
        source_bytes=source_bytes,
        source_fingerprint=source_fingerprint,
    )


def _write_request(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _decode_result(path: Path) -> DatasetPreparationResult:
    payload = json.loads(read_bounded_text(path, max_bytes=MAX_RESULT_BYTES))
    if type(payload) is not dict:
        raise RuntimeError("dataset preparation sidecar returned an invalid result")
    if not payload.get("success"):
        raise RuntimeError(str(payload.get("error", "dataset preparation failed")))
    report_raw = payload.get("preflight") or {}
    issues = [ValidationIssue(**entry) for entry in report_raw.get("issues", [])]
    report = ValidationReport(
        valid=bool(report_raw.get("valid")),
        issues=issues,
        stats=dict(report_raw.get("stats", {})),
    )
    return DatasetPreparationResult(
        role_dataset_dirs={
            str(key): str(value)
            for key, value in dict(payload["role_dataset_dirs"]).items()
        },
        roles=tuple(TrainingRole(value) for value in payload["roles"]),
        measured_reference_body_px=float(
            payload.get("measured_reference_body_px", 0.0)
        ),
        preflight=report,
    )


def prepare_role_datasets_contained(
    orchestrator: Any,
    request: DatasetPreparationRequest,
    *,
    log: Callable[[str], None],
    status: Callable[[str], None],
    should_cancel: Callable[[], bool],
) -> DatasetPreparationResult:
    """Run inspection and preparation in one memory-limited CPU sidecar."""

    workspace = Path(orchestrator.workspace_root).expanduser().resolve()
    payload = _bounded_request_payload(request)
    budget = assess_preparation_budget(workspace, request)
    job_id = uuid.uuid4().hex
    control = Path(tempfile.mkdtemp(prefix=f"hydra-dataset-prep-{job_id}-"))
    request_path = control / "request.json"
    result_path = control / "result.json"
    staging_root = workspace / f".dataset-preparation-{job_id}.staging"
    final_root = workspace / "prepared" / f"dataset-preparation-{job_id}"
    _write_request(request_path, payload)
    command = (
        sys.executable,
        "-m",
        "hydra_suite.detectkit.jobs.dataset_preparation_child",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
        "--staging-root",
        str(staging_root),
        "--final-root",
        str(final_root),
        "--disk-required-bytes",
        str(budget.disk_required_bytes),
    )
    limits = ProcessMemoryLimits(
        soft_host_bytes=budget.soft_host_bytes,
        hard_host_bytes=budget.hard_host_bytes,
        max_processes=64,
    )
    launch = build_limited_launch(command, limits, accelerator_kind=AcceleratorKind.CPU)
    plan = ContainmentPlan(
        launch=launch,
        job_name="DetectKit dataset preparation",
        minimum_system_available_bytes=budget.minimum_system_available_bytes,
        poll_interval_seconds=0.25,
    )

    def final_live_check() -> None:
        live_memory = psutil.virtual_memory()
        if (
            int(live_memory.available) - budget.minimum_system_available_bytes
            < budget.hard_host_bytes
        ):
            raise DatasetLimitError(
                "Available host memory changed before dataset preparation; "
                "the immutable hard limit no longer preserves the host reserve."
            )
        live_files, live_bytes, live_fingerprint = _scan_source_footprint(
            request.sources
        )
        if (
            live_files != budget.source_files
            or live_bytes != budget.source_bytes
            or live_fingerprint != budget.source_fingerprint
        ):
            raise DatasetLimitError(
                "A dataset source changed after initial admission; refusing to "
                "widen or alter the immutable preparation budget."
            )
        if shutil.disk_usage(workspace).free < budget.disk_required_bytes:
            raise DatasetLimitError(
                "Available disk space changed before dataset preparation and no "
                "longer satisfies the transient-space budget."
            )

    def cleanup(*, remove_final: bool = False) -> None:
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(control, ignore_errors=True)
        if remove_final:
            shutil.rmtree(final_root, ignore_errors=True)

    try:
        sidecar = SupervisedSidecar(
            plan,
            prelaunch_check=final_live_check,
            output_max_lines=OUTPUT_MAX_LINES,
            output_max_chars=OUTPUT_MAX_CHARS,
        )
    except WorkloadStillOwnedError as exc:
        exc.recovery_cleanup = lambda: cleanup(remove_final=True)
        raise
    except Exception:
        cleanup(remove_final=True)
        raise
    requested_cancel = False
    try:
        while True:
            lines, eof, output_error = sidecar.output.drain(timeout=0.1)
            for line in lines:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    log(line.rstrip())
                    continue
                message = str(record.get("message", ""))[:32_768]
                if record.get("type") == "status":
                    status(message)
                elif message:
                    log(message)
            if output_error is not None:
                raise RuntimeError(f"dataset preparation output failed: {output_error}")
            if should_cancel():
                requested_cancel = True
                sidecar.cancel(plan.terminate_grace_seconds)
                raise DatasetPreparationCancelled("Dataset preparation cancelled.")
            if (
                eof
                and sidecar.process is not None
                and sidecar.process.poll() is not None
            ):
                break
        supervised = sidecar.wait(requested_cancel=requested_cancel)
    except WorkloadStillOwnedError as exc:
        exc.recovery_cleanup = lambda: cleanup(remove_final=True)
        raise
    except DatasetPreparationCancelled:
        cleanup(remove_final=True)
        raise
    except Exception:
        try:
            sidecar.cancel(plan.terminate_grace_seconds)
        except WorkloadStillOwnedError as exc:
            exc.recovery_cleanup = lambda: cleanup(remove_final=True)
            raise
        cleanup(remove_final=True)
        raise

    try:
        if supervised.classified_exit.kind is not ExitKind.SUCCESS:
            detail = ""
            if result_path.exists():
                try:
                    failed = json.loads(
                        read_bounded_text(result_path, max_bytes=MAX_RESULT_BYTES)
                    )
                    if type(failed) is dict:
                        detail = str(failed.get("error", ""))[:32_768]
                except (OSError, ValueError, TypeError):
                    detail = ""
            message = supervised.classified_exit.message
            if detail:
                message = f"{message}: {detail}"
            raise DatasetPreparationSidecarError(
                supervised.classified_exit.kind, message
            )
        decoded = _decode_result(result_path)
    except Exception:
        cleanup(remove_final=True)
        raise
    cleanup()
    return decoded
