"""Qt-free DetectKit dataset preparation and training-session workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError
from hydra_suite.runtime.safe_text import (
    bounded_terminal_text,
    sanitize_terminal_text_fields,
)
from hydra_suite.training.contracts import (
    Sam3LoraParams,
    SourceDataset,
    SplitConfig,
    TrainingRole,
    TrainingRunSpec,
    ValidationReport,
)
from hydra_suite.training.dataset_builders import role_min_level
from hydra_suite.training.dataset_inspector import inspect_obb_or_detect_dataset
from hydra_suite.training.validation import validate_ultralytics_dataset

if TYPE_CHECKING:
    from hydra_suite.detectkit.config.training import (
        DetectTrainingPlan,
        SliceTrainingConfig,
    )

LogCallback = Callable[[str], None]
StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[str, int, int], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class DatasetPreparationRequest:
    """Immutable inputs for deriving every selected DetectKit role dataset."""

    sources: tuple[SourceDataset, ...]
    roles: tuple[TrainingRole, ...]
    class_names: tuple[str, ...]
    split: SplitConfig
    seed: int
    dedup: bool
    crop_pad_ratio: float
    min_crop_size_px: int
    enforce_square: bool
    imgsz_by_role: tuple[tuple[str, int], ...]
    slice_settings: "SliceTrainingConfig"
    sam3_params: Sam3LoraParams | None = None

    def imgsz_for(self, role: TrainingRole) -> int:
        return dict(self.imgsz_by_role).get(role.value, 640)


@dataclass(frozen=True, slots=True)
class DatasetPreparationResult:
    role_dataset_dirs: dict[str, str]
    roles: tuple[TrainingRole, ...]
    measured_reference_body_px: float = 0.0


@dataclass(frozen=True, slots=True)
class RoleTrainingEntry:
    role: TrainingRole
    spec: TrainingRunSpec
    publish_metadata: dict[str, object]

    def __getitem__(self, key: str):
        """Retain the dialog worker's legacy dict-style test/debug access."""
        if key == "role":
            return self.role
        if key == "spec":
            return self.spec
        if key == "publish_meta":
            return self.publish_metadata
        raise KeyError(key)


class DatasetPreparationCancelled(RuntimeError):
    """Raised after cancellation is observed at a safe preparation boundary."""


def preflight_sources(sources: tuple[SourceDataset, ...]) -> ValidationReport:
    """Validate DetectKit sources using each source's native geometry level."""

    issues = []
    stats: dict[str, object] = {"sources": []}
    for source in sources:
        inspection = inspect_obb_or_detect_dataset(source.path)
        label_mode = {
            "aabb": "detect",
            "obb": "obb",
            "polygon": "segment",
        }.get(source.level, "obb")
        report = validate_ultralytics_dataset(
            inspection,
            label_mode=label_mode,
            require_single_class=False,
            require_train_val=False,
            min_train=1,
            min_val=0,
        )
        stats["sources"].append(
            {
                "path": source.path,
                "level": source.level,
                "valid": report.valid,
                "stats": report.stats,
            }
        )
        issues.extend(report.issues)
    return ValidationReport(
        valid=not any(issue.severity == "error" for issue in issues),
        issues=issues,
        stats=stats,
    )


def prepare_role_datasets(
    orchestrator,
    request: DatasetPreparationRequest,
    *,
    log: LogCallback,
    status: StatusCallback,
    should_cancel: CancelCheck,
) -> DatasetPreparationResult:
    """Build every selected role dataset without importing Qt."""

    def check_cancelled() -> None:
        if should_cancel():
            raise DatasetPreparationCancelled("Dataset preparation cancelled.")

    role_dataset_dirs: dict[str, str] = {}
    merged_by_level = {}
    measured_reference_body_px = 0.0

    for role_index, role in enumerate(request.roles, start=1):
        check_cancelled()
        status(f"Preparing dataset {role_index}/{len(request.roles)}: {role.value}…")

        if role is TrainingRole.SEMANTIC_SAM3:
            if len(request.sources) != 1:
                raise ValueError(
                    "SAM3 concept training supports exactly one labeled source "
                    "dataset at a time."
                )
            if request.sam3_params is None:
                raise ValueError("SAM3 dataset preparation requires SAM3 parameters.")
            source = request.sources[0]
            build = orchestrator.build_role_dataset(
                role,
                source.path,
                sam3_params=request.sam3_params,
                seed=request.seed,
                split=request.split,
            )
            check_cancelled()
            role_dataset_dirs[role.value] = build.dataset_dir
            log(
                f"Prepared [{role.value}] dataset from {source.path}: "
                f"{build.dataset_dir}"
            )
            continue

        required_level = role_min_level(role)
        merged = merged_by_level.get(required_level)
        if merged is None:
            status(f"Merging and deduplicating {required_level.label} sources…")
            merged = orchestrator.build_merged_obb_dataset(
                list(request.sources),
                class_names=list(request.class_names),
                split_cfg=request.split,
                seed=request.seed,
                dedup=request.dedup,
                target_level=required_level,
            )
            check_cancelled()
            merged_by_level[required_level] = merged
            used_count = len(merged.stats.get("source_items", {}))
            log(
                f"Merged {required_level.label} dataset from "
                f"{used_count}/{len(request.sources)} compatible source(s): "
                f"{merged.dataset_dir}"
            )

        role_source_dir = merged.dataset_dir
        slicing = request.slice_settings
        if (
            role
            in {
                TrainingRole.OBB_DIRECT,
                TrainingRole.DETECT_DIRECT,
                TrainingRole.SEGMENT_DIRECT,
            }
            and slicing.enabled
        ):
            from hydra_suite.training.sliced_dataset import SliceBuildParams

            status(f"Slicing the {role.value} dataset…")
            params = SliceBuildParams(
                geometry_mode=slicing.geometry_mode,
                imgsz=request.imgsz_for(role),
                object_tile_fraction=slicing.object_tile_fraction,
                slice_width=slicing.slice_width,
                slice_height=slicing.slice_height,
                overlap=slicing.overlap,
                min_area_ratio=slicing.min_area_ratio,
                negative_tile_fraction=slicing.negative_tile_fraction,
                target_sizes=slicing.target_sizes_for(request.imgsz_for(role)),
                full_frame_mix=slicing.full_frame_mix,
                reference_body_px=slicing.reference_body_px,
            )
            sliced = orchestrator.build_sliced_obb_dataset(
                merged.dataset_dir,
                level=required_level,
                params=params,
                seed=request.seed,
            )
            check_cancelled()
            role_source_dir = sliced.dataset_dir
            log(f"Sliced dataset: {sliced.dataset_dir}")
            if measured_reference_body_px <= 0.0:
                measured_reference_body_px = float(
                    sliced.stats.get("measured_reference_body_px", 0.0)
                )

        status(f"Generating the {role.value} role dataset…")
        build = orchestrator.build_role_dataset(
            role,
            role_source_dir,
            class_names=list(request.class_names),
            crop_pad_ratio=request.crop_pad_ratio,
            min_crop_size_px=request.min_crop_size_px,
            enforce_square=request.enforce_square,
            merged_level=required_level,
        )
        check_cancelled()
        role_dataset_dirs[role.value] = build.dataset_dir
        log(f"Prepared [{role.value}] dataset: {build.dataset_dir}")

    return DatasetPreparationResult(
        role_dataset_dirs=role_dataset_dirs,
        roles=request.roles,
        measured_reference_body_px=measured_reference_body_px,
    )


def _infer_size_token(model_path: str) -> str:
    name = Path(str(model_path or "")).name.lower()
    for token in (
        "26n",
        "26s",
        "26m",
        "26l",
        "26x",
        "11n",
        "11s",
        "11m",
        "11l",
        "11x",
    ):
        if token in name:
            return token
    return "unknown"


def build_role_entries(
    plan: "DetectTrainingPlan", role_dataset_dirs: dict[str, str]
) -> list[RoleTrainingEntry]:
    """Translate a plan and prepared datasets into generic training specs."""

    entries: list[RoleTrainingEntry] = []
    for role_config in plan.roles:
        role = role_config.role
        dataset_dir = str(role_dataset_dirs.get(role.value, "")).strip()
        if not dataset_dir:
            raise ValueError(f"No prepared dataset for role: {role.value}")

        if role is TrainingRole.SEMANTIC_SAM3:
            if plan.sam3_params is None:
                raise ValueError("SAM3 training requires SAM3 parameters")
            spec = TrainingRunSpec(
                role=role,
                source_datasets=list(plan.sources),
                derived_dataset_dir=dataset_dir,
                base_model="sam3",
                hyperparams=replace(
                    plan.hyperparams,
                    epochs=plan.sam3_params.epochs,
                    imgsz=role_config.imgsz,
                ),
                device=plan.device,
                seed=plan.seed,
                publish_policy=plan.publish_policy,
                sam3_params=plan.sam3_params,
            )
        else:
            spec = TrainingRunSpec(
                role=role,
                source_datasets=list(plan.sources),
                derived_dataset_dir=dataset_dir,
                base_model=role_config.base_model,
                hyperparams=replace(plan.hyperparams, imgsz=role_config.imgsz),
                device=plan.device,
                seed=plan.seed,
                augmentation_profile=plan.augmentation_profile,
                publish_policy=plan.publish_policy,
            )

        training_params: dict[str, object] = {"imgsz": role_config.imgsz}
        if role in {TrainingRole.SEQ_CROP_OBB, TrainingRole.SEQ_CROP_SEGMENT}:
            training_params.update(
                {
                    "crop_pad_ratio": plan.crop_pad_ratio,
                    "min_crop_size_px": plan.min_crop_size_px,
                    "enforce_square": plan.enforce_square,
                }
            )
        entries.append(
            RoleTrainingEntry(
                role=role,
                spec=spec,
                publish_metadata={
                    "size": _infer_size_token(role_config.base_model),
                    "species": plan.species,
                    "model_info": f"{plan.model_tag}_{role.value}",
                    "training_params": training_params,
                },
            )
        )
    return entries


def run_role_entries(
    orchestrator,
    entries: list[RoleTrainingEntry],
    *,
    log: LogCallback,
    progress: ProgressCallback,
    should_cancel: CancelCheck,
    role_started: Callable[[str], None] | None = None,
    role_finished: Callable[[str, bool, str], None] | None = None,
) -> list[dict]:
    """Run role entries sequentially, preserving parent-run lineage."""

    results: list[dict] = []
    parent_run_id = ""
    for entry in entries:
        if should_cancel():
            break
        if role_started is not None:
            role_started(entry.role.value)

        def role_log(message: str, role=entry.role) -> None:
            log(f"[{role.value}] {bounded_terminal_text(message)}")

        def role_progress(current: int, total: int, role=entry.role) -> None:
            progress(role.value, int(current), int(total))

        try:
            result = orchestrator.run_role_training(
                entry.spec,
                parent_run_id=parent_run_id,
                publish_metadata=entry.publish_metadata,
                log_cb=role_log,
                progress_cb=role_progress,
                should_cancel=should_cancel,
            )
        except WorkloadStillOwnedError:
            # This exception owns the live sidecar and its leases. Stop the
            # role sequence and preserve the exact recovery handle for the
            # worker/UI owner.
            raise
        except Exception as exc:
            result = {
                "run_id": "",
                "success": False,
                "error": bounded_terminal_text(exc, include_exception_type=False),
                "published_registry_key": "",
                "published_model_path": "",
            }
        result["role"] = entry.role.value
        sanitized_result = sanitize_terminal_text_fields(result)
        assert isinstance(sanitized_result, dict)
        result = sanitized_result
        results.append(result)
        ok = bool(result.get("success", False))
        message = (
            f"run_id={bounded_terminal_text(result.get('run_id', ''))}"
            if ok
            else result.get("error")
            or f"exit={bounded_terminal_text(result.get('exit_code', 'unknown'))}"
        )
        if role_finished is not None:
            role_finished(entry.role.value, ok, bounded_terminal_text(message))
        if result.get("run_id"):
            parent_run_id = bounded_terminal_text(result["run_id"])
    return results
