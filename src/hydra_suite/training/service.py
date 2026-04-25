"""High-level orchestration service for MAT role-aware training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .contracts import (
    DatasetBuildResult,
    SourceDataset,
    SplitConfig,
    TrainingRole,
    TrainingRunSpec,
    ValidationReport,
)
from .dataset_builders import merge_obb_sources, prepare_role_dataset
from .dataset_inspector import inspect_obb_or_detect_dataset
from .model_publish import (
    classifier_metadata_for_artifact,
    publish_trained_model,
    write_classifier_multihead_manifest,
)
from .registry import (
    create_run_record,
    dataset_fingerprint,
    finalize_run_record,
    new_run_id,
)
from .runner import run_training
from .validation import (
    format_validation_report,
    validate_obb_dataset,
    validate_role_dataset,
)

_MULTIHEAD_CLASSIFIER_ROLES = {
    TrainingRole.CLASSIFY_MULTIHEAD_YOLO,
    TrainingRole.CLASSIFY_MULTIHEAD_TINY,
    TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM,
}


def _result_artifact_paths(result: dict) -> list[str]:
    artifact_paths = result.get("artifact_paths")
    if isinstance(artifact_paths, list):
        return [str(path) for path in artifact_paths if str(path).strip()]
    artifact_path = str(result.get("artifact_path", "") or "").strip()
    return [artifact_path] if artifact_path else []


def _publish_training_artifacts(
    *,
    spec: TrainingRunSpec,
    artifact_paths: list[str],
    publish_metadata: dict[str, object],
    run_id: str,
    dataset_fingerprint_value: str,
) -> tuple[str, str]:
    if not artifact_paths:
        return "", ""

    raw_recommended_threshold = publish_metadata.get(
        "recommended_confidence_threshold",
        publish_metadata.get("prediction_confidence_threshold"),
    )
    try:
        recommended_confidence_threshold = (
            min(1.0, max(0.0, float(raw_recommended_threshold)))
            if raw_recommended_threshold is not None
            else None
        )
    except (TypeError, ValueError):
        recommended_confidence_threshold = None

    training_params = publish_metadata.get("training_params")
    base_kwargs = {
        "role": spec.role,
        "size": str(publish_metadata.get("size", "") or "unknown"),
        "species": str(publish_metadata.get("species", "") or "species"),
        "model_info": str(
            publish_metadata.get("model_info", "") or f"{spec.role.value}_{run_id}"
        ),
        "trained_from_run_id": run_id,
        "dataset_fingerprint": dataset_fingerprint_value,
        "base_model": spec.base_model,
        "training_params": (
            dict(training_params) if isinstance(training_params, dict) else None
        ),
    }

    if len(artifact_paths) == 1 or spec.role not in _MULTIHEAD_CLASSIFIER_ROLES:
        classifier_meta = None
        try:
            classifier_meta = classifier_metadata_for_artifact(artifact_paths[0])
        except Exception:
            classifier_meta = None
        if (
            isinstance(classifier_meta, dict)
            and recommended_confidence_threshold is not None
        ):
            classifier_meta["recommended_confidence_threshold"] = (
                recommended_confidence_threshold
            )
        return publish_trained_model(
            artifact_path=artifact_paths[0],
            classifier_v2_meta=classifier_meta,
            **base_kwargs,
        )

    configured_factor_names = publish_metadata.get("factor_names")
    factor_names = (
        [str(name) for name in configured_factor_names]
        if isinstance(configured_factor_names, list)
        else []
    )
    scheme_name = str(publish_metadata.get("scheme_name", "") or "classkit")
    published_key = ""
    published_factor_paths: list[Path] = []
    factor_entries: list[dict[str, object]] = []
    used_factor_names: set[str] = set()
    bundle_input_size: tuple[int, int] | None = None
    bundle_monochrome = False
    bundle_confidence_threshold: float | None = recommended_confidence_threshold

    for index, artifact_path in enumerate(artifact_paths):
        classifier_meta = classifier_metadata_for_artifact(artifact_path)
        if (
            isinstance(classifier_meta, dict)
            and recommended_confidence_threshold is not None
        ):
            classifier_meta["recommended_confidence_threshold"] = (
                recommended_confidence_threshold
            )
        candidate_name = ""
        if index < len(factor_names):
            candidate_name = factor_names[index]
        elif classifier_meta.get("factor_names"):
            candidate_name = str(classifier_meta["factor_names"][0])
        factor_name = candidate_name.strip() or f"factor_{index + 1}"
        if factor_name in used_factor_names:
            factor_name = f"factor_{index + 1}"
        used_factor_names.add(factor_name)

        key, published_path = publish_trained_model(
            artifact_path=artifact_path,
            scheme_name=scheme_name,
            factor_index=index,
            factor_name=factor_name,
            classifier_v2_meta=classifier_meta,
            **base_kwargs,
        )
        if not published_key:
            published_key = key
        published_factor_paths.append(Path(published_path))

        input_size = classifier_meta.get("input_size") or [224, 224]
        if bundle_input_size is None:
            bundle_input_size = (int(input_size[0]), int(input_size[1]))
        bundle_monochrome = bool(classifier_meta.get("monochrome", bundle_monochrome))
        recommended_confidence_threshold = classifier_meta.get(
            "recommended_confidence_threshold"
        )
        if recommended_confidence_threshold is not None:
            try:
                threshold_value = float(recommended_confidence_threshold)
            except (TypeError, ValueError):
                threshold_value = None
            if threshold_value is not None:
                threshold_value = min(1.0, max(0.0, threshold_value))
                if bundle_confidence_threshold is None:
                    bundle_confidence_threshold = threshold_value
                else:
                    bundle_confidence_threshold = max(
                        bundle_confidence_threshold, threshold_value
                    )
        class_names_per_factor = classifier_meta.get("class_names_per_factor") or [[]]
        factor_entries.append(
            {
                "factor": factor_name,
                "path": Path(published_path),
                "class_names": list(class_names_per_factor[0]),
            }
        )

    manifest_path = published_factor_paths[0].with_suffix(".multihead.json")
    write_classifier_multihead_manifest(
        manifest_path,
        factor_entries=factor_entries,
        input_size=bundle_input_size or (224, 224),
        monochrome=bundle_monochrome,
        recommended_confidence_threshold=bundle_confidence_threshold,
    )
    return published_key, str(manifest_path)


@dataclass(slots=True)
class RoleRunConfig:
    """Role-specific training config values."""

    role: TrainingRole
    enabled: bool = True
    base_model: str = ""
    size: str = "26s"
    species: str = "species"
    model_info: str = "model"


@dataclass(slots=True)
class TrainingSessionResult:
    """Session result summary for UI."""

    merged_dataset: str = ""  # noqa: DC01  (dataclass field)

    role_dataset_dirs: dict[str, str] = field(default_factory=dict)
    run_ids: list[str] = field(default_factory=list)  # noqa: DC01  (dataclass field)
    published_models: dict[str, str] = field(
        default_factory=dict
    )  # noqa: DC01  (dataclass field)
    errors: list[str] = field(default_factory=list)  # noqa: DC01  (dataclass field)


class TrainingOrchestrator:
    """Coordinates validation, dataset derivation, run registry, and publishing."""

    def __init__(self, workspace_root: str | Path):
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def preflight_obb_sources(
        self,
        sources: list[SourceDataset],
        *,
        require_train_val: bool = False,
    ) -> ValidationReport:
        """Validate OBB/detect source datasets, checking splits, class IDs, and annotation integrity."""
        all_issues = []
        stats = {"sources": []}

        for src in sources:
            inspection = inspect_obb_or_detect_dataset(src.path)
            report = validate_obb_dataset(
                inspection,
                require_train_val=require_train_val,
                min_train=1,
                min_val=1,
            )
            stats["sources"].append(
                {
                    "path": src.path,
                    "valid": report.valid,
                    "split_counts": report.stats.get("split_counts", {}),
                    "class_ids": report.stats.get("class_ids", []),
                }
            )
            all_issues.extend(report.issues)

        valid = not any(i.severity == "error" for i in all_issues)
        return ValidationReport(valid=valid, issues=all_issues, stats=stats)

    def build_merged_obb_dataset(
        self,
        sources: list[SourceDataset],
        *,
        class_name: str | None = None,
        class_names: list[str] | None = None,
        split_cfg: SplitConfig,
        seed: int,
        dedup: bool,
    ) -> DatasetBuildResult:
        """Merge multiple OBB source datasets into a single unified dataset with optional deduplication."""
        resolved_class_names = [
            str(name).strip()
            for name in (class_names or [class_name or "object"])
            if str(name).strip()
        ] or ["object"]
        out_root = self.workspace_root / "datasets"
        out_root.mkdir(parents=True, exist_ok=True)
        return merge_obb_sources(
            sources=sources,
            output_root=out_root,
            class_name=resolved_class_names[0],
            class_names=resolved_class_names,
            split_cfg=split_cfg,
            seed=seed,
            dedup=dedup,
            remap_single_class=len(resolved_class_names) == 1,
        )

    def build_role_dataset(
        self,
        role: TrainingRole,
        merged_obb_dataset_dir: str,
        *,
        class_name: str | None = None,
        class_names: list[str] | None = None,
        crop_pad_ratio: float = 0.15,
        min_crop_size_px: int = 64,
        enforce_square: bool = True,
    ) -> DatasetBuildResult:
        """Derive a role-specific dataset (detect, crop-OBB, classify) from a merged OBB dataset."""
        out_root = self.workspace_root / "derived" / role.value
        out_root.mkdir(parents=True, exist_ok=True)
        result = prepare_role_dataset(
            role=role,
            merged_obb_dataset_dir=merged_obb_dataset_dir,
            role_output_root=out_root,
            class_name=class_name,
            class_names=class_names,
            crop_pad_ratio=crop_pad_ratio,
            min_crop_size_px=min_crop_size_px,
            enforce_square=enforce_square,
        )
        report = validate_role_dataset(result.dataset_dir, role)
        result.stats = dict(result.stats)
        result.stats["validation"] = report.to_dict()
        if not report.valid:
            raise RuntimeError(
                f"Derived dataset for role '{role.value}' is not valid for Ultralytics training.\n"
                f"{format_validation_report(report)}"
            )
        return result

    def run_role_training(
        self,
        spec: TrainingRunSpec,
        *,
        parent_run_id: str = "",
        publish_metadata: dict[str, object] | None = None,
        log_cb: Callable[[str], None] | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict:
        """Execute a training run: register in the run registry, train, and optionally publish the model."""
        run_id = new_run_id(spec.role.value)
        run_dir = self.workspace_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ds_fp = dataset_fingerprint(spec.derived_dataset_dir)

        create_run_record(
            spec,
            run_id=run_id,
            run_dir=run_dir,
            dataset_fp=ds_fp,
            parent_run_id=parent_run_id,
        )

        result = run_training(
            spec,
            run_dir,
            log_cb=log_cb,
            progress_cb=progress_cb,
            should_cancel=should_cancel,
        )
        result["run_id"] = run_id
        artifact_paths = _result_artifact_paths(result)

        if not result.get("success", False):
            finalize_run_record(
                run_id,
                status="failed" if not result.get("canceled") else "canceled",
                command=result.get("command", []),
                metrics_paths=(
                    [result.get("metrics_path", "")]
                    if result.get("metrics_path")
                    else []
                ),
                artifact_paths=artifact_paths,
                error_message=(
                    "canceled"
                    if result.get("canceled")
                    else f"exit_code={result.get('exit_code', 'unknown')}"
                ),
            )
            return result

        published_key = ""
        published_path = ""
        if spec.publish_policy.auto_import and artifact_paths:
            published_key, published_path = _publish_training_artifacts(
                spec=spec,
                artifact_paths=artifact_paths,
                publish_metadata=publish_metadata or {},
                run_id=run_id,
                dataset_fingerprint_value=ds_fp,
            )

        finalize_run_record(
            run_id,
            status="completed",
            command=result.get("command", []),
            metrics_paths=(
                [result.get("metrics_path", "")] if result.get("metrics_path") else []
            ),
            artifact_paths=artifact_paths,
            published_model_path=published_path,
            published_registry_entry=published_key,
        )

        result["published_registry_key"] = published_key
        result["published_model_path"] = published_path
        result["dataset_fingerprint"] = ds_fp
        return result
