"""High-level orchestration service for MAT role-aware training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hydra_suite.core.inference.semantic.checkpoints import ensure_checkpoint
from hydra_suite.runtime.process_supervisor import WorkloadStillOwnedError
from hydra_suite.runtime.safe_text import bounded_terminal_text

from .contracts import (
    DatasetBuildResult,
    Sam3LoraParams,
    SourceDataset,
    SplitConfig,
    TrainingRole,
    TrainingRunSpec,
    ValidationReport,
    sam3_prompt_pool_error,
)
from .dataset_builders import merge_obb_sources, prepare_role_dataset
from .dataset_inspector import inspect_obb_or_detect_dataset
from .geometry_levels import GeometryLevel
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
    update_run_record,
)
from .runner import run_training
from .sam3_lora.publish import publish_sam3_model
from .sliced_dataset import SliceBuildParams, build_sliced_obb_dataset
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

_DIRECT_DETECTOR_ROLES = {
    TrainingRole.OBB_DIRECT,
    TrainingRole.DETECT_DIRECT,
    TrainingRole.SEGMENT_DIRECT,
}

_ROLE_SOURCE_STAMP_FILENAME = ".source_stamp.json"


def _guard_single_source_role_dataset(out_root: Path, source_dir: str) -> None:
    """Refuse a `build_role_dataset` call that would silently overwrite a
    previous call's output for a different source.

    Mirrors the DetectKit training dialog's "Multiple Sources Not
    Supported" guard (see `_start_training`), but at the service layer,
    so a programmatic caller gets the same protection the GUI already has.
    """
    stamp_path = out_root / _ROLE_SOURCE_STAMP_FILENAME
    if not stamp_path.exists():
        return
    try:
        previous_source = json.loads(stamp_path.read_text(encoding="utf-8"))["source"]
    except (json.JSONDecodeError, KeyError, OSError):
        # A missing/corrupt stamp must never block a legitimate build; treat
        # it as "no prior source recorded" rather than refusing.
        return
    resolved_current = str(Path(source_dir).expanduser().resolve())
    if previous_source == resolved_current:
        return
    raise ValueError(
        "SAM3 concept training supports exactly one labeled source dataset "
        f"per derived output directory ({out_root}). A previous build "
        f"derived this role's dataset from {previous_source!r}; refusing to "
        f"silently overwrite it with a build from {resolved_current!r}. "
        "Use a separate workspace (or run this role separately per source) "
        "instead of reusing the same derived output directory for multiple "
        "sources."
    )


def _stamp_role_dataset_source(out_root: Path, source_dir: str) -> None:
    """Record the source that produced `out_root`'s contents, so a later
    call can detect a would-be silent overwrite from a different source."""
    stamp_path = out_root / _ROLE_SOURCE_STAMP_FILENAME
    resolved_current = str(Path(source_dir).expanduser().resolve())
    stamp_path.write_text(json.dumps({"source": resolved_current}), encoding="utf-8")


def _result_artifact_paths(result: dict) -> list[str]:
    artifact_paths = result.get("artifact_paths")
    if isinstance(artifact_paths, list):
        return [str(path) for path in artifact_paths if str(path).strip()]
    artifact_path = str(result.get("artifact_path", "") or "").strip()
    return [artifact_path] if artifact_path else []


def _failure_details(result: dict) -> dict[str, object]:
    """Return durable structured diagnostics from a failed runner result."""

    return {
        key: result[key]
        for key in ("failure_kind", "resource_preflight", "containment")
        if key in result and result[key] is not None
    }


def _failure_message(result: dict) -> str:
    if result.get("canceled"):
        return "canceled"
    error_value = result.get("error", "") or ""
    error = bounded_terminal_text(error_value).strip()
    if error:
        return error
    return f"exit_code={bounded_terminal_text(result.get('exit_code', 'unknown'))}"


def _slice_geometry_for_publish(spec: TrainingRunSpec) -> dict | None:
    """Return a direct detector's derived-dataset slice geometry, if any.

    Reads ``<derived_dataset_dir>/manifest.json`` and returns its
    ``slice_geometry`` dict when present and non-empty. Any error (missing
    role match, missing manifest, bad JSON, wrong shape) yields None so that
    publish behavior is unaffected when slicing was not used.
    """
    if spec.role not in _DIRECT_DETECTOR_ROLES:
        return None
    try:
        manifest_path = Path(spec.derived_dataset_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        slice_geometry = manifest.get("slice_geometry")
        if isinstance(slice_geometry, dict) and slice_geometry:
            return slice_geometry
    except Exception:
        return None
    return None


def _publish_training_artifacts(
    *,
    spec: TrainingRunSpec,
    artifact_paths: list[str],
    publish_metadata: dict[str, object],
    run_id: str,
    dataset_fingerprint_value: str,
    log_cb: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[str, str]:
    if not artifact_paths:
        return "", ""

    if spec.role is TrainingRole.SEMANTIC_SAM3:
        # Forked, not extended: publish_trained_model's naming scheme does not
        # fit a promptable-concept checkpoint, and _repo_dir_for_role raises
        # for this role. See the design's "Publish -- a fork, not an extension".
        #
        # The geometry the sidecar needs (tile_px, reference_body_px) is
        # written by the BUILDER, not by publish_metadata -- which carries
        # size/species-style fields only. Read it from the derived dataset dir,
        # or the sidecar ships without geometry and Task 10b's prefill has
        # nothing to prefill (which would make Gate 3 unpassable).
        manifest_path = Path(spec.derived_dataset_dir) / "build_manifest.json"
        manifest = dict(publish_metadata or {})
        if manifest_path.exists():
            manifest.update(json.loads(manifest_path.read_text()))
        return publish_sam3_model(
            run_id=run_id,
            adapters_path=Path(artifact_paths[0]),
            base_checkpoint=ensure_checkpoint("sam3", allow_download=False),
            build_manifest=manifest,
            params=spec.sam3_params,
            source_fingerprint=dataset_fingerprint_value,
            log_cb=log_cb,
            should_cancel=should_cancel,
        )

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
        "slice_geometry": _slice_geometry_for_publish(spec),
        # Deliberately None, not a gap: every role this function currently
        # publishes for (OBB_DIRECT, DETECT_DIRECT, SEGMENT_DIRECT,
        # SEQ_DETECT, SEQ_CROP_OBB, SEQ_CROP_SEGMENT -- the only roles
        # dataset_builders.prepare_role_dataset supports) trains on full
        # frames/SAHI tiles or a legacy pad-ratio/enforce-square crop
        # (derive_crop_obb_dataset_from_obb), never the Layer 1
        # CanonicalGeometry fixed-canvas crop. There is no canonical
        # geometry for these models to be stamped with -- stamping one
        # would misrepresent what the model actually consumes at inference.
        # ClassKit's canonical-crop classify roles publish through
        # classkit/gui/main_window.py::_publish_training_results instead,
        # which DOES pass a real canonical_geometry when the training
        # images' import provenance recovers one.
        "canonical_geometry": None,
    }

    # The imgsz a role actually trained at (default 640) is the only sane
    # fallback for YOLO-classify artifacts with no .v2meta.json sidecar --
    # without it, classifier_metadata_for_artifact silently stamps a
    # hardcoded [224, 224] regardless of what the model was trained at.
    fallback_input_size = (
        int(spec.hyperparams.imgsz),
        int(spec.hyperparams.imgsz),
    )

    if len(artifact_paths) == 1 or spec.role not in _MULTIHEAD_CLASSIFIER_ROLES:
        classifier_meta = None
        try:
            classifier_meta = classifier_metadata_for_artifact(
                artifact_paths[0], fallback_input_size=fallback_input_size
            )
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
    bundle_fit_policy: str | None = None

    for index, artifact_path in enumerate(artifact_paths):
        classifier_meta = classifier_metadata_for_artifact(
            artifact_path, fallback_input_size=fallback_input_size
        )
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
        if bundle_fit_policy is None:
            # Every factor of a bundle is trained together under the same
            # pipeline, so all factors report the same fit_policy -- use the
            # first non-None one we see.
            candidate_fit_policy = classifier_meta.get("fit_policy")
            if candidate_fit_policy is not None:
                bundle_fit_policy = str(candidate_fit_policy)
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
        fit_policy=bundle_fit_policy,
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
        target_level: GeometryLevel = GeometryLevel.OBB,
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
            target_level=target_level,
        )

    def build_sliced_obb_dataset(
        self,
        merged_obb_dataset_dir: str,
        *,
        level: GeometryLevel,
        params: SliceBuildParams,
        seed: int = 42,
    ) -> DatasetBuildResult:
        """Tile a merged OBB dataset into a sliced dataset for SAHI-usable training."""
        out_root = self.workspace_root / "datasets_sliced"
        out_root.mkdir(parents=True, exist_ok=True)
        return build_sliced_obb_dataset(
            merged_obb_dataset_dir, out_root, level=level, params=params, seed=int(seed)
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
        merged_level: GeometryLevel = GeometryLevel.POLYGON,
        sam3_params: "Sam3LoraParams | None" = None,
        seed: int = 42,
        split: SplitConfig | None = None,
    ) -> DatasetBuildResult:
        """Derive a role-specific dataset (detect, crop-OBB, classify) from a merged OBB dataset."""
        out_root = self.workspace_root / "derived" / role.value
        out_root.mkdir(parents=True, exist_ok=True)
        if role is TrainingRole.SEMANTIC_SAM3:
            # SEMANTIC_SAM3 is derived directly per-source (it skips the
            # cross-source merge step other roles use, see
            # `_prepare_role_datasets` in the DetectKit training dialog), but
            # `out_root` is keyed by role only, not by source. A second call
            # for this role with a different source would silently overwrite
            # `train/_annotations.coco.json` while both sources' images stay
            # on disk, so only the last source's annotations would survive.
            # The GUI already refuses this up front; enforce it here too so
            # a programmatic/service-level caller cannot silently lose data
            # (see docs/superpowers/specs/2026-08-31-detectkit-sam3-finetune-design.md).
            _guard_single_source_role_dataset(out_root, merged_obb_dataset_dir)
        result = prepare_role_dataset(
            role=role,
            merged_obb_dataset_dir=merged_obb_dataset_dir,
            role_output_root=out_root,
            class_name=class_name,
            class_names=class_names,
            crop_pad_ratio=crop_pad_ratio,
            min_crop_size_px=min_crop_size_px,
            enforce_square=enforce_square,
            merged_level=merged_level,
            sam3_params=sam3_params,
            seed=seed,
            split=split,
        )
        report = validate_role_dataset(result.dataset_dir, role)
        result.stats = dict(result.stats)
        result.stats["validation"] = report.to_dict()
        if not report.valid:
            raise RuntimeError(
                f"Derived dataset for role '{role.value}' is not valid for Ultralytics training.\n"
                f"{format_validation_report(report)}"
            )
        if role is TrainingRole.SEMANTIC_SAM3:
            _stamp_role_dataset_source(out_root, merged_obb_dataset_dir)
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
        if spec.role is TrainingRole.SEMANTIC_SAM3:
            params = spec.sam3_params
            if params is None:
                raise ValueError("SAM3 training requires sam3_params")
            prompt_error = sam3_prompt_pool_error(
                params.prompt, params.negative_prompts
            )
            if prompt_error is not None:
                raise ValueError(f"Invalid SAM3 prompt configuration: {prompt_error}")
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

        try:
            result = run_training(
                spec,
                run_dir,
                log_cb=log_cb,
                progress_cb=progress_cb,
                should_cancel=should_cancel,
            )
        except WorkloadStillOwnedError as exc:
            # The exception owns the still-live sidecar and canonical leases.
            # Preserve that recovery handle for the caller; collapsing it into
            # a dict would orphan the workload and falsely imply finalization.
            try:
                update_run_record(
                    run_id,
                    {
                        "status": "recovery-required",
                        "error_message": bounded_terminal_text(
                            exc, include_exception_type=False
                        ),
                        "failure_kind": "workload-still-owned",
                        "containment": {"ownership": "retained"},
                    },
                )
            except Exception as registry_exc:  # noqa: BLE001 - preserve owner
                exc.registry_update_error = bounded_terminal_text(
                    registry_exc, include_exception_type=False
                )
            # The GUI recovery owner needs the registry identity so a later
            # successful cleanup can turn this nonterminal record into an
            # explicit failed terminal run.
            exc.run_id = run_id
            raise
        except Exception as exc:
            result = {
                "success": False,
                "error": bounded_terminal_text(exc, include_exception_type=False),
                "failure_kind": "training-exception",
            }
        result["run_id"] = run_id
        result["derived_dataset_dir"] = spec.derived_dataset_dir
        result["dataset_fingerprint"] = ds_fp
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
                error_message=_failure_message(result),
                failure_details=_failure_details(result),
            )
            return result

        published_key = ""
        published_path = ""
        if spec.publish_policy.auto_import and artifact_paths:
            try:
                published_key, published_path = _publish_training_artifacts(
                    spec=spec,
                    artifact_paths=artifact_paths,
                    publish_metadata=publish_metadata or {},
                    run_id=run_id,
                    dataset_fingerprint_value=ds_fp,
                    log_cb=log_cb,
                    should_cancel=should_cancel,
                )
            except WorkloadStillOwnedError as exc:
                try:
                    update_run_record(
                        run_id,
                        {
                            "status": "recovery-required",
                            "error_message": bounded_terminal_text(
                                exc, include_exception_type=False
                            ),
                            "failure_kind": "workload-still-owned",
                            "containment": {"ownership": "retained"},
                        },
                    )
                except Exception as registry_exc:  # noqa: BLE001 - preserve owner
                    exc.registry_update_error = bounded_terminal_text(
                        registry_exc, include_exception_type=False
                    )
                exc.run_id = run_id
                raise
            except Exception as exc:
                result["success"] = False
                result["error"] = bounded_terminal_text(
                    exc, include_exception_type=False
                )
                result["failure_kind"] = getattr(
                    exc, "failure_kind", "publish-exception"
                )
                result["canceled"] = bool(getattr(exc, "canceled", False))
                containment = getattr(exc, "containment", None)
                if isinstance(containment, dict):
                    result["containment"] = containment
                result["published_registry_key"] = ""
                result["published_model_path"] = ""
                finalize_run_record(
                    run_id,
                    status="canceled" if result["canceled"] else "failed",
                    command=result.get("command", []),
                    metrics_paths=(
                        [result.get("metrics_path", "")]
                        if result.get("metrics_path")
                        else []
                    ),
                    artifact_paths=artifact_paths,
                    error_message=_failure_message(result),
                    failure_details=_failure_details(result),
                )
                return result

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
        return result
