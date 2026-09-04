"""Qt-free child implementations for protected DetectKit operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from hydra_suite.core.inference.cache.base import CacheKey

Progress = Callable[[int, str], None]


def _required_text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _cache_key(payload: dict[str, Any]) -> CacheKey:
    raw = payload.get("cache_key")
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "model_path",
        "model_mtime",
        "config_hash",
    }:
        raise ValueError("cache_key has invalid fields")
    return CacheKey(
        schema_version=int(raw["schema_version"]),
        model_path=str(raw["model_path"]),
        model_mtime=float(raw["model_mtime"]),
        config_hash=str(raw["config_hash"]),
    )


def _sequential_dicts(tuples) -> list[dict]:
    import cv2

    detections = []
    for cx, cy, width, height, theta, confidence in tuples:
        corners = cv2.boxPoints(
            (
                (float(cx), float(cy)),
                (float(width), float(height)),
                float(theta) * 180.0 / 3.141592653589793,
            )
        )
        detections.append(
            {
                "class_id": 0,
                "polygon_px": [(float(x), float(y)) for x, y in corners],
                "confidence": float(confidence),
            }
        )
    return detections


def run_dataset_inference(
    payload: dict[str, Any], progress: Progress
) -> dict[str, Any]:
    """Process one image/tile batch at a time into an immutable chunk cache."""
    from hydra_suite.detectkit.gui.constants import IMG_EXTS
    from hydra_suite.detectkit.gui.models import SliceTrainingSettings
    from hydra_suite.detectkit.gui.prediction_preview import (
        dicts_from_obb_result,
        load_torch_model,
        predict_obb_for_frame_sequential,
        predict_preview_detections_for_image,
        predict_sliced_obb_result,
        preview_object_tile_fraction,
    )
    from hydra_suite.detectkit.jobs.prediction_cache import (
        DatasetPredictionWriter,
        write_path_index,
    )

    source_path = Path(_required_text(payload, "source_path")).expanduser().resolve()
    images_dir = source_path / "images"
    image_paths = sorted(
        path.resolve()
        for path in images_dir.rglob("*")
        if path.suffix.lower() in IMG_EXTS
    )
    cache_path = Path(_required_text(payload, "cache_path")).expanduser().resolve()
    key = _cache_key(payload)
    write_path_index(cache_path, image_paths)
    writer = DatasetPredictionWriter(
        cache_path,
        key,
        chunk_size=max(1, min(int(payload.get("chunk_frames", 8)), 64)),
        write_mode="fresh",
    )
    inference_kind = str(payload.get("inference_kind", "obb_direct"))
    model_path = _required_text(payload, "model_path")
    secondary_path = str(payload.get("secondary_model_path", "") or "").strip()
    device = str(payload.get("device", "auto") or "auto")
    threshold = float(payload.get("confidence_threshold", 0.01))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    slice_settings = SliceTrainingSettings.from_dict(
        payload.get("slice_settings") or {}
    )
    total = len(image_paths)
    count = 0
    confidence_sum = 0.0
    class_counts: dict[int, int] = {}

    runner = None
    primary = secondary = None
    primary_device = secondary_device = "cpu"
    if inference_kind == "sequential_segment":
        from hydra_suite.core.inference.runner import InferenceRunner
        from hydra_suite.data.al.inference_adapter import build_obb_config_for_al

        runner = InferenceRunner(
            build_obb_config_for_al(
                inference_kind,
                model_path,
                secondary_path,
                crop_pad_ratio=float(payload.get("crop_pad_ratio", 0.15)),
                confidence_threshold=threshold,
                iou_threshold=0.7,
                max_targets=max(1, min(int(payload.get("max_targets", 300)), 1000)),
                stage2_image_size=max(1, int(payload.get("stage2_image_size", 160))),
            )
        )
    elif secondary_path:
        primary, primary_device = load_torch_model(model_path, device, task="detect")
        secondary, secondary_device = load_torch_model(secondary_path, device)
    else:
        task = {"detect_direct": "detect", "segment_direct": "segment"}.get(
            inference_kind, "obb"
        )
        primary, primary_device = load_torch_model(model_path, device, task=task)

    import cv2

    for frame_idx, image_path in enumerate(image_paths):
        detections: list[dict]
        if runner is not None:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            detections = dicts_from_obb_result(
                runner.detect_batch_raw([frame], [frame_idx])[0]
            )
        elif secondary is not None:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            detections = _sequential_dicts(
                predict_obb_for_frame_sequential(
                    primary,
                    secondary,
                    frame,
                    detect_device=primary_device,
                    obb_device=secondary_device,
                    conf=threshold,
                    iou=0.7,
                    crop_pad_ratio=float(payload.get("crop_pad_ratio", 0.15)),
                )
            )
        elif slice_settings.enabled:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            imgsz = max(1, int(payload.get("imgsz_obb_direct", 640)))
            task = {"detect_direct": "detect", "segment_direct": "segment"}.get(
                inference_kind, "obb"
            )
            obb = predict_sliced_obb_result(
                primary,
                frame,
                geometry_mode=slice_settings.geometry_mode,
                imgsz=imgsz,
                reference_body_px=slice_settings.reference_body_px,
                object_tile_fraction=preview_object_tile_fraction(
                    slice_settings.target_sizes_for(imgsz),
                    slice_settings.object_tile_fraction,
                    imgsz,
                ),
                slice_width=slice_settings.slice_width,
                slice_height=slice_settings.slice_height,
                overlap=slice_settings.overlap,
                merge_threshold=slice_settings.merge_threshold,
                confidence_threshold=threshold,
                task=task,
            )
            detections = dicts_from_obb_result(obb) if obb is not None else []
        else:
            task = {"detect_direct": "detect", "segment_direct": "segment"}.get(
                inference_kind, "obb"
            )
            detections = predict_preview_detections_for_image(
                primary,
                str(image_path),
                device=primary_device,
                confidence_threshold=threshold,
                task=task,
            )
        writer.write_frame(frame_idx, detections)
        for detection in detections:
            count += 1
            confidence_sum += float(detection.get("confidence", 0.0))
            class_id = int(detection.get("class_id", 0))
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        progress(
            int((frame_idx + 1) * 100 / max(1, total)),
            f"Processed {frame_idx + 1}/{total}: {image_path.name}",
        )
    writer.close()
    return {
        "cache_path": str(cache_path),
        "image_count": total,
        "detection_count": count,
        "class_counts": {str(key): value for key, value in class_counts.items()},
        "mean_confidence": confidence_sum / count if count else 0.0,
    }


def run_evaluation(payload: dict[str, Any], progress: Progress) -> dict[str, Any]:
    """Execute one Ultralytics validation wholly in the child."""
    from hydra_suite.detectkit.evaluation import EvaluationCandidate, evaluate_candidate

    raw = payload.get("candidate")
    if not isinstance(raw, dict):
        raise ValueError("candidate must be an object")
    candidate = EvaluationCandidate(**raw)
    progress(1, f"Loading {candidate.run_id}")
    result = evaluate_candidate(
        candidate,
        output_root=_required_text(payload, "output_root"),
        device=str(payload.get("device", "cpu")),
        batch=max(1, int(payload.get("batch", 1))),
        evaluation_id=_required_text(payload, "evaluation_id"),
    )
    progress(100, f"Validated {candidate.run_id}")
    from dataclasses import asdict

    return {"evaluation_result": asdict(result)}


def _semantic_sources(payload: dict[str, Any]):
    from hydra_suite.detectkit.gui.models import OBBSource

    raw = payload.get("sources")
    if not isinstance(raw, list) or len(raw) > 10_000:
        raise ValueError("semantic sources must be a bounded list")
    return [OBBSource.from_dict(item) for item in raw]


def _semantic_instance_cap(payload: dict[str, Any]) -> int:
    requested = int(dict(payload.get("params") or {}).get("max_instances", 0))
    return min(requested if requested > 0 else 300, 1000)


def _semantic_labeler(payload: dict[str, Any], confidence_floor: float):
    from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler
    from hydra_suite.detectkit.jobs.semantic_escalation import labeler_checkpoint_for

    variant = _required_text(payload, "variant")
    device = str(payload.get("device", "auto") or "auto")
    return Sam3SemanticLabeler.from_variant(
        variant,
        device=None if device == "auto" else device,
        checkpoint=labeler_checkpoint_for(variant),
        confidence_floor=float(confidence_floor),
    )


def run_semantic_preview(payload: dict[str, Any], progress: Progress) -> dict[str, Any]:
    from hydra_suite.detectkit.jobs.semantic_artifacts import write_frame_preview
    from hydra_suite.detectkit.jobs.semantic_escalation import (
        DEFAULT_MERGE_IOU,
        DEFAULT_OVERLAP,
        DEFAULT_SEAM_MARGIN_PX,
        cache_confidence_floor,
        preview_random_frame,
    )

    sources = _semantic_sources(payload)
    params = dict(payload.get("params") or {})
    confidence = float(params.get("confidence", 0.35))
    labeler = _semantic_labeler(payload, cache_confidence_floor(confidence))
    result = preview_random_frame(
        labeler,
        sources,
        _required_text(payload, "prompt"),
        reference_body_px=float(params.get("reference_body_px", 0.0)),
        tile_fraction=params.get("tile_fraction"),
        overlap=float(params.get("overlap", DEFAULT_OVERLAP)),
        seam_margin_px=float(params.get("seam_margin_px", DEFAULT_SEAM_MARGIN_PX)),
        merge_iou=float(params.get("merge_iou", DEFAULT_MERGE_IOU)),
        confidence=confidence,
        max_instances=_semantic_instance_cap(payload),
        progress=lambda done, total: progress(
            int(100 * done / max(1, total)), f"Segmenting tile {done}/{total}"
        ),
    )
    output_path = Path(_required_text(payload, "output_path")).expanduser().resolve()
    write_frame_preview(output_path, result)
    return {"preview_path": str(output_path)}


def run_semantic_calibration(
    payload: dict[str, Any], progress: Progress
) -> dict[str, Any]:
    from dataclasses import asdict

    from hydra_suite.core.inference.semantic.calibration import (
        CONFIDENCE_GRID,
        calibrate,
    )
    from hydra_suite.detectkit.gui.calibration_preview_store import (
        save_calibration_previews,
    )
    from hydra_suite.detectkit.jobs.semantic_escalation import (
        CALIBRATION_SAMPLE_FRAMES,
        DEFAULT_MERGE_IOU,
        DEFAULT_OVERLAP,
        DEFAULT_SEAM_MARGIN_PX,
        stratified_calibration_frames,
    )

    sources = _semantic_sources(payload)
    budget = max(
        1,
        min(
            int(payload.get("sample_budget", CALIBRATION_SAMPLE_FRAMES)),
            CALIBRATION_SAMPLE_FRAMES,
        ),
    )
    frames = stratified_calibration_frames(sources, budget=budget)
    if not frames:
        return {"points": [], "sampled_frames": [], "preview_artifact": ""}
    params = dict(payload.get("params") or {})
    labeler = _semantic_labeler(payload, CONFIDENCE_GRID[0])
    previews = []
    points = calibrate(
        labeler,
        frames,
        _required_text(payload, "prompt"),
        reference_body_px=float(params.get("reference_body_px", 0.0)),
        overlap=float(params.get("overlap", DEFAULT_OVERLAP)),
        seam_margin_px=float(params.get("seam_margin_px", DEFAULT_SEAM_MARGIN_PX)),
        merge_iou=float(params.get("merge_iou", DEFAULT_MERGE_IOU)),
        max_instances=_semantic_instance_cap(payload),
        progress=progress,
        preview_sink=previews.extend,
    )
    project_dir = Path(_required_text(payload, "project_dir")).expanduser().resolve()
    requested_artifact = str(payload.get("preview_artifact", "") or "")
    preview_artifact = save_calibration_previews(
        project_dir,
        previews,
        relative_path=requested_artifact or None,
    )
    return {
        "points": [asdict(point) for point in points],
        "sampled_frames": [str(path.resolve()) for path, _records in frames],
        "preview_artifact": preview_artifact,
    }


def run_semantic_escalation_sidecar(
    payload: dict[str, Any], progress: Progress
) -> dict[str, Any]:
    from dataclasses import asdict

    from hydra_suite.detectkit.gui.project import open_project, save_project
    from hydra_suite.detectkit.jobs.semantic_escalation import (
        SemanticEscalationRequest,
        cache_confidence_floor,
        run_semantic_escalation,
    )

    project_dir = Path(_required_text(payload, "project_dir")).expanduser().resolve()
    project = open_project(project_dir)
    if project is None:
        raise RuntimeError(f"Could not open DetectKit project: {project_dir}")
    params = dict(payload.get("params") or {})
    params["max_instances"] = _semantic_instance_cap(payload)
    request = SemanticEscalationRequest(
        project=project,
        source_names=list(payload.get("source_names") or []),
        source_paths=list(payload.get("source_paths") or []),
        variant=_required_text(payload, "variant"),
        prompt=_required_text(payload, "prompt"),
        class_name=str(payload.get("class_name", "")),
        **params,
    )
    labeler = _semantic_labeler(
        payload, cache_confidence_floor(float(request.confidence))
    )
    result = run_semantic_escalation(
        request,
        labeler,
        overwrite=request.overwrite,
        progress=progress,
        on_mutated=lambda: save_project(project),
    )
    save_project(project)
    return {
        "semantic_result": asdict(result),
        "sources": [source.to_dict() for source in project.sources],
    }


def run_active_learning_sidecar(
    payload: dict[str, Any], progress: Progress
) -> dict[str, Any]:
    """Run active-learning model scoring outside the GUI process."""
    from hydra_suite.data.al.acquisition import AcquisitionWeights
    from hydra_suite.data.al.candidate_pool import CandidatePoolConfig
    from hydra_suite.detectkit.gui.project import open_project, save_project
    from hydra_suite.detectkit.jobs.al_worker import (
        ALDetectorSpec,
        ALRequest,
        run_active_learning,
    )

    project_dir = Path(_required_text(payload, "project_dir")).expanduser().resolve()
    project = open_project(project_dir)
    if project is None:
        raise RuntimeError(f"Could not open DetectKit project: {project_dir}")
    detector = ALDetectorSpec(**dict(payload.get("detector") or {}))
    raw_weights = payload.get("weights_override")
    request = ALRequest(
        input_kind=str(payload.get("input_kind", "project")),
        input_path=_required_text(payload, "input_path"),
        project=project,
        budget=max(1, int(payload.get("budget", 1))),
        preset=str(payload.get("preset", "balanced")),
        weights_override=(
            AcquisitionWeights(**dict(raw_weights)) if raw_weights is not None else None
        ),
        expected_count=max(0, int(payload.get("expected_count", 0))),
        detector=detector,
        diversity_window=max(0, int(payload.get("diversity_window", 30))),
        probabilistic=bool(payload.get("probabilistic", True)),
        candidate_pool=CandidatePoolConfig(**dict(payload.get("candidate_pool") or {})),
        base_conf=float(payload.get("base_conf", 0.25)),
        base_iou=float(payload.get("base_iou", 0.7)),
        export_level=str(payload.get("export_level", "obb")),
        export_levels=list(payload.get("export_levels") or ["obb"]),
        native_level=str(payload.get("native_level", "obb")),
    )
    result = run_active_learning(request, progress=progress)
    save_project(project)
    source = next(
        source for source in project.sources if source.path == result.source_path
    )
    return {
        "source_path": result.source_path,
        "n_picked": result.n_picked,
        "selected_frames": result.selected_frames,
        "source": source.to_dict(),
    }


OPERATIONS = {
    "active-learning": run_active_learning_sidecar,
    "dataset-inference": run_dataset_inference,
    "evaluation": run_evaluation,
    "semantic-escalation": run_semantic_escalation_sidecar,
    "semantic-calibration": run_semantic_calibration,
    "semantic-preview": run_semantic_preview,
}
