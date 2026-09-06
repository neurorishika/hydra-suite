"""Production seam between a TrackerKit execution plan and the core tuner.

This module deliberately performs resolution before ``InferenceRunner`` exists.
It may inspect files and lightweight video/runtime metadata, but it never loads a
model framework or candidate model in the production process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from hydra_suite.core.inference.config import InferenceConfig
from hydra_suite.paths import get_data_dir
from hydra_suite.runtime.memory_profiles import (
    MemoryMeasurement,
    MemoryProfileStore,
    PressureSettings,
    ProfileIdentity,
    merge_records,
    profile_store_path,
    records_for,
)
from hydra_suite.runtime.resource_budget import (
    AcceleratorKind,
    ResourceObservation,
    ResourcePolicy,
)
from hydra_suite.utils.slice_geometry import tile_size_for_mode

from .candidates import AdmissionContext, CandidatePlanner, MemoryCostModel
from .coordinator import AutotuneCoordinator, AutotuneRequest, ResolveResult
from .fingerprint import (
    AcceleratorFingerprint,
    DetectorFingerprint,
    FrameFingerprint,
    PipelineFingerprint,
    SchemaFingerprint,
    SliceFingerprint,
    TuningProfileKey,
    WorkloadFingerprint,
    default_software_fingerprint,
    default_system_fingerprint,
    model_fingerprint,
)
from .models import (
    InferenceRuntimeOverlay,
    InferenceTuningProfile,
    InferenceTuningSettings,
)
from .search import TrialExecutor
from .store import InferenceTuningProfileStore

logger = logging.getLogger(__name__)

ExecutionMode = Literal["batch", "streaming", "realtime", "cache_replay"]


@dataclass(frozen=True, slots=True)
class TrackingRunContext:
    """Ephemeral inputs needed to tune one real tracking execution plan."""

    video_path: str | Path
    params: Mapping[str, Any]
    frame_width: int
    frame_height: int
    channels: int = 3
    pixel_format: str = "bgr8"
    decoder_mode: str = "opencv"
    execution_mode: ExecutionMode = "batch"
    start_frame: int = 0
    end_frame: int | None = None
    cached_fields: frozenset[str] = frozenset()
    contention_detected: bool = False
    thermal_throttled: bool = False
    should_cancel: Callable[[], bool] = lambda: False

    def __post_init__(self) -> None:
        if self.frame_width < 1 or self.frame_height < 1 or self.channels < 1:
            raise ValueError(
                "tracking autotune context requires positive frame geometry"
            )
        if self.execution_mode not in {
            "batch",
            "streaming",
            "realtime",
            "cache_replay",
        }:
            raise ValueError("unknown inference execution mode")


def _counts(value: Any, fallback: int) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        output = tuple(max(0, int(item)) for item in value[:512])
        if output:
            return output
    return (max(0, int(fallback)),)


def sample_detection_workload(
    cache_path: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    maximum_frames: int = 512,
) -> tuple[int, ...]:
    """Read bounded per-frame density from an existing cache without models.

    Both zero-detection and populated frames contribute to the signature. A
    missing, corrupt, or legacy cache without a complete written-frame index is
    treated as unavailable instead of inventing a workload identity.
    """

    if maximum_frames < 1:
        raise ValueError("maximum_frames must be positive")
    candidate = Path(cache_path)
    if candidate.is_dir():
        candidate = candidate / "detection.npz"
    try:
        from hydra_suite.core.inference.cache import open_detection_cache_reader

        reader = open_detection_cache_reader(candidate)
        if not reader.is_valid():
            return ()
        counts: dict[int, int] = {}
        for arrays in reader.iter_arrays():
            written = tuple(int(value) for value in arrays.get("written_frames", ()))
            if not written:
                # Modern chunks always retain written frames, including empty
                # ones. Refuse a partial density view when that contract is absent.
                return ()
            for frame in written:
                if frame < start_frame or (end_frame is not None and frame > end_frame):
                    continue
                if frame not in counts and len(counts) < maximum_frames:
                    counts[frame] = 0
            for raw_frame in arrays.get("frame_indices", ()):
                frame = int(raw_frame)
                if frame in counts:
                    counts[frame] += 1
            if len(counts) >= maximum_frames:
                break
        return tuple(counts[frame] for frame in sorted(counts))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return ()


def _active_slice(config: InferenceConfig):
    if config.obb is None:
        return None
    if config.obb.mode == "direct" and config.obb.direct is not None:
        return config.obb.direct.slice
    if config.obb.sequential is not None:
        return config.obb.sequential.stage1_slice
    return None


def _detector_imgsz(config: InferenceConfig, params: Mapping[str, Any]) -> int:
    if config.obb is not None and config.obb.mode == "sequential":
        configured = config.obb.sequential.detect_image_size  # type: ignore[union-attr]
        if configured:
            return int(configured)
    try:
        return max(64, min(4096, int(params.get("YOLO_IMAGE_SIZE", 640))))
    except (TypeError, ValueError):
        return 640


def _model_fingerprints(
    config: InferenceConfig, params: Mapping[str, Any], *, backend: str
):
    models = []
    detector_imgsz = _detector_imgsz(config, params)
    canonical_w, canonical_h = map(int, config.canonical.canvas_wh)
    trt_profile_id = "none"
    if backend == "tensorrt":
        from hydra_suite.core.inference.runtime_artifacts import (
            tensorrt_profile_fingerprint,
        )

        requested_profile_batch = int(
            params.get(
                "INFERENCE_AUTOTUNE_TENSORRT_PROFILE_BATCH_SIZE",
                config.detection_batch_size,
            )
        )
        trt_profile_id = tensorrt_profile_fingerprint(requested_profile_batch)
    if config.obb is not None:
        if config.obb.mode == "direct" and config.obb.direct is not None:
            models.append(
                model_fingerprint(
                    "detector.direct",
                    config.obb.direct.model_path,
                    input_size=(detector_imgsz, detector_imgsz),
                    tensorrt_profile_id=trt_profile_id,
                )
            )
        elif config.obb.sequential is not None:
            seq = config.obb.sequential
            models.extend(
                (
                    model_fingerprint(
                        "detector.stage1",
                        seq.detect_model_path,
                        input_size=(detector_imgsz, detector_imgsz),
                        tensorrt_profile_id=trt_profile_id,
                    ),
                    model_fingerprint(
                        "detector.stage2",
                        seq.obb_model_path,
                        input_size=(seq.stage2_image_size, seq.stage2_image_size),
                        crop_size=(canonical_w, canonical_h),
                        tensorrt_profile_id=trt_profile_id,
                    ),
                )
            )
    if config.headtail is not None:
        models.append(
            model_fingerprint(
                "headtail",
                config.headtail.model_path,
                crop_size=(canonical_w, canonical_h),
            )
        )
    for phase in config.cnn_phases:
        models.append(
            model_fingerprint(
                f"identity.{phase.label}",
                phase.model_path,
                crop_size=(canonical_w, canonical_h),
            )
        )
    if config.pose is not None:
        backend = getattr(config.pose, config.pose.backend, None)
        if backend is not None:
            models.append(
                model_fingerprint(
                    f"pose.{config.pose.backend}",
                    backend.model_path,
                    crop_size=(canonical_w, canonical_h),
                )
            )
    return tuple(sorted(models, key=lambda item: item.role))


def _slice_fingerprint(
    config: InferenceConfig,
    params: Mapping[str, Any],
    frame_width: int,
    frame_height: int,
) -> SliceFingerprint:
    slice_config = _active_slice(config)
    if slice_config is None or not slice_config.enabled:
        return SliceFingerprint(False, 0, 0, 0.0, 0.0, False, "none", "none")
    imgsz = _detector_imgsz(config, params)
    width, height = tile_size_for_mode(
        geometry_mode=slice_config.geometry_mode,
        imgsz=imgsz,
        reference_body_px=slice_config.reference_body_px,
        object_tile_fraction=slice_config.object_tile_fraction,
        slice_width=slice_config.slice_width,
        slice_height=slice_config.slice_height,
    )
    return SliceFingerprint(
        True,
        min(int(width), frame_width),
        min(int(height), frame_height),
        float(slice_config.overlap_width_ratio),
        float(slice_config.overlap_height_ratio),
        bool(slice_config.perform_standard_pred),
        "roi" if params.get("ROI_MASK") is not None else "none",
        (
            f"{float(slice_config.reference_body_px):.6g}:"
            f"{float(slice_config.object_tile_fraction):.6g}"
        ),
    )


def _enabled_stages(config: InferenceConfig) -> tuple[str, ...]:
    stages = ["detector"]
    if config.headtail is not None:
        stages.append("headtail")
    stages.extend(f"identity:{phase.label}" for phase in config.cnn_phases)
    if config.pose is not None:
        stages.append(f"pose:{config.pose.backend}")
    if config.apriltag.enabled:
        stages.append("apriltag")
    return tuple(stages)


def memory_profile_identity(key: TuningProfileKey, field_name: str) -> ProfileIdentity:
    """Map one coordinate to an exact-key runtime memory-profile identity."""

    return ProfileIdentity(
        operation=f"inference-autotune:{field_name}",
        model_identity=f"tuning:{key.digest}",
        backend=key.software.backend,
        device_identity=key.accelerator.device_uuid,
        precision=key.software.precision,
        task=field_name,
        tiling_mode="sliced" if key.slice.enabled else "none",
    )


def record_profile_memory_evidence(
    profile: InferenceTuningProfile,
    *,
    store: MemoryProfileStore | None = None,
) -> None:
    """Persist successful memory observations outside the throughput store."""

    memory_store = store or MemoryProfileStore(profile_store_path("inference"))
    if profile.key.accelerator.device_uuid == "mps-unified":
        accelerator_kind = AcceleratorKind.MPS
    elif profile.key.accelerator.total_vram_bytes > 0:
        accelerator_kind = AcceleratorKind.CUDA
    else:
        accelerator_kind = AcceleratorKind.CPU
    incoming = []
    for evidence in profile.candidates:
        if (
            evidence.failure_class is not None
            or evidence.equivalence is not None
            and not evidence.equivalence.passed
        ):
            continue
        for field_name in evidence.settings.field_names():
            value = evidence.settings.value_for(field_name)
            if value is None:
                continue
            incoming.append(
                MemoryMeasurement(
                    identity=memory_profile_identity(profile.key, field_name),
                    settings=PressureSettings(
                        input_width=profile.key.frame.width,
                        input_height=profile.key.frame.height,
                        batch_size=value,
                        pipeline_depth=evidence.settings.pipeline_depth,
                        tile_chunk=evidence.settings.slice_tile_batch_size or 1,
                        crop_batch=max(
                            (
                                candidate
                                for candidate in (
                                    evidence.settings.pose_batch_size,
                                    evidence.settings.headtail_batch_size,
                                    *(
                                        v
                                        for _label, v in evidence.settings.identity_batch_sizes
                                    ),
                                )
                                if candidate is not None
                            ),
                            default=1,
                        ),
                    ),
                    accelerator_kind=accelerator_kind,
                    host_peak_bytes=evidence.host_peak_bytes,
                    accelerator_allocated_peak_bytes=evidence.accelerator_peak_bytes,
                    accelerator_reserved_peak_bytes=evidence.accelerator_peak_bytes,
                    queue_high_water_bytes=evidence.queue_high_water_bytes,
                    observed_at_unix_ns=profile.last_validation_unix_ns,
                )
            )
    if incoming:
        memory_store.save(merge_records(memory_store.load(), incoming))


def _cost_model(
    config: InferenceConfig,
    params: Mapping[str, Any],
    models,
    *,
    frame_bytes: int,
    slice_tile_width: int,
    slice_tile_height: int,
) -> MemoryCostModel:
    # File bytes are a conservative lower bound for resident weights.  The
    # multiplier covers decoded tensors plus a second copy during preparation;
    # measured profiles, when present, replace this analytical floor.
    weight_bytes = 0
    for model in models:
        try:
            role = model.role
            if role == "detector.direct":
                path = config.obb.direct.model_path  # type: ignore[union-attr]
            elif role == "detector.stage1":
                path = config.obb.sequential.detect_model_path  # type: ignore[union-attr]
            elif role == "detector.stage2":
                path = config.obb.sequential.obb_model_path  # type: ignore[union-attr]
            elif role == "headtail":
                path = config.headtail.model_path  # type: ignore[union-attr]
            elif role.startswith("identity."):
                label = role.split(".", 1)[1]
                path = next(
                    item.model_path for item in config.cnn_phases if item.label == label
                )
            else:
                backend = getattr(config.pose, config.pose.backend)  # type: ignore[union-attr]
                path = backend.model_path
            artifact = Path(path)
            if artifact.is_file():
                weight_bytes += artifact.stat().st_size
            elif artifact.is_dir():
                weight_bytes += sum(
                    item.stat().st_size
                    for item in artifact.rglob("*")
                    if item.is_file()
                )
        except (OSError, StopIteration, AttributeError):
            pass
    detector_side = _detector_imgsz(config, params)
    canonical_w, canonical_h = map(int, config.canonical.canvas_wh)
    tile_pixels = max(1, slice_tile_width * slice_tile_height)
    return MemoryCostModel(
        fixed_host_bytes=max(256 * 1024**2, weight_bytes * 2),
        fixed_accelerator_bytes=weight_bytes * 4,
        detector_frame_host_bytes=frame_bytes,
        detector_frame_accelerator_bytes=detector_side**2 * 3 * 4 * 2,
        tile_accelerator_bytes=tile_pixels * 3 * 4 * 2,
        crop_accelerator_bytes=canonical_w * canonical_h * 3 * 4 * 2,
        queue_host_bytes=frame_bytes,
    )


def build_tracking_autotune_request(
    config: InferenceConfig,
    context: TrackingRunContext,
    *,
    observation: ResourceObservation,
    backend: str,
    device_identity: tuple[str, str, str, int],
    trial_executor: TrialExecutor | None = None,
    memory_records: tuple[MemoryMeasurement, ...] | None = None,
) -> AutotuneRequest:
    """Build the exact key and admission plan without loading any models."""

    params = context.params
    policy = config.inference_autotune
    baseline = InferenceTuningSettings.from_config(config)
    models = _model_fingerprints(config, params, backend=backend)
    slice_key = _slice_fingerprint(
        config, params, context.frame_width, context.frame_height
    )
    configured_targets = max(1, int(params.get("MAX_TARGETS", 1)))
    detection_counts = _counts(
        params.get("INFERENCE_AUTOTUNE_DETECTION_COUNTS"), configured_targets
    )
    crop_counts = _counts(
        params.get("INFERENCE_AUTOTUNE_CROP_COUNTS"), configured_targets
    )
    canonical_w, canonical_h = map(int, config.canonical.canvas_wh)
    workload = WorkloadFingerprint.from_counts(
        configured_targets,
        detection_counts,
        crop_counts,
        (f"{canonical_w}x{canonical_h}",),
    )
    device_uuid, device_model, capability, total_vram = device_identity
    detector_method = str(params.get("DETECTION_METHOD", config.detection_source))
    detector_task = (
        config.obb.direct.model_task
        if config.obb is not None
        and config.obb.mode == "direct"
        and config.obb.direct is not None
        else (
            config.obb.sequential.stage2_task
            if config.obb is not None and config.obb.sequential is not None
            else "background"
        )
    )
    target_classes = tuple(config.obb.target_classes) if config.obb is not None else ()
    confidence = config.obb.confidence_threshold if config.obb is not None else 0.0
    iou = config.obb.iou_threshold if config.obb is not None else 0.0
    maximum = (
        config.obb.max_detections if config.obb is not None else configured_targets
    )
    mode = config.obb.mode if config.obb is not None else "background"
    cache_mask = tuple(sorted(map(str, params.get("RESULT_CACHE_STAGE_MASK", ()))))
    key = TuningProfileKey(
        schema=SchemaFingerprint(),
        system=default_system_fingerprint(
            total_host_bytes=observation.total_host_bytes
        ),
        accelerator=AcceleratorFingerprint(
            device_uuid=str(device_uuid),
            model=str(device_model),
            compute_capability=str(capability),
            total_vram_bytes=int(total_vram),
        ),
        software=default_software_fingerprint(
            backend=backend,
            precision="fp16" if backend == "tensorrt" else "fp32",
            driver=str(params.get("INFERENCE_AUTOTUNE_DRIVER_VERSION", "unknown")),
            cuda=str(params.get("INFERENCE_AUTOTUNE_CUDA_VERSION", "unknown")),
            cudnn=str(params.get("INFERENCE_AUTOTUNE_CUDNN_VERSION", "unknown")),
        ),
        models=models,
        frame=FrameFingerprint(
            width=context.frame_width,
            height=context.frame_height,
            channels=context.channels,
            pixel_format=context.pixel_format,
            resize_factor=float(params.get("RESIZE_FACTOR", 1.0)),
            decoder_mode=context.decoder_mode,
        ),
        detector=DetectorFingerprint(
            detector_method,
            detector_task,
            target_classes,
            float(confidence),
            float(iou),
            int(maximum),
            mode,
        ),
        slice=slice_key,
        pipeline=PipelineFingerprint(
            _enabled_stages(config),
            "backward" if bool(params.get("BACKWARD_MODE", False)) else "forward",
            context.execution_mode,
            cache_mask,
        ),
        workload=workload,
    )

    frame_bytes = context.frame_width * context.frame_height * context.channels
    cached_fields = set(context.cached_fields)
    if context.execution_mode in {"streaming", "realtime"}:
        cached_fields.update(("detection_batch_size", "pipeline_depth"))
    if context.execution_mode == "cache_replay":
        cached_fields.update(baseline.field_names())
    maxima: list[tuple[str, int]] = [
        ("detection_batch_size", 64),
        ("pipeline_depth", 4),
    ]
    if baseline.slice_tile_batch_size is not None:
        maxima.append(("slice_tile_batch_size", 128))
    if baseline.pose_batch_size is not None:
        maxima.append(("pose_batch_size", 256))
    if baseline.headtail_batch_size is not None:
        maxima.append(("headtail_batch_size", 256))
    maxima.extend(
        (field, 256)
        for field in baseline.field_names()
        if field.startswith("identity_batch_size:")
    )
    available_memory_records = (
        MemoryProfileStore(profile_store_path("inference")).load()
        if memory_records is None
        else tuple(memory_records)
    )
    measured_records = tuple(
        (
            field_name,
            records_for(
                available_memory_records,
                memory_profile_identity(key, field_name),
            ),
        )
        for field_name in baseline.field_names()
    )
    planner = CandidatePlanner(
        AdmissionContext(
            observation=observation,
            frame_bytes=frame_bytes,
            crop_count_p95=max(crop_counts),
            hard_maxima=tuple(maxima),
            cost=_cost_model(
                config,
                params,
                models,
                frame_bytes=frame_bytes,
                slice_tile_width=slice_key.tile_width,
                slice_tile_height=slice_key.tile_height,
            ),
            policy=ResourcePolicy(),
            realtime=context.execution_mode == "realtime",
            coreml_obb=backend == "coreml",
            cached_fields=frozenset(cached_fields),
            measured_records=measured_records,
        )
    )
    eligible = True
    allow_cached_reuse = True
    eligibility_reason = None
    if context.execution_mode == "realtime":
        eligible = False
        allow_cached_reuse = False
        eligibility_reason = "realtime inference is not tunable"
    elif context.execution_mode == "cache_replay":
        eligible = False
        allow_cached_reuse = False
        eligibility_reason = "all inference stages are satisfied by reusable caches"
    elif context.contention_detected:
        eligible = False
        eligibility_reason = "another accelerator job is active"
    elif context.thermal_throttled:
        eligible = False
        eligibility_reason = "accelerator is thermally throttled"
    elif (
        policy.mode == "automatic"
        and observation.accelerator_kind is not AcceleratorKind.CUDA
    ):
        eligible = False
        allow_cached_reuse = False
        eligibility_reason = "automatic inference tuning is validated only for CUDA"
    else:
        admission = planner.admit(baseline)
        if not admission.admitted:
            eligible = False
            eligibility_reason = f"baseline admission failed: {admission.reason}"

    manual_fields = frozenset(policy.manual_fields) & frozenset(baseline.field_names())
    shares = params.get("INFERENCE_AUTOTUNE_STAGE_SHARES", {})
    stage_shares = (
        tuple((str(name), float(value)) for name, value in shares.items())
        if isinstance(shares, Mapping)
        else ()
    )
    return AutotuneRequest(
        key=key,
        baseline=baseline,
        planner=planner,
        mode=policy.mode,
        manual_fields=manual_fields,
        budget_seconds=policy.budget_seconds,
        singleflight_wait_seconds=policy.singleflight_wait_seconds,
        eligible=eligible,
        allow_cached_reuse=allow_cached_reuse,
        eligibility_reason=eligibility_reason,
        stage_shares=stage_shares,
        should_cancel=context.should_cancel,
    )


def resolve_tracking_inference_config(
    config: InferenceConfig,
    context: TrackingRunContext,
    *,
    observation: ResourceObservation,
    backend: str,
    device_identity: tuple[str, str, str, int],
    trial_executor: TrialExecutor | None = None,
    store: InferenceTuningProfileStore | None = None,
) -> tuple[InferenceConfig, InferenceRuntimeOverlay, ResolveResult]:
    """Resolve and apply a detached overlay before production model loading."""

    baseline = InferenceTuningSettings.from_config(config)
    if config.inference_autotune.mode == "off":
        overlay = InferenceRuntimeOverlay.baseline(
            baseline,
            status="disabled",
            reason="automatic inference tuning is disabled",
        )
        result = ResolveResult(overlay)
        return config, overlay, result
    try:
        request = build_tracking_autotune_request(
            config,
            context,
            observation=observation,
            backend=backend,
            device_identity=device_identity,
            trial_executor=trial_executor,
        )
        profile_store = store or InferenceTuningProfileStore(
            get_data_dir() / "inference_tuning_profiles"
        )
        result = AutotuneCoordinator(
            profile_store, trial_executor=trial_executor
        ).resolve(request)
        if result.profile is not None and result.overlay.status in {
            "calibrated",
            "recorded",
        }:
            record_profile_memory_evidence(result.profile)
        return result.overlay.apply(config), result.overlay, result
    except Exception as exc:
        logger.exception("Inference autotune pre-load resolution failed safely")
        overlay = InferenceRuntimeOverlay.baseline(
            baseline,
            status="fallback",
            reason=f"pre-load resolution failed: {type(exc).__name__}",
        )
        return config, overlay, ResolveResult(overlay)
