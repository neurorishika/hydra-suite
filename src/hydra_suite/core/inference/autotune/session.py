"""Qt-free entry points for inference calibration and profile lookup.

Split out of ``core/tracking/worker.py`` so calibration is an explicit act a
user triggers, not a side effect of starting a tracking run. Both entry points
build their fingerprint through ``build_autotune_context`` so a profile
produced by ``calibrate`` is found by ``lookup`` -- key agreement between the
two is the whole correctness argument for this feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from hydra_suite.core.inference.config import InferenceConfig


@dataclass(frozen=True, slots=True)
class AutotuneContext:
    """Everything both entry points need, derived exactly once."""

    config: InferenceConfig
    run_context: Any  # TrackingRunContext
    params: dict
    backend: str
    probe: Any  # RuntimeResourceProbe
    device_identity: tuple[str, str, str, int]


def build_autotune_context(
    config: InferenceConfig,
    params: dict,
    *,
    video_path: str,
    frame_width: int,
    frame_height: int,
    start_frame: int,
    end_frame: int,
    realtime: bool,
    cache_dir=None,
    use_cached_detections: bool = False,
    cache_read_only_replay: bool = False,
    should_cancel: Callable[[], bool] = lambda: False,
    status_callback: Callable[[str], None] = lambda _message: None,
) -> AutotuneContext:
    """Build the one context both ``lookup`` and ``calibrate`` fingerprint from."""

    from hydra_suite.core.inference.autotune.device import probe_runtime_resources
    from hydra_suite.core.inference.autotune.integration import (
        TrackingRunContext,
        sample_detection_workload,
    )
    from hydra_suite.runtime.resolver import RuntimeResolver, detect_platform
    from hydra_suite.runtime.resource_budget import AcceleratorKind

    platform_info = detect_platform()
    resolved = RuntimeResolver(config.runtime_tier, platform_info).resolve("obb")
    kind = {
        "cuda": AcceleratorKind.CUDA,
        "mps": AcceleratorKind.MPS,
        "cpu": AcceleratorKind.CPU,
    }[resolved.device]
    probe = probe_runtime_resources(kind)
    ephemeral_params = dict(params)
    ephemeral_params["INFERENCE_AUTOTUNE_DRIVER_VERSION"] = probe.driver_version
    prior_counts = ()
    if cache_dir:
        prior_counts = sample_detection_workload(
            cache_dir, start_frame=start_frame, end_frame=end_frame
        )
        if (
            prior_counts
            and "INFERENCE_AUTOTUNE_DETECTION_COUNTS" not in ephemeral_params
        ):
            ephemeral_params["INFERENCE_AUTOTUNE_DETECTION_COUNTS"] = prior_counts
            ephemeral_params["INFERENCE_AUTOTUNE_CROP_COUNTS"] = prior_counts
    cached_fields = frozenset()
    if use_cached_detections and prior_counts:
        cached_fields = frozenset(("detection_batch_size", "slice_tile_batch_size"))
        ephemeral_params["RESULT_CACHE_STAGE_MASK"] = ("detector",)
    run_context = TrackingRunContext(
        video_path=video_path,
        params=ephemeral_params,
        frame_width=max(1, int(frame_width)),
        frame_height=max(1, int(frame_height)),
        channels=3,
        decoder_mode="nvdec" if config.runtime_tier == "gpu_fast" else "opencv",
        execution_mode=(
            "cache_replay"
            if cache_read_only_replay
            else ("realtime" if realtime else "batch")
        ),
        start_frame=start_frame,
        end_frame=end_frame,
        cached_fields=cached_fields,
        contention_detected=probe.contention_detected,
        thermal_throttled=probe.thermal_throttled,
        should_cancel=should_cancel,
        status_callback=status_callback,
    )
    return AutotuneContext(
        config=config,
        run_context=run_context,
        params=ephemeral_params,
        backend=resolved.backend,
        probe=probe,
        device_identity=(
            probe.device_uuid,
            probe.device_model,
            probe.compute_capability,
            int(probe.observation.total_accelerator_bytes or 0),
        ),
    )


def lookup(ctx: AutotuneContext):
    """Find and apply a validated profile. Never measures, never claims a lock."""

    from hydra_suite.core.inference.autotune.integration import (
        resolve_tracking_inference_config,
    )

    return resolve_tracking_inference_config(
        ctx.config,
        ctx.run_context,
        observation=ctx.probe.observation,
        backend=ctx.backend,
        device_identity=ctx.device_identity,
        trial_executor=None,
    )


def calibrate(ctx: AutotuneContext, *, budget_seconds: float):
    """Measure and persist a profile for this exact context."""

    from hydra_suite.core.inference.autotune.integration import (
        build_tracking_autotune_request,
        resolve_tracking_inference_config,
    )
    from hydra_suite.core.inference.autotune.sidecar import (
        ARTIFACT_BUILD_ALLOWANCE_SECONDS,
        ContainedTrialExecutor,
        SidecarTrialSpec,
    )

    preflight = build_tracking_autotune_request(
        ctx.config,
        ctx.run_context,
        observation=ctx.probe.observation,
        backend=ctx.backend,
        device_identity=ctx.device_identity,
    )
    artifact_batch_size = max(
        (
            value
            for field in ("detection_batch_size", "slice_tile_batch_size")
            for value in preflight.planner.values_for(field, preflight.baseline)
        ),
        default=preflight.baseline.detection_batch_size,
    )
    if ctx.backend == "tensorrt":
        ctx.params["INFERENCE_AUTOTUNE_TENSORRT_PROFILE_BATCH_SIZE"] = (
            artifact_batch_size
        )
    executor = ContainedTrialExecutor(
        SidecarTrialSpec(
            video_path=ctx.run_context.video_path,
            params=ctx.params,
            observation=ctx.probe.observation,
            resource_probe=ctx.probe,
            start_frame=ctx.run_context.start_frame,
            end_frame=ctx.run_context.end_frame,
            budget_seconds=budget_seconds,
            runtime_artifact_batch_size=(
                artifact_batch_size if ctx.backend == "tensorrt" else None
            ),
            # B3: the accelerated tiers build their engine INSIDE the child.
            # The spec's own figure for a cold TensorRT profile build is
            # 255-310 s, which alone exceeds the 120 s per-trial measurement
            # cap -- so without this the baseline trial could never complete
            # and the tuner could not start on TensorRT at all. Grant the
            # build its own window; torch tiers keep the default 0.
            artifact_build_allowance_seconds=(
                ARTIFACT_BUILD_ALLOWANCE_SECONDS
                if ctx.backend in ("tensorrt", "coreml")
                else 0.0
            ),
        )
    )
    return resolve_tracking_inference_config(
        ctx.config,
        ctx.run_context,
        observation=ctx.probe.observation,
        backend=ctx.backend,
        device_identity=ctx.device_identity,
        trial_executor=executor,
    )
