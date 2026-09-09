"""Qt-free entry points for inference calibration and profile lookup.

Split out of ``core/tracking/worker.py`` so calibration is an explicit act a
user triggers, not a side effect of starting a tracking run. Both entry points
build their fingerprint through ``build_autotune_context`` so a profile
produced by ``calibrate`` is found by ``lookup`` -- key agreement between the
two is the whole correctness argument for this feature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from hydra_suite.core.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AutotuneContext:
    """Everything both entry points need, derived exactly once."""

    config: InferenceConfig
    run_context: Any  # TrackingRunContext
    params: dict
    backend: str
    probe: Any  # RuntimeResourceProbe
    device_identity: tuple[str, str, str, int]
    # Derived ONCE in ``build_autotune_context`` and consumed by both entry
    # points. Deliberately has NO default: a hand-built context that skipped
    # the derivation would make ``lookup`` and ``calibrate`` agree at the
    # merely-configured batch size, silently keying every TensorRT profile
    # under the wrong ``tensorrt_profile_id`` -- the exact divergence this
    # field exists to make impossible.
    artifact_batch_size: int


TENSORRT_PROFILE_BATCH_PARAM = "INFERENCE_AUTOTUNE_TENSORRT_PROFILE_BATCH_SIZE"


def derive_artifact_batch_size(
    config: InferenceConfig,
    run_context,
    *,
    probe,
    backend: str,
    device_identity: tuple[str, str, str, int],
) -> int:
    """Largest batch any candidate could ask a runtime artifact to serve.

    Uses ``static_max_for`` (not the live-memory-filtered ``values_for``) so
    the value cannot drift with free memory. ``static_max_for`` returns
    ``None`` for a field the project does not have -- ``slice_tile_batch_size``
    is ``None`` on every non-SAHI project (``InferenceTuningSettings
    .from_config``) -- so ``None`` must be filtered out before ``max``, or
    every non-sliced project raises ``TypeError``.
    """

    from hydra_suite.core.inference.autotune.integration import (
        build_tracking_autotune_request,
    )

    preflight = build_tracking_autotune_request(
        config,
        run_context,
        observation=probe.observation,
        backend=backend,
        device_identity=device_identity,
    )
    sizes = [
        value
        for value in (
            preflight.planner.static_max_for(field, preflight.baseline)
            for field in ("detection_batch_size", "slice_tile_batch_size")
        )
        if value is not None
    ]
    return max(sizes, default=preflight.baseline.detection_batch_size)


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
    device_identity = (
        probe.device_uuid,
        probe.device_model,
        probe.compute_capability,
        int(probe.observation.total_accelerator_bytes or 0),
    )
    # Derived HERE, once, and the TensorRT param injected HERE, once -- before
    # anybody builds the real request. When ``calibrate`` derived it privately
    # and ``lookup`` did not, the two keyed every TensorRT profile differently
    # (calibrate on the candidate maximum, lookup on the configured batch) and
    # no profile was ever findable on gpu_fast/CUDA. ``ephemeral_params`` is
    # the same dict object ``run_context`` holds, so this mutation is visible
    # through both.
    artifact_batch_size = derive_artifact_batch_size(
        config,
        run_context,
        probe=probe,
        backend=resolved.backend,
        device_identity=device_identity,
    )
    if resolved.backend == "tensorrt":
        ephemeral_params[TENSORRT_PROFILE_BATCH_PARAM] = artifact_batch_size
    return AutotuneContext(
        config=config,
        run_context=run_context,
        params=ephemeral_params,
        backend=resolved.backend,
        probe=probe,
        device_identity=device_identity,
        artifact_batch_size=artifact_batch_size,
    )


def calibration_key_digest(ctx: AutotuneContext) -> str:
    """The ONE profile-key digest this context calibrates and looks up under.

    Derives nothing of its own: ``build_autotune_context`` already folded
    ``artifact_batch_size`` into ``ctx.params`` (on TensorRT), so this is
    simply the key ``lookup`` and ``calibrate`` both build from. Any
    derivation repeated here could drift from theirs -- and did.
    """

    from hydra_suite.core.inference.autotune.integration import (
        build_tracking_autotune_request,
    )

    request = build_tracking_autotune_request(
        ctx.config,
        ctx.run_context,
        observation=ctx.probe.observation,
        backend=ctx.backend,
        device_identity=ctx.device_identity,
    )
    return request.key.digest


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
        resolve_tracking_inference_config,
    )
    from hydra_suite.core.inference.autotune.sidecar import (
        ARTIFACT_BUILD_ALLOWANCE_SECONDS,
        ContainedTrialExecutor,
        SidecarTrialSpec,
    )

    # Derived and injected by ``build_autotune_context``, never re-derived
    # here: that duplication is what let calibrate and lookup key TensorRT
    # profiles differently.
    artifact_batch_size = ctx.artifact_batch_size
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
    effective, overlay, result = resolve_tracking_inference_config(
        ctx.config,
        ctx.run_context,
        observation=ctx.probe.observation,
        backend=ctx.backend,
        device_identity=ctx.device_identity,
        trial_executor=executor,
    )
    _close_density_bridge(result)
    return effective, overlay, result


def _close_density_bridge(result) -> None:
    """Write the measured-key twin of an estimate-keyed profile immediately.

    The store already re-keys an estimated-density profile on the first real
    production sample (``store.observe_production_throughput``), deliberately
    leaving the estimated record in place so the next brand-new video still
    gets a warm start. Calibration has already measured real density in its
    own trials, so there is no reason to make the user pay a whole extra
    tracking run before a cached run can find anything.

    Every step -- including reading ``profile.key``/``profile.candidates`` --
    is inside the ``try``: a bridge-closing failure must never propagate out
    of a successful ``calibrate()`` call and destroy the minutes-long
    calibration the user just waited for.
    """

    try:
        profile = getattr(result, "profile", None)
        if profile is None or not profile.key.workload.density_is_estimated:
            return
        candidate = _authoritative_candidate(profile)
        counts = candidate.detection_counts if candidate is not None else ()
        if not counts:
            logger.warning(
                "Calibration density bridge not closed: the winning candidate "
                "(profile_id=%s) carries no measured detection counts",
                getattr(profile, "profile_id", "?"),
            )
            return
        from hydra_suite.core.inference.autotune.store import (
            InferenceTuningProfileStore,
        )

        InferenceTuningProfileStore().observe_production_throughput(
            profile.profile_id,
            candidate.median_throughput,
            detection_counts=counts,
            crop_counts=counts,
        )
    except Exception:
        logger.warning("Could not close the calibration density bridge", exc_info=True)


def _authoritative_candidate(profile):
    """The most authoritative evidence for the winning settings vector.

    ``search.py`` records the SAME settings vector at multiple phases
    (``baseline``, ``stage``, ``full``, ``final_validation``) -- e.g. every
    run where the winner equals the baseline. Two other call sites already
    treat phase as load-bearing for exactly this reason:
    ``coordinator.py``'s ``_reuse`` trusts only a ``full`` measurement to
    authorize a winner, and ``store.py``'s regression check filters to
    ``{"full", "final_validation", "baseline"}`` and takes the LAST (most
    recent, most authoritative) match. This mirrors that pattern instead of
    taking the first candidate that happens to match -- which, for a
    baseline-wins outcome, would silently be the LEAST validated evidence.
    """

    matches = [c for c in profile.candidates if c.settings == profile.selected]
    authoritative = [c for c in matches if c.phase in {"final_validation", "full"}]
    if authoritative:
        return authoritative[-1]
    return matches[-1] if matches else None
