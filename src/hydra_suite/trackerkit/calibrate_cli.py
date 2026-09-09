"""Headless entry point for ``trackerkit calibrate`` (Qt-free).

The CLI equivalent of the GUI's Calibrate button: build the exact same
engine params ``track`` would build (via the shared
``load_tracker_cli_session``/``build_engine_params`` path), fingerprint them
through ``build_autotune_context``, and hand them to
``core.inference.autotune.session.calibrate``. Using the same builder as
``track`` is the whole correctness argument -- the key a calibration run
writes a profile under must be the key a later ``track`` run looks it up
under, and that only holds if both paths build params identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Statuses that mean "a validated profile now exists for this key" -- the
# success set for the CLI's exit code. Everything else (fallback, cancelled,
# deferred_due_to_contention, unavailable, singleflight_wait_timeout, ...)
# means the user does not yet have a usable profile.
_SUCCESS_STATUSES = frozenset({"calibrated", "cache_hit", "cache_hit_after_wait"})


@dataclass(frozen=True, slots=True)
class CalibrationContextInputs:
    """The values ``build_autotune_context`` needs beyond ``config``/``params``.

    Extracted to its own function (``derive_context_inputs``) so a test can
    call the REAL production derivation directly, rather than reimplementing
    it -- the failure mode this guards (Task 10 review Finding 1) is exactly
    two independently-written derivations silently drifting apart.
    """

    cache_dir: Path
    start_frame: int
    end_frame: int
    realtime: bool
    use_cached_detections: bool


def derive_context_inputs(
    video_path: str,
    params: dict[str, Any],
    probe: Any,
    *,
    use_cached_detections: bool,
) -> CalibrationContextInputs:
    """Derive the non-params/config inputs to ``build_autotune_context``.

    Mirrors ``TrackingWorker.run_tracking``'s equivalent block exactly:
    ``cache_dir`` via the same ``build_inference_cache_dir`` fallback
    ``_resolve_cache_dir`` uses when no explicit cache dir is set (true on
    every CLI ``track`` invocation); frame bounds via the SAME shared
    ``clamp_frame_range`` helper ``worker.py`` calls; ``realtime`` reduced
    the same way ``effective_realtime_tracking_mode`` is on a plain (non-
    preview, non-backward) run.
    """
    from hydra_suite.core.inference.config import clamp_frame_range
    from hydra_suite.utils.video_artifacts import build_inference_cache_dir

    start_frame, end_frame = clamp_frame_range(
        params.get("START_FRAME", 0),
        params.get("END_FRAME", None),
        probe.total_frames,
    )
    return CalibrationContextInputs(
        cache_dir=build_inference_cache_dir(video_path),
        start_frame=start_frame,
        end_frame=end_frame,
        realtime=bool(params.get("TRACKING_REALTIME_MODE", False)),
        use_cached_detections=bool(use_cached_detections),
    )


def run_calibrate_cli(
    video_path: str,
    *,
    config_path: str | None = None,
    budget_seconds: float,
    inference_autotune_manual: Sequence[str] | None = None,
) -> int:
    """Measure and persist a validated inference profile for one video/config.

    Returns 0 when the resulting status means a profile now exists
    (``calibrated``/``cache_hit``/``cache_hit_after_wait``), non-zero
    otherwise. Always logs the coordinator's ``reason`` so a user can tell
    "already had one" from "declined because contended" from "budget
    expired".
    """
    from hydra_suite.core.inference.autotune import session as autotune_session
    from hydra_suite.core.inference.config import build_inference_config_from_params
    from hydra_suite.trackerkit.cli_config import (
        apply_inference_autotune_override,
        load_tracker_cli_config,
        load_tracker_cli_session,
    )

    video = str(video_path).strip()
    if not video:
        raise ValueError("A video path is required.")
    if not Path(video).is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if config_path and not Path(config_path).is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # I4: manual fields feed ``compute_baseline_digest`` ->
    # ``key.baseline_digest``, so they are part of the profile key. ``track``
    # merges them into its config through ``apply_inference_autotune_override``
    # (batch_plan.py); calibrate must merge them the SAME way through the SAME
    # helper, or the key it writes under is not the key ``track
    # --inference-autotune-manual ...`` looks up.
    if inference_autotune_manual:
        config_data = apply_inference_autotune_override(
            load_tracker_cli_config(config_path),
            manual_fields=tuple(inference_autotune_manual),
        )
        session = load_tracker_cli_session(
            video, config_path=config_path, config_data=config_data
        )
    else:
        session = load_tracker_cli_session(video, config_path=config_path)
    params = session.params
    probe = session.video_probe

    inference_cfg = build_inference_config_from_params(params)
    inputs = derive_context_inputs(
        video, params, probe, use_cached_detections=session.use_cached_detections
    )

    logger.info(
        "Calibrating inference throughput for %s (budget=%.0fs, frames=%s-%s)",
        video,
        budget_seconds,
        inputs.start_frame,
        inputs.end_frame,
    )

    ctx = autotune_session.build_autotune_context(
        inference_cfg,
        params,
        video_path=video,
        frame_width=int(probe.width or 1),
        frame_height=int(probe.height or 1),
        start_frame=inputs.start_frame,
        end_frame=inputs.end_frame,
        realtime=inputs.realtime,
        cache_dir=inputs.cache_dir,
        use_cached_detections=inputs.use_cached_detections,
        cache_read_only_replay=False,
        status_callback=lambda message: logger.info("Calibration: %s", message),
    )

    _effective, overlay, _result = autotune_session.calibrate(
        ctx, budget_seconds=float(budget_seconds)
    )

    if overlay is None:
        logger.error("Calibration produced no result.")
        print("Calibration failed: no result produced.")
        return 1

    status = str(overlay.status).strip().lower()
    logger.info(
        "Calibration finished: status=%s profile=%s reason=%s requested=%s "
        "admitted=%s effective=%s",
        overlay.status,
        overlay.profile_id,
        overlay.reason,
        overlay.requested.to_dict(),
        overlay.admitted.to_dict(),
        overlay.effective.to_dict(),
    )
    print(
        f"Calibration status: {overlay.status} "
        f"(profile={overlay.profile_id}, reason={overlay.reason})"
    )
    print(f"Effective vector: {overlay.effective.to_dict()}")
    # I5 (GUI parity): the detection-cache mask is part of the pipeline
    # fingerprint, so this profile is only findable by runs in the same cache
    # mode. Say which one, so a later miss is explicable.
    if ctx.run_context.cached_fields:
        print(
            "Cache mode: covers runs that REUSE the detection cache "
            "(cached detections on)."
        )
    else:
        print(
            "Cache mode: covers runs with NO detection cache. With "
            "detection-cache reuse enabled, calibrate again after this "
            "video's first tracking run."
        )

    return 0 if status in _SUCCESS_STATUSES else 1
