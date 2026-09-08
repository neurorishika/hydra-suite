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
from pathlib import Path

logger = logging.getLogger(__name__)

# Statuses that mean "a validated profile now exists for this key" -- the
# success set for the CLI's exit code. Everything else (fallback, cancelled,
# deferred_due_to_contention, unavailable, singleflight_wait_timeout, ...)
# means the user does not yet have a usable profile.
_SUCCESS_STATUSES = frozenset({"calibrated", "cache_hit", "cache_hit_after_wait"})


def run_calibrate_cli(
    video_path: str,
    *,
    config_path: str | None = None,
    budget_seconds: float,
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
    from hydra_suite.trackerkit.cli_config import load_tracker_cli_session
    from hydra_suite.utils.video_artifacts import build_inference_cache_dir

    video = str(video_path).strip()
    if not video:
        raise ValueError("A video path is required.")
    if not Path(video).is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if config_path and not Path(config_path).is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    session = load_tracker_cli_session(video, config_path=config_path)
    params = session.params
    probe = session.video_probe

    inference_cfg = build_inference_config_from_params(params)

    total_frames = int(probe.total_frames or 1)
    start_frame = int(params.get("START_FRAME", 0) or 0)
    end_frame_param = params.get("END_FRAME")
    end_frame = (
        int(end_frame_param)
        if end_frame_param is not None
        else max(0, total_frames - 1)
    )

    cache_dir = build_inference_cache_dir(video)

    logger.info(
        "Calibrating inference throughput for %s (budget=%.0fs, frames=%s-%s)",
        video,
        budget_seconds,
        start_frame,
        end_frame,
    )

    ctx = autotune_session.build_autotune_context(
        inference_cfg,
        params,
        video_path=video,
        frame_width=int(probe.width or 1),
        frame_height=int(probe.height or 1),
        start_frame=start_frame,
        end_frame=end_frame,
        realtime=bool(params.get("TRACKING_REALTIME_MODE", False)),
        cache_dir=cache_dir,
        use_cached_detections=bool(session.use_cached_detections),
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

    return 0 if status in _SUCCESS_STATUSES else 1
