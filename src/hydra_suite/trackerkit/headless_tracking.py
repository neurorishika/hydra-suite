"""Headless (Qt-free) tracking session runner for the TrackerKit CLI.

Drives the Qt-free ``TrackingEngineCore`` on plain threads for the forward
(and optional backward) pass, then hands the raw trajectories to the Qt-free
``TrackingSessionCore`` for post-processing/merge/export. No PySide6 import
anywhere in this module — this is the executable definition of "Qt-free CLI".
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from typing import Any, Callable

import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.tracking.session import (
    SessionCallbacks,
    SessionResult,
    TrackingSessionCore,
)
from hydra_suite.core.tracking.worker import TrackingEngineCore
from hydra_suite.data.csv_writer import CSVWriterThread
from hydra_suite.trackerkit.cli_config import TrackerCliSession
from hydra_suite.trackerkit.tracking_cache import plan_tracking_cache

logger = logging.getLogger(__name__)


def build_tracking_csv_header(
    save_confidence_metrics: bool, identity_method: str = "none_disabled"
) -> list[str]:
    """Build the raw tracking CSV header used by the GUI path."""
    if save_confidence_metrics:
        base_cols = [
            "TrackID",
            "TrajectoryID",
            "Index",
            "X",
            "Y",
            "Theta",
            "FrameID",
            "State",
            "DetectionConfidence",
            "AssignmentConfidence",
            "PositionUncertainty",
            "DetectionID",
        ]
    else:
        base_cols = [
            "TrackID",
            "TrajectoryID",
            "Index",
            "X",
            "Y",
            "Theta",
            "FrameID",
            "State",
            "DetectionID",
        ]
    header = list(base_cols) + C.identity_realtime_columns()
    if str(identity_method).strip().lower() == "apriltags":
        header.extend(
            [
                "DetectedTagID",
                "DetectedTagLabel",
                "DetectedTagConf",
                "DetectedTagHamming",
            ]
        )
    return header


def _read_raw_trajectories(raw_csv_path: str) -> pd.DataFrame | None:
    """Load raw tracked rows written by the engine into a DataFrame.

    Mirrors the CSV that ``PostProcessWorker`` used to read; the session's
    post-processing consumes exactly this raw shape.
    """
    if not os.path.exists(raw_csv_path):
        return None
    return pd.read_csv(raw_csv_path)


def _run_engine_pass(
    session: TrackerCliSession,
    *,
    params: dict[str, Any],
    raw_csv_path: str,
    backward_mode: bool,
    detection_cache_path: str,
    use_cached_detections: bool,
    should_stop: Callable[[], bool],
) -> tuple[bool, list[float], pd.DataFrame | None, dict[str, str | None]]:
    """Run one tracking pass on a plain thread; return
    (success, fps, raw_df, analysis_cache_paths).

    ``analysis_cache_paths`` carries the pose/properties analysis-cache paths
    the engine wrote this pass (``individual_properties_cache_path``,
    ``detected_properties_cache_path``) -- the same values the GUI reads off
    the live worker (``TrackingOrchestrator._collect_worker_props_path``) and
    threads into the post-tracking ``paths`` dict. The rich export (and thus
    the offline identity fragment solver) is skipped when none of these are
    set, so the headless path must forward them for parity with the GUI.

    The engine writes its raw rows through ``CSVWriterThread`` (already a plain
    thread). We run ``run_tracking`` on a worker thread and join with a timeout
    so a SIGINT-driven ``should_stop`` can request a clean stop mid-pass.
    """
    direction = "backward" if backward_mode else "forward"
    csv_writer = CSVWriterThread(
        raw_csv_path,
        header=build_tracking_csv_header(
            session.save_confidence_metrics,
            identity_method=session.identity_method,
        ),
    )
    csv_writer.start()

    captured: dict[str, Any] = {"success": False, "fps_list": [], "finished": False}

    def _on_finished(success: bool, fps_list: list[Any], _full_traj: list[Any]) -> None:
        captured["success"] = bool(success)
        captured["fps_list"] = [f for f in (fps_list or []) if f and f > 0]
        captured["finished"] = True

    engine = TrackingEngineCore(
        session.video_path,
        csv_writer_thread=csv_writer,
        video_output_path=None,
        backward_mode=backward_mode,
        detection_cache_path=detection_cache_path,
        preview_mode=False,
        use_cached_detections=use_cached_detections,
        on_finished=_on_finished,
        on_progress=lambda pct, msg: logger.info(
            "[track %s] %d%% %s", direction, int(pct), msg
        ),
        on_warning=lambda title, msg: logger.warning("%s: %s", title, msg),
    )
    engine.set_parameters(dict(params))

    def _target() -> None:
        # TrackingEngineCore.run_tracking() does not guard every exception at the
        # top level (the old QThread wrapper did). Guard here so a crash still
        # produces finished(False) instead of a lost exception + a hung join.
        try:
            engine.run_tracking()
        except Exception:
            logger.exception("Tracking engine crashed during %s pass", direction)
            if not captured["finished"]:
                _on_finished(False, [], [])

    thread = threading.Thread(target=_target, name=f"tracking-engine-{direction}")
    thread.start()
    while thread.is_alive():
        if should_stop():
            engine.stop()
        thread.join(timeout=0.2)

    csv_writer.stop()
    csv_writer.join(timeout=10)

    # Parity with the GUI (TrackingOrchestrator._collect_worker_props_path):
    # surface the engine's analysis-cache paths so the post-tracking rich
    # export has a source and can run the offline identity fragment solver.
    analysis_cache_paths: dict[str, str | None] = {
        "individual_properties_cache_path": getattr(
            engine, "individual_properties_cache_path", None
        ),
        "detected_properties_cache_path": getattr(
            engine, "detected_properties_cache_path", None
        ),
    }

    if not captured["success"]:
        return False, [], None, analysis_cache_paths
    raw_df = _read_raw_trajectories(raw_csv_path)
    return True, list(captured["fps_list"]), raw_df, analysis_cache_paths


def _install_sigint_stop() -> tuple[threading.Event, Any, bool]:
    """Install a SIGINT handler that sets a stop event. Returns (event, prev, installed).

    ``signal.signal`` only works on the main thread; under a pytest worker it
    raises ``ValueError`` — in that case we skip installation and the session
    simply never self-cancels (fine for tests).
    """
    stop_event = threading.Event()
    previous = None
    installed = False
    try:
        previous = signal.getsignal(signal.SIGINT)

        def _handler(_signum, _frame):
            logger.warning(
                "SIGINT received - requesting clean stop of tracking session."
            )
            stop_event.set()

        signal.signal(signal.SIGINT, _handler)
        installed = True
    except (ValueError, OSError):
        installed = False
    return stop_event, previous, installed


def _restore_sigint(previous: Any, installed: bool) -> None:
    if installed and previous is not None:
        try:
            signal.signal(signal.SIGINT, previous)
        except (ValueError, OSError):
            pass


def run_headless_tracking_session(
    session: TrackerCliSession,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run a TrackerKit session without any Qt state.

    ``should_stop`` overrides the built-in SIGINT flag (used by tests / GUI reuse).
    """
    if should_stop is None:
        stop_event, previous_handler, installed = _install_sigint_stop()
        effective_should_stop: Callable[[], bool] = stop_event.is_set
    else:
        stop_event, previous_handler, installed = None, None, False
        effective_should_stop = should_stop

    try:
        cache_plan = plan_tracking_cache(
            session.video_path,
            params=dict(session.params),
            preferred_output_dir=os.path.dirname(session.raw_csv_path),
            use_cached_detections=session.use_cached_detections,
        )
        params = dict(session.params)
        params["INFERENCE_MODEL_ID"] = cache_plan.inference_model_id
        if cache_plan.engine_model_id:
            params["ENGINE_MODEL_ID"] = cache_plan.engine_model_id
        detection_cache_path = cache_plan.detection_cache_path

        raw_base, raw_ext = os.path.splitext(session.raw_csv_path)
        if session.enable_backward_tracking:
            forward_raw_csv = f"{raw_base}_forward{raw_ext}"
            backward_raw_csv = f"{raw_base}_backward{raw_ext}"
        else:
            forward_raw_csv = session.raw_csv_path
            backward_raw_csv = None

        forward_ok, fps_fwd, forward_df, forward_cache_paths = _run_engine_pass(
            session,
            params=params,
            raw_csv_path=forward_raw_csv,
            backward_mode=False,
            detection_cache_path=detection_cache_path,
            use_cached_detections=session.use_cached_detections,
            should_stop=effective_should_stop,
        )
        if not forward_ok:
            return {
                "success": False,
                "lines": [],
                "error": "An error occurred during forward tracking. Check logs for details.",
                "final_csv": None,
            }

        backward_df = None
        fps_bwd: list[float] = []
        if session.enable_backward_tracking:
            backward_ok, fps_bwd, backward_df, _ = _run_engine_pass(
                session,
                params=params,
                raw_csv_path=backward_raw_csv,
                backward_mode=True,
                detection_cache_path=detection_cache_path,
                use_cached_detections=False,
                should_stop=effective_should_stop,
            )
            if not backward_ok:
                return {
                    "success": False,
                    "lines": [],
                    "error": "An error occurred during backward tracking. Check logs for details.",
                    "final_csv": None,
                }

        callbacks = SessionCallbacks(
            progress=lambda pct, msg: logger.info("[post] %d%% %s", int(pct), msg),
            status=lambda msg: logger.info("[post] %s", msg),
            warning=lambda title, msg: logger.warning("%s: %s", title, msg),
            stage_changed=lambda name: logger.debug("[post] stage: %s", name),
            should_stop=effective_should_stop,
        )
        service = TrackingSessionCore(
            video_path=session.video_path,
            config=session.config,
            params=params,
            paths={
                "raw_csv_path": session.raw_csv_path,
                "final_csv_path": session.final_csv_path,
                "detection_cache_path": detection_cache_path,
                # Analysis-cache paths from the forward pass -- without these
                # the rich export finds no source and skips (so the offline
                # identity fragment solver never runs). Parity with the GUI's
                # post-tracking paths dict (orchestrators/tracking.py).
                "individual_properties_cache_path": forward_cache_paths.get(
                    "individual_properties_cache_path"
                ),
                "detected_properties_cache_path": forward_cache_paths.get(
                    "detected_properties_cache_path"
                ),
            },
            callbacks=callbacks,
        )
        result: SessionResult = service.run_post_tracking(
            forward_df, backward_trajectories=backward_df
        )

        if not result.success:
            return {
                "success": False,
                "lines": list(result.summary_lines or []),
                "error": result.error or "Tracker session failed.",
                "final_csv": result.final_csv_path,
            }

        lines = list(result.summary_lines or [])
        lines.insert(0, f"video={os.path.basename(session.video_path)}")
        fps_all = [f for f in (fps_fwd + fps_bwd) if f and f > 0]
        if fps_all:
            lines.append(f"avg_fps={sum(fps_all) / len(fps_all):.1f}")
        return {
            "success": True,
            "lines": lines,
            "error": None,
            "final_csv": result.final_csv_path,
        }
    finally:
        _restore_sigint(previous_handler, installed)
