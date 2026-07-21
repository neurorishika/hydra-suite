"""Headless tracking session runner shared by CLI-oriented flows."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

from hydra_suite.core.post.processing import interpolate_trajectories
from hydra_suite.data.csv_writer import CSVWriterThread
from hydra_suite.trackerkit.cli_config import TrackerCliSession
from hydra_suite.trackerkit.gui.workers.merge_worker import MergeWorker
from hydra_suite.trackerkit.gui.workers.postprocess_worker import PostProcessWorker
from hydra_suite.trackerkit.gui.workers.tracking_worker import TrackingWorker
from hydra_suite.trackerkit.tracking_cache import plan_tracking_cache


def ensure_headless_qt_application():
    """Return a Qt application object suitable for headless worker/event-loop usage."""

    app = QCoreApplication.instance()
    if app is not None:
        return app

    app = QCoreApplication([])
    app.setApplicationName("TrackerKit CLI")
    return app


def build_tracking_csv_header(
    save_confidence_metrics: bool, identity_method: str = "none_disabled"
) -> list[str]:
    """Build the raw tracking CSV header used by the GUI path."""
    if save_confidence_metrics:
        header = [
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
            "IdentityAssignedID",
            "IdentityAssignedLabel",
            "IdentityAssignedConfidence",
            "IdentityPosteriorMargin",
            "IdentityEntropy",
            "IdentityCommitted",
            "IdentityEvidenceSources",
            "IdentityConflictFlag",
            "IdentitySlotLockLabel",
        ]
    else:
        header = [
            "TrackID",
            "TrajectoryID",
            "Index",
            "X",
            "Y",
            "Theta",
            "FrameID",
            "State",
            "DetectionID",
            "IdentityAssignedID",
            "IdentityAssignedLabel",
            "IdentityAssignedConfidence",
            "IdentityPosteriorMargin",
            "IdentityEntropy",
            "IdentityCommitted",
            "IdentityEvidenceSources",
            "IdentityConflictFlag",
            "IdentitySlotLockLabel",
        ]
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


def save_trajectories_to_csv(trajectories, output_path: str) -> bool:
    """Persist post-processed trajectories in the same shape as the GUI path."""
    if trajectories is None:
        return False
    if not isinstance(trajectories, pd.DataFrame):
        raise TypeError("Expected post-processed trajectories as a pandas DataFrame.")
    if trajectories.empty:
        return False

    df_to_save = trajectories.copy()
    for column in ["X", "Y", "FrameID"]:
        if column in df_to_save.columns:
            df_to_save[column] = pd.to_numeric(df_to_save[column], errors="coerce")
            df_to_save[column] = df_to_save[column].round().astype("Int64")

    df_to_save = df_to_save.drop(
        columns=[
            column for column in ["TrackID", "Index"] if column in df_to_save.columns
        ],
        errors="ignore",
    )
    base_columns = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
    ordered_columns = base_columns + [
        column for column in df_to_save.columns if column not in base_columns
    ]
    df_to_save[ordered_columns].to_csv(output_path, index=False)
    return True


def _run_tracking_worker(
    session: TrackerCliSession,
    *,
    params: dict[str, Any],
    raw_csv_path: str,
    backward_mode: bool,
    detection_cache_path: str,
    use_cached_detections: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {"success": False, "fps_list": []}
    loop = QEventLoop()
    csv_writer = CSVWriterThread(
        raw_csv_path,
        header=build_tracking_csv_header(
            session.save_confidence_metrics,
            identity_method=session.identity_method,
        ),
    )
    csv_writer.start()

    worker = TrackingWorker(
        session.video_path,
        csv_writer_thread=csv_writer,
        video_output_path=None,
        backward_mode=backward_mode,
        detection_cache_path=detection_cache_path,
        preview_mode=False,
        use_cached_detections=use_cached_detections,
    )
    worker.set_parameters(dict(params))

    def _on_finished(success: bool, fps_list: list[Any], _full_traj: list[Any]) -> None:
        result["success"] = bool(success)
        result["fps_list"] = list(fps_list or [])
        QTimer.singleShot(0, loop.quit)

    worker.finished_signal.connect(_on_finished)
    worker.start()
    loop.exec()
    worker.wait()

    csv_writer.stop()
    csv_writer.join(timeout=10)
    return result


def _run_postprocess_worker(csv_path: str, params: dict[str, Any], clean: bool):
    result: dict[str, Any] = {"error": None, "trajectories": None}
    loop = QEventLoop()
    worker = PostProcessWorker(csv_path, dict(params), clean=clean)

    def _on_finished(processed_trajectories) -> None:
        result["trajectories"] = processed_trajectories
        QTimer.singleShot(0, loop.quit)

    def _on_error(message: str) -> None:
        result["error"] = message
        QTimer.singleShot(0, loop.quit)

    worker.finished_signal.connect(_on_finished)
    worker.error_signal.connect(_on_error)
    worker.start()
    loop.exec()
    worker.wait()

    if result["error"]:
        raise RuntimeError(
            f"Error during trajectory post-processing: {result['error']}"
        )
    if result["trajectories"] is None:
        raise RuntimeError("Post-processing did not produce trajectories.")
    return result["trajectories"]


def _run_merge_worker(
    forward_trajs,
    backward_trajs,
    *,
    total_frames: int,
    params: dict[str, Any],
    resize_factor: float,
    interp_method: str,
    max_gap: int,
    heading_flip_max_burst: int,
):
    result: dict[str, Any] = {"error": None, "trajectories": None}
    loop = QEventLoop()
    worker = MergeWorker(
        forward_trajs,
        backward_trajs,
        total_frames,
        dict(params),
        resize_factor,
        interp_method,
        max_gap,
        tag_cache_path=None,
        heading_flip_max_burst=heading_flip_max_burst,
        directed_heading_posthoc=bool(
            params.get("DIRECTED_ORIENT_POSTHOC_CONSISTENCY", False)
        ),
        enable_profiling=bool(params.get("ENABLE_PROFILING", False)),
        profile_export_path=None,
    )

    def _on_finished(merged_trajectories) -> None:
        result["trajectories"] = merged_trajectories
        QTimer.singleShot(0, loop.quit)

    def _on_error(message: str) -> None:
        result["error"] = message
        QTimer.singleShot(0, loop.quit)

    worker.finished_signal.connect(_on_finished)
    worker.error_signal.connect(_on_error)
    worker.start()
    loop.exec()
    worker.wait()

    if result["error"]:
        raise RuntimeError(f"Error during trajectory merging: {result['error']}")
    if result["trajectories"] is None:
        raise RuntimeError("Trajectory merge did not produce output.")
    return result["trajectories"]


def _run_forward_only(
    session: TrackerCliSession, *, params: dict[str, Any], detection_cache_path: str
):
    tracking_result = _run_tracking_worker(
        session,
        params=params,
        raw_csv_path=session.raw_csv_path,
        backward_mode=False,
        detection_cache_path=detection_cache_path,
        use_cached_detections=session.use_cached_detections,
    )
    if not tracking_result.get("success"):
        return {
            "success": False,
            "error": "An error occurred during tracking. Check logs for details.",
        }

    processed = _run_postprocess_worker(
        session.raw_csv_path,
        params,
        clean=session.enable_postprocessing,
    )
    interp_method = str(session.interpolation_method or "None").strip().lower()
    if interp_method != "none":
        max_gap = max(1, round(session.interpolation_max_gap_seconds * params["FPS"]))
        processed = interpolate_trajectories(
            processed,
            method=interp_method,
            max_gap=max_gap,
            heading_flip_max_burst=session.heading_flip_max_burst,
            directed_heading_posthoc=bool(
                params.get("DIRECTED_ORIENT_POSTHOC_CONSISTENCY", False)
            ),
        )

    save_trajectories_to_csv(processed, session.final_csv_path)
    fps_list = [fps for fps in tracking_result.get("fps_list", []) if fps and fps > 0]
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else None
    lines = [
        f"video={os.path.basename(session.video_path)}",
        f"raw_csv={session.raw_csv_path}",
        f"final_csv={session.final_csv_path}",
        f"detection_cache={detection_cache_path}",
    ]
    if avg_fps is not None:
        lines.append(f"avg_fps={avg_fps:.1f}")
    return {"success": True, "lines": lines}


def _run_forward_backward(
    session: TrackerCliSession, *, params: dict[str, Any], detection_cache_path: str
):
    raw_base, raw_ext = os.path.splitext(session.raw_csv_path)
    forward_raw_csv = f"{raw_base}_forward{raw_ext}"
    backward_raw_csv = f"{raw_base}_backward{raw_ext}"
    forward_processed_csv = f"{raw_base}_forward_processed{raw_ext}"
    backward_processed_csv = f"{raw_base}_backward_processed{raw_ext}"
    final_csv_path = f"{raw_base}_final{raw_ext}"

    forward_result = _run_tracking_worker(
        session,
        params=params,
        raw_csv_path=forward_raw_csv,
        backward_mode=False,
        detection_cache_path=detection_cache_path,
        use_cached_detections=session.use_cached_detections,
    )
    if not forward_result.get("success"):
        return {
            "success": False,
            "error": "An error occurred during forward tracking. Check logs for details.",
        }

    forward_processed = _run_postprocess_worker(
        forward_raw_csv,
        params,
        clean=session.enable_postprocessing,
    )
    save_trajectories_to_csv(forward_processed, forward_processed_csv)

    backward_result = _run_tracking_worker(
        session,
        params=params,
        raw_csv_path=backward_raw_csv,
        backward_mode=True,
        detection_cache_path=detection_cache_path,
        use_cached_detections=False,
    )
    if not backward_result.get("success"):
        return {
            "success": False,
            "error": "An error occurred during backward tracking. Check logs for details.",
        }

    backward_processed = _run_postprocess_worker(
        backward_raw_csv,
        params,
        clean=session.enable_postprocessing,
    )
    save_trajectories_to_csv(backward_processed, backward_processed_csv)

    interp_method = str(session.interpolation_method or "None").strip().lower()
    max_gap = max(1, round(session.interpolation_max_gap_seconds * params["FPS"]))
    total_frames = int(session.video_probe.total_frames or 0)
    if total_frames <= 0:
        total_frames = int(params.get("END_FRAME") or 0) + 1
    merged = _run_merge_worker(
        forward_processed,
        backward_processed,
        total_frames=total_frames,
        params=params,
        resize_factor=float(params.get("RESIZE_FACTOR", 1.0)),
        interp_method=interp_method,
        max_gap=max_gap,
        heading_flip_max_burst=session.heading_flip_max_burst,
    )
    save_trajectories_to_csv(merged, final_csv_path)

    fps_list = [
        fps
        for fps in (
            forward_result.get("fps_list", []) + backward_result.get("fps_list", [])
        )
        if fps and fps > 0
    ]
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else None
    lines = [
        f"video={os.path.basename(session.video_path)}",
        f"forward_csv={forward_processed_csv}",
        f"backward_csv={backward_processed_csv}",
        f"final_csv={final_csv_path}",
        f"detection_cache={detection_cache_path}",
    ]
    if avg_fps is not None:
        lines.append(f"avg_fps={avg_fps:.1f}")
    return {"success": True, "lines": lines}


def run_headless_tracking_session(session: TrackerCliSession) -> dict[str, Any]:
    """Run a TrackerKit session without GUI state for supported configs."""
    ensure_headless_qt_application()

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

    if session.enable_backward_tracking:
        return _run_forward_backward(
            session,
            params=params,
            detection_cache_path=cache_plan.detection_cache_path,
        )
    return _run_forward_only(
        session,
        params=params,
        detection_cache_path=cache_plan.detection_cache_path,
    )
