"""Qt-free active-learning dataset generation. Extracted from
trackerkit/gui/workers/dataset_worker.py (Slice 3)."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from hydra_suite.core.inference.cache import open_detection_cache_reader
from hydra_suite.data.dataset_generation import FrameQualityScorer, export_dataset

logger = logging.getLogger(__name__)


def _emit(progress, value, message):
    if progress is not None:
        progress(value, message)


def _stopped(should_stop) -> bool:
    return bool(should_stop is not None and should_stop())


def generate_active_learning_dataset(
    *,
    video_path,
    csv_path,
    detection_cache_path,
    output_dir,
    dataset_name,
    class_name,
    params,
    max_frames,
    diversity_window,
    include_context,
    probabilistic,
    progress=None,
    should_stop=None,
    export_levels=None,
    class_names=None,
) -> dict:
    """Score frames and export an active-learning dataset. Pure/Qt-free."""
    detection_cache = None
    try:
        if _stopped(should_stop):
            return {"success": False, "cancelled": True}
        _emit(progress, 5, "Initializing dataset generation...")

        _emit(progress, 10, "Loading tracking data...")
        df = pd.read_csv(csv_path)

        _emit(progress, 15, "Initializing quality scorer...")
        scorer = FrameQualityScorer(params)
        if detection_cache_path and os.path.exists(detection_cache_path):
            try:
                detection_cache = open_detection_cache_reader(detection_cache_path)
                if not detection_cache.is_valid():
                    detection_cache = None
            except Exception:
                detection_cache = None

        _emit(progress, 20, "Scoring frames...")
        unique_frames = df["FrameID"].unique()
        total_unique = len(unique_frames)

        for idx, frame_id in enumerate(unique_frames):
            if _stopped(should_stop):
                return {"success": False, "cancelled": True}
            if idx % 100 == 0:
                pct = 20 + int((idx / total_unique) * 30) if total_unique else 20
                _emit(progress, pct, f"Scoring frames ({idx}/{total_unique})...")

            frame_data = df[df["FrameID"] == frame_id]
            raw_meas, raw_shapes, raw_confidences, raw_obb_corners = [], [], [], []
            used_detection_cache = False
            if detection_cache is not None:
                try:
                    obb = detection_cache.read_frame(int(frame_id))
                    if obb is None:
                        raise KeyError(f"frame {frame_id} not cached")
                    raw_meas = np.concatenate(
                        [obb.centroids, obb.angles[:, None]], axis=1
                    ).tolist()
                    raw_shapes = obb.shapes.tolist()
                    raw_confidences = obb.confidences.tolist()
                    raw_obb_corners = obb.corners.tolist()
                    used_detection_cache = True
                except Exception:
                    raw_meas, raw_shapes, raw_confidences, raw_obb_corners = (
                        [],
                        [],
                        [],
                        [],
                    )

            detection_count = len(raw_meas) if used_detection_cache else len(frame_data)
            detection_data = {
                "confidences": (
                    raw_confidences
                    if raw_confidences
                    else (
                        frame_data["DetectionConfidence"].tolist()
                        if "DetectionConfidence" in frame_data.columns
                        else []
                    )
                ),
                "count": detection_count,
                "measurements": raw_meas,
                "shapes": raw_shapes,
                "obb_corners": raw_obb_corners,
            }
            tracking_data = {
                "lost_tracks": int((frame_data["State"] == "lost").sum()),
                "assignment_confidences": (
                    frame_data["AssignmentConfidence"].tolist()
                    if "AssignmentConfidence" in frame_data.columns
                    else []
                ),
                "uncertainties": (
                    frame_data["PositionUncertainty"].tolist()
                    if "PositionUncertainty" in frame_data.columns
                    else []
                ),
            }
            scorer.score_frame(frame_id, detection_data, tracking_data)

        if _stopped(should_stop):
            return {"success": False, "cancelled": True}
        _emit(progress, 50, "Selecting challenging frames...")
        selected_frames = scorer.get_worst_frames(
            max_frames, diversity_window, probabilistic=probabilistic
        )
        if not selected_frames:
            return {
                "success": False,
                "error": "No frames met the quality criteria for export.",
            }

        _emit(progress, 60, f"Exporting {len(selected_frames)} frames...")
        if _stopped(should_stop):
            return {"success": False, "cancelled": True}
        manifest = export_dataset(
            video_path=video_path,
            csv_path=csv_path,
            frame_ids=selected_frames,
            output_dir=output_dir,
            dataset_name=dataset_name,
            class_name=class_name,
            params=params,
            include_context=include_context,
            export_levels=export_levels,
            class_names=class_names,
        )
        dataset_dir = manifest["round_dir"]
        if _stopped(should_stop):
            return {
                "success": False,
                "cancelled": True,
                "num_frames": len(selected_frames),
                "dir": dataset_dir,
                "manifest": manifest,
            }
        _emit(progress, 100, "Dataset generation complete!")
        return {
            "success": True,
            "num_frames": len(selected_frames),
            "dir": dataset_dir,
            "manifest": manifest,
        }
    except Exception as e:
        logger.exception("Error during dataset generation")
        return {"success": False, "error": str(e)}
    # No `finally: detection_cache.close()` -- unlike the legacy DetectionCache
    # (whose read-mode close() is a harmless mmap release), DetectionCacheHandle
    # is write-oriented: close() flushes `_buffer` and, for a reader that never
    # buffered a write, that means clobbering the on-disk cache with an empty
    # frame set. This handle is opened read-only above and must never be closed.
