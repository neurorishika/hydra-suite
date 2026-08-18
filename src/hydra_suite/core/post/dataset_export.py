"""Qt-free active-learning dataset generation. Extracted from
trackerkit/gui/workers/dataset_worker.py (Slice 3)."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import pandas as pd

from hydra_suite.core.inference.cache import open_detection_cache_reader
from hydra_suite.data.al.candidate_pool import CandidatePoolConfig, build_candidate_pool
from hydra_suite.data.al.frame_source import FrameRef, VideoFrameSource
from hydra_suite.data.dataset_generation import FrameQualityScorer, export_dataset

logger = logging.getLogger(__name__)


def _emit(progress, value, message):
    if progress is not None:
        progress(value, message)


def _stopped(should_stop) -> bool:
    return bool(should_stop is not None and should_stop())


class _SelectedFrameSource:
    """FrameSource restricted to an explicit frame-id list.

    Dedup runs over the SELECTED frames plus their context, never the whole
    video: perceptual hashing 100k video frames is prohibitive, and the
    near-duplicate problem lives entirely in `include_context`'s +/-1 frame
    expansion anyway (dedup runs before that expansion happens -- see
    `generate_active_learning_dataset`). A future "simplification" that swaps
    this for a plain `VideoFrameSource` would silently hash the whole video
    and make exports unusably slow.
    """

    def __init__(self, video_path: str, frame_ids) -> None:
        self._inner = VideoFrameSource(video_path)
        self._ids = sorted(int(f) for f in frame_ids)
        self._source_id = f"selected:{len(self._ids)}"

    def __iter__(self):
        for fid in self._ids:
            yield FrameRef(source_id=self._source_id, frame_id=fid, path=None)

    def read(self, ref):
        return self._inner.read(ref)

    def length(self) -> int:
        return len(self._ids)


def _dedup_selected_frames(video_path, frame_ids, method, threshold):
    """Drop perceptually near-duplicate picks. Returns the surviving ids.

    Scoped to `frame_ids` only (see `_SelectedFrameSource`) -- never the whole
    video.
    """
    if str(method).strip().lower() == "none" or len(frame_ids) < 2:
        return list(frame_ids)
    source = _SelectedFrameSource(str(video_path), frame_ids)
    cfg = CandidatePoolConfig(
        dedup_method=str(method).strip().lower(),
        dedup_threshold=int(threshold),
    )
    kept = build_candidate_pool(source, cfg)
    return [ref.frame_id for ref in kept]


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
    dedup_method: str = "phash",
    dedup_threshold: int = 8,
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
        cap = cv2.VideoCapture(str(video_path))
        try:
            frame_shape = (
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            )
        finally:
            cap.release()
        scorer = FrameQualityScorer(params, frame_shape=frame_shape)
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
            observed = scorer.explain_scores()
            detail = ", ".join(f"{k}={v:.2f}" for k, v in sorted(observed.items()))
            return {
                "success": False,
                "error": (
                    "No frames scored above the minimum selection score. "
                    f"Highest severity observed per signal: {detail or 'none'}. "
                    "Lower 'Min selection score' to export the best available "
                    "frames, or accept that tracking found nothing difficult."
                ),
                "channel_maxima": observed,
            }

        before_dedup = len(selected_frames)
        selected_frames = _dedup_selected_frames(
            video_path, selected_frames, dedup_method, dedup_threshold
        )
        if len(selected_frames) < before_dedup:
            _emit(
                progress,
                55,
                f"Perceptual dedup dropped {before_dedup - len(selected_frames)} "
                f"near-duplicate frames.",
            )

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
