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
        self.unreadable_count = 0

    def __iter__(self):
        for fid in self._ids:
            yield FrameRef(source_id=self._source_id, frame_id=fid, path=None)

    def read(self, ref):
        img = self._inner.read(ref)
        if img is None:
            self.unreadable_count += 1
        return img

    def length(self) -> int:
        return len(self._ids)


def _dedup_selected_frames(video_path, frame_ids, method, threshold):
    """Drop perceptually near-duplicate picks.

    Returns `(kept_ids, error, unreadable_count)`.

    Scoped to `frame_ids` only (see `_SelectedFrameSource`) -- never the whole
    video. `error` is `None` on success; it is set (and `kept_ids` is empty)
    when dedup leaves nothing to export, distinguishing "the video could not
    be read" from "every pick collapsed as a near-duplicate" -- collapsing
    those into one silent empty result would misreport unreadable frames as
    "near-duplicates" to the user (see `generate_active_learning_dataset`).

    `unreadable_count` is returned even when `kept_ids` is non-empty: a
    partial dropout (some frames unreadable, the rest genuine survivors or
    dropped as duplicates) must not have its unreadable frames silently
    counted as "near-duplicates" in the caller's drop message either.
    """
    if str(method).strip().lower() == "none" or len(frame_ids) < 2:
        return list(frame_ids), None, 0
    total = len(frame_ids)
    source = _SelectedFrameSource(str(video_path), frame_ids)
    cfg = CandidatePoolConfig(
        dedup_method=str(method).strip().lower(),
        dedup_threshold=int(threshold),
    )
    kept = build_candidate_pool(source, cfg)
    kept_ids = [ref.frame_id for ref in kept]
    unreadable = source.unreadable_count
    if unreadable:
        logger.warning(
            "Perceptual dedup could not read %d/%d selected frames from %s",
            unreadable,
            total,
            video_path,
        )
    if kept_ids:
        return kept_ids, None, unreadable

    if unreadable >= total:
        error = (
            f"Could not read any of the {total} selected frames from the "
            "video for perceptual dedup (file missing, moved, or unreadable "
            "at export time). Check the video path, or disable dedup, and "
            "try again."
        )
    elif unreadable:
        error = (
            f"Perceptual dedup left no frames to export: {unreadable}/{total} "
            "selected frames could not be read from the video, and the "
            "remaining readable frames collapsed as near-duplicates. Check "
            "the video path, raise the dedup threshold, or disable dedup."
        )
    else:
        error = (
            f"Perceptual dedup collapsed all {total} selected frames into "
            "near-duplicates (dedup_threshold may be too permissive). Raise "
            "the threshold or disable dedup."
        )
    return [], error, unreadable


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
            # An unopenable video reports 0x0. Handing (0, 0) to the scorer is
            # worse than handing it nothing: `score_crowd` would compute every
            # edge margin against a zero-sized frame and call every detection
            # maximally close to the border. `None` degrades honestly to
            # edge_score 0.0 instead.
            frame_shape = (
                (
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                )
                if cap.isOpened()
                else None
            )
        finally:
            cap.release()
        if frame_shape is not None and min(frame_shape) <= 0:
            frame_shape = None
        if frame_shape is None:
            logger.warning(
                "Could not read frame dimensions from %s; edge scoring disabled "
                "for this run.",
                video_path,
            )
        # ORIGINAL video space -- FrameQualityScorer converts it (and
        # REFERENCE_BODY_SIZE) to RESIZE_FACTOR working space, which is the
        # space the cached `obb_corners` below actually live in.
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
        # Pre-compute frame grouping to avoid O(n²) per-frame DataFrame scans
        frames_by_id = {int(fid): sub for fid, sub in df.groupby("FrameID")}

        for idx, frame_id in enumerate(unique_frames):
            if _stopped(should_stop):
                return {"success": False, "cancelled": True}
            if idx % 100 == 0:
                pct = 20 + int((idx / total_unique) * 30) if total_unique else 20
                _emit(progress, pct, f"Scoring frames ({idx}/{total_unique})...")

            frame_data = frames_by_id.get(int(frame_id))
            if frame_data is None:
                continue
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
        selected_frames, dedup_error, dedup_unreadable = _dedup_selected_frames(
            video_path, selected_frames, dedup_method, dedup_threshold
        )
        if dedup_error is not None:
            return {"success": False, "error": dedup_error}
        if len(selected_frames) < before_dedup:
            # Two independent causes can drop a pick: it was a genuine
            # near-duplicate of another pick, or it could not be read from
            # the video at all. Report each with its own count -- rolling
            # them into one "near-duplicate" figure would misattribute
            # unreadable frames as duplicates (see _dedup_selected_frames).
            duplicate_dropped = before_dedup - len(selected_frames) - dedup_unreadable
            sentences = []
            if duplicate_dropped > 0:
                sentences.append(
                    f"Perceptual dedup dropped {duplicate_dropped} "
                    "near-duplicate frames."
                )
            if dedup_unreadable > 0:
                plural = "s" if dedup_unreadable != 1 else ""
                sentences.append(
                    f"{dedup_unreadable} frame{plural} could not be read "
                    "from the video."
                )
            _emit(progress, 55, " ".join(sentences))

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
    # No `finally: detection_cache.close()` -- nothing here needs flushing.
    # Closing is now SAFE regardless: handles from `open_detection_cache_reader`
    # carry `read_only=True`, and `DetectionCacheHandle.close()` returns early
    # for those instead of clobbering the on-disk cache with an empty frame set
    # keyed by the reader's placeholder key. That clobbering is exactly what
    # `interpolated_crops._cleanup_backends` used to do on every run, wiping
    # detection.npz and forcing a full re-inference on the next run.
