"""
Dataset generation utilities for active learning.
Identifies challenging frames and exports them for annotation.
"""

import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.data.al.escalation import (
    LabelRecord,
    achievable_levels,
    records_from_obb_result,
)
from hydra_suite.data.al.export import ExportedFrame, export_al_dataset
from hydra_suite.utils.geometry import clamp01 as _clamp01
from hydra_suite.utils.geometry import (
    obb_corners_from_dims as _detection_corners_from_dims,
)
from hydra_suite.utils.geometry import polygon_overlap_ratio as _polygon_overlap_ratio
from hydra_suite.utils.geometry_levels import GeometryLevel

logger = logging.getLogger(__name__)

_TASK_LEVELS = {
    "segment": GeometryLevel.POLYGON,
    "obb": GeometryLevel.OBB,
    "detect": GeometryLevel.AABB,
}


def resolve_native_level(params) -> GeometryLevel:
    """The geometry level the configured detection source can actually produce.

    Never claims a level the model did not compute: a rotated quad is OBB, not
    a polygon. bg-sub produces true foreground contours, so it reaches POLYGON.
    """
    method = str(params.get("DETECTION_METHOD", "background_subtraction")).lower()
    if method == "background_subtraction":
        return GeometryLevel.POLYGON
    if method != "yolo_obb":
        return GeometryLevel.OBB

    mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
    if mode == "sequential":
        task = str(params.get("YOLO_OBB_STAGE2_TASK", "obb")).strip().lower()
    else:
        task = str(params.get("YOLO_OBB_DIRECT_TASK", "obb")).strip().lower()
    return _TASK_LEVELS.get(task, GeometryLevel.OBB)


class FrameQualityScorer:
    """Tracker-side adapter that produces ALSignals and selects worst frames.

    Public API (`score_frame`, `get_worst_frames`) is preserved for callers; the
    underlying ranking now lives in `hydra_suite.data.al.acquisition`.
    """

    def __init__(self, params):
        from hydra_suite.data.al.acquisition import PRESETS, AcquisitionWeights

        self.params = params
        self.frame_signals: dict = {}
        self.max_targets = params.get("MAX_TARGETS", 4)
        self.conf_threshold = params.get("DATASET_CONF_THRESHOLD", 0.5)
        self.reference_body_size = max(
            float(params.get("REFERENCE_BODY_SIZE", 20.0)), 1.0
        )

        # Public legacy boolean flags (kept for backward compat).
        self.use_confidence = bool(params.get("METRIC_LOW_CONFIDENCE", True))
        self.use_count_mismatch = bool(params.get("METRIC_COUNT_MISMATCH", True))
        self.use_assignment_cost = bool(params.get("METRIC_HIGH_ASSIGNMENT_COST", True))
        self.use_track_loss = bool(params.get("METRIC_TRACK_LOSS", True))
        self.use_uncertainty = bool(params.get("METRIC_HIGH_UNCERTAINTY", False))
        self.use_fragmented_detections = bool(
            params.get("METRIC_FRAGMENTED_DETECTIONS", True)
        )

        self._enabled = {
            "uncertainty": self.use_confidence,
            "count": self.use_count_mismatch,
            "assignment": self.use_assignment_cost,
            "track_loss": self.use_track_loss,
            "position_uncertainty": self.use_uncertainty,
            "crowd": self.use_fragmented_detections,
        }

        preset_name = params.get("DATASET_AL_PRESET", "tracker_default")
        base = PRESETS.get(preset_name, PRESETS["tracker_default"])
        self._weights = AcquisitionWeights(
            uncertainty=base.uncertainty if self._enabled["uncertainty"] else 0.0,
            nms_instability=0.0,
            count=base.count if self._enabled["count"] else 0.0,
            crowd=base.crowd if self._enabled["crowd"] else 0.0,
            edge=base.edge,
            assignment=base.assignment if self._enabled["assignment"] else 0.0,
            track_loss=base.track_loss if self._enabled["track_loss"] else 0.0,
            position_uncertainty=(
                base.position_uncertainty
                if self._enabled["position_uncertainty"]
                else 0.0
            ),
        )

        # Backward-compat scalar map for legacy callers.
        self.frame_scores = defaultdict(lambda: {"score": 0.0, "metrics": {}})

    def score_frame(self, frame_id, detection_data=None, tracking_data=None):
        from hydra_suite.data.al.signals import (
            ALSignals,
            score_count_deviation,
            score_crowd,
            score_uncertainty,
        )

        detection_data = detection_data or {}
        tracking_data = tracking_data or {}

        # ------------------------------------------------------------------
        # New pipeline: build ALSignals and store for get_worst_frames.
        # ------------------------------------------------------------------
        confidences = detection_data.get("confidences") or []
        mean_conf, margin = score_uncertainty(
            confidences, conf_floor=self.conf_threshold
        )

        n_dets = int(detection_data.get("count", len(confidences)))
        count_dev = score_count_deviation(n_dets, self.max_targets)

        obb_corners = self._extract_obb_corners(detection_data)
        if obb_corners:
            crowd, edge = score_crowd(obb_corners, frame_shape=(1, 1))
        else:
            crowd, edge = 0.0, 0.0

        extras: dict[str, float] = {}
        ac = tracking_data.get("assignment_confidences") or []
        if ac:
            extras["assignment"] = max(0.0, 1.0 - float(np.mean(ac)))
        elif tracking_data.get("assignment_costs"):
            costs = tracking_data["assignment_costs"]
            extras["assignment"] = float(min(np.mean(costs) / 50.0, 1.0))

        lost = int(tracking_data.get("lost_tracks", 0))
        if lost > 0:
            extras["track_loss"] = float(min(lost / max(self.max_targets, 1), 1.0))

        unc = tracking_data.get("uncertainties") or []
        if unc:
            extras["position_uncertainty"] = float(min(np.mean(unc) / 50.0, 1.0))

        signal = ALSignals(
            frame_id=int(frame_id),
            n_detections=n_dets,
            mean_confidence=mean_conf,
            margin=margin,
            count_deviation=count_dev,
            crowd_score=crowd,
            edge_score=edge,
            extras=extras,
        )
        self.frame_signals[int(frame_id)] = signal

        # ------------------------------------------------------------------
        # Legacy pipeline: compute scalar score + structured metrics dict
        # so that frame_scores[fid]["score"] and frame_scores[fid]["metrics"]
        # match the pre-refactor API that the tests verify.
        # ------------------------------------------------------------------
        metrics: dict = {}
        legacy_score = 0.0
        legacy_score += self._score_confidence(detection_data, metrics)
        legacy_score += self._score_count_mismatch(detection_data, metrics)
        legacy_score += self._score_assignment_cost(tracking_data, metrics)
        legacy_score += self._score_track_loss(tracking_data, metrics)
        legacy_score += self._score_uncertainty(tracking_data, metrics)
        legacy_score += self._score_fragmented_detections(detection_data, metrics)

        self.frame_scores[int(frame_id)] = {"score": legacy_score, "metrics": metrics}
        return legacy_score

    # ------------------------------------------------------------------
    # Legacy per-metric scorers (restored for backward compat)
    # ------------------------------------------------------------------

    def _score_confidence(self, detection_data, metrics):
        """Score based on low detection confidence. Returns weighted score."""
        if not (self.use_confidence and "confidences" in detection_data):
            return 0.0
        confidences = detection_data["confidences"]
        if not confidences:
            return 0.0
        valid_confs = [c for c in confidences if not np.isnan(c)]
        if not valid_confs:
            return 0.0
        avg_conf = np.mean(valid_confs)
        if avg_conf >= self.conf_threshold:
            return 0.0
        denom = max(self.conf_threshold, 1e-6)
        conf_score = (self.conf_threshold - avg_conf) / denom
        metrics["low_confidence"] = {
            "min": min(valid_confs),
            "avg": avg_conf,
            "score": conf_score,
        }
        return conf_score * 0.4

    def _score_count_mismatch(self, detection_data, metrics):
        """Score based on detection count mismatch. Returns weighted score."""
        if not (self.use_count_mismatch and "count" in detection_data):
            return 0.0
        det_count = detection_data["count"]
        if det_count == self.max_targets:
            return 0.0
        if det_count < self.max_targets:
            count_score = (self.max_targets - det_count) / self.max_targets
            weighted = count_score * 0.3
        else:
            count_score = (
                min((det_count - self.max_targets) / self.max_targets, 1.0) * 0.5
            )
            weighted = count_score * 0.15
        metrics["count_mismatch"] = {
            "expected": self.max_targets,
            "actual": det_count,
            "score": count_score if det_count < self.max_targets else count_score * 0.5,
        }
        return weighted

    def _score_assignment_cost(self, tracking_data, metrics):
        """Score based on high assignment cost. Returns weighted score."""
        if not self.use_assignment_cost:
            return 0.0
        costs = tracking_data.get("assignment_costs") or []
        if costs:
            avg_cost = np.mean(costs)
            cost_score = min(avg_cost / 50.0, 1.0)
            metrics["high_assignment_cost"] = {
                "avg": avg_cost,
                "max": max(costs),
                "score": cost_score,
                "source": "assignment_cost",
            }
            return cost_score * 0.15

        confidences = tracking_data.get("assignment_confidences") or []
        valid_confidences = [
            float(confidence) for confidence in confidences if np.isfinite(confidence)
        ]
        if not valid_confidences:
            return 0.0

        avg_confidence = np.mean(valid_confidences)
        difficulty_score = 1.0 - float(np.clip(avg_confidence, 0.0, 1.0))
        metrics["high_assignment_cost"] = {
            "avg_confidence": avg_confidence,
            "score": difficulty_score,
            "source": "assignment_confidence",
        }
        return difficulty_score * 0.15

    def _score_track_loss(self, tracking_data, metrics):
        """Score based on track losses. Returns weighted score."""
        if not (self.use_track_loss and "lost_tracks" in tracking_data):
            return 0.0
        lost_count = tracking_data["lost_tracks"]
        if lost_count <= 0:
            return 0.0
        loss_score = min(lost_count / self.max_targets, 1.0)
        metrics["track_loss"] = {"count": lost_count, "score": loss_score}
        return loss_score * 0.1

    def _score_uncertainty(self, tracking_data, metrics):
        """Score based on high position uncertainty. Returns weighted score."""
        if not (self.use_uncertainty and "uncertainties" in tracking_data):
            return 0.0
        uncertainties = tracking_data["uncertainties"]
        if not uncertainties:
            return 0.0
        avg_uncertainty = np.mean(uncertainties)
        unc_score = min(avg_uncertainty / 50.0, 1.0)
        metrics["high_uncertainty"] = {"avg": avg_uncertainty, "score": unc_score}
        return unc_score * 0.05

    def _score_fragmented_detections(self, detection_data, metrics):
        """Score frames with suspiciously duplicated or fragmented detections."""
        if not self.use_fragmented_detections:
            return 0.0

        measurements = detection_data.get("measurements") or []
        if len(measurements) < 2:
            return 0.0

        shapes = detection_data.get("shapes") or []
        obb_corners = detection_data.get("obb_corners") or []

        geometries = []
        major_axes = []
        for det_idx, measurement in enumerate(measurements):
            if measurement is None or len(measurement) < 3:
                continue

            cx = float(measurement[0])
            cy = float(measurement[1])
            theta = float(measurement[2])

            corners = None
            if det_idx < len(obb_corners) and obb_corners[det_idx] is not None:
                corners_candidate = np.asarray(obb_corners[det_idx], dtype=np.float32)
                if corners_candidate.size >= 8:
                    corners = corners_candidate.reshape(4, 2)

            if corners is not None:
                width = float(np.linalg.norm(corners[1] - corners[0]))
                height = float(np.linalg.norm(corners[2] - corners[1]))
            elif det_idx < len(shapes) and len(shapes[det_idx]) >= 2:
                area = max(float(shapes[det_idx][0]), 1.0)
                aspect_ratio = float(shapes[det_idx][1])
                width, height = _dims_from_shape(area, aspect_ratio)
                corners = _detection_corners_from_dims(cx, cy, width, height, theta)
            else:
                width = self.reference_body_size * 2.2
                height = self.reference_body_size * 0.8
                corners = _detection_corners_from_dims(cx, cy, width, height, theta)

            major_axis = max(width, height)
            major_axes.append(major_axis)
            geometries.append(
                {
                    "index": det_idx,
                    "center": np.array([cx, cy], dtype=np.float32),
                    "corners": corners,
                    "major_axis": major_axis,
                }
            )

        if len(geometries) < 2:
            return 0.0

        typical_major_axis = float(
            np.median(major_axes) if major_axes else self.reference_body_size * 2.2
        )
        typical_major_axis = max(typical_major_axis, 1.0)

        suspicious_pairs = []
        best_pair = None
        best_pair_score = 0.0

        for left, right in combinations(geometries, 2):
            center_distance = float(np.linalg.norm(left["center"] - right["center"]))
            proximity_threshold = max(typical_major_axis * 0.65, 1.0)
            proximity_score = _clamp01(1.0 - (center_distance / proximity_threshold))

            overlap_score = _polygon_overlap_ratio(
                left["corners"],
                right["corners"],
            )
            pair_major_axis = (left["major_axis"] + right["major_axis"]) / 2.0
            smallness_score = _clamp01(1.0 - (pair_major_axis / typical_major_axis))

            pair_score = _clamp01(
                0.5 * proximity_score + 0.3 * overlap_score + 0.2 * smallness_score
            )
            if pair_score >= 0.45:
                suspicious_pairs.append(pair_score)
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = {
                    "pair": [left["index"], right["index"]],
                    "distance": center_distance,
                    "overlap": overlap_score,
                    "smallness": smallness_score,
                }

        if best_pair is None or best_pair_score <= 0.0:
            return 0.0

        fragmentation_score = _clamp01(
            best_pair_score + min(0.1 * max(len(suspicious_pairs) - 1, 0), 0.2)
        )
        metrics["fragmented_detections"] = {
            **best_pair,
            "score": fragmentation_score,
            "suspicious_pairs": len(suspicious_pairs),
            "typical_major_axis": typical_major_axis,
        }
        return fragmentation_score * 0.3

    def get_worst_frames(self, max_frames, diversity_window=30, probabilistic=True):
        from hydra_suite.data.al.acquisition import select

        signals = list(self.frame_signals.values())
        rng = np.random.default_rng() if probabilistic else None
        return select(
            signals,
            weights=self._weights,
            k=int(max_frames),
            diversity_window=int(diversity_window),
            probabilistic=bool(probabilistic),
            rng=rng,
            min_score=float(self.params.get("DATASET_MIN_SELECTION_SCORE", 0.0)),
        )

    def _extract_obb_corners(self, detection_data):
        corners = detection_data.get("obb_corners") or []
        out: list[np.ndarray] = []
        for c in corners:
            if c is None:
                continue
            arr = np.asarray(c, dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] >= 3:
                out.append(arr)
        return out


def _init_detection_runner(params):
    """Build a detection-only InferenceRunner for dataset label extraction.

    Unlike the legacy version this supports every detection source, not just
    `yolo_obb`: returning None for bg-sub meant every exported label was a
    fabricated reference-size box.
    """
    method = str(params.get("DETECTION_METHOD", "background_subtraction")).lower()
    native_level = resolve_native_level(params)
    try:
        from ..core.inference.runner import InferenceRunner

        if method == "background_subtraction":
            # build_inference_config_from_params never builds a BgSubConfig
            # (it only ever wires up `obb=`), so a bgsub-backed InferenceConfig
            # has to be constructed explicitly here, mirroring the live
            # tracking path in core/tracking/worker.py (bgsub branch).
            from ..core.inference.config import (
                BgSubConfig,
                InferenceConfig,
                migrate_runtime_to_tier,
            )

            _compute_runtime = str(params.get("COMPUTE_RUNTIME", "cpu"))
            _raw_tier = str(params.get("RUNTIME_TIER", "") or "").strip().lower()
            _runtime_tier = (
                _raw_tier
                if _raw_tier in {"cpu", "gpu", "gpu_fast"}
                else migrate_runtime_to_tier({_compute_runtime})
            )
            bgsub_cfg = BgSubConfig.from_params(params)
            if native_level is GeometryLevel.POLYGON:
                bgsub_cfg.emit_native_geometry = True
            cfg = InferenceConfig(
                obb=None,
                bgsub=bgsub_cfg,
                runtime_tier=_runtime_tier,
                detection_batch_size=int(params.get("DETECTION_BATCH_SIZE", 1) or 1),
            )
        else:
            from ..core.inference.config import build_obb_only_config

            mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
            task = (
                (
                    str(params.get("YOLO_OBB_STAGE2_TASK", "obb"))
                    if mode == "sequential"
                    else str(params.get("YOLO_OBB_DIRECT_TASK", "obb"))
                )
                .strip()
                .lower()
            )
            model_path = str(
                params.get(
                    "YOLO_OBB_DIRECT_MODEL_PATH",
                    params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt"),
                )
                or "yolo26s-obb.pt"
            )
            cfg = build_obb_only_config(
                model_path,
                runtime_tier=str(params.get("RUNTIME_TIER", "") or "") or None,
                confidence_threshold=float(
                    params.get("DATASET_YOLO_CONFIDENCE_THRESHOLD", 0.05)
                ),
                iou_threshold=float(params.get("DATASET_YOLO_IOU_THRESHOLD", 0.5)),
                max_targets=max(1, int(params.get("MAX_TARGETS", 8))),
                mode=mode,
                model_task=task,
                emit_native_geometry=(native_level is GeometryLevel.POLYGON),
            )

        runner = InferenceRunner(cfg)
        logger.info(
            "Detection runner initialized for dataset export (method=%s, level=%s)",
            method,
            native_level.label,
        )
        return runner
    except Exception as e:
        logger.warning(
            "Could not initialize detection runner: %s. "
            "Labels will fall back to reference-size approximation.",
            e,
        )
        return None


def _expand_frame_ids(frame_ids, include_context, total_frames):
    """Expand frame list with +/-1 context frames if requested."""
    frames_to_export = set()
    for frame_id in frame_ids:
        frames_to_export.add(frame_id)
        if include_context:
            if frame_id > 0:
                frames_to_export.add(frame_id - 1)
            if frame_id < total_frames - 1:
                frames_to_export.add(frame_id + 1)
    return sorted(frames_to_export)


def _read_and_resize_frame(cap, frame_id, params, first_frame_shape):
    """Read a video frame, resize for detection, and validate shape.

    Returns (original_frame, detection_frame, updated_first_shape) or None if
    the frame could not be read.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not (ret and frame is not None and frame.size > 0):
        logger.warning(f"Could not read frame {frame_id}, skipping")
        return None

    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    if resize_factor != 1.0 and resize_factor > 0:
        h, w = frame.shape[:2]
        new_w, new_h = int(w * resize_factor), int(h * resize_factor)
        if new_w > 0 and new_h > 0:
            frame_for_detection = cv2.resize(frame, (new_w, new_h))
        else:
            logger.warning(
                f"Invalid resize dimensions for frame {frame_id}, using original"
            )
            frame_for_detection = frame
    else:
        frame_for_detection = frame

    if frame_for_detection.size == 0:
        logger.warning(f"Frame {frame_id} has zero size, skipping")
        return None

    if first_frame_shape is None:
        first_frame_shape = frame_for_detection.shape
    elif frame_for_detection.shape != first_frame_shape:
        h, w = first_frame_shape[:2]
        frame_for_detection = cv2.resize(frame_for_detection, (w, h))
        logger.debug(
            f"Resized frame {frame_id} to match batch dimensions: {first_frame_shape}"
        )

    return frame, frame_for_detection, first_frame_shape


def _dims_from_shape(area, aspect_ratio, obb_corners=None, det_idx=None):
    """Compute (w, h) from ellipse area and aspect ratio, with OBB fallback."""
    if aspect_ratio > 0:
        w_det = np.sqrt(area * aspect_ratio / np.pi) * 2
        h_det = w_det / aspect_ratio
    elif obb_corners is not None and det_idx is not None and det_idx < len(obb_corners):
        corners = obb_corners[det_idx]
        w_det = np.linalg.norm(corners[1] - corners[0])
        h_det = np.linalg.norm(corners[2] - corners[1])
    else:
        w_det = h_det = np.sqrt(area / np.pi) * 2
    return w_det, h_det


def _csv_scale_back(df, resize_factor, frame_width, frame_height):
    """Determine scale factor to map CSV coordinates back to original space."""
    if not (resize_factor and resize_factor < 1.0):
        return 1.0
    try:
        max_x = df["X"].max()
        max_y = df["Y"].max()
        if (
            max_x <= frame_width * resize_factor * 1.05
            and max_y <= frame_height * resize_factor * 1.05
        ):
            return 1.0 / resize_factor
    except Exception:
        pass
    return 1.0


def _open_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    return cap


def _detect_records_for_frames(runner, frames, params, native_level):
    """Run detection over `frames` and return {frame_id: [LabelRecord]}."""
    if runner is None:
        return {}
    batch_size = _get_detector_batch_size(runner)
    out: dict[int, list[LabelRecord]] = {}
    frame_ids = sorted(frames)
    for start in range(0, len(frame_ids), batch_size):
        chunk = frame_ids[start : start + batch_size]
        images = [frames[fid] for fid in chunk]
        try:
            results = runner.detect_batch(images, frame_indices=list(chunk))
        except Exception as e:
            logger.warning("Detection failed for batch starting at %s: %s", chunk[0], e)
            continue
        for fid, obb in zip(chunk, results):
            out[fid] = records_from_obb_result(obb, native_level)
    return out


def _select_records_for_frame(rows, frame_records, params, scale_back):
    """Placeholder pairing: returns detector records unchanged. Task 13 replaces
    this with mutual-exclusion matching and strict drops."""
    return list(frame_records), {"lost": 0, "unmatched": 0}


def export_dataset(
    video_path,
    csv_path,
    frame_ids,
    output_dir,
    dataset_name,
    class_name,
    params,
    include_context: bool = True,
    export_levels=None,
    class_names=None,
):
    """Export selected frames and labels as an escalated AL dataset.

    Returns the manifest dict from `export_al_dataset` (previously a directory
    path string).
    """
    from datetime import datetime

    import pandas as pd

    native_level = resolve_native_level(params)
    allowed = achievable_levels(native_level)
    levels = list(export_levels) if export_levels else list(allowed)
    unsupported = [lvl for lvl in levels if lvl not in allowed]
    if unsupported:
        raise ValueError(
            f"requested levels {[lvl.label for lvl in unsupported]} exceed the "
            f"native level {native_level.label!r} of the configured detector"
        )

    resolved_class_names = (
        list(class_names) if class_names else [class_name or "object"]
    )

    cap = _open_video(video_path)
    runner = _init_detection_runner(params)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        df = pd.read_csv(csv_path)
        selected = {int(f) for f in frame_ids}
        frames_to_export = _expand_frame_ids(frame_ids, include_context, total_frames)

        images: dict[int, np.ndarray] = {}
        detection_frames: dict[int, np.ndarray] = {}
        for fid in frames_to_export:
            read = _read_and_resize_frame(cap, fid, params, None)
            if read is None:
                continue
            original, for_detection, _shape = read
            images[fid] = original
            detection_frames[fid] = for_detection

        records_by_frame = _detect_records_for_frames(
            runner, detection_frames, params, native_level
        )

        resize_factor = params.get("RESIZE_FACTOR", 1.0)
        scale_back = _csv_scale_back(df, resize_factor, frame_width, frame_height)
        rows_by_frame = {int(fid): sub for fid, sub in df.groupby("FrameID")}

        exported: list[ExportedFrame] = []
        for fid in sorted(images):
            records, drops = _select_records_for_frame(
                rows_by_frame.get(fid),
                records_by_frame.get(fid, []),
                params,
                scale_back,
            )
            exported.append(
                ExportedFrame(
                    frame_id=fid,
                    image_name=f"f{fid:06d}.jpg",
                    records=records,
                    is_context=fid not in selected,
                    drops=drops,
                )
            )
    finally:
        cap.release()
        if runner is not None:
            runner.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{dataset_name}_{timestamp}" if str(dataset_name).strip() else timestamp
    round_dir = Path(output_dir).resolve() / name

    provenance = {
        "source_video": str(video_path),
        "source_csv": str(csv_path),
        "detection_method": params.get("DETECTION_METHOD"),
        "model_path": params.get("YOLO_OBB_DIRECT_MODEL_PATH"),
        "model_task": params.get("YOLO_OBB_DIRECT_TASK"),
        "export_confidence_threshold": params.get(
            "DATASET_YOLO_CONFIDENCE_THRESHOLD", 0.05
        ),
        "export_iou_threshold": params.get("DATASET_YOLO_IOU_THRESHOLD", 0.5),
        "acquisition_preset": params.get("DATASET_AL_PRESET"),
        "image_width": frame_width,
        "image_height": frame_height,
        "note": (
            "Labels come from a dedicated export detection pass at lower "
            "confidence than tracking, so they may differ from the tracked "
            "detections. Review before training."
        ),
    }

    manifest = export_al_dataset(
        round_dir=round_dir,
        frames=exported,
        images=images,
        native_level=native_level,
        levels=levels,
        class_names=resolved_class_names,
        provenance=provenance,
    )
    logger.info("Dataset exported to %s (%d frames)", round_dir, len(exported))
    return manifest


def _get_detector_batch_size(runner):
    """Return the batch size to use for detection."""
    if runner is not None and getattr(runner, "config", None) is not None:
        return max(1, int(getattr(runner.config, "detection_batch_size", 1)))
    return 1
