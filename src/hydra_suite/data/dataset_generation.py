"""
Dataset generation utilities for active learning.
Identifies challenging frames and exports them for annotation.
"""

import json
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.utils.geometry import clamp01 as _clamp01
from hydra_suite.utils.geometry import (
    obb_corners_from_dims as _detection_corners_from_dims,
)
from hydra_suite.utils.geometry import polygon_overlap_ratio as _polygon_overlap_ratio

logger = logging.getLogger(__name__)


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


def _make_dataset_dir(output_dir, dataset_name):
    """Create timestamped dataset directory structure and return paths."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dataset_name and str(dataset_name).strip():
        dataset_name_with_timestamp = f"{dataset_name}_{timestamp}"
    else:
        dataset_name_with_timestamp = timestamp

    output_path = Path(output_dir).resolve()
    if not output_path.exists():
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f"Could not create output directory {output_path}: {e}")

    dataset_dir = output_path / dataset_name_with_timestamp
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir, images_dir, labels_dir


def _init_detection_runner(params):
    """Build a detection-only InferenceRunner for dataset dimension extraction.

    Returns None for non-yolo_obb methods (dimension extraction then falls back
    to reference-size approximation, as before).
    """
    detection_method = params.get("DETECTION_METHOD", "background_subtraction")
    if detection_method != "yolo_obb":
        return None
    try:
        from ..core.inference.config import build_obb_only_config
        from ..core.inference.runner import InferenceRunner

        model_path = str(
            params.get(
                "YOLO_OBB_DIRECT_MODEL_PATH",
                params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt"),
            )
            or "yolo26s-obb.pt"
        )
        cfg = build_obb_only_config(
            model_path,
            compute_runtime=str(params.get("COMPUTE_RUNTIME", "cpu")),
            confidence_threshold=float(
                params.get("DATASET_YOLO_CONFIDENCE_THRESHOLD", 0.05)
            ),
            iou_threshold=float(params.get("DATASET_YOLO_IOU_THRESHOLD", 0.5)),
            max_targets=max(1, int(params.get("MAX_TARGETS", 8))),
            mode=str(params.get("YOLO_OBB_MODE", "direct")).strip().lower(),
        )
        runner = InferenceRunner(cfg)
        logger.info("Detection runner initialized for dimension extraction")
        return runner
    except Exception as e:
        logger.warning(
            f"Could not initialize detection runner: {e}. Using reference size approximation."
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


def _measurements_to_detections(meas, shapes, resize_factor, obb_corners=None):
    """Convert detector measurements+shapes into a {(cx,cy): (w,h,theta)} dict."""
    scale_back = 1.0 / resize_factor
    yolo_detections = {}
    for det_idx, measurement in enumerate(meas):
        cx_det, cy_det, angle_rad = measurement
        area, aspect_ratio = shapes[det_idx]
        w_det, h_det = _dims_from_shape(
            area, aspect_ratio, obb_corners=obb_corners, det_idx=det_idx
        )
        cx_det *= scale_back
        cy_det *= scale_back
        w_det *= scale_back
        h_det *= scale_back
        corners_det = None
        if obb_corners is not None and det_idx < len(obb_corners):
            corners_det = np.asarray(obb_corners[det_idx], dtype=np.float32).copy()
            corners_det[:, 0] *= scale_back
            corners_det[:, 1] *= scale_back
        yolo_detections[(cx_det, cy_det)] = {
            "width": w_det,
            "height": h_det,
            "theta": angle_rad,
            "corners": corners_det,
        }
    return yolo_detections


def _detect_batch(runner, batch_frames, batch_frame_ids, valid_batch_indices, params):
    """Run OBB detection on a batch via InferenceRunner, returning detection dicts."""
    if runner is None or not batch_frames:
        return [{}] * len(batch_frames)
    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    try:
        results = runner.detect_batch(batch_frames, frame_indices=list(batch_frame_ids))
    except Exception as e:
        logger.warning(f"Detection failed: {e}")
        return [{}] * len(batch_frames)
    out = []
    for obb in results:
        meas = np.concatenate([obb.centroids, obb.angles[:, None]], axis=1)
        out.append(
            _measurements_to_detections(meas, obb.shapes, resize_factor, obb.corners)
        )
    return out


def _match_yolo_detection(cx, cy, yolo_detections, frame_id):
    """Find closest YOLO detection and return matched geometry details."""
    if not yolo_detections:
        return None, None, None, "reference_size"

    min_dist = float("inf")
    matched_detection = None
    for (cx_det, cy_det), detection in yolo_detections.items():
        dist = np.sqrt((cx - cx_det) ** 2 + (cy - cy_det) ** 2)
        if dist < min_dist:
            min_dist = dist
            matched_detection = detection

    if min_dist < 50 and matched_detection is not None:
        if isinstance(matched_detection, dict):
            w = matched_detection.get("width")
            h = matched_detection.get("height")
            corners = matched_detection.get("corners")
        else:
            w, h, _theta = matched_detection
            corners = None
        logger.debug(
            f"Frame {frame_id}: Matched tracking to YOLO detection (dist={min_dist:.1f})"
        )
        return w, h, corners, "yolo_match"
    return None, None, None, "reference_size"


def _format_obb_corners(corners, frame_width, frame_height):
    """Format raw pixel-space OBB corners as a YOLO OBB annotation line."""
    corners_arr = np.asarray(corners, dtype=np.float32).reshape(4, 2).copy()
    corners_arr[:, 0] = np.clip(corners_arr[:, 0] / frame_width, 0.0, 1.0)
    corners_arr[:, 1] = np.clip(corners_arr[:, 1] / frame_height, 0.0, 1.0)

    return (
        f"0 {corners_arr[0, 0]:.6f} {corners_arr[0, 1]:.6f} "
        f"{corners_arr[1, 0]:.6f} {corners_arr[1, 1]:.6f} "
        f"{corners_arr[2, 0]:.6f} {corners_arr[2, 1]:.6f} "
        f"{corners_arr[3, 0]:.6f} {corners_arr[3, 1]:.6f}\n"
    )


def _compute_obb_corners(cx, cy, w, h, theta, frame_width, frame_height):
    """Compute normalized OBB corners and return an OBB annotation line."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    hw, hh = w / 2.0, h / 2.0

    corners_local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    corners = corners_local @ rotation_matrix.T + np.array([cx, cy])

    corners_norm = corners.copy()
    corners_norm[:, 0] /= frame_width
    corners_norm[:, 1] /= frame_height

    return (
        f"0 {corners_norm[0, 0]:.6f} {corners_norm[0, 1]:.6f} "
        f"{corners_norm[1, 0]:.6f} {corners_norm[1, 1]:.6f} "
        f"{corners_norm[2, 0]:.6f} {corners_norm[2, 1]:.6f} "
        f"{corners_norm[3, 0]:.6f} {corners_norm[3, 1]:.6f}\n"
    )


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


def _write_frame_annotations(
    frame_id,
    frame,
    df,
    yolo_detections,
    params,
    images_dir,
    labels_dir,
    frame_width,
    frame_height,
    scale_back,
):
    """Save one frame image and its YOLO OBB annotation file.

    Returns (image_filename, label_filename, annotations_list).
    """
    import pandas as pd

    image_filename = f"f{frame_id:06d}.jpg"
    cv2.imwrite(str(images_dir / image_filename), frame)

    label_filename = f"f{frame_id:06d}.txt"
    label_path = labels_dir / label_filename

    frame_detections = df[df["FrameID"] == frame_id]
    annotations = []

    with open(label_path, "w") as f:
        for _, detection in frame_detections.iterrows():
            if pd.isna(detection["X"]) or pd.isna(detection["Y"]):
                continue

            cx = detection["X"] * scale_back
            cy = detection["Y"] * scale_back
            theta = detection["Theta"]

            w, h, matched_corners, dimension_source = _match_yolo_detection(
                cx, cy, yolo_detections, frame_id
            )
            if w is None or h is None:
                ref_size = params.get("REFERENCE_BODY_SIZE", 20.0)
                w = ref_size * 2.2
                h = ref_size * 0.8
                logger.debug(f"Frame {frame_id}: Using reference size approximation")

            if matched_corners is not None:
                obb_line = _format_obb_corners(
                    matched_corners,
                    frame_width,
                    frame_height,
                )
            else:
                obb_line = _compute_obb_corners(
                    cx, cy, w, h, theta, frame_width, frame_height
                )
            f.write(obb_line)

            track_id = -1
            if "TrackID" in detection:
                track_id = int(detection["TrackID"])
            elif "TrajectoryID" in detection:
                track_id = int(detection["TrajectoryID"])

            annotations.append(
                {
                    "track_id": track_id,
                    "x": float(cx),
                    "y": float(cy),
                    "theta": float(theta),
                    "dimension_source": dimension_source,
                    "state": detection.get("State", "unknown"),
                }
            )

    return image_filename, label_filename, annotations


def _write_dataset_files(
    dataset_dir, dataset_name, class_name, metadata, exported_count
):
    """Write classes.txt, metadata.json, and README.md for the dataset."""
    classes_path = dataset_dir / "classes.txt"
    with open(classes_path, "w") as f:
        f.write(f"{class_name}\n")
    logger.info(f"Created classes.txt with class: {class_name}")

    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    readme_path = dataset_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"# {dataset_name}\n\n")
        f.write("This dataset was automatically generated for active learning.\n\n")
        f.write("## Contents\n\n")
        f.write(f"- **images/**: {exported_count} exported frames\n")
        f.write("- **labels/**: YOLO OBB format annotations (initial, needs review)\n")
        f.write(f"- **classes.txt**: Object class definition ({class_name})\n")
        f.write("- **metadata.json**: Detailed frame and annotation metadata\n\n")
        f.write("## Next Steps\n\n")
        f.write("1. **Review and correct annotations** using x-AnyLabeling:\n")
        f.write("   - Use the 'Open in X-AnyLabeling' button in the tracker GUI\n")
        f.write("   - Or manually run: xanylabeling --filename ./images\n")
        f.write("   - Review and correct the OBB annotations\n\n")
        f.write("2. **Train improved YOLO model**:\n")
        f.write("   - Combine this dataset with your existing training data\n")
        f.write("   - Use YOLO training scripts with the corrected annotations\n")
        f.write("   - Update your model in the tracker configuration\n\n")
        f.write("3. **Iterate**:\n")
        f.write("   - Run tracking with the new model\n")
        f.write("   - Generate another dataset if needed\n")
        f.write("   - Repeat until performance is satisfactory\n")


def export_dataset(
    video_path: object,
    csv_path: object,
    frame_ids: object,
    output_dir: object,
    dataset_name: object,
    class_name: object,
    params: object,
    include_context: object = True,
    _yolo_results_dict: object = None,
) -> object:
    """
    Export selected frames and annotations as a training dataset.

    Args:
        video_path: Path to source video
        csv_path: Path to tracking CSV (for reading annotations)
        frame_ids: List of frame IDs to export
        output_dir: Directory to save dataset
        dataset_name: Name for the dataset
        class_name: Name of the object class (for classes.txt file)
        params: Parameters dict (for accessing RESIZE_FACTOR and REFERENCE_BODY_SIZE)
        include_context: Include ±1 frames around each selected frame
        yolo_results_dict: Optional dict of {frame_id: yolo_detections} for YOLO format export

    Returns:
        zip_path: Path to created zip file
    """
    import pandas as pd

    logger.info(f"Starting dataset export for {len(frame_ids)} frames")

    dataset_dir, images_dir, labels_dir = _make_dataset_dir(output_dir, dataset_name)
    runner = _init_detection_runner(params)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    df = pd.read_csv(csv_path)
    frames_to_export = _expand_frame_ids(frame_ids, include_context, total_frames)
    logger.info(f"Exporting {len(frames_to_export)} frames (including context)")

    exported_count = 0
    metadata = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "source_video": str(video_path),
        "source_csv": str(csv_path),
        "total_frames": len(frames_to_export),
        "image_width": frame_width,
        "image_height": frame_height,
        "frames": [],
    }

    # Determine batch size for YOLO processing
    batch_size = _get_detector_batch_size(runner)

    # Process frames in batches
    frame_batches = [
        frames_to_export[i : i + batch_size]
        for i in range(0, len(frames_to_export), batch_size)
    ]

    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    scale_back = _csv_scale_back(df, resize_factor, frame_width, frame_height)

    try:
        for batch_idx, batch_frame_ids in enumerate(frame_batches):
            batch_frames, batch_frames_original, valid_batch_indices = (
                _read_batch_frames(cap, batch_frame_ids, params)
            )
            if not batch_frames:
                continue

            try:
                batch_yolo_detections = _detect_batch(
                    runner, batch_frames, batch_frame_ids, valid_batch_indices, params
                )
            except Exception as e:
                logger.error(f"YOLO detection failed for batch {batch_idx}: {e}")
                batch_yolo_detections = [{}] * len(batch_frames)

            for frame_idx, (frame_id, frame) in enumerate(batch_frames_original):
                yolo_detections = batch_yolo_detections[frame_idx]
                dataset_conf = params.get("DATASET_YOLO_CONFIDENCE_THRESHOLD", 0.05)
                dataset_iou = params.get("DATASET_YOLO_IOU_THRESHOLD", 0.5)
                logger.debug(
                    f"Frame {frame_id}: Found {len(yolo_detections)} YOLO detections "
                    f"(conf={dataset_conf:.2f}, iou={dataset_iou:.2f})"
                )

                image_filename, label_filename, annotations = _write_frame_annotations(
                    frame_id,
                    frame,
                    df,
                    yolo_detections,
                    params,
                    images_dir,
                    labels_dir,
                    frame_width,
                    frame_height,
                    scale_back,
                )

                metadata["frames"].append(
                    {
                        "frame_id": int(frame_id),
                        "image_file": image_filename,
                        "label_file": label_filename,
                        "annotations": annotations,
                    }
                )
                exported_count += 1
    finally:
        cap.release()
        if runner is not None:
            runner.close()

    _write_dataset_files(
        dataset_dir, dataset_name, class_name, metadata, exported_count
    )

    logger.info(f"Dataset exported successfully to {dataset_dir}")
    logger.info(f"Exported {exported_count} frames with annotations")

    return str(dataset_dir)


def _get_detector_batch_size(runner):
    """Return the batch size to use for detection."""
    if runner is not None and getattr(runner, "config", None) is not None:
        return max(1, int(getattr(runner.config, "detection_batch_size", 1)))
    return 1


def _read_batch_frames(cap, batch_frame_ids, params):
    """Read and preprocess all frames in a batch for detection.

    Returns (batch_frames, batch_frames_original, valid_batch_indices).
    """
    batch_frames = []
    batch_frames_original = []
    valid_batch_indices = []
    first_frame_shape = None

    for idx, frame_id in enumerate(batch_frame_ids):
        result = _read_and_resize_frame(cap, frame_id, params, first_frame_shape)
        if result is None:
            continue
        frame, frame_for_detection, first_frame_shape = result
        batch_frames_original.append((frame_id, frame))
        batch_frames.append(frame_for_detection)
        valid_batch_indices.append(idx)

    return batch_frames, batch_frames_original, valid_batch_indices
