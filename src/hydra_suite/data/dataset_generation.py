"""
Dataset generation utilities for active learning.
Identifies challenging frames and exports them for annotation.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameQualityScorer:
    """
    Scores frames based on various quality metrics to identify
    challenging frames that would benefit from additional training data.
    """

    def __init__(self, params):
        self.params = params
        self.frame_scores = defaultdict(lambda: {"score": 0.0, "metrics": {}})
        self.max_targets = params.get("MAX_TARGETS", 4)
        self.conf_threshold = params.get("DATASET_CONF_THRESHOLD", 0.5)

        # Enabled metrics
        self.use_confidence = params.get("METRIC_LOW_CONFIDENCE", True)
        self.use_count_mismatch = params.get("METRIC_COUNT_MISMATCH", True)
        self.use_assignment_cost = params.get("METRIC_HIGH_ASSIGNMENT_COST", True)
        self.use_track_loss = params.get("METRIC_TRACK_LOSS", True)
        self.use_uncertainty = params.get("METRIC_HIGH_UNCERTAINTY", False)

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
        if not (self.use_assignment_cost and "assignment_costs" in tracking_data):
            return 0.0
        costs = tracking_data["assignment_costs"]
        if not costs:
            return 0.0
        avg_cost = np.mean(costs)
        cost_score = min(avg_cost / 50.0, 1.0)
        metrics["high_assignment_cost"] = {
            "avg": avg_cost,
            "max": max(costs),
            "score": cost_score,
        }
        return cost_score * 0.15

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

    def score_frame(
        self: object,
        frame_id: object,
        detection_data: object = None,
        tracking_data: object = None,
    ) -> object:
        """
        Score a single frame based on enabled quality metrics.

        Args:
            frame_id: Frame number
            detection_data: Dict with detection information
                - confidences: List of detection confidence scores
                - count: Number of detections
            tracking_data: Dict with tracking information
                - assignment_costs: List of assignment costs
                - lost_tracks: Number of lost tracks
                - uncertainties: List of position uncertainties

        Returns:
            score: Higher score = more problematic frame (0.0 to 1.0+)
        """
        if detection_data is None:
            detection_data = {}
        if tracking_data is None:
            tracking_data = {}

        metrics = {}
        score = (
            self._score_confidence(detection_data, metrics)
            + self._score_count_mismatch(detection_data, metrics)
            + self._score_assignment_cost(tracking_data, metrics)
            + self._score_track_loss(tracking_data, metrics)
            + self._score_uncertainty(tracking_data, metrics)
        )

        self.frame_scores[frame_id] = {"score": score, "metrics": metrics}
        return score

    def get_worst_frames(
        self: object,
        max_frames: object,
        diversity_window: object = 30,
        probabilistic: object = True,
    ) -> object:
        """
        Select the worst N frames with visual diversity constraint.

        Args:
            max_frames: Maximum number of frames to select
            diversity_window: Minimum frame separation to ensure diversity
            probabilistic: If True, use rank-based probabilistic sampling.
                          If False, use greedy selection (worst frames first).

        Returns:
            selected_frames: List of frame IDs sorted by score (worst first)
        """
        if not self.frame_scores:
            return []

        # Sort frames by score (descending)
        sorted_frames = sorted(
            self.frame_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )

        if probabilistic:
            # Rank-based weighted sampling for better variety
            # Higher rank (worse frame) = higher probability
            candidates = sorted_frames.copy()
            selected = []

            while len(selected) < max_frames and candidates:
                # Calculate rank-based weights for remaining candidates
                # weight = 1 / (rank + 1), normalized
                weights = np.array([1.0 / (i + 1) for i in range(len(candidates))])
                weights = weights / weights.sum()  # Normalize to probabilities

                # Sample one frame
                idx = np.random.choice(len(candidates), p=weights)
                frame_id, data = candidates[idx]

                # Check diversity constraint
                if all(abs(frame_id - sel) >= diversity_window for sel in selected):
                    selected.append(frame_id)

                # Remove this candidate (and nearby frames to speed up)
                # Remove frames within diversity_window to avoid checking them repeatedly
                candidates = [
                    (fid, fdata)
                    for fid, fdata in candidates
                    if abs(fid - frame_id) >= diversity_window
                ]

            logger.info(
                f"Probabilistically selected {len(selected)} frames out of {len(sorted_frames)} "
                f"with diversity window of {diversity_window} frames"
            )
        else:
            # Greedy selection: take worst frames first
            selected = []
            for frame_id, data in sorted_frames:
                if len(selected) >= max_frames:
                    break

                # Check if this frame is far enough from already selected frames
                if all(abs(frame_id - sel) >= diversity_window for sel in selected):
                    selected.append(frame_id)

            logger.info(
                f"Greedily selected {len(selected)} frames out of {len(sorted_frames)} "
                f"with diversity window of {diversity_window} frames"
            )

        return selected


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


def _init_yolo_detector(params):
    """Initialize YOLO detector if detection method is yolo_obb."""
    from ..core.detectors import create_detector

    detection_method = params.get("DETECTION_METHOD", "background_subtraction")
    if detection_method != "yolo_obb":
        return None
    try:
        detector = create_detector(params)
        logger.info("YOLO detector initialized for dimension extraction")
        return detector
    except Exception as e:
        logger.warning(
            f"Could not initialize YOLO detector: {e}. Using reference size approximation."
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


def _run_batched_detection(detector, batch_frames, batch_frame_ids, params):
    """Run batched YOLO detection and return list of detection dicts."""
    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    batch_results = detector.detect_objects_batched(
        batch_frames,
        start_frame_idx=(batch_frame_ids[0] if batch_frame_ids else 0),
    )
    batch_yolo_detections = []
    for meas, _sizes, shapes, _confidences, obb_corners in batch_results:
        batch_yolo_detections.append(
            _measurements_to_detections(meas, shapes, resize_factor, obb_corners)
        )
    return batch_yolo_detections


def _run_single_frame_detections(
    detector, batch_frames, batch_frame_ids, valid_batch_indices, params
):
    """Fallback: run detection on each frame individually."""
    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    results = [{}] * len(batch_frames)
    for frame_idx, frame_for_detection in enumerate(batch_frames):
        fid = batch_frame_ids[valid_batch_indices[frame_idx]]
        try:
            meas, _sizes, shapes, _yolo_results, _confidences = detector.detect_objects(
                frame_for_detection, fid
            )
            results[frame_idx] = _measurements_to_detections(
                meas, shapes, resize_factor
            )
        except Exception as inner_e:
            logger.warning(
                f"Single-frame detection also failed for frame {fid}: {inner_e}"
            )
            results[frame_idx] = {}
    return results


def _detect_batch(detector, batch_frames, batch_frame_ids, valid_batch_indices, params):
    """Run YOLO detection on a batch, with batched->single-frame fallback."""
    if detector is None or not batch_frames:
        return [{}] * len(batch_frames)

    original_conf = params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25)
    original_iou = params.get("YOLO_IOU_THRESHOLD", 0.7)
    dataset_conf = params.get("DATASET_YOLO_CONFIDENCE_THRESHOLD", 0.05)
    dataset_iou = params.get("DATASET_YOLO_IOU_THRESHOLD", 0.5)

    params["YOLO_CONFIDENCE_THRESHOLD"] = dataset_conf
    params["YOLO_IOU_THRESHOLD"] = dataset_iou
    try:
        if hasattr(detector, "detect_objects_batched"):
            return _run_batched_detection(
                detector, batch_frames, batch_frame_ids, params
            )
        raise AttributeError("Batched detection not available")
    except (AttributeError, Exception) as e:
        if not isinstance(e, AttributeError):
            logger.warning(
                f"Batched detection failed: {e}, falling back to single-frame processing"
            )
        return _run_single_frame_detections(
            detector, batch_frames, batch_frame_ids, valid_batch_indices, params
        )
    finally:
        params["YOLO_CONFIDENCE_THRESHOLD"] = original_conf
        params["YOLO_IOU_THRESHOLD"] = original_iou


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
    detector = _init_yolo_detector(params)

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
    batch_size = _get_detector_batch_size(detector)

    # Process frames in batches
    frame_batches = [
        frames_to_export[i : i + batch_size]
        for i in range(0, len(frames_to_export), batch_size)
    ]

    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    scale_back = _csv_scale_back(df, resize_factor, frame_width, frame_height)

    for batch_idx, batch_frame_ids in enumerate(frame_batches):
        batch_frames, batch_frames_original, valid_batch_indices = _read_batch_frames(
            cap, batch_frame_ids, params
        )
        if not batch_frames:
            continue

        try:
            batch_yolo_detections = _detect_batch(
                detector, batch_frames, batch_frame_ids, valid_batch_indices, params
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

    cap.release()

    _write_dataset_files(
        dataset_dir, dataset_name, class_name, metadata, exported_count
    )

    logger.info(f"Dataset exported successfully to {dataset_dir}")
    logger.info(f"Exported {exported_count} frames with annotations")

    return str(dataset_dir)


def _get_detector_batch_size(detector):
    """Return the batch size to use for YOLO processing."""
    batch_size = 1
    if (
        detector is not None
        and hasattr(detector, "use_tensorrt")
        and detector.use_tensorrt
        and hasattr(detector, "tensorrt_batch_size")
    ):
        batch_size = detector.tensorrt_batch_size
        logger.info(f"Using TensorRT batch processing with batch size {batch_size}")
    return batch_size


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
