"""InterpolatedCropsWorker — per-animal interpolated crop export worker."""

import gc
import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Signal

from hydra_suite.core.identity.dataset.generator import IndividualDatasetGenerator
from hydra_suite.core.identity.properties.export import (
    POSE_SUMMARY_COLUMNS,
    build_pose_keypoint_labels,
    flatten_cnn_prediction_row,
    flatten_pose_keypoints_row,
    pose_wide_columns_for_labels,
)
from hydra_suite.data.detection_cache import DetectionCache
from hydra_suite.utils.geometry import wrap_angle_degs
from hydra_suite.widgets.workers import BaseWorker

from .merge_worker import _write_csv_artifact, _write_roi_npz

logger = logging.getLogger(__name__)


class InterpolatedCropsWorker(BaseWorker):
    """Worker thread for interpolating occluded crops without blocking the UI."""

    progress_signal = Signal(int, str)
    finished_signal = Signal(dict)

    def __init__(
        self,
        csv_path,
        video_path,
        detection_cache_path,
        params,
        enable_profiling=False,
        profile_export_path=None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.params = params
        self.enable_profiling = enable_profiling
        self.profile_export_path = profile_export_path
        self._stop_requested = False

    def stop(self):
        """Request cooperative cancellation."""
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    @staticmethod
    def _interp_angle(theta_start, theta_end, t):
        deg0 = math.degrees(theta_start)
        deg1 = math.degrees(theta_end)
        candidates = (deg1, deg1 + 180.0, deg1 - 180.0)
        best_delta = None
        for cand in candidates:
            delta = wrap_angle_degs(cand - deg0)
            if best_delta is None or abs(delta) < abs(best_delta):
                best_delta = delta
        return math.radians(deg0 + (best_delta or 0.0) * t)

    @staticmethod
    @staticmethod
    def _size_from_obb_corners(obb_corners, idx):
        if not (obb_corners and idx < len(obb_corners)):
            return None, None
        c = np.asarray(obb_corners[idx], dtype=np.float32)
        if c.shape[0] < 4:
            return None, None
        w = float(np.linalg.norm(c[1] - c[0]))
        h = float(np.linalg.norm(c[2] - c[1]))
        if w < h:
            w, h = h, w
        return w, h

    @staticmethod
    def _size_from_shapes(shapes, idx):
        if not (shapes and idx < len(shapes)):
            return None, None
        area, aspect_ratio = shapes[idx][0], shapes[idx][1]
        if aspect_ratio > 0 and area > 0:
            ax2 = math.sqrt(4 * area / (math.pi * aspect_ratio))
            ax1 = aspect_ratio * ax2
            return ax1, ax2
        return None, None

    def _get_detection_size(detection_cache, frame_id, detection_id):
        if detection_cache is None or detection_id is None or pd.isna(detection_id):
            return None, None
        try:
            _, _, shapes, _, obb_corners, detection_ids, *_ = detection_cache.get_frame(
                int(frame_id)
            )
        except Exception:
            return None, None

        idx = None
        try:
            for i, did in enumerate(detection_ids):
                if int(did) == int(detection_id):
                    idx = i
                    break
        except Exception:
            idx = None

        if idx is None:
            return None, None

        w, h = InterpolatedCropsWorker._size_from_obb_corners(obb_corners, idx)
        if w is not None:
            return w, h

        return InterpolatedCropsWorker._size_from_shapes(shapes, idx)

    def _init_pose_backend(self, output_dir):
        """Initialize pose estimation backend. Returns (backend, kpt_source_names, kpt_labels)."""
        if not bool(self.params.get("ENABLE_POSE_EXTRACTOR", False)):
            return None, [], []
        from hydra_suite.core.identity.pose.api import (
            build_runtime_config,
            create_pose_backend_from_config,
        )

        try:
            pose_config = build_runtime_config(
                self.params, out_root=str(Path(output_dir).expanduser())
            )
            backend = create_pose_backend_from_config(pose_config)
            backend.warmup()
            kpt_source_names = list(getattr(backend, "output_keypoint_names", []) or [])
            kpt_labels = build_pose_keypoint_labels(
                kpt_source_names, len(kpt_source_names)
            )
            return backend, kpt_source_names, kpt_labels
        except Exception as exc:
            logger.warning(
                "Interpolated pose analysis disabled (backend init failed): %s",
                exc,
            )
            return None, [], []

    def _init_apriltag_detector(self):
        """Initialize AprilTag detector if configured. Returns detector or None."""
        apriltag_enabled = (
            bool(self.params.get("USE_APRILTAGS", False))
            or str(self.params.get("IDENTITY_METHOD", "")).lower() == "apriltags"
        )
        if not apriltag_enabled:
            return None
        try:
            from hydra_suite.core.identity.classification.apriltag import (
                AprilTagConfig,
                AprilTagDetector,
            )

            return AprilTagDetector(AprilTagConfig.from_params(self.params))
        except Exception as exc:
            logger.warning("Interpolated AprilTag analysis disabled: %s", exc)
            return None

    def _init_cnn_backends(self):
        """Initialize CNN identity backends. Returns (backends_list, labels_list)."""
        cnn_backends = []
        cnn_labels = []
        cnn_classifiers_cfg = self.params.get("CNN_CLASSIFIERS", [])
        if not cnn_classifiers_cfg:
            return cnn_backends, cnn_labels
        try:
            from hydra_suite.core.identity.classification.cnn import (
                CNNIdentityBackend,
                CNNIdentityConfig,
            )

            compute_rt = str(
                self.params.get(
                    "CNN_COMPUTE_RUNTIME", self.params.get("COMPUTE_RUNTIME", "cpu")
                )
            )
            for cnn_cfg_dict in cnn_classifiers_cfg:
                model_path = str(cnn_cfg_dict.get("model_path", ""))
                if not model_path or not os.path.exists(model_path):
                    continue
                label = str(cnn_cfg_dict.get("label", "cnn_identity"))
                cnn_cfg = CNNIdentityConfig(
                    model_path=model_path,
                    confidence=float(cnn_cfg_dict.get("confidence", 0.5)),
                    scoring_mode=str(cnn_cfg_dict.get("scoring_mode", "atomic")),
                    batch_size=int(cnn_cfg_dict.get("batch_size", 64)),
                )
                try:
                    backend = CNNIdentityBackend(
                        cnn_cfg,
                        model_path=model_path,
                        compute_runtime=compute_rt,
                    )
                    cnn_backends.append(backend)
                    cnn_labels.append(label)
                except Exception as exc:
                    logger.warning(
                        "Interpolated CNN identity '%s' disabled: %s",
                        label,
                        exc,
                    )
        except Exception as exc:
            logger.warning("Interpolated CNN identity analysis disabled: %s", exc)
        return cnn_backends, cnn_labels

    def _init_headtail_analyzer(self):
        """Initialize head-tail direction analyzer. Returns analyzer or None.

        Raises:
            ClassifierFormatError: if the configured model path exists but
                cannot be loaded by any supported backend.  Callers in
                BaseWorker.run() forward this to the ``error`` signal.
        """
        headtail_model_path = str(self.params.get("YOLO_HEADTAIL_MODEL_PATH", ""))
        if not headtail_model_path or not os.path.exists(headtail_model_path):
            return None
        from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

        _ht_runtime = str(
            self.params.get(
                "HEADTAIL_COMPUTE_RUNTIME",
                self.params.get("COMPUTE_RUNTIME", "cpu"),
            )
        )
        analyzer = HeadTailAnalyzer(
            model_path=headtail_model_path,
            compute_runtime=_ht_runtime,
            conf_threshold=float(self.params.get("YOLO_HEADTAIL_CONF_THRESHOLD", 0.5)),
            batch_size=max(1, int(self.params.get("HEADTAIL_BATCH_SIZE", 64))),
            reference_aspect_ratio=float(
                self.params.get("REFERENCE_ASPECT_RATIO", 2.0)
            ),
        )
        if not analyzer.is_available:
            analyzer.close()
            return None
        return analyzer

    def _init_interpolation_backends(self, output_dir):
        """Initialize optional analysis backends after eligible gap tasks exist."""
        pose_backend, pose_kpt_source_names, pose_kpt_labels = self._init_pose_backend(
            output_dir
        )
        apriltag_detector = self._init_apriltag_detector()
        cnn_backends, cnn_labels = self._init_cnn_backends()
        headtail_analyzer = self._init_headtail_analyzer()
        interp_cnn_rows = {label: [] for label in cnn_labels}
        return (
            pose_backend,
            pose_kpt_source_names,
            pose_kpt_labels,
            apriltag_detector,
            cnn_backends,
            cnn_labels,
            headtail_analyzer,
            interp_cnn_rows,
        )

    @staticmethod
    def _empty_artifact_paths():
        return {
            "mapping_path": None,
            "roi_csv_path": None,
            "roi_npz_path": None,
            "pose_csv_path": None,
            "tag_csv_path": None,
            "cnn_csv_paths": {},
            "headtail_csv_path": None,
        }

    def _load_and_validate_csv(self):
        """Load CSV and validate required columns. Returns DataFrame or None."""
        df = pd.read_csv(self.csv_path)
        if "FrameID" not in df.columns and "Frame" in df.columns:
            df = df.rename(columns={"Frame": "FrameID"})
        if "TrajectoryID" not in df.columns and "Trajectory" in df.columns:
            df = df.rename(columns={"Trajectory": "TrajectoryID"})
        if df.empty or "FrameID" not in df.columns or "State" not in df.columns:
            return None
        for col in ("FrameID", "X", "Y", "Theta"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _process_occluded_run(
        self,
        group,
        traj_id,
        last_valid_idx,
        i,
        j,
        detection_cache,
        position_scale,
        size_scale,
        frame_tasks,
        interp_runs,
        interp_gaps,
    ):
        prev_row = group.iloc[last_valid_idx]
        next_row = group.iloc[j]
        if (
            pd.isna(prev_row["X"])
            or pd.isna(prev_row["Y"])
            or pd.isna(next_row["X"])
            or pd.isna(next_row["Y"])
        ):
            return interp_runs, interp_gaps, j

        f0 = int(prev_row["FrameID"])
        f1 = int(next_row["FrameID"])
        if f1 - f0 <= 1:
            return interp_runs, interp_gaps, j
        interp_runs += 1

        interp_total = max(0, f1 - f0 - 1)
        interp_gaps += interp_total

        det_id_prev = (
            prev_row["DetectionID"] if "DetectionID" in group.columns else None
        )
        det_id_next = (
            next_row["DetectionID"] if "DetectionID" in group.columns else None
        )

        w0, h0 = self._get_detection_size(detection_cache, f0, det_id_prev)
        w1, h1 = self._get_detection_size(detection_cache, f1, det_id_next)

        if w0 is None or h0 is None or w1 is None or h1 is None:
            ref_size = self.params.get("REFERENCE_BODY_SIZE", 20.0)
            w0 = w0 or ref_size * 2.2
            h0 = h0 or ref_size * 0.8
            w1 = w1 or ref_size * 2.2
            h1 = h1 or ref_size * 0.8

        for k in range(i, j):
            if self._should_stop():
                return None
            row = group.iloc[k]
            f = int(row["FrameID"])
            t = (f - f0) / (f1 - f0)
            cx = float(prev_row["X"]) + t * (
                float(next_row["X"]) - float(prev_row["X"])
            )
            cy = float(prev_row["Y"]) + t * (
                float(next_row["Y"]) - float(prev_row["Y"])
            )
            theta = self._interp_angle(
                float(prev_row["Theta"]), float(next_row["Theta"]), t
            )
            w = w0 + t * (w1 - w0)
            h = h0 + t * (h1 - h0)

            interp_index = max(1, f - f0)

            frame_tasks[f].append(
                {
                    "frame_id": f,
                    "cx": cx * position_scale,
                    "cy": cy * position_scale,
                    "w": w * size_scale,
                    "h": h * size_scale,
                    "theta": theta,
                    "traj_id": traj_id,
                    "interp_from": (f0, f1),
                    "interp_index": interp_index,
                    "interp_total": interp_total,
                }
            )

        return interp_runs, interp_gaps, j

    def _scan_trajectory_gaps(
        self,
        traj_id,
        group,
        detection_cache,
        position_scale,
        size_scale,
        frame_tasks,
        interp_runs,
        interp_gaps,
    ):
        group = group.sort_values("FrameID").reset_index(drop=True)
        states = group["State"].astype(str).str.strip().str.lower()
        states = states.where(~states.str.contains("occluded", na=False), "occluded")
        traj_occluded = int((states == "occluded").sum())

        last_valid_idx = None
        i = 0
        while i < len(group):
            if self._should_stop():
                return None
            if states[i] != "occluded":
                if not pd.isna(group.at[i, "X"]) and not pd.isna(group.at[i, "Y"]):
                    last_valid_idx = i
                i += 1
                continue

            if last_valid_idx is None:
                i += 1
                continue

            j = i
            while j < len(group) and states[j] == "occluded":
                j += 1
            if j >= len(group):
                break

            run_result = self._process_occluded_run(
                group,
                traj_id,
                last_valid_idx,
                i,
                j,
                detection_cache,
                position_scale,
                size_scale,
                frame_tasks,
                interp_runs,
                interp_gaps,
            )
            if run_result is None:
                return None
            interp_runs, interp_gaps, i = run_result

        return interp_runs, interp_gaps, traj_occluded

    def _detect_interpolation_gaps(
        self, df, detection_cache, position_scale, size_scale
    ):
        """Scan trajectories for occluded gaps and build per-frame task lists.

        Returns (frame_tasks, occluded_rows, interp_runs, interp_gaps) or None if stopped.
        """
        occluded_rows = 0
        interp_gaps = 0
        interp_runs = 0
        frame_tasks = defaultdict(list)

        for traj_id, group in df.groupby("TrajectoryID"):
            if self._should_stop():
                return None
            result = self._scan_trajectory_gaps(
                traj_id,
                group,
                detection_cache,
                position_scale,
                size_scale,
                frame_tasks,
                interp_runs,
                interp_gaps,
            )
            if result is None:
                return None
            interp_runs, interp_gaps, traj_occluded = result
            occluded_rows += traj_occluded

        return frame_tasks, occluded_rows, interp_runs, interp_gaps

    def _flush_pose_batch(
        self,
        pose_backend,
        pending_crops,
        pending_entries,
        interp_pose_rows,
        pose_kpt_source_names,
        pose_kpt_labels,
        profiler,
    ):
        """Run pose inference on accumulated crops and append results."""
        from hydra_suite.core.canonicalization.crop import (
            invert_keypoints as _invert_kpts,
        )

        profiler.tick("interp_pose_inference")
        pose_results = pose_backend.predict_batch(pending_crops)
        profiler.tock("interp_pose_inference")
        for pidx, entry in enumerate(pending_entries):
            pose_out = pose_results[pidx] if pidx < len(pose_results) else None
            pose_mean_conf = 0.0
            pose_valid_fraction = 0.0
            pose_num_valid = 0
            pose_num_keypoints = 0
            pose_wide = {}
            if pose_out is not None:
                pose_mean_conf = float(getattr(pose_out, "mean_conf", 0.0))
                pose_valid_fraction = float(getattr(pose_out, "valid_fraction", 0.0))
                pose_num_valid = int(getattr(pose_out, "num_valid", 0))
                pose_num_keypoints = int(getattr(pose_out, "num_keypoints", 0))
                keypoints = getattr(pose_out, "keypoints", None)
                crop_info = entry.get("crop_info") or {}
                if keypoints is not None and len(keypoints) > 0:
                    gkpts = np.asarray(keypoints, dtype=np.float32).copy()
                    _M_inv = crop_info.get("M_inverse")
                    if _M_inv is not None and crop_info.get("canonical"):
                        gkpts = _invert_kpts(gkpts, _M_inv).astype(np.float32)
                    else:
                        crop_bbox = crop_info.get("crop_bbox")
                        if crop_bbox is not None and len(crop_bbox) >= 2:
                            gkpts[:, 0] += float(crop_bbox[0])
                            gkpts[:, 1] += float(crop_bbox[1])
                    if len(gkpts) > len(pose_kpt_labels):
                        pose_kpt_labels[:] = build_pose_keypoint_labels(
                            pose_kpt_source_names, len(gkpts)
                        )
                    pose_wide = flatten_pose_keypoints_row(gkpts, pose_kpt_labels)

            pose_row = {
                "frame_id": int(entry["task"]["frame_id"]),
                "trajectory_id": int(entry["task"]["traj_id"]),
                "filename": entry["filename"],
                "PoseMeanConf": pose_mean_conf,
                "PoseValidFraction": pose_valid_fraction,
                "PoseNumValid": pose_num_valid,
                "PoseNumKeypoints": pose_num_keypoints,
            }
            pose_row.update(pose_wide)
            interp_pose_rows.append(pose_row)
        pending_crops.clear()
        pending_entries.clear()

    @staticmethod
    def _flush_cnn_batch(
        cnn_backends,
        cnn_labels,
        pending_cnn_crops,
        pending_cnn_entries,
        interp_cnn_rows,
        profiler,
    ):
        """Run CNN identity inference on accumulated crops and append results."""
        profiler.tick("interp_cnn_inference")
        for _bi, _cnn_be in enumerate(cnn_backends):
            _cnn_label = cnn_labels[_bi]
            try:
                _cnn_preds = _cnn_be.predict_batch(pending_cnn_crops)
                for _pi, _pred in enumerate(_cnn_preds):
                    if _pi >= len(pending_cnn_entries):
                        break
                    _ce = pending_cnn_entries[_pi]
                    row = {
                        "frame_id": int(_ce["task"]["frame_id"]),
                        "trajectory_id": int(_ce["task"]["traj_id"]),
                    }
                    row.update(
                        flatten_cnn_prediction_row(
                            _cnn_label,
                            getattr(_pred, "factor_names", ("flat",)),
                            getattr(_pred, "class_names", ()),
                            getattr(_pred, "confidences", ()),
                        )
                    )
                    interp_cnn_rows[_cnn_label].append(row)
            except Exception as exc:
                logger.warning(
                    "Interp CNN '%s' batch failed: %s",
                    _cnn_label,
                    exc,
                )
        profiler.tock("interp_cnn_inference")
        pending_cnn_crops.clear()
        pending_cnn_entries.clear()

    @staticmethod
    def _detect_apriltags_in_frame(
        apriltag_detector,
        frame,
        frame_tasks_f,
        all_corners,
        params,
        interp_tag_rows,
    ):
        """Detect AprilTags in all interpolated crops for one frame."""
        from hydra_suite.core.tracking.pose.pose_pipeline import (
            extract_one_crop as _extract_aabb_crop,
        )

        _tag_crops = []
        _tag_offsets = []
        _tag_det_indices = []
        _tag_tasks = []
        _crop_padding = float(params.get("INDIVIDUAL_CROP_PADDING", 0.1))
        _suppress_foreign = bool(params.get("SUPPRESS_FOREIGN_OBB_REGIONS", True))
        _bg_color = tuple(params.get("INDIVIDUAL_BACKGROUND_COLOR", (0, 0, 0)))
        for ti, task in enumerate(frame_tasks_f):
            aabb_result = _extract_aabb_crop(
                frame,
                all_corners[ti],
                ti,
                _crop_padding,
                all_corners,
                _suppress_foreign,
                _bg_color,
            )
            if aabb_result is not None:
                crop, offset, _ = aabb_result
                _tag_crops.append(crop)
                _tag_offsets.append(offset)
                _tag_det_indices.append(ti)
                _tag_tasks.append(task)
        if _tag_crops:
            tag_obs = apriltag_detector.detect_in_crops(
                _tag_crops,
                _tag_offsets,
                det_indices=_tag_det_indices,
            )
            for obs in tag_obs:
                _ti = obs.det_index
                _ttask = _tag_tasks[_ti] if _ti < len(_tag_tasks) else _tag_tasks[0]
                interp_tag_rows.append(
                    {
                        "frame_id": int(_ttask["frame_id"]),
                        "trajectory_id": int(_ttask["traj_id"]),
                        "tag_id": int(obs.tag_id),
                        "center_x": float(obs.center_xy[0]),
                        "center_y": float(obs.center_xy[1]),
                        "hamming": int(obs.hamming),
                    }
                )

    @staticmethod
    def _detect_headtail_in_frame(
        headtail_analyzer,
        frame,
        frame_tasks_f,
        all_corners,
        interp_headtail_rows,
    ):
        """Detect head-tail directions for all interpolated detections in one frame."""
        ht_results = headtail_analyzer.analyze_crops([frame], [all_corners])
        if ht_results and ht_results[0]:
            for ti, (heading, conf, directed) in enumerate(ht_results[0]):
                task = frame_tasks_f[ti]
                interp_headtail_rows.append(
                    {
                        "frame_id": int(task["frame_id"]),
                        "trajectory_id": int(task["traj_id"]),
                        "heading_rad": float(heading),
                        "heading_conf": float(conf),
                        "heading_directed": int(directed),
                    }
                )

    @staticmethod
    def _write_interpolation_artifacts(
        gen,
        save_interpolated_outputs,
        cache_interpolated_artifacts,
        interp_rows,
        roi_rows,
        roi_corners,
        interp_pose_rows,
        interp_tag_rows,
        interp_cnn_rows,
        interp_headtail_rows,
        pose_kpt_labels,
    ):
        """Write all interpolation CSV/NPZ artifacts to disk.

        Returns dict of artifact paths.
        """
        result = {
            "mapping_path": None,
            "roi_csv_path": None,
            "roi_npz_path": None,
            "pose_csv_path": None,
            "tag_csv_path": None,
            "cnn_csv_paths": {},
            "headtail_csv_path": None,
        }

        parent = getattr(gen, "run_dir", None)
        if parent is None and gen.crops_dir is not None:
            parent = gen.crops_dir.parent
        if parent is None:
            return result

        if save_interpolated_outputs and interp_rows:
            result["mapping_path"] = _write_csv_artifact(
                parent / "interpolated_mapping.csv",
                [
                    "frame_id",
                    "trajectory_id",
                    "filename",
                    "interp_from_start",
                    "interp_from_end",
                    "interp_index",
                    "interp_total",
                ],
                interp_rows,
            )

        if cache_interpolated_artifacts and roi_rows:
            result["roi_csv_path"] = _write_csv_artifact(
                parent / "interpolated_rois.csv",
                [
                    "frame_id",
                    "trajectory_id",
                    "filename",
                    "cx",
                    "cy",
                    "w",
                    "h",
                    "theta",
                    "interp_from_start",
                    "interp_from_end",
                    "interp_index",
                    "interp_total",
                ],
                roi_rows,
            )
            result["roi_npz_path"] = _write_roi_npz(
                parent / "interpolated_rois.npz", roi_rows, roi_corners
            )

        if save_interpolated_outputs and interp_pose_rows:
            pose_fieldnames = [
                "frame_id",
                "trajectory_id",
                "filename",
                *POSE_SUMMARY_COLUMNS,
                *pose_wide_columns_for_labels(pose_kpt_labels),
            ]
            result["pose_csv_path"] = _write_csv_artifact(
                parent / "interpolated_pose.csv", pose_fieldnames, interp_pose_rows
            )

        if interp_tag_rows:
            result["tag_csv_path"] = _write_csv_artifact(
                parent / "interpolated_tags.csv",
                [
                    "frame_id",
                    "trajectory_id",
                    "tag_id",
                    "center_x",
                    "center_y",
                    "hamming",
                ],
                interp_tag_rows,
            )

        cnn_csv_paths = {}
        for _cnn_label, _cnn_rows in interp_cnn_rows.items():
            if _cnn_rows:
                fieldnames = ["frame_id", "trajectory_id"]
                for _cnn_row in _cnn_rows:
                    for _key in _cnn_row:
                        if _key not in fieldnames:
                            fieldnames.append(_key)
                path = _write_csv_artifact(
                    parent / f"interpolated_cnn_{_cnn_label}.csv",
                    fieldnames,
                    _cnn_rows,
                )
                if path is not None:
                    cnn_csv_paths[_cnn_label] = str(path)
        result["cnn_csv_paths"] = cnn_csv_paths

        if interp_headtail_rows:
            result["headtail_csv_path"] = _write_csv_artifact(
                parent / "interpolated_headtail.csv",
                [
                    "frame_id",
                    "trajectory_id",
                    "heading_rad",
                    "heading_conf",
                    "heading_directed",
                ],
                interp_headtail_rows,
            )

        return result

    @staticmethod
    def _cleanup_backends(
        cap,
        detection_cache,
        pose_backend,
        apriltag_detector,
        cnn_backends,
        headtail_analyzer,
    ):
        """Safely close all backends and resources."""
        for resource in (
            cap,
            detection_cache,
            pose_backend,
            apriltag_detector,
            headtail_analyzer,
        ):
            if resource is not None:
                try:
                    if hasattr(resource, "release"):
                        resource.release()
                    elif hasattr(resource, "close"):
                        resource.close()
                except Exception:
                    pass
        for _be in cnn_backends or []:
            try:
                _be.close()
            except Exception:
                pass

    def _build_prefetcher(self, cap, needed_frames, total_frames):
        from hydra_suite.utils.frame_prefetcher import (
            SequentialScanPrefetcher,
            SparseFramePrefetcher,
        )

        _frame_range = needed_frames[-1] - needed_frames[0] + 1
        _density = total_frames / max(_frame_range, 1)
        _use_sequential = (
            _density >= 0.05 or (_frame_range / max(total_frames, 1)) < 100
        )
        if _use_sequential:
            logger.info(
                "Interpolation: sequential scan (%d needed / %d range, "
                "density=%.2f%%)",
                total_frames,
                _frame_range,
                _density * 100,
            )
            return SequentialScanPrefetcher(cap, needed_frames, buffer_size=8)
        logger.info(
            "Interpolation: sparse seek (%d needed / %d range, " "density=%.2f%%)",
            total_frames,
            _frame_range,
            _density * 100,
        )
        return SparseFramePrefetcher(cap, needed_frames, buffer_size=4)

    def _compute_frame_corners_and_affines(
        self, tasks, _compute_native_scale, _canonical_ref_ar, _canonical_padding
    ):
        from hydra_suite.core.identity.geometry import ellipse_to_obb_corners as _e2obb

        corners = [_e2obb(t["cx"], t["cy"], t["w"], t["h"], t["theta"]) for t in tasks]
        affines = []
        for _c in corners:
            try:
                _M, _cw, _ch, _ = _compute_native_scale(
                    _c, _canonical_ref_ar, _canonical_padding
                )
                affines.append((_M, _cw, _ch))
            except (ValueError, Exception):
                affines.append(None)
        return corners, affines

    def _process_single_task(
        self,
        task,
        task_idx,
        frame,
        _frame_all_corners,
        _frame_affines,
        gen,
        save_interpolated_outputs,
        _extract_canonical,
        cnn_backends,
        pose_backend,
        interp_saved,
        interp_rows,
        roi_rows,
        roi_corners,
        _pending_crops,
        _pending_entries,
        _pending_cnn_crops,
        _pending_cnn_entries,
    ):
        corners = _frame_all_corners[task_idx]
        _aff = _frame_affines[task_idx]
        filename = ""
        if save_interpolated_outputs:
            filename = gen.save_interpolated_crop(
                frame=frame,
                frame_id=task["frame_id"],
                cx=task["cx"],
                cy=task["cy"],
                w=task["w"],
                h=task["h"],
                theta=task["theta"],
                traj_id=task["traj_id"],
                interp_from=task["interp_from"],
                interp_index=task["interp_index"],
                interp_total=task["interp_total"],
                canonical_affine=(_aff[0] if _aff is not None else None),
            )
        if save_interpolated_outputs and filename:
            interp_saved += 1
            interp_rows.append(
                {
                    "frame_id": int(task["frame_id"]),
                    "trajectory_id": int(task["traj_id"]),
                    "filename": filename,
                    "interp_from_start": int(task["interp_from"][0]),
                    "interp_from_end": int(task["interp_from"][1]),
                    "interp_index": int(task["interp_index"]),
                    "interp_total": int(task["interp_total"]),
                }
            )
            roi_rows.append(
                {
                    "frame_id": int(task["frame_id"]),
                    "trajectory_id": int(task["traj_id"]),
                    "filename": filename,
                    "cx": float(task["cx"]),
                    "cy": float(task["cy"]),
                    "w": float(task["w"]),
                    "h": float(task["h"]),
                    "theta": float(task["theta"]),
                    "interp_from_start": int(task["interp_from"][0]),
                    "interp_from_end": int(task["interp_from"][1]),
                    "interp_index": int(task["interp_index"]),
                    "interp_total": int(task["interp_total"]),
                }
            )
            roi_corners.append(corners)
        if pose_backend is not None:
            pose_crop, pose_crop_info = self._extract_pose_crop(
                task_idx,
                frame,
                _frame_all_corners,
                _aff,
                corners,
                gen,
                _extract_canonical,
            )
            if pose_crop is not None and pose_crop.size > 0:
                _pending_crops.append(pose_crop)
                _pending_entries.append(
                    {"task": task, "filename": filename, "crop_info": pose_crop_info}
                )
            if cnn_backends and pose_crop is not None and pose_crop.size > 0:
                _pending_cnn_crops.append(pose_crop)
                _pending_cnn_entries.append({"task": task})
        return interp_saved

    def _extract_pose_crop(
        self,
        task_idx,
        frame,
        _frame_all_corners,
        _aff,
        corners,
        gen,
        _extract_canonical,
    ):
        pose_crop = None
        pose_crop_info = None
        try:
            _other_corners = [
                c for ci, c in enumerate(_frame_all_corners) if ci != task_idx
            ]
            if _aff is not None:
                _M_pose, _cw_pose, _ch_pose = _aff
                _foreign = _other_corners if _other_corners else None
                pose_crop = _extract_canonical(
                    frame,
                    _M_pose,
                    _cw_pose,
                    _ch_pose,
                    bg_color=gen.background_color,
                    foreign_corners=_foreign,
                )
                _M_inv = cv2.invertAffineTransform(_M_pose).astype(np.float32)
                pose_crop_info = {
                    "crop_size": (_cw_pose, _ch_pose),
                    "M_inverse": _M_inv,
                    "canonical": True,
                }
            else:
                pose_crop, pose_crop_info = gen._extract_obb_masked_crop(
                    frame,
                    corners,
                    frame.shape[0],
                    frame.shape[1],
                    other_corners_list=(_other_corners if _other_corners else None),
                )
        except Exception:
            pose_crop = None
            pose_crop_info = None
        return pose_crop, pose_crop_info

    def _build_finished_payload(
        self,
        interp_saved,
        interp_gaps,
        artifact_paths,
        interp_pose_rows,
        interp_tag_rows,
        interp_cnn_rows,
        interp_headtail_rows,
        save_interpolated_outputs,
        *,
        occluded_rows=0,
        interp_runs=0,
        eligible_frames=0,
        eligible_rows=0,
        roi_rows_count=0,
        no_work_reason="",
    ):
        def _str_or_none(p):
            return str(p) if p else None

        cnn_rows_produced = int(sum(len(rows) for rows in interp_cnn_rows.values()))

        return {
            "saved": interp_saved,
            "gaps": interp_gaps,
            "occluded_rows": int(occluded_rows),
            "interp_runs": int(interp_runs),
            "eligible_frames": int(eligible_frames),
            "eligible_rows": int(eligible_rows),
            "roi_rows_cached": int(roi_rows_count),
            "pose_rows_produced": int(len(interp_pose_rows)),
            "tag_rows_produced": int(len(interp_tag_rows)),
            "cnn_rows_produced": cnn_rows_produced,
            "headtail_rows_produced": int(len(interp_headtail_rows)),
            "no_work_reason": str(no_work_reason or ""),
            "mapping_path": _str_or_none(artifact_paths["mapping_path"]),
            "roi_csv_path": _str_or_none(artifact_paths["roi_csv_path"]),
            "roi_npz_path": _str_or_none(artifact_paths["roi_npz_path"]),
            "pose_csv_path": _str_or_none(artifact_paths["pose_csv_path"]),
            "pose_rows": (
                interp_pose_rows
                if (interp_pose_rows and not save_interpolated_outputs)
                else None
            ),
            "tag_csv_path": _str_or_none(artifact_paths["tag_csv_path"]),
            "tag_rows": (
                interp_tag_rows
                if (interp_tag_rows and not artifact_paths["tag_csv_path"])
                else None
            ),
            "cnn_csv_paths": (
                artifact_paths["cnn_csv_paths"]
                if artifact_paths["cnn_csv_paths"]
                else None
            ),
            "cnn_rows": (
                interp_cnn_rows
                if (
                    any(interp_cnn_rows.values())
                    and not artifact_paths["cnn_csv_paths"]
                )
                else None
            ),
            "headtail_csv_path": _str_or_none(artifact_paths["headtail_csv_path"]),
            "headtail_rows": (
                interp_headtail_rows
                if (interp_headtail_rows and not artifact_paths["headtail_csv_path"])
                else None
            ),
        }

    def _process_single_frame(
        self,
        f,
        idx,
        frame,
        total_frames,
        frame_tasks,
        gen,
        save_interpolated_outputs,
        _compute_native_scale,
        _canonical_ref_ar,
        _canonical_padding,
        _extract_canonical,
        pose_backend,
        cnn_backends,
        cnn_labels,
        apriltag_detector,
        headtail_analyzer,
        interp_saved,
        interp_rows,
        roi_rows,
        roi_corners,
        interp_pose_rows,
        interp_tag_rows,
        interp_cnn_rows,
        interp_headtail_rows,
        _pending_crops,
        _pending_entries,
        _pending_cnn_crops,
        _pending_cnn_entries,
        _pose_batch_size,
        _cnn_batch_size,
        pose_kpt_source_names,
        pose_kpt_labels,
        profiler,
    ):
        _frame_all_corners, _frame_affines = self._compute_frame_corners_and_affines(
            frame_tasks[f], _compute_native_scale, _canonical_ref_ar, _canonical_padding
        )
        for task_idx, task in enumerate(frame_tasks[f]):
            interp_saved = self._process_single_task(
                task,
                task_idx,
                frame,
                _frame_all_corners,
                _frame_affines,
                gen,
                save_interpolated_outputs,
                _extract_canonical,
                cnn_backends,
                pose_backend,
                interp_saved,
                interp_rows,
                roi_rows,
                roi_corners,
                _pending_crops,
                _pending_entries,
                _pending_cnn_crops,
                _pending_cnn_entries,
            )

        if apriltag_detector is not None and frame_tasks[f]:
            self._detect_apriltags_in_frame(
                apriltag_detector,
                frame,
                frame_tasks[f],
                _frame_all_corners,
                self.params,
                interp_tag_rows,
            )

        if headtail_analyzer is not None and frame_tasks[f] and _frame_all_corners:
            self._detect_headtail_in_frame(
                headtail_analyzer,
                frame,
                frame_tasks[f],
                _frame_all_corners,
                interp_headtail_rows,
            )

        if (
            pose_backend is not None
            and _pending_crops
            and (len(_pending_crops) >= _pose_batch_size or idx == total_frames)
        ):
            if self._should_stop():
                return None
            self._flush_pose_batch(
                pose_backend,
                _pending_crops,
                _pending_entries,
                interp_pose_rows,
                pose_kpt_source_names,
                pose_kpt_labels,
                profiler,
            )

        if (
            cnn_backends
            and _pending_cnn_crops
            and (len(_pending_cnn_crops) >= _cnn_batch_size or idx == total_frames)
        ):
            self._flush_cnn_batch(
                cnn_backends,
                cnn_labels,
                _pending_cnn_crops,
                _pending_cnn_entries,
                interp_cnn_rows,
                profiler,
            )

        if idx % 25 == 0 or idx == total_frames:
            progress = int((idx / total_frames) * 100)
            self.progress_signal.emit(
                progress, f"Interpolating occlusions... {idx}/{total_frames}"
            )
            del frame
        return interp_saved

    def _run_frame_tasks_loop(
        self,
        frame_tasks,
        cap,
        gen,
        save_interpolated_outputs,
        _compute_native_scale,
        _canonical_ref_ar,
        _canonical_padding,
        _extract_canonical,
        pose_backend,
        cnn_backends,
        cnn_labels,
        apriltag_detector,
        headtail_analyzer,
        interp_saved,
        interp_rows,
        roi_rows,
        roi_corners,
        interp_pose_rows,
        interp_tag_rows,
        interp_cnn_rows,
        interp_headtail_rows,
        pose_kpt_source_names,
        pose_kpt_labels,
        profiler,
    ):
        needed_frames = sorted(frame_tasks.keys())
        total_frames = len(needed_frames)
        _pose_batch_size = int(self.params.get("INTERP_POSE_INFERENCE_BATCH_SIZE", 64))
        _pending_crops: list = []
        _pending_entries: list = []
        _cnn_batch_size = 64
        _pending_cnn_crops: list = []
        _pending_cnn_entries: list = []

        _prefetcher = self._build_prefetcher(cap, needed_frames, total_frames)
        _prefetcher.start()
        for idx in range(1, total_frames + 1):
            if self._should_stop():
                _prefetcher.stop()
                return None
            _pf_item = _prefetcher.read()
            if _pf_item is None:
                break
            f, ret, frame = _pf_item
            if not ret or frame is None:
                continue
            result = self._process_single_frame(
                f,
                idx,
                frame,
                total_frames,
                frame_tasks,
                gen,
                save_interpolated_outputs,
                _compute_native_scale,
                _canonical_ref_ar,
                _canonical_padding,
                _extract_canonical,
                pose_backend,
                cnn_backends,
                cnn_labels,
                apriltag_detector,
                headtail_analyzer,
                interp_saved,
                interp_rows,
                roi_rows,
                roi_corners,
                interp_pose_rows,
                interp_tag_rows,
                interp_cnn_rows,
                interp_headtail_rows,
                _pending_crops,
                _pending_entries,
                _pending_cnn_crops,
                _pending_cnn_entries,
                _pose_batch_size,
                _cnn_batch_size,
                pose_kpt_source_names,
                pose_kpt_labels,
                profiler,
            )
            if result is None:
                return None
            interp_saved = result
        _prefetcher.stop()
        return interp_saved

    def _compute_position_scale(
        self, df, resize_factor, frame_width, frame_height, default
    ):
        try:
            max_x = df["X"].dropna().max()
            max_y = df["Y"].dropna().max()
            if (
                resize_factor
                and resize_factor < 1.0
                and max_x <= frame_width * resize_factor * 1.05
                and max_y <= frame_height * resize_factor * 1.05
            ):
                return 1.0 / resize_factor
        except Exception:
            return 1.0
        return default

    def _validate_and_setup(self, profiler):
        if self._should_stop():
            return None
        if not self.csv_path or not os.path.exists(self.csv_path):
            self.finished_signal.emit({"saved": 0, "gaps": 0})
            return None
        if not self.video_path or not os.path.exists(self.video_path):
            self.finished_signal.emit({"saved": 0, "gaps": 0})
            return None

        output_dir = self.params.get("INDIVIDUAL_DATASET_OUTPUT_DIR")
        if not output_dir:
            self.finished_signal.emit({"saved": 0, "gaps": 0})
            return None

        df = self._load_and_validate_csv()
        if df is None:
            self.finished_signal.emit({"saved": 0, "gaps": 0})
            return None

        resize_factor = self.params.get("RESIZE_FACTOR", 1.0)
        position_scale = 1.0
        size_scale = 1.0 / resize_factor if resize_factor else 1.0

        detection_cache = None
        if self.detection_cache_path and os.path.exists(self.detection_cache_path):
            detection_cache = DetectionCache(self.detection_cache_path, mode="r")

        save_interpolated_outputs = bool(
            self.params.get("ENABLE_INDIVIDUAL_IMAGE_SAVE", False)
        )
        generate_oriented_videos = bool(
            self.params.get("FINAL_MEDIA_EXPORT_VIDEOS_ENABLED", False)
            or self.params.get("GENERATE_ORIENTED_TRACK_VIDEOS", False)
        )
        if not save_interpolated_outputs and generate_oriented_videos:
            output_dir = self.params.get(
                "FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR",
                self.params.get(
                    "ORIENTED_TRACK_VIDEO_OUTPUT_DIR",
                    output_dir,
                ),
            )
        cache_interpolated_artifacts = bool(
            save_interpolated_outputs or generate_oriented_videos
        )
        gen_params = dict(self.params or {})
        gen_params["ENABLE_INDIVIDUAL_DATASET"] = cache_interpolated_artifacts
        gen_params["ENABLE_INDIVIDUAL_IMAGE_SAVE"] = save_interpolated_outputs

        gen = IndividualDatasetGenerator(
            gen_params,
            output_dir,
            Path(self.video_path).stem,
            (
                self.params.get("INDIVIDUAL_DATASET_NAME", "individual_dataset")
                if save_interpolated_outputs
                else ""
            ),
        )
        gen.enabled = cache_interpolated_artifacts

        _canonical_ref_ar = gen._canonical_ref_ar
        _canonical_padding = gen._canonical_padding

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished_signal.emit({"saved": 0, "gaps": 0})
            return None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        position_scale = self._compute_position_scale(
            df, resize_factor, frame_width, frame_height, position_scale
        )

        if "TrajectoryID" not in df.columns:
            logger.warning("Interpolated crops skipped: CSV missing TrajectoryID.")
            self.finished_signal.emit({"saved": 0, "gaps": 0})
            return None

        profiler.phase_end("interp_setup")
        return (
            df,
            cap,
            detection_cache,
            gen,
            output_dir,
            save_interpolated_outputs,
            cache_interpolated_artifacts,
            position_scale,
            size_scale,
            _canonical_ref_ar,
            _canonical_padding,
        )

    def execute(self):
        """Generate interpolated crops for occluded trajectory gaps."""
        from hydra_suite.core.canonicalization.crop import (
            compute_native_scale_affine as _compute_native_scale,
        )
        from hydra_suite.core.canonicalization.crop import (
            extract_canonical_crop as _extract_canonical,
        )
        from hydra_suite.core.tracking.profiler import TrackingProfiler

        profiler = TrackingProfiler(enabled=self.enable_profiling)
        profiler.phase_start("interp_setup")

        pose_backend = None
        detection_cache = None
        cap = None
        cnn_backends = []
        cnn_labels = []
        apriltag_detector = None
        headtail_analyzer = None
        pose_kpt_source_names = []
        pose_kpt_labels = []
        interp_cnn_rows = {}
        try:
            setup = self._validate_and_setup(profiler)
            if setup is None:
                return
            (
                df,
                cap,
                detection_cache,
                gen,
                output_dir,
                save_interpolated_outputs,
                cache_interpolated_artifacts,
                position_scale,
                size_scale,
                _canonical_ref_ar,
                _canonical_padding,
            ) = setup

            interp_saved = 0
            interp_rows = []
            interp_pose_rows = []
            interp_tag_rows = []
            interp_headtail_rows = []
            roi_rows = []
            roi_corners = []

            profiler.phase_start("interp_gap_detection")

            gap_result = self._detect_interpolation_gaps(
                df,
                detection_cache,
                position_scale,
                size_scale,
            )
            if gap_result is None:
                return
            frame_tasks, occluded_rows, interp_runs, interp_gaps = gap_result

            logger.info(
                f"Interpolated occlusion rows: {occluded_rows} "
                f"(runs: {interp_runs}, gaps: {interp_gaps})"
            )
            del df
            gc.collect()

            eligible_frames = int(len(frame_tasks))
            eligible_rows = int(sum(len(tasks) for tasks in frame_tasks.values()))

            profiler.phase_end("interp_gap_detection")
            profiler.phase_start("interp_crop_extraction")

            if frame_tasks:
                (
                    pose_backend,
                    pose_kpt_source_names,
                    pose_kpt_labels,
                    apriltag_detector,
                    cnn_backends,
                    cnn_labels,
                    headtail_analyzer,
                    interp_cnn_rows,
                ) = self._init_interpolation_backends(output_dir)
                interp_saved = self._run_frame_tasks_loop(
                    frame_tasks,
                    cap,
                    gen,
                    save_interpolated_outputs,
                    _compute_native_scale,
                    _canonical_ref_ar,
                    _canonical_padding,
                    _extract_canonical,
                    pose_backend,
                    cnn_backends,
                    cnn_labels,
                    apriltag_detector,
                    headtail_analyzer,
                    interp_saved,
                    interp_rows,
                    roi_rows,
                    roi_corners,
                    interp_pose_rows,
                    interp_tag_rows,
                    interp_cnn_rows,
                    interp_headtail_rows,
                    pose_kpt_source_names,
                    pose_kpt_labels,
                    profiler,
                )
                if interp_saved is None:
                    return
            else:
                logger.info(
                    "Interpolated post-pass found no eligible bounded gaps; skipping backend initialization."
                )

            profiler.phase_end("interp_crop_extraction")
            profiler.phase_start("interp_finalize")

            artifact_paths = self._empty_artifact_paths()
            if frame_tasks:
                artifact_paths = self._write_interpolation_artifacts(
                    gen,
                    save_interpolated_outputs,
                    cache_interpolated_artifacts,
                    interp_rows,
                    roi_rows,
                    roi_corners,
                    interp_pose_rows,
                    interp_tag_rows,
                    interp_cnn_rows,
                    interp_headtail_rows,
                    pose_kpt_labels,
                )
            if cache_interpolated_artifacts:
                gen.finalize()

            profiler.phase_end("interp_finalize")
            profiler.log_final_summary()
            if self.profile_export_path:
                profiler.export_summary(self.profile_export_path)

            if not self._should_stop():
                self.finished_signal.emit(
                    self._build_finished_payload(
                        interp_saved,
                        interp_gaps,
                        artifact_paths,
                        interp_pose_rows,
                        interp_tag_rows,
                        interp_cnn_rows,
                        interp_headtail_rows,
                        save_interpolated_outputs,
                        occluded_rows=occluded_rows,
                        interp_runs=interp_runs,
                        eligible_frames=eligible_frames,
                        eligible_rows=eligible_rows,
                        roi_rows_count=len(roi_rows),
                        no_work_reason=(
                            "no_occluded_rows"
                            if occluded_rows == 0
                            else "no_eligible_gaps" if eligible_rows == 0 else ""
                        ),
                    )
                )
        except Exception:
            self.finished_signal.emit({"saved": 0, "gaps": 0})
        finally:
            self._cleanup_backends(
                cap,
                detection_cache,
                pose_backend,
                apriltag_detector,
                cnn_backends,
                headtail_analyzer,
            )
