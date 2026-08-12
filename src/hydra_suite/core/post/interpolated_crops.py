"""Qt-free interpolated-crop extraction pipeline.

Moved out of ``trackerkit/gui/workers/crops_worker.py`` (Slice 2, Task 6):
this module owns the entire per-animal interpolated-crop generation
pipeline as a pure function, ``run_interpolated_crops``, with no Qt or
app-layer dependency. The worker becomes a thin wrapper that calls this
function and re-emits its return value as the ``finished_signal`` payload.
"""

import gc
import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hydra_suite.core.canonicalization.fit import apply_fit
from hydra_suite.core.canonicalization.geometry import (
    ClippingStats,
    canonical_geometry_from_params,
)
from hydra_suite.core.individual.dataset.generator import IndividualDatasetGenerator
from hydra_suite.core.individual.properties.export import (
    POSE_SUMMARY_COLUMNS,
    build_pose_keypoint_labels,
    flatten_cnn_prediction_row,
    flatten_pose_keypoints_row,
    pose_wide_columns_for_labels,
)
from hydra_suite.core.inference.api import load_pose_backend
from hydra_suite.core.post.merge import write_csv_artifact as _write_csv_artifact
from hydra_suite.core.post.merge import write_roi_npz as _write_roi_npz
from hydra_suite.data.detection_cache import DetectionCache
from hydra_suite.utils.geometry import wrap_angle_degs

logger = logging.getLogger(__name__)


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

    w, h = _size_from_obb_corners(obb_corners, idx)
    if w is not None:
        return w, h

    return _size_from_shapes(shapes, idx)


def _init_pose_backend(params, output_dir):
    """Initialize pose estimation backend. Returns (backend, kpt_source_names, kpt_labels).

    Routes through ``core/inference/api.load_pose_backend`` (the shared shim
    over ``stages/pose.load_pose_model`` — the single source of the pose
    runtime golden rule) instead of duplicating the runtime-flavor ladder
    here. ``load_pose_model`` still honors the SLEAP-flavor debug override
    internally.
    """
    if not bool(params.get("ENABLE_POSE_EXTRACTOR", False)):
        return None, [], []
    try:
        from hydra_suite.core.individual.pose.utils import load_skeleton_from_json

        backend_family = str(params.get("POSE_MODEL_TYPE", "yolo")).strip().lower()
        model_path = str(params.get("POSE_MODEL_DIR", ""))
        min_valid_conf = float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
        batch_size = int(params.get("POSE_BATCH_SIZE", 4))
        skeleton_file = str(params.get("POSE_SKELETON_FILE", "") or "")
        keypoint_names, skeleton_edges = load_skeleton_from_json(skeleton_file)

        # Runtime comes solely from RUNTIME_TIER (Runtime Gen-2 FT1): the
        # COMPUTE_RUNTIME param family was retired. load_pose_backend only
        # needs the tier bucket (it re-derives the tier via
        # migrate_runtime_to_tier), so a resolved runtime string is passed.
        if backend_family == "sleap":
            pose_stage = "sleap_pose"
        elif backend_family == "vitpose":
            pose_stage = "vitpose_pose"
        else:
            pose_stage = "yolo_pose"
        compute_runtime = _resolved_runtime_string(params, pose_stage)

        backend = load_pose_backend(
            backend_family=backend_family,
            model_path=model_path,
            compute_runtime=compute_runtime,
            keypoint_names=list(keypoint_names),
            skeleton_edges=skeleton_edges,
            batch_size=max(1, batch_size),
            min_valid_confidence=min_valid_conf,
            out_root=str(Path(output_dir).expanduser()),
            sleap_env=str(params.get("POSE_SLEAP_ENV", "sleap")),
            sleap_batch=max(1, batch_size),
            sleap_max_instances=int(params.get("POSE_SLEAP_MAX_INSTANCES", 1)),
        )

        # NOTE: do NOT call backend.warmup() here -- load_pose_backend
        # (-> stages/pose.load_pose_model) already warms the backend it
        # returns. A second warmup() is redundant, and for the SLEAP
        # service backend it is not idempotent w.r.t. ownership: the
        # first warmup sets _service_started_here=True, a second sees
        # was_running=True and flips it back to False, so close() later
        # skips shutdown and leaks the SLEAP service subprocess.
        kpt_source_names = list(getattr(backend, "output_keypoint_names", []) or [])
        kpt_labels = build_pose_keypoint_labels(kpt_source_names, len(kpt_source_names))
        return backend, kpt_source_names, kpt_labels
    except Exception as exc:
        logger.warning(
            "Interpolated pose analysis disabled (backend init failed): %s",
            exc,
        )
        return None, [], []


def _resolve_backend(params, stage: str):
    """Resolve the runtime tier in ``params`` to a concrete ResolvedBackend.

    The single Gen-2 authority (``RuntimeResolver``) owns the tier -> backend
    decision; this pipeline no longer threads per-stage compute_runtime strings.
    Falls back to the CPU tier when no valid ``RUNTIME_TIER`` is present.
    """
    from hydra_suite.runtime.resolver import RuntimeResolver, detect_platform

    tier = str(params.get("RUNTIME_TIER", "cpu") or "cpu").strip().lower()
    if tier not in {"cpu", "gpu", "gpu_fast"}:
        tier = "cpu"
    return RuntimeResolver(tier, detect_platform()).resolve(stage)


def _resolved_runtime_string(params, stage: str) -> str:
    """Resolve ``RUNTIME_TIER`` to a compute-runtime string for ``stage``.

    Used only where a downstream shim (``load_pose_backend``) still takes a
    runtime string rather than a ``ResolvedBackend``; the string maps back to
    the correct tier via ``migrate_runtime_to_tier``.
    """
    resolved = _resolve_backend(params, stage)
    if resolved.backend == "tensorrt":
        return "tensorrt"
    if resolved.backend == "coreml":
        return "coreml"
    return resolved.device  # cpu / cuda / mps


def _init_apriltag_detector(params):
    """Initialize AprilTag detector if configured. Returns detector or None."""
    apriltag_enabled = (
        bool(params.get("USE_APRILTAGS", False))
        or str(params.get("IDENTITY_METHOD", "")).lower() == "apriltags"
    )
    if not apriltag_enabled:
        return None
    try:
        from hydra_suite.core.individual.classification.apriltag import (
            AprilTagConfig,
            AprilTagDetector,
        )

        return AprilTagDetector(AprilTagConfig.from_params(params))
    except Exception as exc:
        logger.warning("Interpolated AprilTag analysis disabled: %s", exc)
        return None


def _init_cnn_backends(params):
    """Initialize CNN identity backends. Returns (backends_list, labels_list)."""
    cnn_backends = []
    cnn_labels = []
    cnn_classifiers_cfg = params.get("CNN_CLASSIFIERS", [])
    if not cnn_classifiers_cfg:
        return cnn_backends, cnn_labels
    try:
        from hydra_suite.core.individual.classification.cnn import (
            CNNIdentityBackend,
            CNNIdentityConfig,
        )

        cnn_resolved = _resolve_backend(params, "cnn")
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
                    resolved=cnn_resolved,
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


def _init_headtail_analyzer(params, geometry):
    """Initialize head-tail direction analyzer. Returns analyzer or None.

    Raises:
        ClassifierFormatError: if the configured model path exists but
            cannot be loaded by any supported backend.  Callers forward
            this to the ``error`` signal.
    """
    headtail_model_path = str(params.get("YOLO_HEADTAIL_MODEL_PATH", ""))
    if not headtail_model_path or not os.path.exists(headtail_model_path):
        return None
    from hydra_suite.core.individual.classification.headtail import HeadTailAnalyzer

    analyzer = HeadTailAnalyzer(
        model_path=headtail_model_path,
        resolved=_resolve_backend(params, "head_tail"),
        conf_threshold=float(params.get("YOLO_HEADTAIL_CONF_THRESHOLD", 0.5)),
        batch_size=max(1, int(params.get("HEADTAIL_BATCH_SIZE", 64))),
        geometry=geometry,
    )
    if not analyzer.is_available:
        analyzer.close()
        return None
    return analyzer


def _init_interpolation_backends(params, output_dir, geometry):
    """Initialize optional analysis backends after eligible gap tasks exist."""
    pose_backend, pose_kpt_source_names, pose_kpt_labels = _init_pose_backend(
        params, output_dir
    )
    apriltag_detector = _init_apriltag_detector(params)
    cnn_backends, cnn_labels = _init_cnn_backends(params)
    headtail_analyzer = _init_headtail_analyzer(params, geometry)
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


def _load_and_validate_csv(csv_path):
    """Load CSV and validate required columns. Returns DataFrame or None."""
    df = pd.read_csv(csv_path)
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
    params,
    should_stop,
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
    def _stop():
        return bool(should_stop()) if should_stop is not None else False

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

    det_id_prev = prev_row["DetectionID"] if "DetectionID" in group.columns else None
    det_id_next = next_row["DetectionID"] if "DetectionID" in group.columns else None

    w0, h0 = _get_detection_size(detection_cache, f0, det_id_prev)
    w1, h1 = _get_detection_size(detection_cache, f1, det_id_next)

    if w0 is None or h0 is None or w1 is None or h1 is None:
        ref_size = params.get("REFERENCE_BODY_SIZE", 20.0)
        w0 = w0 or ref_size * 2.2
        h0 = h0 or ref_size * 0.8
        w1 = w1 or ref_size * 2.2
        h1 = h1 or ref_size * 0.8

    for k in range(i, j):
        if _stop():
            return None
        row = group.iloc[k]
        f = int(row["FrameID"])
        t = (f - f0) / (f1 - f0)
        cx = float(prev_row["X"]) + t * (float(next_row["X"]) - float(prev_row["X"]))
        cy = float(prev_row["Y"]) + t * (float(next_row["Y"]) - float(prev_row["Y"]))
        theta = _interp_angle(float(prev_row["Theta"]), float(next_row["Theta"]), t)
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
    params,
    should_stop,
    traj_id,
    group,
    detection_cache,
    position_scale,
    size_scale,
    frame_tasks,
    interp_runs,
    interp_gaps,
):
    def _stop():
        return bool(should_stop()) if should_stop is not None else False

    group = group.sort_values("FrameID").reset_index(drop=True)
    states = group["State"].astype(str).str.strip().str.lower()
    states = states.where(~states.str.contains("occluded", na=False), "occluded")
    traj_occluded = int((states == "occluded").sum())

    last_valid_idx = None
    i = 0
    while i < len(group):
        if _stop():
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

        run_result = _process_occluded_run(
            params,
            should_stop,
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
    params, should_stop, df, detection_cache, position_scale, size_scale
):
    """Scan trajectories for occluded gaps and build per-frame task lists.

    Returns (frame_tasks, occluded_rows, interp_runs, interp_gaps) or None if stopped.
    """

    def _stop():
        return bool(should_stop()) if should_stop is not None else False

    occluded_rows = 0
    interp_gaps = 0
    interp_runs = 0
    frame_tasks = defaultdict(list)

    for traj_id, group in df.groupby("TrajectoryID"):
        if _stop():
            return None
        result = _scan_trajectory_gaps(
            params,
            should_stop,
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
    pose_backend,
    pending_crops,
    pending_entries,
    interp_pose_rows,
    pose_kpt_source_names,
    pose_kpt_labels,
    profiler,
    geometry,
):
    """Run pose inference on accumulated crops and append results.

    Pre-fits every canonical (Layer 1) crop through Layer 2
    (``fit_to_model_input`` / ``apply_fit``) before handing it to the backend
    -- the SAME call shape ``core/inference/stages/pose.py`` uses, so this
    module's interpolated-frame keypoints agree on content scale with the
    tracked-frame keypoints produced by the same model.  Every entry here is
    Layer 1 canonical by construction: ``_extract_pose_crop`` skips (rather
    than produces) a crop for the degenerate-OBB case where no rigid Layer 1
    transform exists, so there is no non-canonical entry left to special-case.
    An ``apply_fit`` failure on an otherwise-canonical crop is treated the
    same way: there is no honest crop to feed the backend (a raw canvas crop
    would be anisotropically resized by the backend, and there is no fit to
    invert for the keypoint back-projection), so the detection is dropped
    exactly like the degenerate-OBB case in ``_extract_pose_crop``.
    """
    from hydra_suite.core.canonicalization.crop import invert_keypoints as _invert_kpts
    from hydra_suite.core.canonicalization.fit import fit_affine, fit_to_model_input
    from hydra_suite.core.inference.stages.pose import compose_affine, model_input_wh

    class _BackendHolder:
        backend = pose_backend

    model_wh = model_input_wh(_BackendHolder, geometry)
    fit = fit_to_model_input(geometry.canvas_wh, model_wh)
    fit_m = fit_affine(fit)

    fitted_crops = []
    kept_entries = []
    for _crop, _entry in zip(pending_crops, pending_entries):
        _crop_info = _entry.get("crop_info") or {}
        if _crop_info.get("canonical"):
            try:
                fitted_crop = apply_fit(_crop, fit)
            except Exception:
                logger.warning(
                    "Interp pose: skipping frame_id=%s traj_id=%s -- Layer 2 "
                    "fit (apply_fit) failed for an otherwise-canonical crop; "
                    "the old raw-crop fallback fed the backend a "
                    "wrongly-scaled crop and inverted a fit that was never "
                    "applied instead of skipping.",
                    _entry["task"]["frame_id"],
                    _entry["task"]["traj_id"],
                    exc_info=True,
                )
                continue
            fitted_crops.append(fitted_crop)
            kept_entries.append(_entry)
        else:
            fitted_crops.append(_crop)
            kept_entries.append(_entry)

    profiler.tick("interp_pose_inference")
    pose_results = pose_backend.predict_batch(fitted_crops)
    profiler.tock("interp_pose_inference")
    for pidx, entry in enumerate(kept_entries):
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
                _M_align = crop_info.get("M_forward")
                if _M_align is not None and crop_info.get("canonical"):
                    # Keypoints come back in MODEL-input coords, so the
                    # back-projection must undo Layer 2 (fit) then Layer 1
                    # (canonical align) -- invert the composed transform.
                    _m_total = compose_affine(fit_m, _M_align)
                    _M_inv = cv2.invertAffineTransform(_m_total.astype(np.float32))
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


def _flush_cnn_batch(
    cnn_backends,
    cnn_labels,
    pending_cnn_crops,
    pending_cnn_entries,
    interp_cnn_rows,
    profiler,
    geometry,
):
    """Run CNN identity inference on accumulated crops and append results.

    Pre-fits the shared Layer 1 canonical crops through Layer 2
    (``fit_to_model_input`` / ``apply_fit``) per classifier, exactly as
    ``core/inference/stages/cnn.py`` does -- each classifier may have a
    different input size, so the fit is computed and applied fresh for every
    backend rather than shared across them.  Without this,
    ``core/individual/classification/backend.py`` would ANISOTROPICALLY stretch
    the canonical crop to the model's input.
    """
    from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input

    profiler.tick("interp_cnn_inference")
    for _bi, _cnn_be in enumerate(cnn_backends):
        _cnn_label = cnn_labels[_bi]
        try:
            _in_h, _in_w = _cnn_be.metadata.input_size  # documents (H, W)
            _fit = fit_to_model_input(geometry.canvas_wh, (_in_w, _in_h))
            _fitted_cnn_crops = [apply_fit(_c, _fit) for _c in pending_cnn_crops]
            _cnn_preds = _cnn_be.predict_batch(_fitted_cnn_crops)
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


def _build_prefetcher(cap, needed_frames, total_frames):
    from hydra_suite.utils.frame_prefetcher import (
        SequentialScanPrefetcher,
        SparseFramePrefetcher,
    )

    _frame_range = needed_frames[-1] - needed_frames[0] + 1
    _density = total_frames / max(_frame_range, 1)
    _use_sequential = _density >= 0.05 or (_frame_range / max(total_frames, 1)) < 100
    if _use_sequential:
        logger.info(
            "Interpolation: sequential scan (%d needed / %d range, " "density=%.2f%%)",
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


def _compute_frame_corners_and_affines(tasks, geometry, clipping_stats):
    """Layer 1 affine per task against the ONE project-wide canonical canvas.

    ``clipping_stats`` accumulates the per-detection overflow so a too-small
    ``canonical_margin`` produces a visible end-of-run signal instead of
    silently truncated animals -- the same guard the core tracking loop
    applies (``core/tracking/worker.py``).
    """
    from hydra_suite.core.canonicalization.geometry import canonical_affine
    from hydra_suite.core.individual.geometry import ellipse_to_obb_corners as _e2obb

    corners = [_e2obb(t["cx"], t["cy"], t["w"], t["h"], t["theta"]) for t in tasks]
    affines = []
    for _c in corners:
        try:
            _M, _theta, _clipped = canonical_affine(_c, geometry)
        except ValueError:
            affines.append(None)
            continue
        if clipping_stats is not None:
            clipping_stats.record(_c, geometry)
        affines.append((_M, geometry.canvas_w, geometry.canvas_h))
    return corners, affines


def _process_single_task(
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
        pose_crop, pose_crop_info = _extract_pose_crop(
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
    task_idx,
    frame,
    _frame_all_corners,
    _aff,
    corners,
    gen,
    _extract_canonical,
):
    """Extract the pose/CNN crop for one task via Layer 1 canonicalization.

    ``_aff`` is None exactly when ``canonical_affine`` raised
    (``core/canonicalization/geometry.py::_axes`` -- a degenerate OBB with a
    zero-length edge). There is no rigid Layer 1 transform for a degenerate
    box, and the un-canonicalized ``_extract_obb_masked_crop`` fallback this
    used to feed the backend produces an arbitrary axis-aligned aspect ratio
    for which Layer 2's ``fit_to_model_input`` cannot honestly be computed (it
    assumes the source is the fixed canonical canvas) -- feeding it anyway
    would hand the backend a wrongly-scaled crop, exactly the defect class
    this work removes. A genuinely degenerate OBB has no salvageable animal
    geometry to recover either way, so this loudly skips the detection.
    """
    if _aff is None:
        logger.warning(
            "Interp pose/CNN: skipping task_idx=%s -- degenerate OBB has no "
            "Layer 1 canonical transform (canonical_affine raised); the old "
            "masked-crop fallback fed the backend an un-canonicalized, "
            "wrongly-scaled crop instead of skipping.",
            task_idx,
        )
        return None, None
    pose_crop = None
    pose_crop_info = None
    try:
        _other_corners = [
            c for ci, c in enumerate(_frame_all_corners) if ci != task_idx
        ]
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
            # Layer 1 forward affine (image -> canonical canvas). Kept so the
            # Layer 2 (model-fit) affine can be composed onto it at flush
            # time -- see ``_flush_pose_batch``.
            "M_forward": np.asarray(_M_pose, dtype=np.float64),
            "canonical": True,
        }
    except Exception:
        pose_crop = None
        pose_crop_info = None
    return pose_crop, pose_crop_info


def _build_finished_payload(
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
            artifact_paths["cnn_csv_paths"] if artifact_paths["cnn_csv_paths"] else None
        ),
        "cnn_rows": (
            interp_cnn_rows
            if (any(interp_cnn_rows.values()) and not artifact_paths["cnn_csv_paths"])
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
    params,
    should_stop,
    progress,
    f,
    idx,
    frame,
    total_frames,
    frame_tasks,
    gen,
    save_interpolated_outputs,
    geometry,
    clipping_stats,
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
    def _stop():
        return bool(should_stop()) if should_stop is not None else False

    def _emit(v, m):
        if progress is not None:
            progress(v, m)

    _frame_all_corners, _frame_affines = _compute_frame_corners_and_affines(
        frame_tasks[f], geometry, clipping_stats
    )
    for task_idx, task in enumerate(frame_tasks[f]):
        interp_saved = _process_single_task(
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
        _detect_apriltags_in_frame(
            apriltag_detector,
            frame,
            frame_tasks[f],
            _frame_all_corners,
            params,
            interp_tag_rows,
        )

    if headtail_analyzer is not None and frame_tasks[f] and _frame_all_corners:
        _detect_headtail_in_frame(
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
        if _stop():
            return None
        _flush_pose_batch(
            pose_backend,
            _pending_crops,
            _pending_entries,
            interp_pose_rows,
            pose_kpt_source_names,
            pose_kpt_labels,
            profiler,
            geometry,
        )

    if (
        cnn_backends
        and _pending_cnn_crops
        and (len(_pending_cnn_crops) >= _cnn_batch_size or idx == total_frames)
    ):
        _flush_cnn_batch(
            cnn_backends,
            cnn_labels,
            _pending_cnn_crops,
            _pending_cnn_entries,
            interp_cnn_rows,
            profiler,
            geometry,
        )

    if idx % 25 == 0 or idx == total_frames:
        progress_pct = int((idx / total_frames) * 100)
        _emit(progress_pct, f"Interpolating occlusions... {idx}/{total_frames}")
        del frame
    return interp_saved


def _run_frame_tasks_loop(
    params,
    should_stop,
    progress,
    frame_tasks,
    cap,
    gen,
    save_interpolated_outputs,
    geometry,
    clipping_stats,
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
    def _stop():
        return bool(should_stop()) if should_stop is not None else False

    needed_frames = sorted(frame_tasks.keys())
    total_frames = len(needed_frames)
    _pose_batch_size = int(params.get("INTERP_POSE_INFERENCE_BATCH_SIZE", 64))
    _pending_crops: list = []
    _pending_entries: list = []
    _cnn_batch_size = 64
    _pending_cnn_crops: list = []
    _pending_cnn_entries: list = []

    _prefetcher = _build_prefetcher(cap, needed_frames, total_frames)
    _prefetcher.start()
    for idx in range(1, total_frames + 1):
        if _stop():
            _prefetcher.stop()
            return None
        _pf_item = _prefetcher.read()
        if _pf_item is None:
            break
        f, ret, frame = _pf_item
        if not ret or frame is None:
            continue
        result = _process_single_frame(
            params,
            should_stop,
            progress,
            f,
            idx,
            frame,
            total_frames,
            frame_tasks,
            gen,
            save_interpolated_outputs,
            geometry,
            clipping_stats,
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


def _compute_position_scale(df, resize_factor, frame_width, frame_height, default):
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


def _validate_and_setup(
    csv_path, video_path, detection_cache_path, params, profiler, should_stop
):
    def _stop():
        return bool(should_stop()) if should_stop is not None else False

    if _stop():
        return None
    if not csv_path or not os.path.exists(csv_path):
        return None
    if not video_path or not os.path.exists(video_path):
        return None

    output_dir = params.get("INDIVIDUAL_DATASET_OUTPUT_DIR")
    if not output_dir:
        return None

    df = _load_and_validate_csv(csv_path)
    if df is None:
        return None

    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    position_scale = 1.0
    size_scale = 1.0 / resize_factor if resize_factor else 1.0

    detection_cache = None
    if detection_cache_path and os.path.exists(detection_cache_path):
        detection_cache = DetectionCache(detection_cache_path, mode="r")

    save_interpolated_outputs = bool(params.get("ENABLE_INDIVIDUAL_IMAGE_SAVE", False))
    generate_oriented_videos = bool(
        params.get("FINAL_MEDIA_EXPORT_VIDEOS_ENABLED", False)
        or params.get("GENERATE_ORIENTED_TRACK_VIDEOS", False)
    )
    if not save_interpolated_outputs and generate_oriented_videos:
        output_dir = params.get(
            "FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR",
            params.get(
                "ORIENTED_TRACK_VIDEO_OUTPUT_DIR",
                output_dir,
            ),
        )
    cache_interpolated_artifacts = bool(
        save_interpolated_outputs or generate_oriented_videos
    )
    gen_params = dict(params or {})
    gen_params["ENABLE_INDIVIDUAL_DATASET"] = cache_interpolated_artifacts
    gen_params["ENABLE_INDIVIDUAL_IMAGE_SAVE"] = save_interpolated_outputs

    gen = IndividualDatasetGenerator(
        gen_params,
        output_dir,
        Path(video_path).stem,
        (
            params.get("INDIVIDUAL_DATASET_NAME", "individual_dataset")
            if save_interpolated_outputs
            else ""
        ),
    )
    gen.enabled = cache_interpolated_artifacts

    # The ONE project-wide Layer 1 canvas, built once from params. Every crop
    # site below shares it -- there is no per-detection or per-worker canvas.
    geometry = canonical_geometry_from_params(params)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    position_scale = _compute_position_scale(
        df, resize_factor, frame_width, frame_height, position_scale
    )

    if "TrajectoryID" not in df.columns:
        logger.warning("Interpolated crops skipped: CSV missing TrajectoryID.")
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
        geometry,
    )


def run_interpolated_crops(
    csv_path,
    video_path,
    detection_cache_path,
    params,
    *,
    enable_profiling=False,
    profile_export_path=None,
    progress=None,
    should_stop=None,
):
    """Generate interpolated crops for occluded trajectory gaps.

    Pure, Qt-free entry point extracted from
    ``InterpolatedCropsWorker.execute()``. Returns the finished-payload dict
    that the worker used to emit via ``finished_signal``.
    """
    from hydra_suite.core.canonicalization.crop import (
        extract_canonical_crop as _extract_canonical,
    )
    from hydra_suite.core.tracking.profiler import TrackingProfiler

    def _stop():
        return bool(should_stop()) if should_stop is not None else False

    # Run-scoped canonical-crop clipping accumulator; reported at the end of
    # the pass, mirroring TrackingWorker's end-of-run summary.
    clipping_stats = ClippingStats()
    profiler = TrackingProfiler(enabled=enable_profiling)
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
        setup = _validate_and_setup(
            csv_path, video_path, detection_cache_path, params, profiler, should_stop
        )
        if setup is None:
            return {"saved": 0, "gaps": 0}
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
            geometry,
        ) = setup

        interp_saved = 0
        interp_rows = []
        interp_pose_rows = []
        interp_tag_rows = []
        interp_headtail_rows = []
        roi_rows = []
        roi_corners = []

        profiler.phase_start("interp_gap_detection")

        gap_result = _detect_interpolation_gaps(
            params,
            should_stop,
            df,
            detection_cache,
            position_scale,
            size_scale,
        )
        if gap_result is None:
            return {"saved": 0, "gaps": 0}
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
            ) = _init_interpolation_backends(params, output_dir, geometry)
            interp_saved = _run_frame_tasks_loop(
                params,
                should_stop,
                progress,
                frame_tasks,
                cap,
                gen,
                save_interpolated_outputs,
                geometry,
                clipping_stats,
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
                return {"saved": 0, "gaps": 0}
        else:
            logger.info(
                "Interpolated post-pass found no eligible bounded gaps; skipping backend initialization."
            )

        profiler.phase_end("interp_crop_extraction")
        profiler.phase_start("interp_finalize")

        artifact_paths = _empty_artifact_paths()
        if frame_tasks:
            artifact_paths = _write_interpolation_artifacts(
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

        # Report canonical-crop clipping, if any -- mirrors TrackingWorker's
        # end-of-run clipping_stats summary (core/tracking/worker.py).
        # ``canonical_margin`` is the operator's only dial against truncated
        # animals, so this must never be silent.
        _clip_msg = clipping_stats.summary()
        if _clip_msg:
            logger.warning("Canonicalization clipping summary: %s", _clip_msg)

        profiler.log_final_summary()
        if profile_export_path:
            profiler.export_summary(profile_export_path)

        if not _stop():
            return _build_finished_payload(
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
        return {"saved": 0, "gaps": 0}
    except Exception:
        return {"saved": 0, "gaps": 0}
    finally:
        _cleanup_backends(
            cap,
            detection_cache,
            pose_backend,
            apriltag_detector,
            cnn_backends,
            headtail_analyzer,
        )
