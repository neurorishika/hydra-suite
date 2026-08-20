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
from hydra_suite.core.inference.cache import open_detection_cache_reader
from hydra_suite.core.inference.stages.apriltag import run_apriltag
from hydra_suite.core.inference.stages.crops import extract_aabb_crops
from hydra_suite.core.post.merge import write_csv_artifact as _write_csv_artifact
from hydra_suite.core.post.merge import write_roi_npz as _write_roi_npz
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
        obb = detection_cache.read_frame(int(frame_id))
    except Exception:
        return None, None
    if obb is None:
        return None, None

    idx = None
    try:
        for i, did in enumerate(obb.detection_ids):
            if int(did) == int(detection_id):
                idx = i
                break
    except Exception:
        idx = None

    if idx is None:
        return None, None

    w, h = _size_from_obb_corners(list(obb.corners), idx)
    if w is not None:
        return w, h

    return _size_from_shapes(list(obb.shapes), idx)


def _init_interpolation_backends(params, output_dir, geometry):
    """Build the InferenceConfig/RuntimeContext and load the four
    downstream-stage models (headtail/CNN/pose/AprilTag) via their public
    stage loaders -- the SAME loaders ``Pipeline`` uses, so the interpolated
    path shares the tier->backend resolution and model-loading code instead
    of hand-rolling its own runtime-flavor ladder (design spec
    "Model-loading glue"; see the plan's "Plan-level deviation from the spec
    text" note for why the per-stage loaders are called directly here rather
    than ``runner.py``'s private ``_load_all_models`` -- that function always
    loads an OBB/bgsub detector too, which this pass never uses).

    Returns (cfg, runtime, pose_model, apriltag_model, cnn_models,
    cnn_labels, headtail_model). Any model whose config/params disable it is
    None (pose, apriltag, headtail) or empty (cnn_models/cnn_labels), mirroring
    today's opt-in behavior -- CNN no longer depends on pose being enabled
    (design spec, bug fix #1: the CNN/pose decoupling is now real, since
    ``run_cnn_batch`` builds its own classifier crops independently, unlike
    the old ``_pending_cnn_crops.append(pose_crop)``).
    """
    # NOTE: `geometry` is unused in this body -- none of the four stage
    # loaders need the canonical-geometry object at load time (only at
    # crop-extraction time, later). Kept in the signature because the sole
    # call site's tuple-unpack convention and existing tests already pass it
    # positionally; not removed here to avoid a wider signature churn.
    from hydra_suite.core.inference.config import (
        AprilTagConfig,
        build_inference_config_from_params,
    )
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.stages.apriltag import load_apriltag_model
    from hydra_suite.core.inference.stages.cnn import load_cnn_model
    from hydra_suite.core.inference.stages.headtail import load_headtail_model
    from hydra_suite.core.inference.stages.pose import load_pose_model

    try:
        cfg = build_inference_config_from_params(params)
        runtime = RuntimeContext.from_config(cfg)
    except Exception:
        # Unlike the four per-model try/except blocks below (which degrade
        # ONE signal and keep going), a failure here -- e.g.
        # PoseModelUnresolvedError when ENABLE_POSE_EXTRACTOR is set but no
        # usable pose model resolves, or any other InferenceConfigError --
        # used to propagate all the way up to `run_interpolated_crops`'s
        # outer blanket `except Exception: return {"saved": 0, "gaps": 0}`,
        # which has NO logging at all: the entire interpolated-crop
        # post-pass (crops, ROI CSV/NPZ, pose/tag/cnn/headtail CSVs --
        # everything) would silently vanish with zero diagnostic trace. Log
        # loudly and degrade to "no models available" instead, matching this
        # function's existing return contract (mirrors the old code's
        # graceful degradation: a failed pose-backend init logged and kept
        # the other three signals running -- here nothing about config-
        # building is stage-specific, so all four degrade together, but the
        # caller still runs the frame-task loop and saves crops/ROI).
        logger.exception(
            "Interpolated post-pass: failed to build the inference config/"
            "runtime; disabling pose/AprilTag/CNN/head-tail analysis for "
            "this run (crop/ROI extraction is unaffected)."
        )
        from types import SimpleNamespace

        degraded_cfg = SimpleNamespace(
            pose=None,
            apriltag=AprilTagConfig(enabled=False),
            headtail=None,
            cnn_phases=[],
        )
        return degraded_cfg, None, None, None, [], [], None

    pose_model = None
    if cfg.pose is not None:
        try:
            pose_model = load_pose_model(
                cfg.pose,
                runtime,
                out_root=str(Path(output_dir).expanduser()),
            )
        except Exception as exc:
            logger.warning(
                "Interpolated pose analysis disabled (backend init failed): %s",
                exc,
            )
            pose_model = None

    apriltag_model = None
    # `cfg.apriltag.enabled` == `bool(params.get("USE_APRILTAGS", False))` only
    # (see `build_inference_config_from_params`, config.py) -- it deliberately
    # does NOT also check `IDENTITY_METHOD == "apriltags"` the way the old
    # hand-rolled interpolated-crops enablement check did. This is intentional
    # parity with `Pipeline`'s own real-detection enablement check
    # (`pipeline.py`'s `self.stages.apriltag_model is not None`, built from the
    # SAME `cfg.apriltag.enabled` field with no `IDENTITY_METHOD` fallback of
    # its own) -- not a functional regression versus real detections.
    if cfg.apriltag.enabled:
        try:
            apriltag_model = load_apriltag_model(cfg.apriltag)
        except Exception as exc:
            logger.warning("Interpolated AprilTag analysis disabled: %s", exc)
            apriltag_model = None

    cnn_models = []
    cnn_labels = []
    for cnn_cfg in cfg.cnn_phases:
        try:
            cnn_models.append(load_cnn_model(cnn_cfg, runtime))
            cnn_labels.append(cnn_cfg.label)
        except Exception as exc:
            logger.warning(
                "Interpolated CNN identity '%s' disabled: %s", cnn_cfg.label, exc
            )

    headtail_model = None
    if cfg.headtail is not None:
        try:
            headtail_model = load_headtail_model(cfg.headtail, runtime)
        except Exception as exc:
            logger.warning("Interpolated head-tail analysis disabled: %s", exc)
            headtail_model = None

    return (
        cfg,
        runtime,
        pose_model,
        apriltag_model,
        cnn_models,
        cnn_labels,
        headtail_model,
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
        # Geometry-sourcing priority (design spec "Geometry sourcing",
        # NaN-triggered, not max_gap-triggered): if mechanism (1)'s
        # trajectory interpolation already filled this row's X/Y/Theta
        # (honoring the user's interpolation_method and heading-flip
        # correction), use it directly instead of re-deriving a bespoke
        # linear/± 180 degree estimate. Only fall back to independent
        # linear interpolation when the CSV row is genuinely NaN here
        # (interpolation_method="None" -- the GUI default -- or a gap
        # beyond max_gap).
        row_x = row["X"] if "X" in group.columns else float("nan")
        row_y = row["Y"] if "Y" in group.columns else float("nan")
        row_theta = row["Theta"] if "Theta" in group.columns else float("nan")
        if not (pd.isna(row_x) or pd.isna(row_y) or pd.isna(row_theta)):
            cx = float(row_x)
            cy = float(row_y)
            theta = float(row_theta)
        else:
            cx = float(prev_row["X"]) + t * (
                float(next_row["X"]) - float(prev_row["X"])
            )
            cy = float(prev_row["Y"]) + t * (
                float(next_row["Y"]) - float(prev_row["Y"])
            )
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


def _flush_pose_cnn_window(
    pending_frames,
    pending_obbs,
    pending_tasks_by_frame,
    pose_model,
    cnn_models,
    cnn_labels,
    cfg,
    runtime,
    geometry,
    interp_pose_rows,
    interp_cnn_rows,
    profiler,
    background_color=(0, 0, 0),
):
    """Run pose + CNN inference over a window of (frame, synthetic OBB) pairs.

    Calls the SAME stage functions ``Pipeline`` calls for real detections
    (``pipeline.py:367-387``): ``extract_canonical_crops_batch`` then
    ``run_pose_batch`` for pose, ``run_cnn_batch`` per CNN phase for CNN --
    instead of this module's old hand-rolled crop extraction + batching
    (``_flush_pose_batch``/``_flush_cnn_batch``). ``suppress_foreign`` for the
    pose call is read from ``cfg.pose.suppress_foreign_regions`` -- the SAME
    config knob ``Pipeline`` reads for real detections (``pipeline.py:369-
    370``) -- so a user who disables foreign suppression gets that honored for
    interpolated crops too, not a hardcoded ``True`` (design spec, AprilTag/
    foreign-suppression decisions). Pose and CNN crops are now genuinely independent
    (CNN via ``extract_classifier_crops_batch_np`` inside ``run_cnn_batch``,
    not a reused pose crop) -- design spec bug fix #1.

    ``flatten_cnn_prediction_row`` (``properties/export.py``) expects a
    PRE-COMPUTED argmax class name + confidence per factor -- it just indexes
    ``class_names[idx]``/``confidences[idx]``, it does not run its own
    argmax. ``CNNDetectionPrediction.factors`` (``result.py``) carries raw
    probability vectors (``CNNFactorPrediction.raw_probabilities``) plus the
    full per-factor class-name list, so the argmax is done here for each
    factor before calling ``flatten_cnn_prediction_row``, mirroring the same
    conversion ``frame_result_bridge._cnn_det_pred_to_class_prediction`` does
    for the live tracking path.
    """
    from hydra_suite.core.inference.stages.cnn import run_cnn_batch
    from hydra_suite.core.inference.stages.crops import extract_canonical_crops_batch
    from hydra_suite.core.inference.stages.pose import run_pose_batch

    if not pending_frames:
        return

    if profiler is not None:
        profiler.tick("interp_pose_inference")
    if pose_model is not None:
        crop_batch = extract_canonical_crops_batch(
            pending_frames,
            pending_obbs,
            geometry,
            runtime,
            suppress_foreign=(
                cfg.pose.suppress_foreign_regions if cfg.pose is not None else False
            ),
            # Match `gen.save_interpolated_crop`'s (`_process_single_frame`)
            # background color -- both paths must agree so the images saved
            # to disk and the crops actually fed to the pose/CNN models are
            # the same crop, not two differently-padded versions.
            background_color=tuple(background_color),
        )
        pose_by_frame = run_pose_batch(
            crop_batch, pose_model, cfg.pose, runtime, geometry
        )
        for frame_idx, tasks in zip(
            (obb.frame_idx for obb in pending_obbs), pending_tasks_by_frame
        ):
            pose_result = pose_by_frame.get(frame_idx)
            if pose_result is None:
                continue
            for i, task in enumerate(tasks):
                kpts = (
                    pose_result.keypoints[i] if i < len(pose_result.keypoints) else None
                )
                # `_assemble_pose_result` (stages/pose.py) pre-allocates a
                # zero-filled (n, K, 3) array and simply `continue`s past a
                # detection the backend found nothing for -- so `kpts` here
                # is NEVER None, even for a fabricated all-zero (x=0, y=0,
                # conf=0) "no detection" case. `valid_mask[i]` is the ONLY
                # signal that distinguishes a real backend result from that
                # fabrication (it is only set True when the backend actually
                # returned keypoints AND enough of them cleared the
                # confidence gate) -- gate on it so a backend miss produces
                # NO `PoseKpt_*` columns and no false `PoseSource="interp"`
                # stamp, matching the old ``_flush_pose_batch``'s ``keypoints
                # is not None and len(keypoints) > 0`` semantics.
                is_valid = (
                    bool(pose_result.valid_mask[i])
                    if i < len(pose_result.valid_mask)
                    else False
                )
                pose_wide = {}
                pose_mean_conf = pose_valid_fraction = 0.0
                pose_num_valid = pose_num_keypoints = 0
                pose_source = None
                if kpts is not None and is_valid:
                    conf_col = kpts[:, 2]
                    pose_num_keypoints = int(kpts.shape[0])
                    valid_mask = conf_col >= float(cfg.pose.min_keypoint_confidence)
                    pose_num_valid = int(valid_mask.sum())
                    pose_mean_conf = (
                        float(conf_col.mean()) if pose_num_keypoints else 0.0
                    )
                    pose_valid_fraction = (
                        pose_num_valid / pose_num_keypoints
                        if pose_num_keypoints
                        else 0.0
                    )
                    pose_wide = flatten_pose_keypoints_row(
                        kpts,
                        build_pose_keypoint_labels(
                            pose_model.keypoint_names, pose_num_keypoints
                        ),
                    )
                    pose_source = "interp"
                pose_row = {
                    "frame_id": int(task["frame_id"]),
                    "trajectory_id": int(task["traj_id"]),
                    # Always empty (unlike the old per-frame `_flush_pose_batch`,
                    # which had the saved crop's filename in scope): this is a
                    # windowed batch flush over many frames' tasks, with no
                    # per-detection filename in scope at flush time. Nothing
                    # downstream reads this column (verified: no reader of
                    # interpolated_pose.csv's "filename" grep-hits anywhere in
                    # src/), so left empty rather than threading it through.
                    "filename": "",
                    "PoseMeanConf": pose_mean_conf,
                    "PoseValidFraction": pose_valid_fraction,
                    "PoseNumValid": pose_num_valid,
                    "PoseNumKeypoints": pose_num_keypoints,
                    "PoseSource": pose_source,
                }
                pose_row.update(pose_wide)
                interp_pose_rows.append(pose_row)
    if profiler is not None:
        profiler.tock("interp_pose_inference")

    if profiler is not None:
        profiler.tick("interp_cnn_inference")
    for cnn_model, cnn_label, cnn_cfg in zip(cnn_models, cnn_labels, cfg.cnn_phases):
        try:
            cnn_by_frame = run_cnn_batch(
                pending_frames, pending_obbs, cnn_model, cnn_cfg, runtime, geometry
            )
        except Exception as exc:
            logger.warning("Interp CNN '%s' batch failed: %s", cnn_label, exc)
            continue
        for frame_idx, tasks in zip(
            (obb.frame_idx for obb in pending_obbs), pending_tasks_by_frame
        ):
            cnn_result = cnn_by_frame.get(frame_idx)
            if cnn_result is None:
                continue
            for i, task in enumerate(tasks):
                pred = next(
                    (p for p in cnn_result.predictions if p.det_index == i), None
                )
                if pred is None:
                    continue
                factor_names = []
                class_names = []
                confidences = []
                for factor in pred.factors:
                    factor_names.append(factor.factor_name)
                    probs = np.asarray(factor.raw_probabilities, dtype=np.float32)
                    if probs.size == 0:
                        class_names.append(None)
                        confidences.append(0.0)
                        continue
                    best_idx = int(np.argmax(probs))
                    best_conf = float(probs[best_idx])
                    if 0 <= best_idx < len(factor.class_names):
                        class_names.append(factor.class_names[best_idx])
                    else:
                        class_names.append(None)
                    confidences.append(best_conf)
                row = {
                    "frame_id": int(task["frame_id"]),
                    "trajectory_id": int(task["traj_id"]),
                }
                row.update(
                    flatten_cnn_prediction_row(
                        cnn_label,
                        factor_names,
                        class_names,
                        confidences,
                    )
                )
                row[f"CNN_{cnn_label}_Source"] = "interp"
                interp_cnn_rows.setdefault(cnn_label, []).append(row)
    if profiler is not None:
        profiler.tock("interp_cnn_inference")


def _detect_apriltags_in_frame(apriltag_model, cfg, frame, obb, tasks, interp_tag_rows):
    """Detect AprilTags in one frame's interpolated crops via the SAME
    ``extract_aabb_crops``/``run_apriltag`` ``Pipeline`` uses for real
    detections (``pipeline.py:389-398``) -- no batch variant exists for
    AprilTag (design spec, "Key architectural finding"), so this stays
    per-frame like today.

    Per the design spec's AprilTag/foreign-suppression decision: unlike the
    old hand-rolled path (which foreign-masked other synthetic tasks' AABB
    regions via ``SUPPRESS_FOREIGN_OBB_REGIONS``), ``extract_aabb_crops`` has
    no suppression parameter at all -- interpolated AprilTag crops lose
    foreign-suppression of other interpolated tasks, deliberately matching
    what real detections already get.
    """
    if not tasks:
        return
    aabb_crops = extract_aabb_crops(frame, obb, padding=cfg.crop_padding)
    result = run_apriltag(aabb_crops, obb, apriltag_model, cfg)
    for tag_id, det_idx in zip(result.tag_ids, result.det_indices):
        if det_idx >= len(tasks):
            continue
        task = tasks[det_idx]
        interp_tag_rows.append(
            {
                "frame_id": int(task["frame_id"]),
                "trajectory_id": int(task["traj_id"]),
                "tag_id": int(tag_id),
            }
        )


def _flush_headtail_window(
    pending_frames,
    pending_obbs,
    pending_tasks_by_frame,
    headtail_model,
    cfg,
    runtime,
    geometry,
    interp_headtail_rows,
):
    """Run head-tail classification over a window via ``run_headtail_batch``
    -- the SAME function ``Pipeline`` calls for real detections
    (``pipeline.py:342-350``). Switches from the old per-frame
    ``HeadTailAnalyzer.analyze_crops`` to the windowed batch path: a
    materially different crop-construction path, registered as an expected
    difference in the design spec's Testing section (verify equivalence
    empirically on the characterization golden, not byte-identity).
    """
    from hydra_suite.core.inference.stages.headtail import run_headtail_batch

    if not pending_frames or headtail_model is None:
        return
    headtail_by_frame = run_headtail_batch(
        pending_frames, pending_obbs, headtail_model, cfg.headtail, runtime, geometry
    )
    for frame_idx, tasks in zip(
        (obb.frame_idx for obb in pending_obbs), pending_tasks_by_frame
    ):
        result = headtail_by_frame.get(frame_idx)
        if result is None:
            continue
        for i, task in enumerate(tasks):
            if i >= len(result.heading_hints):
                continue
            interp_headtail_rows.append(
                {
                    "frame_id": int(task["frame_id"]),
                    "trajectory_id": int(task["traj_id"]),
                    "heading_rad": float(result.heading_hints[i]),
                    "heading_conf": float(result.heading_confidences[i]),
                    "heading_directed": int(result.directed_mask[i]),
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
            "PoseSource",
            *pose_wide_columns_for_labels(pose_kpt_labels),
        ]
        result["pose_csv_path"] = _write_csv_artifact(
            parent / "interpolated_pose.csv", pose_fieldnames, interp_pose_rows
        )

    if interp_tag_rows:
        result["tag_csv_path"] = _write_csv_artifact(
            parent / "interpolated_tags.csv",
            ["frame_id", "trajectory_id", "tag_id"],
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


def _close_resource(resource):
    """Best-effort ``release()``/``close()`` on one resource, swallowing errors."""
    if resource is None:
        return
    try:
        if hasattr(resource, "release"):
            resource.release()
        elif hasattr(resource, "close"):
            resource.close()
    except Exception:
        pass


def _close_model_backend(model):
    """Close the REAL backend behind a stage-model wrapper, not the wrapper.

    ``PoseModel.close()``/``CNNModel.close()``/``HeadTailModel.close()``/
    ``AprilTagModel.close()`` (``core/inference/stages/*.py``) are ALL
    ``def close(self): pass`` -- shared infra also used by
    ``Pipeline``/``InferenceRunner`` for real detections, out of scope to
    change here. Calling ``.close()`` on the wrapper is therefore a no-op:
    for the SLEAP service backend in particular, the underlying
    ``model.backend`` is what actually reaches ``shutdown_sleap_service()``
    and terminates the subprocess. Reach into ``.backend`` (pose/CNN/
    head-tail models) or ``.detector`` (AprilTag's field is named
    differently) and close/release THAT object instead.
    """
    if model is None:
        return
    underlying = getattr(model, "backend", None)
    if underlying is None:
        underlying = getattr(model, "detector", None)
    _close_resource(underlying)


def _cleanup_backends(
    cap,
    detection_cache,
    pose_model,
    apriltag_model,
    cnn_models,
    headtail_model,
):
    """Safely close all loaded resources."""
    for resource in (cap, detection_cache):
        _close_resource(resource)
    for model in (pose_model, apriltag_model, headtail_model):
        _close_model_backend(model)
    for model in cnn_models or []:
        _close_model_backend(model)


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


def _filter_degenerate_and_get_corners(tasks, geometry, clipping_stats):
    """Degenerate-OBB pre-filter + corner geometry for one frame's tasks.

    Delegates the pre-filter itself to
    ``synthetic_detections.filter_degenerate_tasks`` (design spec, "Error
    handling") -- this wrapper now just also returns the per-task OBB
    corners for callers that still need the raw geometry (the
    interpolated-crop image-save path in ``_process_single_frame``).
    """
    from hydra_suite.core.individual.geometry import ellipse_to_obb_corners as _e2obb
    from hydra_suite.core.post.synthetic_detections import filter_degenerate_tasks

    kept_tasks = filter_degenerate_tasks(tasks, geometry, clipping_stats)
    corners = [_e2obb(t["cx"], t["cy"], t["w"], t["h"], t["theta"]) for t in kept_tasks]
    return kept_tasks, corners


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
    apriltag_model,
    apriltag_cfg,
    interp_saved,
    interp_rows,
    roi_rows,
    roi_corners,
    interp_tag_rows,
    _pending_frames,
    _pending_obbs,
    _pending_tasks_by_frame,
):
    # NOTE: `params` and `should_stop` are unused in this body -- per-frame
    # work here has no stop-checkpoint of its own (the caller's loop already
    # checks `should_stop` before/after calling this) and reads no params
    # directly. Kept in the signature because the caller passes them
    # positionally alongside the other loop state and existing tests call
    # this function with the same positional shape; not removed here to
    # avoid a wider signature churn.
    def _emit(v, m):
        if progress is not None:
            progress(v, m)

    kept_tasks, corners = _filter_degenerate_and_get_corners(
        frame_tasks[f], geometry, clipping_stats
    )

    for task_idx, task in enumerate(kept_tasks):
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
                canonical_affine=None,
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
            roi_corners.append(corners[task_idx])

    if kept_tasks:
        from hydra_suite.core.post.synthetic_detections import (
            build_synthetic_obb_result,
        )

        obb = build_synthetic_obb_result(f, kept_tasks)
        if apriltag_model is not None:
            _detect_apriltags_in_frame(
                apriltag_model, apriltag_cfg, frame, obb, kept_tasks, interp_tag_rows
            )
        _pending_frames.append(frame)
        _pending_obbs.append(obb)
        _pending_tasks_by_frame.append(kept_tasks)

    if idx % 25 == 0 or idx == total_frames:
        progress_pct = int((idx / total_frames) * 100)
        _emit(progress_pct, f"Interpolating occlusions... {idx}/{total_frames}")
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
    cfg,
    runtime,
    pose_model,
    cnn_models,
    cnn_labels,
    apriltag_model,
    headtail_model,
    interp_saved,
    interp_rows,
    roi_rows,
    roi_corners,
    interp_pose_rows,
    interp_tag_rows,
    interp_cnn_rows,
    interp_headtail_rows,
    profiler,
):
    def _stop():
        return bool(should_stop()) if should_stop is not None else False

    needed_frames = sorted(frame_tasks.keys())
    total_frames = len(needed_frames)
    # NOTE: unlike POSE_BATCH_SIZE/CNN batch knobs (which bound a batch of
    # individual crops), this now bounds the number of FULL DECODED FRAMES
    # buffered in `_pending_frames` per inference window (each frame can
    # carry many per-animal tasks) -- on a 4K clip a large value here is
    # multiple GB resident. Default kept small and memory-safe; this key has
    # no GUI/param-builder exposure yet (hand-edit the params dict only).
    window_batch_size = int(params.get("INTERP_POSE_INFERENCE_BATCH_SIZE", 8))
    _pending_frames: list = []
    _pending_obbs: list = []
    _pending_tasks_by_frame: list = []

    def _flush_window():
        _flush_pose_cnn_window(
            _pending_frames,
            _pending_obbs,
            _pending_tasks_by_frame,
            pose_model,
            cnn_models,
            cnn_labels,
            cfg,
            runtime,
            geometry,
            interp_pose_rows,
            interp_cnn_rows,
            profiler,
            background_color=getattr(gen, "background_color", (0, 0, 0)),
        )
        _flush_headtail_window(
            _pending_frames,
            _pending_obbs,
            _pending_tasks_by_frame,
            headtail_model,
            cfg,
            runtime,
            geometry,
            interp_headtail_rows,
        )
        _pending_frames.clear()
        _pending_obbs.clear()
        _pending_tasks_by_frame.clear()

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
            apriltag_model,
            cfg.apriltag,
            interp_saved,
            interp_rows,
            roi_rows,
            roi_corners,
            interp_tag_rows,
            _pending_frames,
            _pending_obbs,
            _pending_tasks_by_frame,
        )
        # `_process_single_frame` always returns an int (its internal
        # `_stop()`-based early-return path was removed) -- no None check
        # needed here.
        interp_saved = result
        if len(_pending_frames) >= window_batch_size:
            if _stop():
                _prefetcher.stop()
                return None
            _flush_window()
    # Flush any frames still buffered when the loop exits -- whether it ran
    # to completion, `break`-ed out on prefetcher exhaustion, or the last
    # iteration(s) `continue`-ed past a bad read. The old `idx == total_frames`
    # trigger only fired from inside a loop iteration, so a `break`/`continue`
    # on the final index silently dropped up to `window_batch_size` frames'
    # worth of pose/CNN/head-tail rows -- this final flush is unconditional so
    # that can no longer happen. Buffered frames are freed here (list.clear()
    # inside _flush_window), not via any per-iteration `del`.
    if _pending_frames:
        if _stop():
            _prefetcher.stop()
            return None
        _flush_window()
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
        detection_cache = open_detection_cache_reader(detection_cache_path)

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
    from hydra_suite.core.tracking.pose.pose_pipeline import (
        reset_degenerate_padding_warning,
    )
    from hydra_suite.core.tracking.profiler import TrackingProfiler

    # Re-arm the once-per-run degenerate-padding warning: this is the entry
    # point that reaches ``extract_one_crop``, and the guard it fires is a
    # module global that otherwise stays suppressed after the first run in a
    # long-lived (GUI) process ever logs it.
    reset_degenerate_padding_warning()

    def _stop():
        return bool(should_stop()) if should_stop is not None else False

    # Run-scoped canonical-crop clipping accumulator; reported at the end of
    # the pass, mirroring TrackingWorker's end-of-run summary.
    clipping_stats = ClippingStats()
    profiler = TrackingProfiler(enabled=enable_profiling)
    profiler.phase_start("interp_setup")

    pose_model = None
    detection_cache = None
    cap = None
    cnn_models = []
    cnn_labels = []
    apriltag_model = None
    headtail_model = None
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

        pose_kpt_labels = []
        if frame_tasks:
            (
                cfg,
                runtime,
                pose_model,
                apriltag_model,
                cnn_models,
                cnn_labels,
                headtail_model,
            ) = _init_interpolation_backends(params, output_dir, geometry)
            if pose_model is not None:
                pose_kpt_labels = build_pose_keypoint_labels(
                    pose_model.keypoint_names, pose_model.n_keypoints
                )
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
                cfg,
                runtime,
                pose_model,
                cnn_models,
                cnn_labels,
                apriltag_model,
                headtail_model,
                interp_saved,
                interp_rows,
                roi_rows,
                roi_corners,
                interp_pose_rows,
                interp_tag_rows,
                interp_cnn_rows,
                interp_headtail_rows,
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
        # Any future silent-failure class must be visible in logs, even when
        # graceful degradation isn't possible for it -- this blanket handler
        # used to swallow everything with zero diagnostic trace, which is
        # exactly how a whole-pass loss (crops, ROI, pose/tag/cnn/headtail
        # CSVs -- everything) could vanish silently (see
        # `_init_interpolation_backends`'s own try/except for the specific
        # config-build failure this was first found from).
        logger.exception(
            "Interpolated post-pass failed; returning an empty payload "
            "(saved=0, gaps=0)."
        )
        return {"saved": 0, "gaps": 0}
    finally:
        _cleanup_backends(
            cap,
            detection_cache,
            pose_model,
            apriltag_model,
            cnn_models,
            headtail_model,
        )
