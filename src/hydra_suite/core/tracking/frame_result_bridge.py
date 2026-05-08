"""Helpers that bridge InferenceRunner FrameResult objects to the tracking-loop
live feature stores and legacy measurement format.

These replace the UnifiedPrecompute callback wiring (Task 18 integration).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hydra_suite.core.inference.result import AprilTagResult, CNNResult, PoseResult
    from hydra_suite.core.tracking.live_features import (
        LiveCNNIdentityStore,
        LivePosePropertiesStore,
        LiveTagObservationStore,
    )


def frame_result_to_meas(
    centroids: np.ndarray,
    angles: np.ndarray,
) -> list:
    """Convert OBBResult centroids + axis angles to legacy meas list.

    Each element is a (3,) float32 array [cx, cy, theta] matching the format
    expected by the Kalman filter, Hungarian assigner, and trajectory writer.

    The angle passed here should be the OBB *axis angle* (in [0, pi)) so the
    downstream tracking layer can call ``resolve_tracking_theta`` to pick
    between ``theta`` and ``theta + pi`` based on motion history and the
    head-tail classifier hints (passed separately via ``raw_heading_hints``).
    Passing the already-merged ``resolved_headings`` here would short-circuit
    that resolution and produce results inconsistent with legacy parity.

    Args:
        centroids: (D, 2) array of [cx, cy] centroid positions.
        angles: (D,) array of OBB axis angles in radians, in [0, pi).

    Returns:
        List of D numpy arrays, each shape (3,) with [cx, cy, theta].
    """
    n = len(centroids)
    meas = []
    for i in range(n):
        meas.append(
            np.array(
                [
                    float(centroids[i, 0]),
                    float(centroids[i, 1]),
                    float(angles[i]),
                ],
                dtype=np.float32,
            )
        )
    return meas


def _cnn_det_pred_to_class_prediction(pred, factor_names: tuple):  # -> ClassPrediction
    """Convert a CNNDetectionPrediction to a ClassPrediction for live stores.

    For flat single-factor models the top class + confidence are taken from
    argmax(raw_probabilities).  For multi-factor models each factor's top class
    and confidence are recorded independently.
    """
    from hydra_suite.core.identity.classification.cnn import ClassPrediction

    names = []
    confs = []
    for factor in pred.factors:
        probs = np.asarray(factor.raw_probabilities, dtype=np.float32)
        if len(probs) == 0:
            names.append(None)
            confs.append(0.0)
        else:
            best_idx = int(np.argmax(probs))
            class_name = (
                factor.class_names[best_idx]
                if best_idx < len(factor.class_names)
                else None
            )
            names.append(class_name)
            confs.append(float(probs[best_idx]))

    return ClassPrediction(
        det_index=pred.det_index,
        factor_names=factor_names,
        class_names=tuple(names),
        confidences=tuple(confs),
    )


def populate_live_cnn_store(
    store: "LiveCNNIdentityStore",
    cnn_results: "list[CNNResult]",
    detection_ids: np.ndarray,
    frame_idx: int,
    phase_label: str,
) -> None:
    """Push CNN predictions from a FrameResult into a LiveCNNIdentityStore.

    Converts CNNDetectionPrediction → ClassPrediction so the live store's
    ``load()`` method returns the same format as a legacy CNNIdentityCache.

    Args:
        store: The LiveCNNIdentityStore instance for this CNN phase.
        cnn_results: List of CNNResult (one per CNN phase) from FrameResult.cnn.
        detection_ids: (D,) int64 detection IDs for the filtered detections.
        frame_idx: Current video frame index.
        phase_label: CNN phase label string; used to select the matching phase
            from cnn_results.
    """
    # Select the right CNN phase by label
    phase_result = None
    for r in cnn_results:
        if r.label == phase_label:
            phase_result = r
            break
    if phase_result is None or not phase_result.predictions:
        store.update_frame(frame_idx, [])
        return

    # Build factor_names tuple from first prediction
    first = phase_result.predictions[0]
    factor_names = (
        tuple(f.factor_name for f in first.factors) if first.factors else ("flat",)
    )

    class_preds = [
        _cnn_det_pred_to_class_prediction(pred, factor_names)
        for pred in phase_result.predictions
    ]
    store.update_frame(frame_idx, class_preds)


def populate_live_pose_store(
    store: "LivePosePropertiesStore",
    pose: "PoseResult | None",
    detection_ids: np.ndarray,
    frame_idx: int,
) -> None:
    """Push PoseResult keypoints into a LivePosePropertiesStore.

    The store records keypoints per detection ID so the tracking loop can look
    up pose features for a detection by its integer ID.

    Args:
        store: The LivePosePropertiesStore instance.
        pose: PoseResult from FrameResult, or None if pose inference was not run.
        detection_ids: (D,) int64 detection IDs aligned to the pose keypoints.
        frame_idx: Current video frame index.
    """
    if pose is None or len(detection_ids) == 0:
        store.update_frame(frame_idx, [], [])
        return

    ids = [int(did) for did in detection_ids]
    kpts_list = []
    for i in range(pose.keypoints.shape[0]):
        if bool(pose.valid_mask[i]):
            kpts_list.append(pose.keypoints[i])  # (K, 3)
        else:
            kpts_list.append(None)

    store.update_frame(frame_idx, ids, kpts_list)


def populate_live_tag_store(
    store: "LiveTagObservationStore",
    apriltag: "AprilTagResult | None",
    detection_ids: np.ndarray,
    frame_idx: int,
) -> None:
    """Push AprilTagResult into a LiveTagObservationStore.

    Args:
        store: The LiveTagObservationStore instance.
        apriltag: AprilTagResult from FrameResult, or None if not enabled.
        detection_ids: (D,) int64 detection IDs for the filtered detections.
        frame_idx: Current video frame index.
    """
    if apriltag is None or len(apriltag.tag_ids) == 0:
        store.update_frame(frame_idx, [], [], [], [])
        return

    tag_ids = list(apriltag.tag_ids)
    det_indices = list(apriltag.det_indices)
    centers = apriltag.centers  # (T, 2)
    corners = apriltag.corners  # (T, 4, 2)

    centers_xy = [
        (float(centers[i, 0]), float(centers[i, 1])) for i in range(len(tag_ids))
    ]
    corners_list = [corners[i] for i in range(len(tag_ids))]

    # hammings defaults to 0 (AprilTagResult doesn't expose Hamming distance)
    hammings = [0] * len(tag_ids)

    store.update_frame(
        frame_idx,
        tag_ids=tag_ids,
        centers_xy=centers_xy,
        corners=corners_list,
        det_indices=det_indices,
        hammings=hammings,
    )


def build_density_cache_dict(
    runner,  # InferenceRunner (avoided forward ref for flake8)
    start_frame: int,
    end_frame: int,
) -> dict:
    """Build the {frame_idx: (meas_arr, confs_arr, sizes_arr)} dict needed by
    the confidence density map computation, reading from the InferenceRunner's
    detection cache.

    Args:
        runner: An InferenceRunner whose caches are already open (batch pass done).
        start_frame: First frame index to include.
        end_frame: Last frame index to include (inclusive).

    Returns:
        Dict mapping frame_idx → (meas_arr (F, 3), confs_arr (F,), sizes_arr (F,)).
    """

    result: dict = {}
    if runner.cache_dir is None:
        return result

    # Ensure caches are open
    if runner._caches is None:
        from hydra_suite.core.inference.runner import _open_caches as _oc

        runner._caches = _oc(runner.config, runner.cache_dir)

    det_cache = runner._caches.detection
    if det_cache is None:
        return result

    # Load the raw NPZ data to get all frame indices at once.
    if not det_cache.is_valid():
        return result

    if det_cache._data is None:
        import numpy as _np

        det_cache._data = dict(_np.load(det_cache.path))

    d = det_cache._data
    all_fi = d.get("frame_indices", np.zeros(0, np.int32))
    unique_fi = sorted(
        {int(fi) for fi in all_fi if start_frame <= int(fi) <= end_frame}
    )

    for fi in unique_fi:
        obb = det_cache.read_frame(fi)
        if obb is None or obb.num_detections == 0:
            result[fi] = (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
            continue
        # meas_arr: (D, 3) — [cx, cy, angle]; angles are raw OBB axis angles
        meas_arr = np.column_stack(
            [
                obb.centroids,
                obb.angles.reshape(-1, 1),
            ]
        ).astype(np.float32)
        confs_arr = obb.confidences.astype(np.float32)
        sizes_arr = obb.sizes.astype(np.float32)
        result[fi] = (meas_arr, confs_arr, sizes_arr)

    return result
