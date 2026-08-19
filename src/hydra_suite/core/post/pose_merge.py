"""Pose-source merge + quality post-pass (Qt-free).

Moved out of ``trackerkit/gui/orchestrators/tracking.py`` as part of the
headless Qt-free refactor. Widget/state reads that used to go through
``self._mw``/``self._panels`` are now explicit args or fields on
``PoseSourceState``.
"""

import glob as _glob
import json
import logging
import os
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PoseSourceState:
    individual_properties_cache_path: str | None = None
    detected_properties_cache_path: str | None = None
    detection_cache_path: str | None = None
    interpolated_pose_csv_path: str | None = None
    interpolated_pose_df: object | None = None
    interpolated_tag_csv_path: str | None = None
    interpolated_tag_df: object | None = None
    interpolated_cnn_csv_paths: dict | None = None
    interpolated_cnn_dfs: dict | None = None
    interpolated_headtail_csv_path: str | None = None
    interpolated_headtail_df: object | None = None
    inference_cache_dir: str | None = None
    """Directory holding the InferenceRunner caches (``.inference_cache_<stem>/``).

    Source of the detection-keyed CNN predictions merged below. Unset means no
    detection-keyed CNN merge -- the pre-restoration behavior."""


def check_pose_export_sources(state):
    """Return (has_other_analyses, cache_path, cache_available, interp_pose_path,
    interp_available, interp_pose_df_mem, interp_mem_available)."""
    _detected_props_path = str(state.detected_properties_cache_path or "").strip()
    _has_detected_props = bool(
        _detected_props_path and os.path.exists(_detected_props_path)
    )
    _has_interp_tag = bool(
        state.interpolated_tag_csv_path
        or isinstance(state.interpolated_tag_df, pd.DataFrame)
    )
    _has_interp_cnn = bool(
        state.interpolated_cnn_csv_paths or state.interpolated_cnn_dfs
    )
    _has_interp_ht = bool(
        state.interpolated_headtail_csv_path
        or isinstance(state.interpolated_headtail_df, pd.DataFrame)
    )
    _has_other_analyses = (
        _has_detected_props or _has_interp_tag or _has_interp_cnn or _has_interp_ht
    )
    cache_path = str(state.individual_properties_cache_path or "").strip()
    cache_available = bool(cache_path and os.path.exists(cache_path))
    interp_pose_path = str(state.interpolated_pose_csv_path or "").strip()
    interp_available = bool(interp_pose_path and os.path.exists(interp_pose_path))
    interp_pose_df_mem = state.interpolated_pose_df
    interp_mem_available = (
        isinstance(interp_pose_df_mem, pd.DataFrame) and not interp_pose_df_mem.empty
    )
    return (
        _has_other_analyses,
        cache_path,
        cache_available,
        interp_pose_path,
        interp_available,
        interp_pose_df_mem,
        interp_mem_available,
    )


def resolve_current_tag_cache_path(params, detection_cache_path):
    """Return the best available detected AprilTag cache path for this session."""
    if not bool(params.get("USE_APRILTAGS", False)):
        return ""
    if not detection_cache_path or not os.path.exists(str(detection_cache_path)):
        return ""
    pattern = str(detection_cache_path).replace(".npz", "") + "_tags_*.npz"
    candidates = sorted(_glob.glob(pattern))
    return str(candidates[-1]) if candidates else ""


def merge_pose_sources_into_df(
    trajectories_df,
    sources,
    state,
    *,
    params,
    min_valid_conf,
    ignore_keypoints,
):
    """Merge pose cache, interpolated pose, AprilTag, CNN, and head-tail into trajectories_df."""
    from hydra_suite.core.individual.properties.export import (
        augment_trajectories_with_detected_properties_cache,
        augment_trajectories_with_pose_cache,
        merge_interpolated_pose_df,
    )

    (
        cache_path,
        cache_available,
        interp_pose_path,
        interp_available,
        interp_pose_df_mem,
        interp_mem_available,
    ) = sources[1:]

    with_pose_df = trajectories_df
    _detected_props_path = str(state.detected_properties_cache_path or "").strip()
    if _detected_props_path and os.path.exists(_detected_props_path):
        with_pose_df = augment_trajectories_with_detected_properties_cache(
            with_pose_df,
            _detected_props_path,
        )

    _tag_cache_path = resolve_current_tag_cache_path(params, state.detection_cache_path)
    if _tag_cache_path and os.path.exists(_tag_cache_path):
        try:
            from hydra_suite.core.individual.properties.export import (
                augment_trajectories_with_detected_apriltag_cache,
            )

            _tag_labels = [
                str(_lbl) for _lbl in (params.get("TAG_IDENTITY_LABELS", []) or [])
            ]
            with_pose_df = augment_trajectories_with_detected_apriltag_cache(
                with_pose_df,
                _tag_cache_path,
                tag_labels=_tag_labels,
            )
        except Exception:
            logger.debug(
                "Detection-level AprilTag augmentation skipped.", exc_info=True
            )

    if cache_available:
        _resize_factor = float(params.get("RESIZE_FACTOR", 1.0))
        _coord_scale = (
            1.0 / _resize_factor if _resize_factor and _resize_factor != 1.0 else 1.0
        )
        with_pose_df = augment_trajectories_with_pose_cache(
            with_pose_df,
            cache_path,
            ignore_keypoints=ignore_keypoints,
            min_valid_conf=min_valid_conf,
            coordinate_scale=_coord_scale,
        )
    if interp_available:
        interp_pose_df = pd.read_csv(interp_pose_path)
        with_pose_df = merge_interpolated_pose_df(with_pose_df, interp_pose_df)
    elif interp_mem_available:
        with_pose_df = merge_interpolated_pose_df(with_pose_df, interp_pose_df_mem)

    _interp_tag_path = str(state.interpolated_tag_csv_path or "").strip()
    _interp_tag_df = state.interpolated_tag_df
    try:
        from hydra_suite.core.individual.properties.export import (
            merge_interpolated_apriltag_df,
        )

        if _interp_tag_path and os.path.exists(_interp_tag_path):
            _tag_df = pd.read_csv(_interp_tag_path)
            with_pose_df = merge_interpolated_apriltag_df(with_pose_df, _tag_df)
        elif isinstance(_interp_tag_df, pd.DataFrame) and not _interp_tag_df.empty:
            with_pose_df = merge_interpolated_apriltag_df(with_pose_df, _interp_tag_df)
    except Exception:
        logger.debug("Interpolated AprilTag merge skipped.", exc_info=True)

    # Detection-keyed CNN predictions, mirroring the detected-properties and
    # detected-AprilTag merges above. These columns went missing when the
    # Gen-2 inference migration replaced the V3 CNNIdentityCache (whose reader
    # was os.path.exists-guarded, so its loss was silent) with the CNN stage's
    # own per-detection cache. Everything downstream that reads
    # `CNN_<label>_Class` -- the identity evidence summary, UniqueIdentityKey,
    # and the non-identifying-class report -- was starved by that gap.
    _inference_cache_dir = str(state.inference_cache_dir or "").strip()
    if _inference_cache_dir and os.path.isdir(_inference_cache_dir):
        try:
            from hydra_suite.core.individual.properties.export import (
                augment_trajectories_with_detected_cnn_cache,
            )

            for _cfg in params.get("CNN_CLASSIFIERS", []) or []:
                _label = str(_cfg.get("label", "") or "").strip()
                if not _label:
                    continue
                _cnn_cache = os.path.join(_inference_cache_dir, f"cnn_{_label}.npz")
                if not os.path.exists(_cnn_cache):
                    continue
                with_pose_df = augment_trajectories_with_detected_cnn_cache(
                    with_pose_df, _cnn_cache, _label
                )
        except Exception:
            logger.debug("Detection-level CNN augmentation skipped.", exc_info=True)

    _interp_cnn_paths = state.interpolated_cnn_csv_paths or {}
    _interp_cnn_dfs = state.interpolated_cnn_dfs or {}
    try:
        from hydra_suite.core.individual.properties.export import (
            merge_interpolated_cnn_df,
        )

        _all_cnn_labels = set(_interp_cnn_paths.keys()) | set(_interp_cnn_dfs.keys())
        for _cnn_label in _all_cnn_labels:
            _cnn_path = str(_interp_cnn_paths.get(_cnn_label, "")).strip()
            if _cnn_path and os.path.exists(_cnn_path):
                _cnn_df = pd.read_csv(_cnn_path)
                with_pose_df = merge_interpolated_cnn_df(
                    with_pose_df, _cnn_df, label=_cnn_label
                )
            elif _cnn_label in _interp_cnn_dfs:
                _cnn_df = _interp_cnn_dfs[_cnn_label]
                if isinstance(_cnn_df, pd.DataFrame) and not _cnn_df.empty:
                    with_pose_df = merge_interpolated_cnn_df(
                        with_pose_df, _cnn_df, label=_cnn_label
                    )
    except Exception:
        logger.debug("Interpolated CNN merge skipped.", exc_info=True)

    _interp_ht_path = str(state.interpolated_headtail_csv_path or "").strip()
    _interp_ht_df = state.interpolated_headtail_df
    try:
        from hydra_suite.core.individual.properties.export import (
            merge_interpolated_headtail_df,
        )

        if _interp_ht_path and os.path.exists(_interp_ht_path):
            _ht_df = pd.read_csv(_interp_ht_path)
            with_pose_df = merge_interpolated_headtail_df(with_pose_df, _ht_df)
        elif isinstance(_interp_ht_df, pd.DataFrame) and not _interp_ht_df.empty:
            with_pose_df = merge_interpolated_headtail_df(with_pose_df, _interp_ht_df)
    except Exception:
        logger.debug("Interpolated head-tail merge skipped.", exc_info=True)

    return with_pose_df


def apply_pose_quality_postprocessing(
    with_pose_df,
    pose_labels,
    params,
    *,
    individual_properties_cache_path,
):
    """Apply quality gating and temporal post-processing to pose-augmented dataframe."""
    from hydra_suite.core.individual.pose.features import resolve_pose_group_indices
    from hydra_suite.core.individual.pose.quality import (
        apply_quality_to_dataframe,
        apply_temporal_pose_postprocessing,
        calibrate_body_length_prior,
        calibrate_edge_length_priors,
    )

    kpt_names = []
    try:
        from hydra_suite.core.individual.properties.cache import (
            IndividualPropertiesCache,
        )

        _cache_path = str(individual_properties_cache_path or "").strip()
        if _cache_path and os.path.exists(_cache_path):
            _cache = IndividualPropertiesCache(_cache_path, mode="r")
            try:
                kpt_names = [
                    str(v)
                    for v in (_cache.metadata.get("pose_keypoint_names", []) or [])
                ]
            finally:
                _cache.close()
    except Exception:
        pass
    anterior_indices = resolve_pose_group_indices(
        params.get("POSE_DIRECTION_ANTERIOR_KEYPOINTS", []), kpt_names
    )
    posterior_indices = resolve_pose_group_indices(
        params.get("POSE_DIRECTION_POSTERIOR_KEYPOINTS", []), kpt_names
    )

    skeleton_edges = []
    try:
        _skel_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
        if _skel_file and os.path.exists(_skel_file):
            with open(_skel_file, "r", encoding="utf-8") as _sf:
                _skel_data = json.load(_sf)
            for _edge in _skel_data.get("skeleton_edges", _skel_data.get("edges", [])):
                if isinstance(_edge, (list, tuple)) and len(_edge) >= 2:
                    try:
                        skeleton_edges.append((int(_edge[0]), int(_edge[1])))
                    except Exception:
                        pass
    except Exception:
        logger.exception("Failed to load skeleton edges for anatomy check; skipping.")
        skeleton_edges = []

    body_length_prior = None
    if anterior_indices and posterior_indices:
        try:
            body_length_prior = calibrate_body_length_prior(
                with_pose_df,
                pose_labels,
                anterior_indices,
                posterior_indices,
                min_valid_conf=float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
            )
            if body_length_prior.is_valid:
                logger.info(
                    "Body-length prior calibrated: median=%.1f px, MAD=%.1f px, n=%d",
                    body_length_prior.median_px,
                    body_length_prior.mad_px,
                    body_length_prior.n_samples,
                )
        except Exception:
            logger.exception(
                "Body-length prior calibration failed; skipping anatomy check."
            )
            body_length_prior = None

    edge_length_priors = None
    if skeleton_edges:
        try:
            edge_length_priors = calibrate_edge_length_priors(
                with_pose_df,
                pose_labels,
                skeleton_edges,
                min_valid_conf=float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
            )
            if edge_length_priors.is_valid:
                logger.info(
                    "Edge-length priors calibrated for %d edges.",
                    len(edge_length_priors.priors),
                )
        except Exception:
            logger.exception(
                "Edge-length prior calibration failed; skipping skeleton check."
            )
            edge_length_priors = None

    try:
        with_pose_df = apply_quality_to_dataframe(
            with_pose_df,
            pose_labels,
            params,
            body_length_prior=body_length_prior,
            anterior_indices=anterior_indices if anterior_indices else None,
            posterior_indices=posterior_indices if posterior_indices else None,
            skeleton_edges=skeleton_edges if skeleton_edges else None,
            edge_length_priors=edge_length_priors,
        )
    except Exception:
        logger.exception("Pose quality gating failed; using unfiltered pose.")

    max_gap = int(params.get("POSE_POSTPROC_MAX_GAP", 5))
    z_threshold = float(params.get("POSE_TEMPORAL_OUTLIER_ZSCORE", 3.0))
    if z_threshold > 0.0 and "TrajectoryID" in with_pose_df.columns:
        try:
            parts = []
            for _, traj_group in with_pose_df.groupby("TrajectoryID", sort=False):
                parts.append(
                    apply_temporal_pose_postprocessing(
                        traj_group,
                        pose_labels,
                        max_gap=max_gap,
                        z_score_threshold=z_threshold,
                    )
                )
            if parts:
                with_pose_df = (
                    pd.concat(parts, ignore_index=True)
                    .sort_values(["TrajectoryID", "FrameID"], kind="stable")
                    .reset_index(drop=True)
                )
        except Exception:
            logger.exception(
                "Pose temporal post-processing failed; using unfiltered pose."
            )
    return with_pose_df
