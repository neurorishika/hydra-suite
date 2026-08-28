"""Pure trajectory-merge functions (Qt-free), extracted from MergeWorker."""

from __future__ import annotations

import csv
import logging

import numpy as np
import pandas as pd

from hydra_suite.core.post.processing import (
    interpolate_trajectories,
    resolve_trajectories,
    trim_positionless_ends,
)
from hydra_suite.utils import profiling_names as N
from hydra_suite.utils.profiling import span

logger = logging.getLogger(__name__)


def convert_resolved_to_dataframe(resolved_trajectories):
    """Convert a list of resolved trajectories to a single DataFrame."""
    if not resolved_trajectories or not isinstance(resolved_trajectories, list):
        return resolved_trajectories
    if isinstance(resolved_trajectories[0], pd.DataFrame):
        for new_id, traj_df in enumerate(resolved_trajectories):
            traj_df["TrajectoryID"] = new_id
        return pd.concat(resolved_trajectories, ignore_index=True)
    logger.warning("Received tuple format from resolve_trajectories, converting...")
    all_data = []
    for traj_id, traj in enumerate(resolved_trajectories):
        for x, y, theta, frame in traj:
            all_data.append(
                {
                    "TrajectoryID": traj_id,
                    "X": x,
                    "Y": y,
                    "Theta": theta,
                    "FrameID": frame,
                }
            )
    return pd.DataFrame(all_data) if all_data else []


def resolve_tag_identities(
    resolved_trajectories, *, tag_cache_path, params, progress=None
):
    """Apply AprilTag identity resolution if a tag cache is available."""
    if not isinstance(resolved_trajectories, pd.DataFrame) or not tag_cache_path:
        return resolved_trajectories
    try:
        from hydra_suite.core.post.tag_identity import detect_tag_swaps
        from hydra_suite.core.post.tag_identity import (
            resolve_tag_identities as _resolve_tag_identities,
        )
        from hydra_suite.data.tag_observation_cache import TagObservationCache

        if progress is not None:
            progress(92, "Resolving tag identities...")
        tag_cache = TagObservationCache(str(tag_cache_path), mode="r")
        resolved_trajectories = _resolve_tag_identities(
            resolved_trajectories, tag_cache, params
        )
        swaps = detect_tag_swaps(resolved_trajectories, tag_cache, params)
        if swaps:
            logger.warning("Detected %d potential tag-swap events", len(swaps))
        tag_cache.close()
    except Exception:
        logger.warning("Tag identity resolution failed (non-fatal)", exc_info=True)
    return resolved_trajectories


def rescale_coordinates(resolved_trajectories, *, resize_factor):
    """Scale coordinates back to original video space."""
    if not isinstance(resolved_trajectories, pd.DataFrame):
        return resolved_trajectories
    logger.info(
        f"Pre-scaling (resize_factor={resize_factor:.3f}): "
        f"X range [{resolved_trajectories['X'].min():.1f}, {resolved_trajectories['X'].max():.1f}], "
        f"Y range [{resolved_trajectories['Y'].min():.1f}, {resolved_trajectories['Y'].max():.1f}]"
    )
    resolved_trajectories[["X", "Y"]] = (
        resolved_trajectories[["X", "Y"]] / resize_factor
    )
    if "Width" in resolved_trajectories.columns:
        resolved_trajectories["Width"] /= resize_factor
    if "Height" in resolved_trajectories.columns:
        resolved_trajectories["Height"] /= resize_factor
    logger.info(
        f"Post-scaling: "
        f"X range [{resolved_trajectories['X'].min():.1f}, {resolved_trajectories['X'].max():.1f}], "
        f"Y range [{resolved_trajectories['Y'].min():.1f}, {resolved_trajectories['Y'].max():.1f}]"
    )
    return resolved_trajectories


def merge_trajectories(
    forward_trajs,
    backward_trajs,
    *,
    total_frames,
    params,
    resize_factor,
    interp_method,
    max_gap,
    tag_cache_path=None,
    heading_flip_max_burst=5,
    directed_heading_posthoc=False,
    fill_all_interior: bool = False,
    enable_profiling=False,
    profile_export_path=None,
    progress=None,
    should_stop=None,
):
    """Merge forward and backward trajectories. Returns merged DataFrame, or None if stopped."""
    from hydra_suite.core.tracking.profiler import TrackingProfiler

    def _stop() -> bool:
        return bool(should_stop()) if should_stop is not None else False

    def _emit(value, message) -> None:
        if progress is not None:
            progress(value, message)

    profiler = TrackingProfiler(enabled=enable_profiling)

    with profiler.armed(), span(N.POST):
        if _stop():
            return None
        profiler.phase_start("post_prepare")
        _emit(10, "Preparing trajectories...")

        def prepare_trajs_for_merge(trajs):
            if isinstance(trajs, pd.DataFrame):
                return [group for _, group in trajs.groupby("TrajectoryID")]
            return trajs

        with span(N.PREPARE):
            forward_prepared = prepare_trajs_for_merge(forward_trajs)
            backward_prepared = prepare_trajs_for_merge(backward_trajs)
        profiler.phase_end("post_prepare")

        if _stop():
            return None
        profiler.phase_start("post_resolve")
        _emit(30, "Resolving trajectory conflicts...")
        with span(N.RESOLVE):
            resolved = resolve_trajectories(
                forward_prepared,
                backward_prepared,
                params=params,
                should_stop=should_stop,
            )
        profiler.phase_end("post_resolve")

        if _stop():
            return None
        _emit(60, "Converting to DataFrame...")
        resolved = convert_resolved_to_dataframe(resolved)

        profiler.phase_start("post_interpolate")
        _emit(75, "Applying interpolation...")
        with span(N.INTERPOLATE):
            if isinstance(resolved, pd.DataFrame) and interp_method != "none":
                resolved = interpolate_trajectories(
                    resolved,
                    method=interp_method,
                    max_gap=max_gap,
                    heading_flip_max_burst=heading_flip_max_burst,
                    directed_heading_posthoc=directed_heading_posthoc,
                    fill_all_interior=fill_all_interior,
                )
            elif isinstance(resolved, pd.DataFrame):
                # No interpolation requested: positions are not fabricated,
                # but leading/trailing position-less rows can never be filled
                # and must not be left in the final CSV as silent NaN.
                resolved = trim_positionless_ends(resolved)
        profiler.phase_end("post_interpolate")

        if _stop():
            return None
        _emit(90, "Scaling to original space...")
        profiler.phase_start("post_tag_identity")
        with span(N.TAG_IDENTITY):
            resolved = resolve_tag_identities(
                resolved,
                tag_cache_path=tag_cache_path,
                params=params,
                progress=progress,
            )
        profiler.phase_end("post_tag_identity")

        profiler.phase_start("post_rescale")
        with span(N.RESCALE):
            resolved = rescale_coordinates(resolved, resize_factor=resize_factor)
        profiler.phase_end("post_rescale")

        if _stop():
            return None
        profiler.log_final_summary()
        if profile_export_path:
            profiler.export_summary(profile_export_path)
        _emit(100, "Merge complete!")
        return resolved


def write_csv_artifact(path, fieldnames, rows):
    """Write a CSV artifact file. Returns the path on success, None on failure."""
    try:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path
    except Exception:
        return None


def write_roi_npz(path, roi_rows, roi_corners):
    """Write ROI data to a compressed NPZ file. Returns path on success, None on failure."""
    try:
        np.savez_compressed(
            str(path),
            frame_id=np.array([r["frame_id"] for r in roi_rows], dtype=np.int64),
            trajectory_id=np.array(
                [r["trajectory_id"] for r in roi_rows], dtype=np.int64
            ),
            filename=np.array([r["filename"] for r in roi_rows], dtype=object),
            cx=np.array([r["cx"] for r in roi_rows], dtype=np.float32),
            cy=np.array([r["cy"] for r in roi_rows], dtype=np.float32),
            w=np.array([r["w"] for r in roi_rows], dtype=np.float32),
            h=np.array([r["h"] for r in roi_rows], dtype=np.float32),
            theta=np.array([r["theta"] for r in roi_rows], dtype=np.float32),
            interp_from_start=np.array(
                [r["interp_from_start"] for r in roi_rows], dtype=np.int64
            ),
            interp_from_end=np.array(
                [r["interp_from_end"] for r in roi_rows], dtype=np.int64
            ),
            interp_index=np.array(
                [r["interp_index"] for r in roi_rows], dtype=np.int64
            ),
            interp_total=np.array(
                [r["interp_total"] for r in roi_rows], dtype=np.int64
            ),
            obb_corners=(
                np.stack(roi_corners).astype(np.float32)
                if roi_corners
                else np.zeros((0, 4, 2), dtype=np.float32)
            ),
        )
        return path
    except Exception:
        return None
