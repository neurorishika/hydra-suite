"""MergeWorker — trajectory merge and CSV export background worker."""

import csv
import logging

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal

from hydra_suite.core.post.processing import (
    interpolate_trajectories,
    resolve_trajectories,
)
from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


class MergeWorker(BaseWorker):
    """Worker thread for merging trajectories without blocking the UI."""

    progress_signal = Signal(int, str)  # progress value, status message
    finished_signal = Signal(object)  # merged trajectories
    error_signal = Signal(str)  # error message

    def __init__(
        self,
        forward_trajs,
        backward_trajs,
        total_frames,
        params,
        resize_factor,
        interp_method,
        max_gap,
        tag_cache_path=None,
        heading_flip_max_burst=5,
        directed_heading_posthoc=False,
        enable_profiling=False,
        profile_export_path=None,
    ):
        super().__init__()
        self.forward_trajs = forward_trajs
        self.backward_trajs = backward_trajs
        self.total_frames = total_frames
        self.params = params
        self.resize_factor = resize_factor
        self.interp_method = interp_method
        self.max_gap = max_gap
        self.tag_cache_path = tag_cache_path
        self.heading_flip_max_burst = heading_flip_max_burst
        self.directed_heading_posthoc = directed_heading_posthoc
        self.enable_profiling = enable_profiling
        self.profile_export_path = profile_export_path
        self._stop_requested = False

    def stop(self):
        """Request cooperative cancellation."""
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    @staticmethod
    def _convert_resolved_to_dataframe(resolved_trajectories):
        """Convert a list of resolved trajectories to a single DataFrame."""
        if not resolved_trajectories or not isinstance(resolved_trajectories, list):
            return resolved_trajectories
        if isinstance(resolved_trajectories[0], pd.DataFrame):
            for new_id, traj_df in enumerate(resolved_trajectories):
                traj_df["TrajectoryID"] = new_id
            return pd.concat(resolved_trajectories, ignore_index=True)
        # Fallback for old tuple format
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

    def _resolve_tag_identities(self, resolved_trajectories):
        """Apply AprilTag identity resolution if a tag cache is available."""
        if (
            not isinstance(resolved_trajectories, pd.DataFrame)
            or self.tag_cache_path is None
        ):
            return resolved_trajectories
        try:
            from hydra_suite.core.post.tag_identity import (
                detect_tag_swaps,
                resolve_tag_identities,
            )
            from hydra_suite.data.tag_observation_cache import TagObservationCache

            self.progress_signal.emit(92, "Resolving tag identities...")
            tag_cache = TagObservationCache(str(self.tag_cache_path), mode="r")
            resolved_trajectories = resolve_tag_identities(
                resolved_trajectories, tag_cache, self.params
            )
            swaps = detect_tag_swaps(resolved_trajectories, tag_cache, self.params)
            if swaps:
                logger.warning("Detected %d potential tag-swap events", len(swaps))
            tag_cache.close()
        except Exception:
            logger.warning(
                "Tag identity resolution failed (non-fatal)",
                exc_info=True,
            )
        return resolved_trajectories

    def _rescale_coordinates(self, resolved_trajectories):
        """Scale coordinates back to original video space."""
        if not isinstance(resolved_trajectories, pd.DataFrame):
            return resolved_trajectories
        logger.info(
            f"Pre-scaling (resize_factor={self.resize_factor:.3f}): "
            f"X range [{resolved_trajectories['X'].min():.1f}, {resolved_trajectories['X'].max():.1f}], "
            f"Y range [{resolved_trajectories['Y'].min():.1f}, {resolved_trajectories['Y'].max():.1f}]"
        )
        resolved_trajectories[["X", "Y"]] = (
            resolved_trajectories[["X", "Y"]] / self.resize_factor
        )
        if "Width" in resolved_trajectories.columns:
            resolved_trajectories["Width"] /= self.resize_factor
        if "Height" in resolved_trajectories.columns:
            resolved_trajectories["Height"] /= self.resize_factor
        logger.info(
            f"Post-scaling: "
            f"X range [{resolved_trajectories['X'].min():.1f}, {resolved_trajectories['X'].max():.1f}], "
            f"Y range [{resolved_trajectories['Y'].min():.1f}, {resolved_trajectories['Y'].max():.1f}]"
        )
        return resolved_trajectories

    def execute(self):
        """Merge forward and backward trajectories."""
        from hydra_suite.core.tracking.profiler import TrackingProfiler

        profiler = TrackingProfiler(enabled=self.enable_profiling)
        try:
            if self._should_stop():
                return
            profiler.phase_start("post_prepare")
            self.progress_signal.emit(10, "Preparing trajectories...")

            def prepare_trajs_for_merge(trajs):
                if isinstance(trajs, pd.DataFrame):
                    return [group for _, group in trajs.groupby("TrajectoryID")]
                return trajs

            forward_prepared = prepare_trajs_for_merge(self.forward_trajs)
            backward_prepared = prepare_trajs_for_merge(self.backward_trajs)
            profiler.phase_end("post_prepare")

            if self._should_stop():
                return
            profiler.phase_start("post_resolve")
            self.progress_signal.emit(30, "Resolving trajectory conflicts...")

            resolved_trajectories = resolve_trajectories(
                forward_prepared,
                backward_prepared,
                params=self.params,
            )
            profiler.phase_end("post_resolve")

            if self._should_stop():
                return
            self.progress_signal.emit(60, "Converting to DataFrame...")

            resolved_trajectories = self._convert_resolved_to_dataframe(
                resolved_trajectories
            )

            profiler.phase_start("post_interpolate")
            self.progress_signal.emit(75, "Applying interpolation...")
            if isinstance(resolved_trajectories, pd.DataFrame):
                if self.interp_method != "none":
                    resolved_trajectories = interpolate_trajectories(
                        resolved_trajectories,
                        method=self.interp_method,
                        max_gap=self.max_gap,
                        heading_flip_max_burst=self.heading_flip_max_burst,
                        directed_heading_posthoc=self.directed_heading_posthoc,
                    )
            profiler.phase_end("post_interpolate")

            if self._should_stop():
                return
            self.progress_signal.emit(90, "Scaling to original space...")

            profiler.phase_start("post_tag_identity")
            resolved_trajectories = self._resolve_tag_identities(resolved_trajectories)
            profiler.phase_end("post_tag_identity")

            profiler.phase_start("post_rescale")
            resolved_trajectories = self._rescale_coordinates(resolved_trajectories)
            profiler.phase_end("post_rescale")

            if not self._should_stop():
                profiler.log_final_summary()
                if self.profile_export_path:
                    profiler.export_summary(self.profile_export_path)
                self.progress_signal.emit(100, "Merge complete!")
                self.finished_signal.emit(resolved_trajectories)

        except Exception as e:
            logger.exception("Error during trajectory merging")
            self.error_signal.emit(str(e))


def _write_csv_artifact(path, fieldnames, rows):
    """Write a CSV artifact file. Returns the path on success, None on failure."""
    try:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path
    except Exception:
        return None


def _write_roi_npz(path, roi_rows, roi_corners):
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
