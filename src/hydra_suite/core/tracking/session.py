"""Qt-free post-tracking session service (Slice 2: analysis chain)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from hydra_suite.core.canonicalization.geometry import canonical_geometry_from_params
from hydra_suite.core.post import dataset_export, media_export
from hydra_suite.core.post.interpolated_crops import run_interpolated_crops
from hydra_suite.core.post.merge import merge_trajectories, rescale_coordinates
from hydra_suite.core.post.pose_merge import (
    PoseSourceState,
    resolve_current_tag_cache_path,
)
from hydra_suite.core.post.processing import (
    interpolate_trajectories,
    process_trajectories_from_csv,
)
from hydra_suite.core.post.rich_export import (
    export_rich_csv,
    relink_and_export_rich_csv,
)
from hydra_suite.core.tracking.errors import TrackingSessionError
from hydra_suite.core.tracking.session_policy import (
    should_export_final_canonical_images,
    should_export_final_media_videos,
    should_run_interpolated_postpass,
)
from hydra_suite.core.tracking.session_summary import build_session_summary_lines

logger = logging.getLogger(__name__)


def csv_has_data_rows(csv_path) -> bool:
    """Return True if *csv_path* has at least one data row beyond the header.

    O(1) in the file size -- reads only the header and the first data line.
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            fh.readline()  # header
            return bool(fh.readline().strip())
    except OSError:
        return False


def detection_cache_has_detections(detection_cache_path) -> bool:
    """Return True if the detection cache recorded at least one detection.

    Reads the raw ``.npz`` directly (no full ``DetectionCache`` instantiation /
    validation) and checks whether any per-frame ``frame_<N>_meas`` array is
    non-empty. Returns False on any read error or a missing cache, so the
    empty-output guard only fires when detections are *positively confirmed* --
    it can never false-positive on a legitimately empty clip.
    """
    try:
        with np.load(str(detection_cache_path), allow_pickle=True) as data:
            for key in data.files:
                if key.startswith("frame_") and key.endswith("_meas"):
                    arr = data[key]
                    if arr is not None and getattr(arr, "shape", (0,))[0] > 0:
                        return True
    except Exception:
        return False
    return False


def enforce_nonempty_forward(raw_csv_path, detection_cache_path) -> None:
    """Fail loud if the forward pass emitted an empty CSV despite detections.

    Invariant: if the detection cache contains detections, a *successful* forward
    tracking pass MUST produce at least one tracked row. A completed run that
    wrote a header-only CSV is a silent pipeline failure (e.g. a crashed
    pose/identity stage or an OOM that still returned exit 0). Raising here turns
    that into a loud abort instead of a valid-looking empty CSV that "falsely
    passes" downstream comparison/benchmark tooling.
    """
    if detection_cache_has_detections(detection_cache_path) and not csv_has_data_rows(
        raw_csv_path
    ):
        raise TrackingSessionError(
            "Forward tracking produced ZERO tracked rows even though the detection "
            "cache contains detections. This indicates a silent pipeline failure "
            "(e.g. a crashed pose/identity stage or an out-of-memory abort). "
            "Refusing to emit an empty tracking CSV. "
            f"csv={raw_csv_path} detection_cache={detection_cache_path}"
        )


def _save_trajectories_to_csv(trajectories, output_path: str) -> bool:
    """Persist post-processed trajectories in the same shape as the GUI path.

    Copied verbatim from ``trackerkit/headless_tracking.py::save_trajectories_to_csv``
    so ``core/`` needs no app-layer import.
    """
    if trajectories is None:
        return False
    if not isinstance(trajectories, pd.DataFrame):
        raise TypeError("Expected post-processed trajectories as a pandas DataFrame.")
    if trajectories.empty:
        return False

    df_to_save = trajectories.copy()
    for column in ["X", "Y", "FrameID"]:
        if column in df_to_save.columns:
            df_to_save[column] = pd.to_numeric(df_to_save[column], errors="coerce")
            df_to_save[column] = df_to_save[column].round().astype("Int64")

    df_to_save = df_to_save.drop(
        columns=[
            column for column in ["TrackID", "Index"] if column in df_to_save.columns
        ],
        errors="ignore",
    )
    base_columns = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
    ordered_columns = base_columns + [
        column for column in df_to_save.columns if column not in base_columns
    ]
    df_to_save[ordered_columns].to_csv(output_path, index=False)
    return True


def _noop1(_a) -> None:
    return None


def _noop2(_a, _b) -> None:
    return None


def _never() -> bool:
    return False


@dataclass
class SessionCallbacks:
    progress: Callable[[int, str], None] = _noop2
    status: Callable[[str], None] = _noop1
    warning: Callable[[str, str], None] = _noop2
    stage_changed: Callable[[str], None] = _noop1
    should_stop: Callable[[], bool] = _never


@dataclass
class SessionResult:
    success: bool
    final_csv_path: str | None
    rich_export_path: str | None
    media_paths: list[str]
    dataset_result: dict | None
    summary_lines: list[str]
    error: str | None


class TrackingSessionCore:
    """Owns the post-tracking analysis chain, Qt-free."""

    def __init__(self, *, video_path, config, params, paths, callbacks=None):
        self.video_path = video_path
        self.config = config
        self.params = params
        self.paths = paths
        self.callbacks = callbacks if callbacks is not None else SessionCallbacks()
        self.pose_state = PoseSourceState(
            detection_cache_path=self.paths.get("detection_cache_path"),
            individual_properties_cache_path=self.paths.get(
                "individual_properties_cache_path"
            ),
            detected_properties_cache_path=self.paths.get(
                "detected_properties_cache_path"
            ),
        )

    def _stopped_result(self) -> SessionResult:
        return SessionResult(False, None, None, [], None, [], None)

    def _postprocess_csv(self, csv_path):
        if not self.config.get("enable_postprocessing"):
            effective_params = dict(self.params)
            effective_params["MIN_TRAJECTORY_LENGTH"] = 1
            effective_params["MAX_VELOCITY_BREAK"] = float("inf")
            effective_params["MAX_OCCLUSION_GAP"] = 0
            effective_params["MAX_VELOCITY_ZSCORE"] = 0.0
        else:
            effective_params = self.params
        processed, _ = process_trajectories_from_csv(
            csv_path, effective_params, should_stop=self.callbacks.should_stop
        )
        return processed

    def _interpolate_and_scale(self, df):
        interp_method = str(self.config.get("interpolation_method", "none")).lower()
        if interp_method != "none":
            max_gap = max(
                1,
                round(
                    float(self.config["interpolation_max_gap_seconds"])
                    * float(self.params["FPS"])
                ),
            )
            df = interpolate_trajectories(
                df,
                method=interp_method,
                max_gap=max_gap,
                heading_flip_max_burst=int(self.config["heading_flip_max_burst"]),
                directed_heading_posthoc=bool(
                    self.params.get("DIRECTED_ORIENT_POSTHOC_CONSISTENCY", False)
                ),
            )
        return rescale_coordinates(
            df, resize_factor=float(self.params.get("RESIZE_FACTOR", 1.0))
        )

    def _merge(self, forward, backward):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            cap.release()
            raise TrackingSessionError(
                f"Cannot open video for merge: {self.video_path}"
            )
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return merge_trajectories(
            forward,
            backward,
            total_frames=total_frames,
            params=self.params,
            resize_factor=float(self.params.get("RESIZE_FACTOR", 1.0)),
            interp_method=str(self.config.get("interpolation_method", "none")).lower(),
            max_gap=max(
                1,
                round(
                    float(self.config["interpolation_max_gap_seconds"])
                    * float(self.params["FPS"])
                ),
            ),
            tag_cache_path=resolve_current_tag_cache_path(
                self.params, self.paths.get("detection_cache_path")
            ),
            heading_flip_max_burst=int(self.config["heading_flip_max_burst"]),
            directed_heading_posthoc=bool(
                self.params.get("DIRECTED_ORIENT_POSTHOC_CONSISTENCY", False)
            ),
            enable_profiling=bool(self.params.get("ENABLE_PROFILING", False)),
            progress=self.callbacks.progress,
            should_stop=self.callbacks.should_stop,
        )

    def _identity_evidence_cache_path(self):
        """Resolve the Phase-3 evidence sidecar path for this run's video, or
        ``None`` (Identity Phase 5: threads the cache path into the offline
        fragment solver without needing the tracking worker's live
        ``InferenceRunner`` instance -- see ``find_identity_evidence_cache_path``).
        """
        if not self.video_path:
            return None
        try:
            from hydra_suite.core.individual.identity.cache import (
                find_identity_evidence_cache_path,
            )

            path = find_identity_evidence_cache_path(self.video_path)
            return str(path) if path is not None else None
        except Exception:
            logger.warning(
                "Failed to resolve identity evidence cache path for %s",
                self.video_path,
                exc_info=True,
            )
            return None

    def _export_rich(self, final_csv):
        return export_rich_csv(
            final_csv,
            self.pose_state,
            params=self.params,
            min_valid_conf=float(self.params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
            ignore_keypoints=self.config.get("pose_ignore_keypoints"),
            identity_evidence_cache_path=self._identity_evidence_cache_path(),
        )

    def _relink_export_rich(self, final_csv):
        return relink_and_export_rich_csv(
            final_csv,
            self.pose_state,
            params=self.params,
            min_valid_conf=float(self.params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
            ignore_keypoints=self.config.get("pose_ignore_keypoints"),
            identity_evidence_cache_path=self._identity_evidence_cache_path(),
        )

    def _run_interp_crops(self, final_csv) -> None:
        from hydra_suite.core.tracking import session_policy

        if not session_policy.should_run_interpolated_postpass(self.config):
            return

        payload = run_interpolated_crops(
            final_csv,
            self.video_path,
            self.pose_state.detection_cache_path,
            self.params,
            enable_profiling=bool(self.params.get("ENABLE_PROFILING", False)),
            progress=self.callbacks.progress,
            should_stop=self.callbacks.should_stop,
        )

        pose_csv = payload.get("pose_csv_path")
        pose_rows = payload.get("pose_rows")
        if pose_csv:
            self.pose_state.interpolated_pose_csv_path = pose_csv
            self.pose_state.interpolated_pose_df = None
        elif pose_rows:
            try:
                self.pose_state.interpolated_pose_df = pd.DataFrame(pose_rows)
                self.pose_state.interpolated_pose_csv_path = None
            except Exception:
                self.pose_state.interpolated_pose_df = None

        tag_csv = payload.get("tag_csv_path")
        tag_rows = payload.get("tag_rows")
        if tag_csv:
            self.pose_state.interpolated_tag_csv_path = tag_csv
            self.pose_state.interpolated_tag_df = None
        elif tag_rows:
            try:
                self.pose_state.interpolated_tag_df = pd.DataFrame(tag_rows)
                self.pose_state.interpolated_tag_csv_path = None
            except Exception:
                self.pose_state.interpolated_tag_df = None

        cnn_csv_paths = payload.get("cnn_csv_paths")
        cnn_rows = payload.get("cnn_rows")
        if cnn_csv_paths:
            self.pose_state.interpolated_cnn_csv_paths = cnn_csv_paths
            self.pose_state.interpolated_cnn_dfs = None
        elif cnn_rows:
            try:
                self.pose_state.interpolated_cnn_dfs = {
                    label: pd.DataFrame(rows)
                    for label, rows in cnn_rows.items()
                    if rows
                }
                self.pose_state.interpolated_cnn_csv_paths = {}
            except Exception:
                self.pose_state.interpolated_cnn_dfs = None

        headtail_csv = payload.get("headtail_csv_path")
        headtail_rows = payload.get("headtail_rows")
        if headtail_csv:
            self.pose_state.interpolated_headtail_csv_path = headtail_csv
            self.pose_state.interpolated_headtail_df = None
        elif headtail_rows:
            try:
                self.pose_state.interpolated_headtail_df = pd.DataFrame(headtail_rows)
                self.pose_state.interpolated_headtail_csv_path = None
            except Exception:
                self.pose_state.interpolated_headtail_df = None

    def _run_dataset_generation(self, final_csv_path):
        """Generate an active-learning dataset inline; return its result dict or None."""
        if not self.config.get("enable_dataset_generation", False):
            return None
        if self.callbacks.should_stop():
            return None
        video_path = self.video_path
        if not video_path or not os.path.exists(video_path):
            self.callbacks.warning(
                "Dataset Generation Error", "Source video file not found."
            )
            return {"success": False, "error": "Source video file not found."}
        csv_path = final_csv_path
        if not csv_path or not os.path.exists(csv_path):
            self.callbacks.warning(
                "Dataset Generation Error", "Tracking CSV file not found."
            )
            return {"success": False, "error": "Tracking CSV file not found."}

        output_dir = os.path.join(
            os.path.dirname(video_path),
            f"{os.path.splitext(os.path.basename(video_path))[0]}_datasets",
            "active_learning",
        )
        os.makedirs(output_dir, exist_ok=True)
        class_name = (
            str(self.config.get("dataset_class_name", "") or "").strip() or "object"
        )

        self.callbacks.stage_changed("dataset_generation")
        return dataset_export.generate_active_learning_dataset(
            video_path=video_path,
            csv_path=csv_path,
            detection_cache_path=self.paths.get("detection_cache_path"),
            output_dir=output_dir,
            dataset_name="",
            class_name=class_name,
            params=self.params,
            max_frames=int(self.config.get("dataset_max_frames", 100)),
            diversity_window=int(self.config.get("dataset_diversity_window", 30)),
            include_context=bool(self.config.get("dataset_include_context", True)),
            probabilistic=bool(self.config.get("dataset_probabilistic_sampling", True)),
            progress=self.callbacks.progress,
            should_stop=self.callbacks.should_stop,
        )

    def _run_final_media_export(self, final_csv_path):
        """Export canonical stills / oriented per-track videos; return written media paths."""
        if self.callbacks.should_stop():
            return []
        export_images = should_export_final_canonical_images(self.config)
        export_videos = should_export_final_media_videos(self.config)
        if not export_images and not export_videos:
            return []
        image_root = self.paths.get("individual_dataset_dir") if export_images else None
        video_root = self.paths.get("final_media_video_dir") if export_videos else None

        self.callbacks.stage_changed("final_media_export")
        result = media_export.export_final_media(
            final_csv_path=final_csv_path,
            config=self.config,
            video_path=self.video_path,
            detection_cache_path=self.paths.get("detection_cache_path"),
            interpolated_roi_npz_path=self.paths.get("interpolated_roi_npz_path"),
            fps=self.paths.get("source_video_fps"),
            image_root=image_root,
            video_root=video_root,
            export_images=export_images,
            export_videos=export_videos,
            padding_fraction=float(self.config.get("individual_crop_padding", 0.1)),
            background_color=tuple(
                self.config.get("individual_background_color", [0, 0, 0])
            ),
            # The session's own project-wide Layer 1 canvas -- without this the
            # exporter falls back to a DEFAULT geometry that silently diverges
            # from the canvas every other stage of this session used.
            geometry=canonical_geometry_from_params(self.params),
            progress=self.callbacks.progress,
            should_stop=self.callbacks.should_stop,
        )
        if not result:
            return []
        media_paths = []
        for key in ("output_dir", "image_output_dir"):
            val = str(result.get(key, "")).strip()
            if val:
                media_paths.append(val)
        return media_paths

    def _run_annotated_video(self, final_csv_path):
        """Render the annotated overlay video; return its path or None."""
        if self.callbacks.should_stop():
            return None
        output_path = str(self.config.get("video_output_path", "") or "").strip()
        if not (self.config.get("video_output_enabled", False) and output_path):
            return None
        trajectories_df, loaded_path = media_export.load_video_trajectories(
            final_csv_path
        )
        if trajectories_df is None or trajectories_df.empty:
            logger.warning(
                "Skipping final video generation: no trajectories loaded from %s",
                final_csv_path,
            )
            return None
        self.callbacks.stage_changed("annotated_video")
        return media_export.render_annotated_video(
            trajectories_df=trajectories_df,
            video_path=self.video_path,
            output_path=output_path,
            params=self.params,
            config=self.config,
            progress=self.callbacks.progress,
            should_stop=self.callbacks.should_stop,
        )

    def run_post_tracking(
        self, forward_trajectories, backward_trajectories=None
    ) -> SessionResult:
        cb = self.callbacks
        try:
            if cb.should_stop():
                return self._stopped_result()

            raw_csv = self.paths.get("raw_csv_path")
            detection_cache = self.paths.get("detection_cache_path", "")
            base, ext = os.path.splitext(raw_csv) if raw_csv else ("", ".csv")
            backward_enabled = bool(self.config.get("enable_backward_tracking"))

            cb.stage_changed("postprocess")
            if raw_csv and detection_cache:
                enforce_nonempty_forward(
                    (f"{base}_forward{ext}" if backward_enabled else raw_csv),
                    detection_cache,
                )
            forward_csv = f"{base}_forward{ext}" if backward_enabled else raw_csv
            forward_processed = self._postprocess_csv(forward_csv)

            if cb.should_stop():
                return self._stopped_result()

            if backward_enabled:
                cb.stage_changed("backward_postprocess")
                backward_processed = self._postprocess_csv(f"{base}_backward{ext}")
                cb.stage_changed("merge")
                final_df = self._merge(forward_processed, backward_processed)
                final_csv = f"{base}_final{ext}"
            else:
                final_df = self._interpolate_and_scale(forward_processed)
                final_csv = f"{base}_forward_processed{ext}"

            if final_df is None or cb.should_stop():
                return self._stopped_result()
            _save_trajectories_to_csv(final_df, final_csv)

            cb.stage_changed("rich_export")
            rich_path = self._export_rich(final_csv)

            if should_run_interpolated_postpass(self.config) and not cb.should_stop():
                cb.stage_changed("interpolated_crops")
                self._run_interp_crops(final_csv)
                rich_path = self._relink_export_rich(final_csv) or rich_path

            # --- Slice 3 export chain ---
            dataset_result = None
            media_paths: list[str] = []
            if not self.callbacks.should_stop():
                dataset_result = self._run_dataset_generation(final_csv)
            if not self.callbacks.should_stop():
                media_paths.extend(self._run_final_media_export(final_csv))
            if not self.callbacks.should_stop():
                annotated = self._run_annotated_video(final_csv)
                if annotated:
                    media_paths.append(annotated)

            result = SessionResult(
                success=True,
                final_csv_path=final_csv,
                rich_export_path=rich_path,
                media_paths=media_paths,
                dataset_result=dataset_result,
                summary_lines=[],
                error=None,
            )
            try:
                traj_count = int(
                    pd.read_csv(final_csv, usecols=["TrajectoryID"])[
                        "TrajectoryID"
                    ].nunique()
                )
            except Exception:
                traj_count = None
            summary_result = {
                "video_path": self.video_path,
                "csv_path": final_csv,
                "trajectory_count": traj_count,
                "dataset": dataset_result,
            }
            result.summary_lines = build_session_summary_lines(
                self.config, summary_result
            )
            cb.stage_changed("done")
            return result
        except TrackingSessionError as e:
            return SessionResult(False, None, None, [], None, [], str(e))
