"""Qt-free post-tracking session service (Slice 2: analysis chain)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from hydra_suite.core.canonicalization.geometry import canonical_geometry_from_params
from hydra_suite.core.individual.dataset.oriented_video import (
    resolve_individual_dataset_dir,
    resolve_oriented_track_video_dir,
)
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
from hydra_suite.core.post.trajectory_writer import (
    user_tracks_path,
    write_base_final_csv,
)
from hydra_suite.core.tracking.errors import TrackingSessionError
from hydra_suite.core.tracking.profiler import TrackingProfiler
from hydra_suite.core.tracking.session_policy import (
    should_export_final_canonical_images,
    should_export_final_media_videos,
    should_run_interpolated_postpass,
)
from hydra_suite.core.tracking.session_summary import build_session_summary_lines
from hydra_suite.utils import profiling_names as N
from hydra_suite.utils.profiling import span
from hydra_suite.utils.video_artifacts import build_inference_cache_dir

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


def _user_mode_intermediate_paths(base: str, ext: str) -> list[str]:
    """Intermediate CSVs to delete after a User-mode run (never the clean tracks.csv).

    Each stem also contributes its rich-export siblings. The rich CSV is
    written in User mode too -- it is the only artifact carrying the identity
    and ``PoseKpt_*`` columns the annotated-video exporter reads -- so every
    stem that can be a rich export's base must have that sibling cleaned up.
    Hardcoding only ``_final_with_individual`` leaked
    ``_forward_processed_with_individual.csv``; the legacy ``_with_pose``
    alias is swept for the same reason.
    """
    from hydra_suite.core.post.rich_export import (
        LEGACY_RICH_EXPORT_SUFFIX,
        RICH_EXPORT_SUFFIX,
    )

    stems = [
        "_final",
        "_forward",
        "_backward",
        "_forward_processed",
        "_tracking_forward",
        "_tracking_backward",
        "_tracking_final",
    ]
    paths = []
    for stem in stems:
        paths.append(f"{base}{stem}{ext}")
        for suffix in (RICH_EXPORT_SUFFIX, LEGACY_RICH_EXPORT_SUFFIX):
            paths.append(f"{base}{stem}{suffix}{ext}")
    return paths


def _save_trajectories_to_csv(trajectories, output_path: str) -> bool:
    """Persist post-processed trajectories (delegates to the shared base-final writer)."""
    return write_base_final_csv(trajectories, output_path)


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
            inference_cache_dir=str(build_inference_cache_dir(video_path)),
        )

        # Set by _run_interp_crops; consumed by _run_final_media_export when the
        # caller did not supply an explicit "interpolated_roi_npz_path".
        self._interpolated_roi_npz_path = None

        # Session-scoped span profiler. The merge / interpolated_crops
        # profilers nested below defer to this one (equal priority), so their
        # subtrees stay in the session tree instead of being split out.
        self._profiler = TrackingProfiler(
            enabled=bool(self.params.get("ENABLE_PROFILING", False))
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
        with span(N.TRAJECTORY_POSTPROC):
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

    def _identity_ran(self) -> bool:
        """Whether an identity/tag method actually ran (not just enabled-with-none)."""
        _method = (
            str(self.config.get("identity_method", "none_disabled")).strip().lower()
        )
        return bool(self.config.get("enable_identity_analysis")) and _method not in (
            "none_disabled",
            "none",
            "",
        )

    def _export_rich(self, final_csv):
        return export_rich_csv(
            final_csv,
            self.pose_state,
            params=self.params,
            min_valid_conf=float(self.params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
            ignore_keypoints=self.config.get("pose_ignore_keypoints"),
            identity_evidence_cache_path=self._identity_evidence_cache_path(),
            debug_mode=bool(self.params.get("DEBUG_MODE", True)),
            fps=self.params.get("FPS"),
            identity_ran=self._identity_ran(),
        )

    def _relink_export_rich(self, final_csv):
        return relink_and_export_rich_csv(
            final_csv,
            self.pose_state,
            params=self.params,
            min_valid_conf=float(self.params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
            ignore_keypoints=self.config.get("pose_ignore_keypoints"),
            identity_evidence_cache_path=self._identity_evidence_cache_path(),
            debug_mode=bool(self.params.get("DEBUG_MODE", True)),
            fps=self.params.get("FPS"),
            identity_ran=self._identity_ran(),
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

        # The interpolated ROI npz feeds the final-media exporter so occluded /
        # interpolated frames still get crops. It used to be captured by the GUI
        # (`current_interpolated_roi_npz_path`); after the Slice-5 cutover
        # nothing read it out of the payload at all.
        roi_npz = str(payload.get("roi_npz_path") or "").strip()
        if roi_npz:
            self._interpolated_roi_npz_path = roi_npz

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

        from hydra_suite.data.al.escalation import achievable_levels
        from hydra_suite.data.dataset_generation import resolve_native_level
        from hydra_suite.utils.geometry_levels import GeometryLevel

        stored_levels = [
            GeometryLevel.from_str(name)
            for name in self.params.get(
                "DATASET_EXPORT_LEVELS", ["polygon", "obb", "aabb"]
            )
        ]
        # An empty stored list is a deliberate "export nothing" choice (every
        # level checkbox unchecked in the GUI) -- honor it rather than
        # silently exporting everything.
        if not stored_levels:
            return {
                "success": False,
                "error": (
                    "No label levels were selected. Enable at least one "
                    "export level (polygon/obb/aabb) to generate a dataset."
                ),
            }
        # A stored level preference the current detector cannot achieve (e.g.
        # "polygon" saved against a since-switched OBB model) is clamped down
        # to what is achievable rather than raised -- a stale stored
        # preference must never break a tracking run. If clamping a
        # NON-EMPTY stored list leaves nothing achievable, fall back to all
        # achievable levels (the stored preference is stale, not a
        # deliberate "export nothing").
        allowed = set(achievable_levels(resolve_native_level(self.params)))
        levels = [lvl for lvl in stored_levels if lvl in allowed] or sorted(
            allowed, reverse=True
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
            export_levels=levels,
            class_names=self.params.get("DATASET_CLASS_NAMES", [class_name]),
            dedup_method=self.params.get("DATASET_DEDUP_METHOD", "phash"),
            dedup_threshold=int(self.params.get("DATASET_DEDUP_THRESHOLD", 8)),
        )

    def _resolve_image_root(self):
        """Per-run directory for final canonical stills.

        Resolved from ``self.params`` -- which every caller already builds via
        ``build_engine_params`` -- so the GUI and the CLI need no extra wiring.
        An explicit ``paths["individual_dataset_dir"]`` still wins.

        Before the Slice-5 cutover the GUI computed this in
        ``main_window._resolve_current_individual_dataset_dir()``; the cutover
        dropped the call and left the ``paths`` key unwritten by every caller,
        so the export silently skipped on every run.
        """
        explicit = self.paths.get("individual_dataset_dir")
        if explicit:
            return Path(explicit).expanduser()
        resolved = resolve_individual_dataset_dir(
            self.params.get("INDIVIDUAL_DATASET_OUTPUT_DIR"),
            self.params.get("INDIVIDUAL_DATASET_NAME"),
            self.params.get("INDIVIDUAL_DATASET_RUN_ID"),
        )
        return Path(resolved).expanduser() if resolved else None

    def _resolve_video_root(self):
        """Per-run directory for orientation-fixed per-track videos.

        Mirrors ``_resolve_image_root``; replaces the dropped
        ``main_window._resolve_current_final_media_video_dir()``.
        """
        explicit = self.paths.get("final_media_video_dir")
        if explicit:
            return Path(explicit).expanduser()
        resolved = resolve_oriented_track_video_dir(
            self.params.get("FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR")
            or self.params.get("ORIENTED_TRACK_VIDEO_OUTPUT_DIR"),
            self.params.get("INDIVIDUAL_DATASET_RUN_ID"),
        )
        return Path(resolved).expanduser() if resolved else None

    def _resolve_source_fps(self):
        """Source-video FPS for the exported per-track videos.

        ``paths["source_video_fps"]`` is never written by any caller, and the
        exporter turns a ``None`` into ``max(0.1, 0.0)`` -- i.e. 0.1 FPS videos.
        Fall back to the ``FPS`` params key the engine builder always emits.
        """
        explicit = self.paths.get("source_video_fps")
        if explicit:
            return float(explicit)
        try:
            fps = float(self.params.get("FPS") or 0.0)
        except (TypeError, ValueError):
            fps = 0.0
        return fps if fps > 0 else None

    def _run_final_media_export(self, final_csv_path):
        """Export canonical stills / oriented per-track videos; return written media paths."""
        if self.callbacks.should_stop():
            return []
        export_images = should_export_final_canonical_images(self.config)
        export_videos = should_export_final_media_videos(self.config)
        if not export_images and not export_videos:
            return []
        image_root = self._resolve_image_root() if export_images else None
        video_root = self._resolve_video_root() if export_videos else None

        self.callbacks.stage_changed("final_media_export")
        result = media_export.export_final_media(
            final_csv_path=final_csv_path,
            config=self.config,
            video_path=self.video_path,
            detection_cache_path=self.paths.get("detection_cache_path"),
            interpolated_roi_npz_path=(
                self.paths.get("interpolated_roi_npz_path")
                or self._interpolated_roi_npz_path
            ),
            fps=self._resolve_source_fps(),
            image_root=image_root,
            video_root=video_root,
            export_images=export_images,
            export_videos=export_videos,
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
            with self._profiler.armed(), span(N.SESSION):
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
                with span(N.POSTPROCESS):
                    forward_processed = self._postprocess_csv(forward_csv)

                if cb.should_stop():
                    return self._stopped_result()

                if backward_enabled:
                    cb.stage_changed("backward_postprocess")
                    with span(N.BACKWARD_POSTPROCESS):
                        backward_processed = self._postprocess_csv(
                            f"{base}_backward{ext}"
                        )
                    cb.stage_changed("merge")
                    with span(N.MERGE):
                        final_df = self._merge(forward_processed, backward_processed)
                    final_csv = f"{base}_final{ext}"
                else:
                    with span(N.INTERPOLATE_AND_SCALE):
                        final_df = self._interpolate_and_scale(forward_processed)
                    final_csv = f"{base}_forward_processed{ext}"

                if final_df is None or cb.should_stop():
                    return self._stopped_result()
                with span(N.WRITE):
                    _save_trajectories_to_csv(final_df, final_csv)

                cb.stage_changed("rich_export")
                with span(N.RICH_EXPORT):
                    rich_path = self._export_rich(final_csv)

                if (
                    should_run_interpolated_postpass(self.config)
                    and not cb.should_stop()
                ):
                    cb.stage_changed("interpolated_crops")
                    with span(N.INTERP_CROPS):
                        self._run_interp_crops(final_csv)
                    with span(N.RELINK):
                        rich_path = self._relink_export_rich(final_csv) or rich_path

                # --- Slice 3 export chain ---
                dataset_result = None
                media_paths: list[str] = []
                if not self.callbacks.should_stop():
                    with span(N.DATASET_GENERATION):
                        dataset_result = self._run_dataset_generation(final_csv)
                if not self.callbacks.should_stop():
                    with span(N.MEDIA_EXPORT):
                        media_paths.extend(self._run_final_media_export(final_csv))
                if not self.callbacks.should_stop():
                    with span(N.ANNOTATED_VIDEO):
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

                # User mode: intermediates are no longer needed once dataset
                # generation, media export, and the annotated video (all of
                # which read the base-final CSV) have finished -- clean them
                # up so only the clean tracks.csv + annotated video remain.
                # NO-OP in debug mode (and thus a no-op for the equivalence
                # gate).
                if not bool(self.params.get("DEBUG_MODE", True)):
                    _expected_tracks_csv = user_tracks_path(final_csv)
                    if os.path.exists(_expected_tracks_csv):
                        for _p in _user_mode_intermediate_paths(base, ext):
                            try:
                                if os.path.exists(_p):
                                    os.remove(_p)
                            except OSError:
                                logger.warning(
                                    "Failed to remove intermediate %s",
                                    _p,
                                    exc_info=True,
                                )
                    else:
                        logger.warning(
                            "User-mode cleanup skipped: expected clean output %s "
                            "was not found. Keeping intermediates as a fallback.",
                            _expected_tracks_csv,
                        )

                cb.stage_changed("done")

            self._profiler.end_frame()
            self._profiler.log_final_summary()
            # Wire the HYDRA_PROFILE dump location -- without this call
            # `profiling_process.set_log_dir` is dead code and the spec's
            # "<video>_logs/ when a session supplies one" clause never
            # holds.
            from hydra_suite.utils.profiling_process import set_log_dir
            from hydra_suite.utils.video_artifacts import build_video_log_dir

            set_log_dir(build_video_log_dir(self.video_path, create=True))
            self._profiler.export_summary(
                build_video_log_dir(self.video_path, create=True)
                / "tracking_profile_session.json"
            )

            return result
        except TrackingSessionError as e:
            return SessionResult(False, None, None, [], None, [], str(e))
