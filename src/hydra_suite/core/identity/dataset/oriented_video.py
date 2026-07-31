"""Orientation-fixed per-track video export for MAT individual outputs."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import pandas as pd

# Correction 20 / Task 17d: rewire to new DetectionCache.
# The new DetectionCacheHandle has a different API (read_frame returns OBBResult).
# We provide a thin compatibility wrapper so the existing _add_actual_tasks code
# needs only one targeted update (the get_frame tuple-unpack → OBBResult access).
try:
    from hydra_suite.core.inference.cache.base import CacheKey as _CacheKey
    from hydra_suite.core.inference.cache.store import (
        DetectionCacheHandle as _NewDetCache,
    )

    class DetectionCache:  # type: ignore[no-redef]
        """Compatibility shim: wraps DetectionCacheHandle to expose legacy-compat API."""

        def __init__(self, path, mode="r"):  # noqa: ARG002
            # CacheKey with empty values — is_valid() checks path existence only.
            _key = _CacheKey(
                schema_version=0,
                model_path="",
                model_mtime=0.0,
                config_hash="",
            )
            self._handle = _NewDetCache(path=path, key=_key)

        def is_compatible(self) -> bool:
            return self._handle.path.exists()

        def get_frame(self, frame_id: int):
            """Return frame data as OBBResult for use by _add_actual_tasks."""
            return self._handle.read_frame(frame_id)

        def close(self) -> None:
            self._handle.close()

except ImportError:
    from ....data.detection_cache import DetectionCache  # type: ignore[no-redef]

from ....utils.video_encoder import VideoEncoder
from ...post.processing import _fix_heading_flips
from ..geometry import ellipse_axes_from_area, ellipse_to_obb_corners
from .naming import build_detection_image_filename, build_interpolated_image_filename

logger = logging.getLogger(__name__)


@dataclass
class FrameTask:
    """Encapsulates the affine transform and geometry for one track crop within a single frame."""

    frame_id: int
    trajectory_id: int
    affine: np.ndarray
    out_w: int
    out_h: int
    center_x: float
    center_y: float
    width: float
    height: float
    theta: float
    corners: np.ndarray
    expanded_corners: np.ndarray
    polygon_index: int
    detection_id: int | None = None
    interpolated: bool = False
    interp_from_start: int | None = None
    interp_from_end: int | None = None
    interp_index: int | None = None
    interp_total: int | None = None


@dataclass
class FrameBundle:
    """Groups all FrameTask objects and polygon corners that belong to a single video frame."""

    tasks: list[FrameTask] = field(default_factory=list)
    polygons: list[np.ndarray] = field(default_factory=list)


@dataclass
class OrientedTrackVideoExportResult:
    """Summary statistics returned after an oriented per-track video export run."""

    output_dir: str
    dataset_dir: str
    image_output_dir: str
    exported_videos: int
    exported_tracks: int
    exported_frames: int
    exported_images: int
    skipped_tracks: int
    missing_rows: int
    missing_detected_rows: int = 0
    missing_interpolated_rows: int = 0
    invalid_geometry_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the export result to a plain dictionary."""
        return asdict(self)


def resolve_individual_dataset_dir(
    output_dir: str | Path | None,
    dataset_name: str | None = None,
    run_id: str | None = None,
) -> Optional[Path]:
    """Resolve the dataset directory used for individual outputs."""
    if not output_dir:
        return None
    root = Path(output_dir).expanduser()
    if not str(root).strip():
        return None

    name_part = str(dataset_name or "").strip()
    run_part = str(run_id or "").strip()
    if run_part:
        folder_name = f"{name_part}_{run_part}" if name_part else run_part
        return root / folder_name

    if not root.exists():
        return None

    if name_part:
        matches = sorted(root.glob(f"{name_part}_*"), key=lambda p: p.name)
    else:
        matches = sorted(
            (p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name
        )
    return matches[-1] if matches else None


def resolve_oriented_track_video_dir(
    output_dir: str | Path | None,
    run_id: str | None = None,
) -> Optional[Path]:
    """Resolve the run directory used for oriented-track video outputs."""
    if not output_dir:
        return None
    root = Path(output_dir).expanduser()
    if not str(root).strip():
        return None

    run_part = str(run_id or "").strip()
    if run_part:
        return root / run_part

    if not root.exists():
        return None

    matches = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    return matches[-1] if matches else None


class OrientedTrackVideoExporter:
    """Build per-track orientation-fixed videos directly from cached geometry."""

    def __init__(
        self,
        dataset_dir: str | Path,
        final_csv_path: str | Path,
        *,
        video_path: str | Path,
        detection_cache_path: str | Path,
        interpolated_roi_npz_path: str | Path | None = None,
        fps: float,
        padding_fraction: float = 0.1,
        background_color: tuple[int, int, int] = (0, 0, 0),
        suppress_foreign_obb: bool = False,
        suppress_foreign_obb_images: bool | None = None,
        suppress_foreign_obb_videos: bool | None = None,
        fix_direction_flips: bool = False,
        heading_flip_max_burst: int = 5,
        enable_affine_stabilization: bool = False,
        stabilization_window: int = 5,
        export_videos: bool = True,
        export_images: bool = False,
        image_output_dir: str | Path | None = None,
        image_interval: int = 1,
        image_format: str = "png",
        output_subdir: str = "oriented_videos",
        codec: str = "mp4v",
    ) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.final_csv_path = Path(final_csv_path).expanduser().resolve()
        self.video_path = Path(video_path).expanduser().resolve()
        self.detection_cache_path = Path(detection_cache_path).expanduser().resolve()
        self.interpolated_roi_npz_path = (
            Path(interpolated_roi_npz_path).expanduser().resolve()
            if interpolated_roi_npz_path
            else None
        )
        self.fps = max(0.1, float(fps or 0.0))
        self.padding_fraction = max(0.0, float(padding_fraction or 0.0))
        self.background_color = self._normalize_background_color(background_color)
        self.suppress_foreign_obb = bool(suppress_foreign_obb)
        self.suppress_foreign_obb_images = bool(
            self.suppress_foreign_obb
            if suppress_foreign_obb_images is None
            else suppress_foreign_obb_images
        )
        self.suppress_foreign_obb_videos = bool(
            self.suppress_foreign_obb
            if suppress_foreign_obb_videos is None
            else suppress_foreign_obb_videos
        )
        self.fix_direction_flips = bool(fix_direction_flips)
        self.heading_flip_max_burst = max(1, int(heading_flip_max_burst or 1))
        self.enable_affine_stabilization = bool(enable_affine_stabilization)
        self.stabilization_window = max(1, int(stabilization_window or 1))
        self.export_videos = bool(export_videos)
        self.export_images = bool(export_images)
        output_subdir = str(output_subdir).strip()
        self.output_dir = (
            (
                self.dataset_dir / output_subdir
                if self.export_videos and output_subdir
                else self.dataset_dir
            )
            if self.export_videos
            else None
        )
        self.image_output_dir = (
            Path(image_output_dir).expanduser().resolve()
            if image_output_dir
            else (self.dataset_dir / "images" if self.export_images else None)
        )
        self.image_interval = max(1, int(image_interval or 1))
        image_format = str(image_format or "png").strip().lower()
        self.image_format = "jpg" if image_format in {"jpg", "jpeg"} else "png"
        self.codec = str(codec or "mp4v")
        self._last_missing_breakdown = self._empty_missing_breakdown()

    @staticmethod
    def _empty_missing_breakdown() -> dict[str, int]:
        return {
            "missing_detected_rows": 0,
            "missing_interpolated_rows": 0,
            "invalid_geometry_rows": 0,
        }

    def export(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> OrientedTrackVideoExportResult:
        """Build orientation-corrected MP4 videos for every trajectory in the final CSV.

        Streams the source video once, warps each frame into a canonical per-animal crop,
        and writes one output file per track to the configured output sub-directory.
        """
        self._emit(progress_callback, 2, "Loading final cleaned trajectories...")
        trajectories_df = self._load_final_dataframe()
        self._emit(progress_callback, 8, "Loading interpolated ROI cache...")
        interp_lookup = self._load_interpolated_roi_lookup()
        self._emit(progress_callback, 12, "Building cached geometry tasks...")
        frame_bundles, track_sizes, missing_rows = self._build_frame_bundles(
            trajectories_df,
            interp_lookup,
            should_stop=should_stop,
            progress_callback=progress_callback,
        )
        missing_breakdown = dict(self._last_missing_breakdown)

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.image_output_dir is not None:
            self.image_output_dir.mkdir(parents=True, exist_ok=True)
        track_ids = sorted(track_sizes)
        if not track_ids:
            logger.warning(
                "No oriented track videos exported: no cached geometry matched final trajectory rows."
            )
            return OrientedTrackVideoExportResult(
                output_dir=str(self.output_dir or ""),
                dataset_dir=str(self.dataset_dir),
                image_output_dir=str(self.image_output_dir or ""),
                exported_videos=0,
                exported_tracks=0,
                exported_frames=0,
                exported_images=0,
                skipped_tracks=0,
                missing_rows=int(missing_rows),
                missing_detected_rows=int(missing_breakdown["missing_detected_rows"]),
                missing_interpolated_rows=int(
                    missing_breakdown["missing_interpolated_rows"]
                ),
                invalid_geometry_rows=int(missing_breakdown["invalid_geometry_rows"]),
            )

        self._emit(
            progress_callback, 35, "Streaming source video for final media export..."
        )
        exported_videos, exported_frames, exported_images, skipped_tracks = (
            self._write_videos(
                frame_bundles,
                track_sizes,
                should_stop=should_stop,
                progress_callback=progress_callback,
            )
        )
        self._emit(progress_callback, 100, "Final canonical media export complete.")
        return OrientedTrackVideoExportResult(
            output_dir=str(self.output_dir or ""),
            dataset_dir=str(self.dataset_dir),
            image_output_dir=str(self.image_output_dir or ""),
            exported_videos=int(exported_videos),
            exported_tracks=int(len(track_ids)),
            exported_frames=int(exported_frames),
            exported_images=int(exported_images),
            skipped_tracks=int(skipped_tracks),
            missing_rows=int(missing_rows),
            missing_detected_rows=int(missing_breakdown["missing_detected_rows"]),
            missing_interpolated_rows=int(
                missing_breakdown["missing_interpolated_rows"]
            ),
            invalid_geometry_rows=int(missing_breakdown["invalid_geometry_rows"]),
        )

    @staticmethod
    def _emit(
        callback: Optional[Callable[[int, str], None]], pct: int, message: str
    ) -> None:
        if callback is None:
            return
        try:
            callback(int(max(0, min(100, pct))), str(message))
        except Exception:
            logger.debug(
                "Progress callback failed during oriented video export.", exc_info=True
            )

    @staticmethod
    def _normalize_background_color(color) -> tuple[int, int, int]:
        if isinstance(color, (list, tuple)) and len(color) == 3:
            return tuple(int(np.clip(v, 0, 255)) for v in color)
        fill = int(np.clip(color if color is not None else 0, 0, 255))
        return (fill, fill, fill)

    @staticmethod
    def _normalize_theta(theta: float) -> float:
        try:
            value = float(theta)
        except Exception:
            value = 0.0
        return value % (2.0 * math.pi)

    @classmethod
    def _collapse_branch_to_reference(
        cls, theta: float, reference_theta: Optional[float]
    ) -> float:
        theta0 = cls._normalize_theta(theta)
        theta1 = cls._normalize_theta(theta0 + math.pi)
        if reference_theta is None:
            return theta0
        try:
            ref = float(reference_theta)
        except Exception:
            return theta0
        if not np.isfinite(ref):
            return theta0
        ref = cls._normalize_theta(ref)
        diff0 = abs(((theta0 - ref) + math.pi) % (2.0 * math.pi) - math.pi)
        diff1 = abs(((theta1 - ref) + math.pi) % (2.0 * math.pi) - math.pi)
        return theta0 if diff0 <= diff1 else theta1

    @classmethod
    def _resolve_task_theta(
        cls,
        row: Any,
        fallback_theta: float,
        *,
        reference_theta: Optional[float] = None,
        directed_heading: Optional[float] = None,
        directed: bool = False,
    ) -> float:
        theta = float("nan")
        if directed:
            try:
                candidate = float(directed_heading)
                if np.isfinite(candidate):
                    theta = candidate
            except Exception:
                pass
        if not np.isfinite(theta):
            row_theta = getattr(row, "Theta", np.nan)
            if pd.notna(row_theta):
                theta = float(row_theta)
            else:
                theta = float(fallback_theta)
        return cls._collapse_branch_to_reference(theta, reference_theta)

    def _load_final_dataframe(self) -> pd.DataFrame:
        if not self.final_csv_path.exists():
            raise FileNotFoundError(
                f"Missing final trajectories CSV: {self.final_csv_path}"
            )
        df = pd.read_csv(self.final_csv_path)
        required = {"TrajectoryID", "FrameID"}
        if not required.issubset(df.columns):
            raise ValueError(
                "Final trajectories CSV must contain TrajectoryID and FrameID columns."
            )
        df = df.copy()
        for col in ("TrajectoryID", "FrameID", "DetectionID", "Theta"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["TrajectoryID", "FrameID"]).sort_values(
            ["FrameID", "TrajectoryID"], kind="stable"
        )

    def _load_interpolated_roi_lookup(self) -> dict[tuple[int, int], dict[str, Any]]:
        path = self.interpolated_roi_npz_path
        if path is None or not path.exists():
            return {}
        payload = np.load(str(path), allow_pickle=True)
        try:
            frame_ids = payload["frame_id"]
            trajectory_ids = payload["trajectory_id"]
            cx = payload["cx"]
            cy = payload["cy"]
            width = payload["w"]
            height = payload["h"]
            theta = payload["theta"]
            obb_corners = payload["obb_corners"]
            interp_from_start = (
                payload["interp_from_start"] if "interp_from_start" in payload else None
            )
            interp_from_end = (
                payload["interp_from_end"] if "interp_from_end" in payload else None
            )
            interp_index = (
                payload["interp_index"] if "interp_index" in payload else None
            )
            interp_total = (
                payload["interp_total"] if "interp_total" in payload else None
            )
            lookup: dict[tuple[int, int], dict[str, Any]] = {}
            total = min(
                len(frame_ids),
                len(trajectory_ids),
                len(cx),
                len(cy),
                len(width),
                len(height),
                len(theta),
                len(obb_corners),
            )
            for idx in range(total):
                lookup[(int(frame_ids[idx]), int(trajectory_ids[idx]))] = {
                    "cx": float(cx[idx]),
                    "cy": float(cy[idx]),
                    "w": float(width[idx]),
                    "h": float(height[idx]),
                    "theta": float(theta[idx]),
                    "obb_corners": np.asarray(obb_corners[idx], dtype=np.float32),
                    "interp_from_start": (
                        int(interp_from_start[idx])
                        if interp_from_start is not None
                        and idx < len(interp_from_start)
                        else int(frame_ids[idx])
                    ),
                    "interp_from_end": (
                        int(interp_from_end[idx])
                        if interp_from_end is not None and idx < len(interp_from_end)
                        else int(frame_ids[idx])
                    ),
                    "interp_index": (
                        int(interp_index[idx])
                        if interp_index is not None and idx < len(interp_index)
                        else 1
                    ),
                    "interp_total": (
                        int(interp_total[idx])
                        if interp_total is not None and idx < len(interp_total)
                        else 1
                    ),
                }
            return lookup
        finally:
            try:
                payload.close()
            except Exception:
                pass

    def _build_frame_bundles(
        self,
        trajectories_df: pd.DataFrame,
        interp_lookup: dict[tuple[int, int], dict[str, Any]],
        *,
        should_stop: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> tuple[dict[int, FrameBundle], dict[int, tuple[int, int]], int]:
        actual_rows_by_frame: dict[int, list[Any]] = defaultdict(list)
        interp_rows_by_frame: dict[int, list[Any]] = defaultdict(list)
        for row in trajectories_df.itertuples(index=False):
            frame_id = int(row.FrameID)
            det_id = getattr(row, "DetectionID", np.nan)
            if pd.notna(det_id):
                actual_rows_by_frame[frame_id].append(row)
            else:
                interp_rows_by_frame[frame_id].append(row)

        frame_bundles: dict[int, FrameBundle] = {}
        track_sizes: dict[int, tuple[int, int]] = {}
        missing_breakdown = self._empty_missing_breakdown()

        actual_frames = sorted(actual_rows_by_frame)
        total_actual = len(actual_frames)
        interp_frames = sorted(interp_rows_by_frame)
        total_interp = len(interp_frames)
        frame_ids = sorted(set(actual_rows_by_frame) | set(interp_rows_by_frame))
        track_theta_state: dict[int, float] = {}

        detection_cache = None
        if actual_rows_by_frame:
            if not self.detection_cache_path.exists():
                raise FileNotFoundError(
                    f"Missing detection cache for oriented track videos: {self.detection_cache_path}"
                )
            detection_cache = DetectionCache(self.detection_cache_path, mode="r")
            if not detection_cache.is_compatible():
                detection_cache.close()
                raise ValueError(
                    f"Incompatible detection cache for oriented track videos: {self.detection_cache_path}"
                )

        actual_index = 0
        interp_index = 0
        try:
            for frame_id in frame_ids:
                if should_stop and should_stop():
                    break

                actual_rows = actual_rows_by_frame.get(frame_id, [])
                if actual_rows and detection_cache is not None:
                    actual_index += 1
                    self._emit(
                        progress_callback,
                        12 + int(16 * actual_index / max(1, total_actual)),
                        f"Preparing cached detection geometry... {actual_index}/{total_actual}",
                    )
                    bundle = frame_bundles.setdefault(frame_id, FrameBundle())
                    actual_missing = self._add_actual_tasks(
                        detection_cache,
                        frame_id,
                        actual_rows,
                        bundle,
                        track_sizes,
                        track_theta_state,
                    )
                    for key, value in actual_missing.items():
                        missing_breakdown[key] += int(value)

                interp_rows = interp_rows_by_frame.get(frame_id, [])
                if interp_rows:
                    interp_index += 1
                    self._emit(
                        progress_callback,
                        28 + int(6 * interp_index / max(1, total_interp)),
                        f"Preparing interpolated geometry... {interp_index}/{total_interp}",
                    )
                    bundle = frame_bundles.setdefault(frame_id, FrameBundle())
                    for row in interp_rows:
                        traj_id = int(row.TrajectoryID)
                        record = interp_lookup.get((frame_id, traj_id))
                        if record is None:
                            missing_breakdown["missing_interpolated_rows"] += 1
                            continue
                        theta = self._resolve_task_theta(
                            row,
                            record.get("theta", 0.0),
                            reference_theta=track_theta_state.get(traj_id),
                        )
                        task = self._build_task(
                            frame_id=frame_id,
                            trajectory_id=traj_id,
                            center_x=float(record["cx"]),
                            center_y=float(record["cy"]),
                            width=float(record["w"]),
                            height=float(record["h"]),
                            theta=theta,
                            corners=np.asarray(record["obb_corners"], dtype=np.float32),
                            polygon_index=len(bundle.polygons),
                            interpolated=True,
                            interp_from_start=int(
                                record.get("interp_from_start", frame_id)
                            ),
                            interp_from_end=int(
                                record.get("interp_from_end", frame_id)
                            ),
                            interp_index=int(record.get("interp_index", 1)),
                            interp_total=int(record.get("interp_total", 1)),
                        )
                        if task is None:
                            missing_breakdown["invalid_geometry_rows"] += 1
                            continue
                        bundle.polygons.append(task.corners)
                        bundle.tasks.append(task)
                        track_theta_state[traj_id] = theta
                        track_sizes[traj_id] = self._merge_canvas_size(
                            track_sizes.get(traj_id), (task.out_w, task.out_h)
                        )
        finally:
            if detection_cache is not None:
                detection_cache.close()

        frame_bundles = {k: v for k, v in frame_bundles.items() if v.tasks}
        frame_bundles, track_sizes = self._apply_track_postprocessing(
            frame_bundles,
            track_sizes,
        )
        self._last_missing_breakdown = missing_breakdown
        return frame_bundles, track_sizes, int(sum(missing_breakdown.values()))

    def _apply_track_postprocessing(
        self,
        frame_bundles: dict[int, FrameBundle],
        track_sizes: dict[int, tuple[int, int]],
    ) -> tuple[dict[int, FrameBundle], dict[int, tuple[int, int]]]:
        if not frame_bundles:
            return frame_bundles, track_sizes
        if not self.fix_direction_flips and not self.enable_affine_stabilization:
            return frame_bundles, track_sizes

        tasks_by_track: dict[int, list[FrameTask]] = defaultdict(list)
        for bundle in frame_bundles.values():
            for task in bundle.tasks:
                tasks_by_track[task.trajectory_id].append(task)

        updated_track_sizes: dict[int, tuple[int, int]] = {}
        window = self._normalized_smoothing_window(self.stabilization_window)
        for trajectory_id, tasks in tasks_by_track.items():
            tasks.sort(key=lambda task: task.frame_id)
            center_x = np.array([task.center_x for task in tasks], dtype=np.float64)
            center_y = np.array([task.center_y for task in tasks], dtype=np.float64)
            width = np.array([task.width for task in tasks], dtype=np.float64)
            height = np.array([task.height for task in tasks], dtype=np.float64)
            theta = np.array([task.theta for task in tasks], dtype=np.float64)

            if self.fix_direction_flips and len(theta) > 0:
                theta = _fix_heading_flips(
                    theta,
                    max_burst=self.heading_flip_max_burst,
                )

            if self.enable_affine_stabilization and len(tasks) > 1:
                center_x = self._smooth_numeric_series(center_x, window)
                center_y = self._smooth_numeric_series(center_y, window)
                width = self._smooth_numeric_series(width, window)
                height = self._smooth_numeric_series(height, window)
                theta = self._smooth_angle_series(theta, window)

            for idx, task in enumerate(tasks):
                affine, out_w, out_h = self._compute_affine(
                    float(center_x[idx]),
                    float(center_y[idx]),
                    max(1.0, float(width[idx])),
                    max(1.0, float(height[idx])),
                    float(theta[idx]),
                )
                task.center_x = float(center_x[idx])
                task.center_y = float(center_y[idx])
                task.width = max(1.0, float(width[idx]))
                task.height = max(1.0, float(height[idx]))
                task.theta = self._normalize_theta(float(theta[idx]))
                task.affine = affine
                task.out_w = out_w
                task.out_h = out_h
                updated_track_sizes[trajectory_id] = self._merge_canvas_size(
                    updated_track_sizes.get(trajectory_id),
                    (out_w, out_h),
                )

        return frame_bundles, updated_track_sizes or track_sizes

    @staticmethod
    def _normalized_smoothing_window(window: int) -> int:
        value = max(1, int(window or 1))
        return value if value % 2 == 1 else value + 1

    @classmethod
    def _smooth_numeric_series(cls, values: np.ndarray, window: int) -> np.ndarray:
        if len(values) < 2 or window <= 1:
            return values.copy()
        half = window // 2
        result = values.astype(np.float64, copy=True)
        for idx in range(len(values)):
            start = max(0, idx - half)
            end = min(len(values), idx + half + 1)
            result[idx] = float(np.nanmedian(values[start:end]))
        return result

    @classmethod
    def _smooth_angle_series(cls, values: np.ndarray, window: int) -> np.ndarray:
        if len(values) < 2 or window <= 1:
            return values.copy()
        unwrapped = np.unwrap(values.astype(np.float64, copy=False))
        half = window // 2
        result = unwrapped.copy()
        for idx in range(len(unwrapped)):
            start = max(0, idx - half)
            end = min(len(unwrapped), idx + half + 1)
            result[idx] = float(np.nanmean(unwrapped[start:end]))
        two_pi = 2.0 * math.pi
        return np.mod(result, two_pi)

    def _add_actual_tasks(
        self,
        detection_cache: DetectionCache,
        frame_id: int,
        rows: list[Any],
        bundle: FrameBundle,
        track_sizes: dict[int, tuple[int, int]],
        track_theta_state: dict[int, float],
    ) -> dict[str, int]:
        missing = {
            "missing_detected_rows": 0,
            "invalid_geometry_rows": 0,
        }
        # Support both new OBBResult API and legacy 12-tuple.
        # Correction 20: new cache returns OBBResult; legacy returns a 12-tuple.
        frame_data = detection_cache.get_frame(frame_id)
        from hydra_suite.core.inference.result import OBBResult as _OBBResult

        if isinstance(frame_data, _OBBResult):
            obb = frame_data
            meas = np.concatenate(
                [obb.centroids, obb.angles[:, None]], axis=1
            )  # (D, 3): cx, cy, theta
            shapes = obb.shapes  # (D, 2)
            obb_corners = obb.corners  # (D, 4, 2)
            detection_ids = obb.detection_ids
            heading_hints = None  # Not stored in OBBResult
            directed_mask = None
        else:
            # Legacy 12-tuple unpack
            (
                meas,
                _sizes,
                shapes,
                _confidences,
                obb_corners,
                detection_ids,
                heading_hints,
                _heading_confidences,
                directed_mask,
                _canonical_affines,
                _canvas_dims,
                _M_inverse,
            ) = frame_data

        det_index = {}
        for idx, det_id in enumerate(
            detection_ids if detection_ids is not None else []
        ):
            try:
                det_index[int(det_id)] = idx
            except Exception:
                continue

        for row in rows:
            traj_id = int(row.TrajectoryID)
            det_id = getattr(row, "DetectionID", np.nan)
            if pd.isna(det_id):
                missing["missing_detected_rows"] += 1
                continue
            idx = det_index.get(int(det_id))
            if idx is None:
                missing["missing_detected_rows"] += 1
                continue
            corners = None
            if obb_corners is not None and idx < len(obb_corners):
                corners = np.asarray(obb_corners[idx], dtype=np.float32)
            elif (
                meas is not None
                and idx < len(meas)
                and shapes is not None
                and idx < len(shapes)
            ):
                shape = shapes[idx]
                if shape is not None and len(shape) >= 2:
                    area = float(shape[0])
                    aspect_ratio = float(shape[1])
                    if area > 0.0 and aspect_ratio > 0.0:
                        theta = float(meas[idx][2]) if len(meas[idx]) > 2 else 0.0
                        cx = float(meas[idx][0])
                        cy = float(meas[idx][1])
                        corners = ellipse_to_obb_corners(
                            cx,
                            cy,
                            *ellipse_axes_from_area(area, aspect_ratio),
                            theta,
                        )
            if corners is None or corners.shape != (4, 2):
                missing["missing_detected_rows"] += 1
                continue
            width, height = self._edge_lengths(corners)
            center = corners.mean(axis=0)
            theta_fallback = (
                float(meas[idx][2]) if idx < len(meas) and len(meas[idx]) > 2 else 0.0
            )
            directed_heading = (
                float(heading_hints[idx])
                if heading_hints is not None and idx < len(heading_hints)
                else None
            )
            directed = bool(
                directed_mask is not None
                and idx < len(directed_mask)
                and directed_mask[idx]
                and directed_heading is not None
                and np.isfinite(directed_heading)
            )
            theta = self._resolve_task_theta(
                row,
                theta_fallback,
                reference_theta=track_theta_state.get(traj_id),
                directed_heading=directed_heading,
                directed=directed,
            )
            task = self._build_task(
                frame_id=frame_id,
                trajectory_id=traj_id,
                center_x=float(center[0]),
                center_y=float(center[1]),
                width=width,
                height=height,
                theta=theta,
                corners=corners,
                polygon_index=len(bundle.polygons),
                detection_id=int(det_id),
            )
            if task is None:
                missing["invalid_geometry_rows"] += 1
                continue
            bundle.polygons.append(task.corners)
            bundle.tasks.append(task)
            track_theta_state[traj_id] = theta
            track_sizes[traj_id] = self._merge_canvas_size(
                track_sizes.get(traj_id), (task.out_w, task.out_h)
            )
        return missing

    def _build_task(
        self,
        *,
        frame_id: int,
        trajectory_id: int,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        theta: float,
        corners: np.ndarray,
        polygon_index: int,
        detection_id: int | None = None,
        interpolated: bool = False,
        interp_from_start: int | None = None,
        interp_from_end: int | None = None,
        interp_index: int | None = None,
        interp_total: int | None = None,
    ) -> Optional[FrameTask]:
        if not np.isfinite(center_x) or not np.isfinite(center_y):
            return None
        if not np.isfinite(width) or not np.isfinite(height):
            return None
        box_w = max(1.0, float(width))
        box_h = max(1.0, float(height))
        affine, out_w, out_h = self._compute_affine(
            center_x,
            center_y,
            box_w,
            box_h,
            float(theta),
        )
        return FrameTask(
            frame_id=int(frame_id),
            trajectory_id=int(trajectory_id),
            affine=affine,
            out_w=out_w,
            out_h=out_h,
            center_x=float(center_x),
            center_y=float(center_y),
            width=float(box_w),
            height=float(box_h),
            theta=self._normalize_theta(float(theta)),
            corners=np.asarray(corners, dtype=np.float32),
            expanded_corners=self._expand_corners(corners, self.padding_fraction),
            polygon_index=int(polygon_index),
            detection_id=(int(detection_id) if detection_id is not None else None),
            interpolated=bool(interpolated),
            interp_from_start=(
                int(interp_from_start) if interp_from_start is not None else None
            ),
            interp_from_end=(
                int(interp_from_end) if interp_from_end is not None else None
            ),
            interp_index=(int(interp_index) if interp_index is not None else None),
            interp_total=(int(interp_total) if interp_total is not None else None),
        )

    def _process_frame_bundle(
        self,
        frame,
        bundle,
        track_sizes,
        writers,
        frames_written_by_track,
        images_written_by_track,
        _canvases,
    ):
        """Write all tasks from a single frame bundle to their track writers."""
        for task in bundle.tasks:
            canvas_size = track_sizes.get(task.trajectory_id)
            if canvas_size is None:
                continue
            rendered_videos = None
            if self.export_videos:
                rendered_videos = self._render_task(
                    frame,
                    task,
                    bundle.polygons,
                    canvas_size,
                    canvas=_canvases.get(task.trajectory_id),
                    suppress_foreign_obb=self.suppress_foreign_obb_videos,
                )
                if rendered_videos is not None:
                    writer = writers.get(task.trajectory_id)
                    if writer is None:
                        writer = self._open_writer(task.trajectory_id, canvas_size)
                        writers[task.trajectory_id] = writer
                    writer.write(rendered_videos)
                    frames_written_by_track[task.trajectory_id] += 1

            if self._should_export_image(task.frame_id):
                rendered_images = rendered_videos
                if rendered_images is None or (
                    self.suppress_foreign_obb_images != self.suppress_foreign_obb_videos
                ):
                    rendered_images = self._render_task(
                        frame,
                        task,
                        bundle.polygons,
                        canvas_size,
                        suppress_foreign_obb=self.suppress_foreign_obb_images,
                    )
                if rendered_images is not None and self._write_image(
                    task, rendered_images
                ):
                    images_written_by_track[task.trajectory_id] += 1

    def _tally_exports(
        self, track_sizes, frames_written_by_track, images_written_by_track
    ):
        """Compute export summary and clean up empty output files."""
        exported_videos = 0
        skipped_tracks = 0
        exported_frames = 0
        exported_images = 0
        for traj_id in track_sizes:
            count = int(frames_written_by_track.get(traj_id, 0))
            exported_images += int(images_written_by_track.get(traj_id, 0))
            if count > 0:
                exported_videos += 1
                exported_frames += count
            elif self.export_videos:
                skipped_tracks += 1
                output_path = self.output_dir / f"trajectory_{int(traj_id):04d}.mp4"
                try:
                    output_path.unlink(missing_ok=True)
                except Exception:
                    pass
        return exported_videos, exported_frames, exported_images, skipped_tracks

    def _write_videos(
        self,
        frame_bundles: dict[int, FrameBundle],
        track_sizes: dict[int, tuple[int, int]],
        *,
        should_stop: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> tuple[int, int, int, int]:
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Missing source video for oriented track export: {self.video_path}"
            )
        needed_frames = sorted(frame_bundles)
        if not needed_frames:
            return 0, 0, 0, 0

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open source video: {self.video_path}")

        writers: dict[int, VideoEncoder] = {}
        frames_written_by_track: dict[int, int] = defaultdict(int)
        images_written_by_track: dict[int, int] = defaultdict(int)
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)

        # Pre-allocate one reusable canvas per track to avoid per-frame np.full()
        _canvases: dict[int, np.ndarray] = {}
        for traj_id, (cw, ch) in track_sizes.items():
            _canvases[traj_id] = np.full(
                (ch, cw, 3), self.background_color, dtype=np.uint8
            )

        def _read_frame(fid):
            """Read a single frame (called from background thread)."""
            nonlocal current_pos
            if fid != current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, frm = cap.read()
            current_pos = int(fid) + 1
            return ok, frm

        try:
            total_frames = len(needed_frames)
            with ThreadPoolExecutor(max_workers=1) as reader:
                future = reader.submit(_read_frame, needed_frames[0])

                for index, frame_id in enumerate(needed_frames, start=1):
                    if should_stop and should_stop():
                        break
                    self._emit(
                        progress_callback,
                        35 + int(60 * index / max(1, total_frames)),
                        f"Writing final canonical media... {index}/{total_frames}",
                    )

                    ok, frame = future.result()
                    if index < total_frames:
                        future = reader.submit(_read_frame, needed_frames[index])

                    if not ok or frame is None:
                        continue
                    bundle = frame_bundles.get(frame_id)
                    if bundle is None:
                        continue
                    self._process_frame_bundle(
                        frame,
                        bundle,
                        track_sizes,
                        writers,
                        frames_written_by_track,
                        images_written_by_track,
                        _canvases,
                    )
        finally:
            cap.release()
            for writer in writers.values():
                try:
                    writer.release()
                except Exception:
                    pass

        return self._tally_exports(
            track_sizes,
            frames_written_by_track,
            images_written_by_track,
        )

    def _should_export_image(self, frame_id: int) -> bool:
        if not self.export_images or self.image_output_dir is None:
            return False
        try:
            return int(frame_id) % self.image_interval == 0
        except Exception:
            return False

    def _write_image(self, task: FrameTask, image: np.ndarray) -> bool:
        if self.image_output_dir is None:
            return False
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        if task.detection_id is not None:
            filename = build_detection_image_filename(
                task.detection_id, self.image_format
            )
        elif task.interpolated:
            filename = build_interpolated_image_filename(
                frame_id=task.frame_id,
                trajectory_id=task.trajectory_id,
                interp_from=(
                    int(task.interp_from_start or task.frame_id),
                    int(task.interp_from_end or task.frame_id),
                ),
                interp_index=int(task.interp_index or 1),
                interp_total=int(task.interp_total or 1),
                extension=self.image_format,
            )
        else:
            filename = (
                f"f{int(task.frame_id):06d}_traj{int(task.trajectory_id):04d}."
                f"{self.image_format}"
            )
        output_path = self.image_output_dir / filename
        try:
            return bool(cv2.imwrite(str(output_path), image))
        except Exception:
            logger.debug(
                "Failed to write canonical image: %s", output_path, exc_info=True
            )
            return False

    def _open_writer(
        self, trajectory_id: int, canvas_size: tuple[int, int]
    ) -> VideoEncoder:
        if self.output_dir is None:
            raise RuntimeError("Video output directory is not configured for export.")
        output_path = self.output_dir / f"trajectory_{int(trajectory_id):04d}.mp4"
        return VideoEncoder(
            output_path,
            fps=self.fps,
            width=canvas_size[0],
            height=canvas_size[1],
        )

    def _render_task(
        self,
        frame: np.ndarray,
        task: FrameTask,
        frame_polygons: list[np.ndarray],
        canvas_size: tuple[int, int],
        canvas: Optional[np.ndarray] = None,
        suppress_foreign_obb: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        warped = cv2.warpAffine(
            frame,
            task.affine,
            (task.out_w, task.out_h),
            flags=getattr(cv2, "INTER_LINEAR", 1),
            borderMode=cv2.BORDER_REPLICATE,
        )
        if warped is None or warped.size == 0:
            return None

        mask = np.zeros((task.out_h, task.out_w), dtype=np.uint8)
        target_poly = self._transform_polygon(
            task.expanded_corners, task.affine, task.out_w, task.out_h
        )
        if target_poly is None:
            return None
        cv2.fillPoly(mask, [target_poly], 255)

        if suppress_foreign_obb is None:
            suppress_foreign_obb = self.suppress_foreign_obb

        if suppress_foreign_obb and len(frame_polygons) > 1:
            for idx, corners in enumerate(frame_polygons):
                if idx == task.polygon_index:
                    continue
                foreign_poly = self._transform_polygon(
                    corners, task.affine, task.out_w, task.out_h
                )
                if foreign_poly is None:
                    continue
                cv2.fillPoly(mask, [foreign_poly], 0)

        masked = np.full_like(warped, self.background_color, dtype=np.uint8)
        valid = mask > 0
        masked[valid] = warped[valid]

        canvas_w, canvas_h = canvas_size
        if masked.shape[1] == canvas_w and masked.shape[0] == canvas_h:
            return masked

        # Reuse pre-allocated canvas when available, otherwise allocate
        if (
            canvas is not None
            and canvas.shape[0] == canvas_h
            and canvas.shape[1] == canvas_w
        ):
            canvas[:] = self.background_color
        else:
            canvas = np.full(
                (canvas_h, canvas_w, 3), self.background_color, dtype=np.uint8
            )
        x0 = max(0, (canvas_w - masked.shape[1]) // 2)
        y0 = max(0, (canvas_h - masked.shape[0]) // 2)
        x1 = min(canvas_w, x0 + masked.shape[1])
        y1 = min(canvas_h, y0 + masked.shape[0])
        canvas[y0:y1, x0:x1] = masked[: y1 - y0, : x1 - x0]
        return canvas

    def _compute_affine(
        self,
        center_x: float,
        center_y: float,
        box_w: float,
        box_h: float,
        theta: float,
    ) -> tuple[np.ndarray, int, int]:
        crop_w = max(8.0, float(box_w) * (1.0 + self.padding_fraction))
        crop_h = max(8.0, float(box_h) * (1.0 + self.padding_fraction))
        out_w, out_h = self._normalize_frame_size(
            int(round(crop_w)), int(round(crop_h))
        )

        cos_a = float(math.cos(theta))
        sin_a = float(math.sin(theta))
        half_w = crop_w * 0.5
        half_h = crop_h * 0.5
        src_pts = np.array(
            [
                [
                    center_x - half_w * cos_a + half_h * sin_a,
                    center_y - half_w * sin_a - half_h * cos_a,
                ],
                [
                    center_x + half_w * cos_a + half_h * sin_a,
                    center_y + half_w * sin_a - half_h * cos_a,
                ],
                [
                    center_x - half_w * cos_a - half_h * sin_a,
                    center_y - half_w * sin_a + half_h * cos_a,
                ],
            ],
            dtype=np.float32,
        )
        dst_pts = np.array(
            [[0, 0], [out_w - 1, 0], [0, out_h - 1]],
            dtype=np.float32,
        )
        affine = cv2.getAffineTransform(src_pts, dst_pts)
        return affine, out_w, out_h

    @staticmethod
    def _normalize_frame_size(width: int, height: int) -> tuple[int, int]:
        out_w = max(8, int(width))
        out_h = max(8, int(height))
        if out_w % 2:
            out_w += 1
        if out_h % 2:
            out_h += 1
        return out_w, out_h

    @staticmethod
    def _expand_corners(corners: np.ndarray, padding_fraction: float) -> np.ndarray:
        arr = np.asarray(corners, dtype=np.float32)
        if arr.shape != (4, 2):
            return arr
        centroid = arr.mean(axis=0)
        expanded = arr.copy()
        for idx in range(4):
            direction = arr[idx] - centroid
            expanded[idx] = centroid + direction * (1.0 + float(padding_fraction))
        return expanded.astype(np.float32)

    @staticmethod
    def _transform_polygon(
        corners: np.ndarray,
        affine: np.ndarray,
        out_w: int,
        out_h: int,
    ) -> Optional[np.ndarray]:
        arr = np.asarray(corners, dtype=np.float32)
        if arr.shape != (4, 2):
            return None
        pts = cv2.transform(arr.reshape(1, 4, 2), affine).reshape(4, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, max(0, out_w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(0, out_h - 1))
        return pts.astype(np.int32)

    @staticmethod
    def _edge_lengths(corners: np.ndarray) -> tuple[float, float]:
        arr = np.asarray(corners, dtype=np.float32)
        width = float(np.linalg.norm(arr[1] - arr[0]))
        height = float(np.linalg.norm(arr[2] - arr[1]))
        return max(1.0, width), max(1.0, height)

    @staticmethod
    def _merge_canvas_size(
        current: Optional[tuple[int, int]],
        candidate: tuple[int, int],
    ) -> tuple[int, int]:
        if current is None:
            return candidate
        return max(int(current[0]), int(candidate[0])), max(
            int(current[1]), int(candidate[1])
        )
