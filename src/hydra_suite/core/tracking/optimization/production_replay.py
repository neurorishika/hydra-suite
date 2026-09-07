"""Read-only autotuner evaluation through the production tracking engine.

The Bayesian search uses a lightweight loop because it may evaluate hundreds of
parameter proposals.  A proposal is not safe to recommend on that evidence
alone: the lightweight loop intentionally omits some production-only features.
This module runs shortlisted candidates through :class:`TrackingEngineCore`
itself and returns only the observed positions that production would export.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from hydra_suite.core.tracking.confidence.confidence_density import (
    ConfidenceDensityCancelled,
)

logger = logging.getLogger(__name__)

_REPLAY_THRESHOLD_TUNING_DIMENSIONS = (
    "YOLO_CONFIDENCE_THRESHOLD",
    "YOLO_IOU_THRESHOLD",
)
_REPLAY_UNVALIDATED_TUNING_DIMENSIONS = {
    "KALMAN_INITIAL_VELOCITY_RETENTION": (
        "Read-only replay has no lifecycle diagnostic proving a young-filter "
        "association in the scored held-out interval; this bootstrap control "
        "is disabled rather than recommending an unvalidated value."
    )
}


def disabled_replay_tuning_dimensions(params: Mapping[str, Any]) -> dict[str, str]:
    """Return tuning dimensions a read-only production replay cannot score.

    Raw OBB caches deliberately preserve detections before the user-facing
    confidence/IoU filters. Downstream head-tail, CNN, pose, and AprilTag
    evidence is currently materialized only for the baseline filtered set.
    Lowering a threshold can therefore admit a raw detection with no evidence;
    changing NMS/cap membership can also attach locally indexed evidence to the
    wrong raw detection. A read-only evaluator must not recommend values based
    on those fabricated inputs.

    The helper is UI-independent so the core optimizer and TrackerKit can
    surface the same explicit explanation before a run starts.
    """

    disabled = dict(_REPLAY_UNVALIDATED_TUNING_DIMENSIONS)
    detection_method = (
        str(params.get("DETECTION_METHOD", "background_subtraction")).strip().lower()
    )
    if detection_method != "yolo_obb":
        reason = (
            "YOLO confidence/IoU are inactive for the selected detection source; "
            "read-only replay will not tune inert dimensions."
        )
        disabled.update({name: reason for name in _REPLAY_THRESHOLD_TUNING_DIMENSIONS})
        return disabled

    downstream_stages: list[str] = []
    if str(params.get("YOLO_HEADTAIL_MODEL_PATH", "") or "").strip():
        downstream_stages.append("head-tail")
    if any(
        isinstance(item, Mapping) and str(item.get("model_path", "") or "").strip()
        for item in (params.get("CNN_CLASSIFIERS", []) or [])
    ):
        downstream_stages.append("CNN")
    if bool(params.get("ENABLE_POSE_EXTRACTOR", False)):
        downstream_stages.append("pose")
    if bool(params.get("USE_APRILTAGS", False)):
        downstream_stages.append("AprilTag")
    if not downstream_stages:
        return disabled

    reason = (
        "Read-only replay cannot faithfully re-index cached downstream "
        f"({', '.join(downstream_stages)}) evidence after confidence/IoU filtering "
        "changes; those tuning dimensions are disabled."
    )
    disabled.update({name: reason for name in _REPLAY_THRESHOLD_TUNING_DIMENSIONS})
    return disabled


def sanitize_replay_tuning_config(
    tuning_config: Mapping[str, bool], params: Mapping[str, Any]
) -> tuple[dict[str, bool], dict[str, str]]:
    """Return a safe copy of selected tuning dimensions and their exclusions."""

    sanitized = {str(name): bool(enabled) for name, enabled in tuning_config.items()}
    disabled = disabled_replay_tuning_dimensions(params)
    for name in disabled:
        if name in sanitized:
            sanitized[name] = False
    return sanitized, disabled


@dataclass(frozen=True)
class ProductionReplayResult:
    """Dense observed positions from one production-engine replay."""

    success: bool
    frame_indices: np.ndarray
    positions: np.ndarray
    error: str | None = None


def cache_directory(cache_path: str | Path) -> Path:
    """Return the InferenceRunner cache directory for a dir or member path."""

    path = Path(cache_path)
    # Normal TrackerKit production runs retain ``detection.npz`` as their
    # artifact path. Existing cache members other than that canonical name are
    # also safe to normalize; a non-existent optimizer build target remains a
    # directory so it can be created by ``InferenceRunner``.
    return path.parent if path.name == "detection.npz" or path.is_file() else path


@dataclass(frozen=True)
class ReplayCacheAdmission:
    """Read-only compatibility result for a production replay cache set.

    A detection member alone cannot support production-faithful replay: every
    enabled downstream member must have the matching cache key, generation,
    and coverage.  ``detection_range_covered`` is intentionally separate so a
    complete cache set for a different clip span is reported honestly.
    """

    cache_set_valid: bool
    detection_range_covered: bool
    missing_frames: tuple[int, ...] = ()
    error: str | None = None

    @property
    def ready(self) -> bool:
        """Whether this cache can replay the requested evidence window."""

        return self.cache_set_valid and self.detection_range_covered


def inspect_replay_cache_admission(
    cache_path: str | Path,
    video_path: str | Path,
    params: Mapping[str, Any],
    start_frame: int,
    end_frame: int,
) -> ReplayCacheAdmission:
    """Inspect a full production replay cache set without running inference.

    This is the single admission contract for TrackerKit's cache-preparation
    shortcut and the optimizer core.  It opens only cache handles and applies
    the same pure cache-set predicate as ``InferenceRunner.caches_all_valid``;
    metadata probing must not initialize an OBB/TRT backend for each candidate
    cache directory the GUI scans.
    """

    if not cache_path:
        return ReplayCacheAdmission(False, False, error="No replay cache path")

    cache_dir = cache_directory(cache_path)
    if not cache_dir.is_dir():
        return ReplayCacheAdmission(
            False, False, error=f"Replay cache directory does not exist: {cache_dir}"
        )

    try:
        from hydra_suite.core.inference.runner import (
            _open_caches,
            cache_set_is_fully_reusable,
            video_signature,
        )
        from hydra_suite.core.tracking.optimization.detection_config import (
            inference_config_for_optimizer_params,
        )

        config = inference_config_for_optimizer_params(dict(params))
        caches = _open_caches(
            config,
            cache_dir,
            video_signature(str(video_path)),
            params.get("ROI_MASK", None),
            read_only=True,
        )
        if not cache_set_is_fully_reusable(caches):
            return ReplayCacheAdmission(False, False)

        detection = caches.detection
        if detection is None:
            return ReplayCacheAdmission(False, False)
        range_covered = bool(detection.covers_frame_range(start_frame, end_frame))
        missing = (
            tuple(
                int(frame)
                for frame in detection.get_missing_frames(start_frame, end_frame)
            )
            if not range_covered
            else ()
        )
        return ReplayCacheAdmission(True, range_covered, missing)
    except Exception as exc:
        logger.warning("Unable to inspect replay evidence cache", exc_info=True)
        return ReplayCacheAdmission(False, False, error=str(exc))


def trajectories_to_positions(
    trajectories: Sequence[Sequence[Sequence[float]]],
    *,
    start_frame: int,
    end_frame: int,
    n_tracks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert production trajectory tuples to a frame-major observed array.

    Production trajectories contain only real matched detections as
    ``(x, y, theta, frame_idx)``.  Missing/unmatched observations remain NaN;
    hidden Kalman posterior positions are deliberately never synthesized.
    """

    if end_frame < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")
    if n_tracks <= 0:
        raise ValueError("n_tracks must be positive")

    frame_indices = np.arange(start_frame, end_frame + 1, dtype=np.int64)
    positions = np.full((len(frame_indices), n_tracks, 2), np.nan, dtype=np.float32)
    for track_idx, track in enumerate(trajectories[:n_tracks]):
        for point in track:
            if len(point) < 4:
                continue
            frame_idx = int(point[3])
            if start_frame <= frame_idx <= end_frame:
                positions[frame_idx - start_frame, track_idx] = (
                    float(point[0]),
                    float(point[1]),
                )
    return frame_indices, positions


class ProductionReplayEvaluator:
    """Evaluate a parameter set with the actual production tracking loop.

    The underlying inference cache is opened read-only.  ``engine_factory`` is
    injectable to keep this adapter independently testable without importing
    or constructing the full tracking runtime.
    """

    def __init__(
        self,
        video_path: str,
        detection_cache_path: str,
        start_frame: int,
        end_frame: int,
        *,
        pre_roll_start: int | None = None,
        cache_provenance_params: Mapping[str, Any] | None = None,
        engine_factory: Callable[..., Any] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        if end_frame < start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        if pre_roll_start is not None and int(pre_roll_start) > int(start_frame):
            raise ValueError("pre_roll_start must not be after start_frame")
        self.video_path = str(video_path)
        self.detection_cache_path = str(detection_cache_path)
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.pre_roll_start = (
            self.start_frame if pre_roll_start is None else max(0, int(pre_roll_start))
        )
        # Keep cache provenance independent from the loop/scoring range. The
        # caller normally passes the full production params; when it does not,
        # ``run`` snapshots params before it replaces START/END.
        self._cache_provenance_params = (
            dict(cache_provenance_params)
            if cache_provenance_params is not None
            else None
        )
        self._engine_factory = engine_factory
        self._should_stop = should_stop or (lambda: False)

    def run(
        self, params: dict[str, Any], *, reverse: bool = False
    ) -> ProductionReplayResult:
        """Run one read-only forward or backward production replay."""

        n_tracks = int(params.get("MAX_TARGETS", 0))
        if n_tracks <= 0:
            raise ValueError("MAX_TARGETS must be positive")

        if self._engine_factory is None:
            from hydra_suite.core.tracking.worker import TrackingEngineCore

            engine_factory = TrackingEngineCore
        else:
            engine_factory = self._engine_factory

        cache_provenance_params = dict(
            self._cache_provenance_params
            if self._cache_provenance_params is not None
            else params
        )

        captured: dict[str, Any] = {
            "finished": False,
            "success": False,
            "trajectories": [],
            "error": None,
        }

        def _on_finished(success, _fps_list, trajectories) -> None:
            captured["finished"] = True
            captured["success"] = bool(success)
            captured["trajectories"] = trajectories or []

        engine_ref: dict[str, Any] = {}

        def _on_progress(_pct: int, _msg: str) -> None:
            if self._should_stop() and "engine" in engine_ref:
                engine_ref["engine"].stop()

        engine = engine_factory(
            self.video_path,
            csv_writer_thread=None,
            video_output_path=None,
            backward_mode=bool(reverse),
            detection_cache_path=self.detection_cache_path,
            # Preview mode deliberately disables production-only association
            # inputs such as live pose stores.  Read-only replay is a separate
            # concern, so exercise the full production path here.
            preview_mode=False,
            use_cached_detections=True,
            on_finished=_on_finished,
            on_progress=_on_progress,
            on_warning=lambda title, msg: captured.__setitem__(
                "error", f"{title}: {msg}"
            ),
            inference_cache_dir=cache_directory(self.detection_cache_path),
            cache_read_only_replay=True,
            inference_cache_provenance_params=cache_provenance_params,
            # The core checks this token at every frame. Retain the progress
            # callback below for compatibility with injected engines, but do
            # not make cancellation depend on its coarse cadence.
            should_stop=self._should_stop,
        )
        engine_ref["engine"] = engine
        replay_params = dict(params)
        replay_params.update(
            {
                "START_FRAME": self.pre_roll_start,
                "END_FRAME": self.end_frame,
                "VISUALIZATION_FREE_MODE": True,
                "ENABLE_VIDEO_OUTPUT": False,
                "ENABLE_PROFILING": False,
                "ENABLE_FRAME_PREFETCH": False,
            }
        )
        engine.set_parameters(replay_params)
        try:
            engine.run_tracking()
        except ConfidenceDensityCancelled:
            # A cancelled density build is never failed candidate evidence.
            # Let the optimizer discard the entire partial search result.
            raise
        except Exception as exc:  # production errors become a rejected candidate
            captured["error"] = str(exc)

        observations = getattr(engine, "replay_observations", None)
        if observations is None:
            observations = captured["trajectories"]
        frame_indices, positions = trajectories_to_positions(
            observations,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            n_tracks=n_tracks,
        )
        success = bool(captured["finished"] and captured["success"])
        error = captured["error"]
        if not captured["finished"] and error is None:
            error = "production replay ended without a completion result"
        return ProductionReplayResult(success, frame_indices, positions, error)
