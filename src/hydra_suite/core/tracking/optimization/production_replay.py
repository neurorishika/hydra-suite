"""Read-only autotuner evaluation through the production tracking engine.

The Bayesian search uses a lightweight loop because it may evaluate hundreds of
parameter proposals.  A proposal is not safe to recommend on that evidence
alone: the lightweight loop intentionally omits some production-only features.
This module runs shortlisted candidates through :class:`TrackingEngineCore`
itself and returns only the observed positions that production would export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


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
    return path.parent if path.name == "detection.npz" else path


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
        engine_factory: Callable[..., Any] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        if end_frame < start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        self.video_path = str(video_path)
        self.detection_cache_path = str(detection_cache_path)
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
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
        )
        engine_ref["engine"] = engine
        replay_params = dict(params)
        replay_params.update(
            {
                "START_FRAME": self.start_frame,
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
        except Exception as exc:  # production errors become a rejected candidate
            captured["error"] = str(exc)

        frame_indices, positions = trajectories_to_positions(
            captured["trajectories"],
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            n_tracks=n_tracks,
        )
        success = bool(captured["finished"] and captured["success"])
        error = captured["error"]
        if not captured["finished"] and error is None:
            error = "production replay ended without a completion result"
        return ProductionReplayResult(success, frame_indices, positions, error)
