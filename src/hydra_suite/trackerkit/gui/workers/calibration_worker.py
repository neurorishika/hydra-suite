"""Background worker for user-triggered inference calibration."""

from __future__ import annotations

from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker


class CalibrationWorker(BaseWorker):
    """Run one calibration search off the GUI thread.

    Builds the exact same ``AutotuneContext`` the CLI's ``calibrate_cli``
    builds (same ``build_autotune_context`` call, same
    ``derive_context_inputs``-derived inputs) so the profile this worker
    persists is found by a later ``track`` run's ``lookup``.
    """

    completed = Signal(dict)

    def __init__(
        self,
        params: dict,
        config,
        *,
        video_path: str,
        budget_seconds: float,
        frame_width: int,
        frame_height: int,
        start_frame: int,
        end_frame: int,
        realtime: bool,
        cache_dir=None,
        use_cached_detections: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._params = dict(params)
        self._config = config
        self._video_path = video_path
        self._budget_seconds = float(budget_seconds)
        self._frame_width = int(frame_width)
        self._frame_height = int(frame_height)
        self._start_frame = int(start_frame)
        self._end_frame = int(end_frame)
        self._realtime = bool(realtime)
        self._cache_dir = cache_dir
        self._use_cached_detections = bool(use_cached_detections)
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation; observed by the running search via should_cancel."""
        self._cancelled = True

    def execute(self) -> None:
        from hydra_suite.core.inference.autotune import session

        ctx = session.build_autotune_context(
            self._config,
            self._params,
            video_path=self._video_path,
            frame_width=self._frame_width,
            frame_height=self._frame_height,
            start_frame=self._start_frame,
            end_frame=self._end_frame,
            # MUST match the CLI: run_calibrate_cli passes inputs.realtime.
            # realtime is part of the workload identity, so hardcoding False
            # here would persist the profile under a key a realtime-mode
            # production run never looks up.
            realtime=self._realtime,
            cache_read_only_replay=False,
            cache_dir=self._cache_dir,
            use_cached_detections=self._use_cached_detections,
            should_cancel=lambda: self._cancelled,
            status_callback=self.status.emit,
        )
        _effective, overlay, _result = session.calibrate(
            ctx, budget_seconds=self._budget_seconds
        )
        if overlay is None:
            raise RuntimeError("Calibration produced no result.")
        self.completed.emit(
            {
                "status": overlay.status,
                "reason": overlay.reason,
                "profile_id": overlay.profile_id,
                "effective": overlay.effective.to_dict(),
            }
        )
