"""SessionWorker — runs TrackingSessionCore.run_post_tracking on a QThread.

Mirrors the other gui/workers BaseWorker subclasses: extra Signals for the
progress/result/error/warning payloads, an `execute()` that builds the Qt-free
service and drives it, and a cooperative `stop()`/`_should_stop()` pair wired to
the service's should_stop callback.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


class SessionWorker(BaseWorker):
    progress_signal = Signal(int, str)  # (percent, message)
    finished_signal = Signal(object)  # SessionResult
    error_signal = Signal(str)
    warning_signal = Signal(str, str)  # (title, message)

    def __init__(
        self,
        *,
        video_path: str,
        config: Any,
        params: dict[str, Any],
        paths: dict[str, Any],
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._config = config
        self._params = params
        self._paths = paths
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    def execute(self) -> None:
        # Import lazily so this module stays importable without pulling the
        # full core graph at GUI import time (matches merge_worker).
        from hydra_suite.core.tracking.session import (
            SessionCallbacks,
            TrackingSessionCore,
        )

        try:
            callbacks = SessionCallbacks(
                progress=lambda pct, msg: self.progress_signal.emit(int(pct), str(msg)),
                status=lambda msg: self.status.emit(str(msg)),
                warning=lambda title, msg: self.warning_signal.emit(
                    str(title), str(msg)
                ),
                stage_changed=lambda name: self.status.emit(str(name)),
                should_stop=self._should_stop,
            )
            service = TrackingSessionCore(
                video_path=self._video_path,
                config=self._config,
                params=self._params,
                paths=self._paths,
                callbacks=callbacks,
            )
            result = service.run_post_tracking(None, None)
            if not self._should_stop():
                self.finished_signal.emit(result)
        except Exception as exc:  # noqa: BLE001 — surface as a Qt error signal
            logger.exception("SessionWorker failed during run_post_tracking")
            self.error_signal.emit(str(exc))
