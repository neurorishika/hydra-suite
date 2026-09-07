"""Qt bridge for the Qt-free batch fan-out scheduler.

The scheduler's callbacks arrive on more than one thread (``job_progress`` and
``job_log`` on each job's own log-reader thread, ``job_started`` and
``job_finished`` on the thread running :func:`run_batch_fanout`). This class
therefore does exactly one thing in every callback: emit a Qt signal with a
bounded, primitive payload. Because the worker object itself lives in the GUI
thread, Qt marshals every one of those emissions back to the GUI thread as a
queued connection -- so no callback ever touches a widget directly.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import Signal

from hydra_suite.runtime.cuda_devices import CudaDevice
from hydra_suite.trackerkit.batch_fanout import (
    FanoutJobResult,
    FanoutOptions,
    LiveChildRegistry,
    run_batch_fanout,
)
from hydra_suite.trackerkit.batch_plan import BatchJobSpec
from hydra_suite.widgets.workers import BaseWorker, bounded_worker_message

# A full UUID is unreadably long in a table cell; the prefix is still unique
# in practice and is what the per-job log header prints.
_GPU_LABEL_CHARS = 12


def gpu_label(gpu: Optional[CudaDevice]) -> str:
    """Short, table-friendly name for a job's pinned GPU (``"-"`` when none)."""
    if gpu is None:
        return "-"
    return str(gpu.uuid)[:_GPU_LABEL_CHARS]


class BatchFanoutWorker(BaseWorker):
    """Runs ``run_batch_fanout`` on a QThread and re-emits its events."""

    job_started = Signal(int, str, str, str)  # index, video, log path, gpu label
    job_progress = Signal(int, int, str)  # index, percent, message
    job_log = Signal(int, str)  # index, line
    job_finished = Signal(int, bool, str)  # index, success, summary-or-error
    fanout_finished = Signal(object)  # FanoutResult

    def __init__(
        self, specs: Sequence[BatchJobSpec], options: FanoutOptions, parent=None
    ) -> None:
        super().__init__(parent)
        self._specs = list(specs)
        self._options = options
        self._stop = threading.Event()
        self._children = LiveChildRegistry()
        self.result = None

    @property
    def specs(self) -> list[BatchJobSpec]:
        """The jobs this fan-out was asked to run (read-only)."""
        return list(self._specs)

    @property
    def options(self) -> FanoutOptions:
        """The scheduler options, including the stop grace periods."""
        return self._options

    # --- cancellation -----------------------------------------------------
    def cancel(self) -> None:
        self._stop.set()

    def stop(self) -> None:
        """Alias for :meth:`cancel` -- ``_request_qthread_stop`` calls ``stop()``."""
        self.cancel()

    def should_stop(self) -> bool:
        return self._stop.is_set() or self.isInterruptionRequested()

    def kill_children_now(self) -> list[int]:
        """SIGKILL every live child's process group; return the pids signalled.

        The escape hatch for a caller that has already spent the cooperative
        stop budget and must proceed anyway -- closing the window while these
        processes live would orphan them, still holding their GPUs. Callable
        from the GUI thread while ``execute`` runs on the worker thread: the
        registry is the only shared state and it is lock-guarded.
        """
        return self._children.kill_all()

    # --- FanoutEvents (called from scheduler / log-reader threads) ---------
    def job_started_cb(
        self, spec: BatchJobSpec, gpu: Optional[CudaDevice], log_path: Path
    ) -> None:
        self.job_started.emit(
            int(spec.index), str(spec.video_path), str(log_path), gpu_label(gpu)
        )

    def job_progress_cb(self, spec: BatchJobSpec, percent: int, message: str) -> None:
        self.job_progress.emit(
            int(spec.index), int(percent), bounded_worker_message(message)
        )

    def job_log_cb(self, spec: BatchJobSpec, line: str) -> None:
        self.job_log.emit(int(spec.index), bounded_worker_message(line))

    def job_finished_cb(self, result: FanoutJobResult) -> None:
        text = (
            " | ".join(result.summary_lines)
            if result.success
            else (result.error or "failed")
        )
        self.job_finished.emit(
            int(result.spec.index), bool(result.success), bounded_worker_message(text)
        )

    # --- BaseWorker -------------------------------------------------------
    def execute(self) -> None:
        events = _EventAdapter(self)
        self.result = run_batch_fanout(
            self._specs,
            self._options,
            events=events,
            should_stop=self.should_stop,
            child_registry=self._children,
        )
        self.fanout_finished.emit(self.result)


class _EventAdapter:
    """``FanoutEvents`` implementation that forwards to the worker's signals."""

    def __init__(self, worker: BatchFanoutWorker) -> None:
        self._w = worker

    def job_started(self, spec, gpu, log_path) -> None:
        self._w.job_started_cb(spec, gpu, log_path)

    def job_progress(self, spec, percent, message) -> None:
        self._w.job_progress_cb(spec, percent, message)

    def job_log(self, spec, line) -> None:
        self._w.job_log_cb(spec, line)

    def job_finished(self, result) -> None:
        self._w.job_finished_cb(result)
