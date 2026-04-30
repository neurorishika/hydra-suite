"""Reusable busy-indicator helpers for cross-kit GUI tasks.

These helpers eliminate UI freezes by running heavy work in a background
``BaseWorker`` thread and showing a window-modal ``QProgressDialog`` while
it runs. The callable receives ``set_status`` and ``set_progress`` callbacks
so it can stream progress updates back to the dialog.

Typical usage::

    def _scan(set_status, set_progress):
        set_status("Scanning…")
        ...
        return result

    self._busy_task = run_with_busy_dialog(
        self,
        _scan,
        title="Loading Project",
        message="Scanning database…",
        on_success=self._handle_loaded,
        on_error=lambda msg: QMessageBox.warning(self, "Load Failed", msg),
    )

The returned ``BusyTask`` keeps the worker and dialog alive for the duration
of the run; assign it to an instance attribute on the parent widget so the
worker is not garbage-collected mid-task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QProgressDialog, QWidget

from .workers import BaseWorker

StatusFn = Callable[[str], None]
ProgressFn = Callable[[int], None]
TaskFn = Callable[[StatusFn, ProgressFn], Any]


class CallableWorker(BaseWorker):
    """BaseWorker that runs an arbitrary callable.

    The callable receives ``(set_status, set_progress)`` callbacks and may
    return any value, which is relayed via the ``success`` signal.
    """

    success = Signal(object)

    def __init__(self, fn: TaskFn) -> None:
        super().__init__()
        self._fn = fn

    def execute(self) -> None:
        result = self._fn(self.status.emit, self.progress.emit)
        self.success.emit(result)


@dataclass
class BusyTask:
    """Handle returned by ``run_with_busy_dialog``.

    Holds references to the worker and dialog so callers can keep them alive
    (assign to a ``self._busy_task`` slot on the parent widget).
    """

    worker: CallableWorker
    dialog: QProgressDialog


def run_with_busy_dialog(
    parent: QWidget,
    fn: TaskFn,
    *,
    title: str,
    message: str,
    on_success: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
    on_finished: Optional[Callable[[], None]] = None,
    determinate: bool = False,
) -> BusyTask:
    """Run *fn* in a background thread with a modal progress dialog.

    Parameters
    ----------
    parent:
        Parent widget. The dialog uses window-modal modality so the rest of
        the app stays painted but inputs are blocked while work runs.
    fn:
        Callable invoked in the worker thread. Signature: ``fn(set_status,
        set_progress)`` where ``set_status(str)`` updates the dialog label
        and ``set_progress(int)`` updates the bar (0–100). Whatever the
        callable returns is forwarded to ``on_success``.
    title:
        Window title for the progress dialog.
    message:
        Initial message shown in the dialog.
    on_success / on_error / on_finished:
        Callbacks bound to the worker's signals. ``on_finished`` runs after
        the dialog is closed regardless of outcome.
    determinate:
        When ``True`` the progress bar shows percent (0–100) using values
        emitted via ``set_progress``. When ``False`` (default) the dialog
        is indeterminate.

    The returned :class:`BusyTask` must be retained by the caller until the
    work completes (otherwise Python may GC the worker mid-run).
    """
    if determinate:
        dialog = QProgressDialog(message, None, 0, 100, parent)
    else:
        dialog = QProgressDialog(message, None, 0, 0, parent)
    dialog.setWindowTitle(title)
    dialog.setCancelButton(None)
    dialog.setMinimumDuration(0)
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.show()

    worker = CallableWorker(fn)
    worker.status.connect(dialog.setLabelText)
    if determinate:
        worker.progress.connect(dialog.setValue)

    if on_success is not None:
        worker.success.connect(on_success)
    if on_error is not None:
        worker.error.connect(on_error)

    def _cleanup() -> None:
        dialog.close()
        if on_finished is not None:
            on_finished()

    worker.finished.connect(_cleanup)
    worker.start()
    return BusyTask(worker=worker, dialog=dialog)


class BusyTaskError(RuntimeError):
    """Raised by ``run_blocking_with_busy_dialog`` when the worker errors."""


def run_blocking_with_busy_dialog(
    parent: QWidget,
    fn: TaskFn,
    *,
    title: str,
    message: str,
    determinate: bool = False,
) -> Any:
    """Run *fn* in a background thread while blocking the caller.

    Unlike :func:`run_with_busy_dialog` this variant blocks the caller by
    spinning a nested Qt event loop, so the calling code can continue
    linearly once the work completes. The UI keeps painting and other
    windows stay alive while the work runs.

    Returns whatever *fn* returns. Raises :class:`BusyTaskError` if *fn*
    raises an exception inside the worker thread.
    """
    from PySide6.QtCore import QEventLoop

    if determinate:
        dialog = QProgressDialog(message, None, 0, 100, parent)
    else:
        dialog = QProgressDialog(message, None, 0, 0, parent)
    dialog.setWindowTitle(title)
    dialog.setCancelButton(None)
    dialog.setMinimumDuration(0)
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)

    worker = CallableWorker(fn)
    worker.status.connect(dialog.setLabelText)
    if determinate:
        worker.progress.connect(dialog.setValue)

    result_box: list[Any] = []
    error_box: list[str] = []
    worker.success.connect(result_box.append)
    worker.error.connect(error_box.append)

    loop = QEventLoop()
    worker.finished.connect(dialog.close)
    worker.finished.connect(loop.quit)

    worker.start()
    dialog.show()
    run_loop = loop.exec
    run_loop()
    worker.wait()

    if error_box:
        raise BusyTaskError(error_box[0])
    return result_box[0] if result_box else None
