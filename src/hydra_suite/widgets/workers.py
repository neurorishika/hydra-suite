"""BaseWorker — standard QThread base class for all background tasks."""

import logging

from PySide6.QtCore import QCoreApplication, QThread, Signal

from hydra_suite.runtime.safe_text import MAX_TERMINAL_TEXT_BYTES, bounded_terminal_text

MAX_WORKER_TERMINAL_MESSAGE_BYTES = MAX_TERMINAL_TEXT_BYTES

logger = logging.getLogger(__name__)


def bounded_worker_message(value: object) -> str:
    """Return a UTF-8-safe, fixed-size Qt signal payload."""

    return bounded_terminal_text(value)


class BaseWorker(QThread):
    """Base class for all background task workers.

    Subclasses implement ``execute()`` only.  ``run()`` is owned by this
    class and guarantees:
    - Unhandled exceptions in ``execute()`` emit ``error`` instead of
      crashing the thread silently.
    - Qt automatically emits the inherited ``QThread.finished`` signal
      when ``run()`` returns, whether execution succeeded or failed.
      Do not redefine ``finished`` in subclasses — it would shadow
      Qt's mechanism and cause missed or double emissions.

    Standard signals
    ----------------
    progress(int)  — 0–100 completion percentage
    status(str)    — human-readable status update
    error(str)     — error message; emitted only on exception
    finished()     — inherited from QThread; emitted automatically by Qt
                     when run() returns (success or failure)
    """

    progress: Signal = Signal(int)
    status: Signal = Signal(str)
    error: Signal = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Keep the exact exception object available to the durable worker
        # owner. Some failures carry recovery state that must not be reduced
        # to the human-readable error signal.
        self.failure_exception: Exception | None = None

    def run(self) -> None:
        """Wrap execute() in error handling, emitting the error signal on failure."""
        self.failure_exception = None
        try:
            self.execute()
        except Exception as exc:  # noqa: BLE001
            self.failure_exception = exc
            self.error.emit(bounded_worker_message(exc))

    def execute(self) -> None:
        """Override in subclasses with the actual work."""
        raise NotImplementedError(f"{type(self).__name__} must implement execute()")


class AbandonableWorker(BaseWorker):
    """BaseWorker whose owner may go away while it is still running.

    Use for advisory background work (dataset analysis, previews) that a
    dialog starts and may close on before it finishes.  The thread is parented
    to the application rather than the dialog and deletes itself once
    finished, so closing (or garbage-collecting) the owner never destroys a
    running ``QThread`` -- which aborts the process.  When the application
    quits, every still-running instance is cancelled and joined.

    Subclasses poll ``is_cancelled()`` between units of work.  Owners abandon a
    run by calling ``cancel()`` and dropping their reference; their result
    slots should ignore signals whose ``sender()`` is no longer the current
    worker.
    """

    def __init__(self) -> None:
        app = QCoreApplication.instance()
        super().__init__(app)
        self._cancel = False
        self.finished.connect(self.deleteLater)
        if app is not None:
            _install_abandonable_quit_hook(app)

    def cancel(self) -> None:
        self._cancel = True

    def is_cancelled(self) -> bool:
        return bool(self._cancel)


_QUIT_HOOK_APPS: set[int] = set()


def _install_abandonable_quit_hook(app: QCoreApplication) -> None:
    if id(app) in _QUIT_HOOK_APPS:
        return
    _QUIT_HOOK_APPS.add(id(app))
    app.aboutToQuit.connect(join_abandonable_workers)


def join_abandonable_workers() -> None:
    """Cancel and wait for every running AbandonableWorker (called on quit)."""
    app = QCoreApplication.instance()
    if app is None:
        return
    running = [w for w in app.findChildren(AbandonableWorker) if w.isRunning()]
    if not running:
        return
    for worker in running:
        worker.cancel()
    logger.info(
        "Waiting for %d background task(s) to stop before quitting.", len(running)
    )
    # Unbounded: cancellation is polled per file, so this waits for at most
    # one in-flight read per worker.  A timeout would not avoid the abort that
    # destroying a running QThread causes; it would only delay it.
    for worker in running:
        worker.wait()
