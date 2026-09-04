"""BaseWorker — standard QThread base class for all background tasks."""

from PySide6.QtCore import QThread, Signal

MAX_WORKER_TERMINAL_MESSAGE_BYTES = 32 * 1024
_TRUNCATED_MESSAGE_SUFFIX = b"\n[message truncated]"


def _safe_exception_text(error: BaseException) -> str:
    """Describe an exception without invoking arbitrary ``__str__`` code."""

    pieces = [type(error).__name__]
    safe_args: list[str] = []
    for arg in error.args[:8]:
        if type(arg) is str:
            safe_args.append(arg[:MAX_WORKER_TERMINAL_MESSAGE_BYTES])
        elif type(arg) is int:
            safe_args.append(repr(arg) if arg.bit_length() <= 8192 else "<large int>")
        elif type(arg) in (float, bool, type(None)):
            safe_args.append(repr(arg))
        else:
            safe_args.append(f"<{type(arg).__name__}>")
    if safe_args:
        pieces.extend((": ", ", ".join(safe_args)))
    return "".join(pieces)


def bounded_worker_message(value: object) -> str:
    """Return a UTF-8-safe, fixed-size Qt signal payload."""

    if isinstance(value, BaseException):
        text = _safe_exception_text(value)
    elif type(value) is str:
        text = value
    else:
        text = f"<{type(value).__name__}>"
    candidate = text[:MAX_WORKER_TERMINAL_MESSAGE_BYTES]
    encoded = candidate.encode("utf-8", errors="replace")
    truncated = (
        len(text) > len(candidate) or len(encoded) > MAX_WORKER_TERMINAL_MESSAGE_BYTES
    )
    if not truncated:
        return candidate
    retained = encoded[
        : MAX_WORKER_TERMINAL_MESSAGE_BYTES - len(_TRUNCATED_MESSAGE_SUFFIX)
    ].decode("utf-8", errors="ignore")
    return retained + _TRUNCATED_MESSAGE_SUFFIX.decode("ascii")


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
