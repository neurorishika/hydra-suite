"""Live log tail for a ``QMainWindow`` status bar.

Installs a root-logger handler that keeps the newest record's message, and a
``QLabel`` in the status bar's *normal* area that mirrors it. Two properties
make this safe for a chatty pipeline:

* **Thread safety without signal floods.** Log records arrive from worker
  threads at arbitrary rates. The handler only stores the latest message under
  a lock (a constant-cost, non-blocking write); a GUI-thread ``QTimer`` polls
  it a few times a second and repaints only on change. Emitting a queued signal
  per record would post thousands of events per second into the GUI event loop.
* **Transient messages still win.** The label sits in the status bar's normal
  area, which Qt hides for as long as a ``showMessage()`` temporary message is
  displayed. Existing toasts therefore override the tail and it reappears by
  itself when they expire.
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel, QMainWindow, QSizePolicy, QStatusBar

#: Poll cadence for the GUI-side refresh. Fast enough to read as live, slow
#: enough that a per-frame logger cannot starve the event loop.
_REFRESH_MS = 200

#: Level -> prefix, so severity is visible without colouring the whole bar.
_LEVEL_PREFIX = {
    logging.WARNING: "⚠ ",
    logging.ERROR: "✖ ",
    logging.CRITICAL: "✖ ",
}


class _LatestRecordHandler(logging.Handler):
    """Logging handler that remembers only the most recent formatted message."""

    def __init__(self) -> None:
        super().__init__()
        self._lock_latest = threading.Lock()
        self._latest = ""

    def emit(self, record: logging.LogRecord) -> None:
        """Store *record* as the latest message. Never raises, never logs."""
        try:
            text = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad log call must not break logging
            return
        # Collapse to one line: the status bar is a single line of text and a
        # traceback's first line is the useful part.
        text = text.strip().splitlines()[0] if text.strip() else ""
        if not text:
            return
        text = _LEVEL_PREFIX.get(record.levelno, "") + text
        with self._lock_latest:
            self._latest = text

    def latest(self) -> str:
        """Return the newest message seen (empty string if none yet)."""
        with self._lock_latest:
            return self._latest


class StatusLogTail:
    """Mirror the newest log line into *window*'s status bar.

    Args:
        window: The main window whose ``statusBar()`` gets the label.
        level: Minimum level to display.
        logger: Logger to attach to (defaults to the root logger).
    """

    def __init__(
        self,
        window: QMainWindow,
        level: int = logging.INFO,
        logger: logging.Logger | None = None,
    ) -> None:
        self._window = window
        self._logger = logger if logger is not None else logging.getLogger()

        self._label = QLabel("")
        self._label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # Ignored width: the label takes whatever the bar gives it and elides,
        # rather than a long log line forcing the window wider.
        self._label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        status_bar: QStatusBar = window.statusBar()
        # Stretch 1 so the label owns the bar's width and elision has room.
        status_bar.addWidget(self._label, 1)

        self._handler = _LatestRecordHandler()
        self._handler.setLevel(level)
        self._logger.addHandler(self._handler)

        self._shown = ""
        self._timer = QTimer(window)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _refresh(self) -> None:
        """Repaint the label when its displayed text would change.

        Compares the *elided* result, so a window resize re-elides the same
        message and an unchanged message costs one string comparison.
        """
        text = self._handler.latest()
        elided = self._elided(text)
        if elided == self._shown:
            return
        self._shown = elided
        self._label.setToolTip(text)
        self._label.setText(elided)

    def _elided(self, text: str) -> str:
        """Elide *text* to the label's current width."""
        width = self._label.width()
        if width <= 0:
            return text
        return QFontMetrics(self._label.font()).elidedText(text, Qt.ElideRight, width)

    def detach(self) -> None:
        """Stop polling and remove the handler. Safe to call more than once."""
        self._timer.stop()
        try:
            self._logger.removeHandler(self._handler)
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass
        self._handler.close()
