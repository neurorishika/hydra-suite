"""Background image loading so the next labeling images are already in memory.

On network storage every image read is a round trip.  Doing it on the UI thread
freezes labeling after each keypress.  :class:`ImagePrefetcher` reads upcoming
images on worker threads (as ``QImage``, which is thread-safe) and hands them to
the UI thread, which turns them into ``QPixmap`` objects for the viewer cache.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Set

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage

Loader = Callable[[str], Optional[QImage]]


class _Signals(QObject):
    loaded = Signal(str, object)  # path, QImage
    done = Signal(str)  # path (loaded or failed)


class _LoadTask(QRunnable):
    def __init__(self, path: str, loader: Loader, signals: _Signals) -> None:
        super().__init__()
        self._path, self._loader, self._signals = path, loader, signals

    def run(self) -> None:
        image = None
        try:
            image = self._loader(self._path)
        except Exception:
            image = None
        try:
            if image is not None and not image.isNull():
                self._signals.loaded.emit(self._path, image)
        except RuntimeError:  # receiver destroyed during shutdown
            return
        try:
            self._signals.done.emit(self._path)
        except RuntimeError:
            pass


class ImagePrefetcher(QObject):
    """Load images on a small thread pool and report them on the UI thread."""

    image_ready = Signal(str, object)  # path, QImage

    def __init__(self, parent: Optional[QObject] = None, threads: int = 4) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, int(threads)))
        self._pending: Set[str] = set()
        self._signals = _Signals(self)
        self._signals.loaded.connect(self.image_ready)
        self._signals.done.connect(self._pending.discard)

    def request(
        self,
        paths: Iterable[str],
        loader: Loader,
        skip: Callable[[str], bool] = lambda p: False,
    ) -> int:
        """Queue ``paths`` (nearest first); returns how many were queued."""
        queued = 0
        for path in paths:
            if path in self._pending or skip(path):
                continue
            self._pending.add(path)
            # Earlier paths are needed sooner: higher priority.
            self._pool.start(_LoadTask(path, loader, self._signals), 1000 - queued)
            queued += 1
        return queued

    def shutdown(self) -> None:
        self._pool.clear()
        self._pool.waitForDone(2000)
