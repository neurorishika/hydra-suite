"""
Frame prefetching utility for asynchronous video frame loading.

This module provides a thread-based frame prefetcher that reads video frames
in the background while the main tracking thread processes the current frame,
reducing I/O wait times and improving overall throughput.
"""

import logging
import math
import queue
import threading

import cv2

logger = logging.getLogger(__name__)


def _safe_cap_float(cap, prop_id, default=-1.0):
    """Read a numeric capture property safely for logging."""
    try:
        value = cap.get(prop_id)
    except Exception:
        return float(default)
    try:
        value = float(value)
    except Exception:
        return float(default)
    return value if math.isfinite(value) else float(default)


def _enqueue_with_retry(q, item, stop_event, timeout=0.1):
    """Put *item* into *q*, retrying until success or *stop_event* is set.

    Returns True if enqueued, False if stopped.
    """
    while not stop_event.is_set():
        try:
            q.put(item, timeout=timeout)
            return True
        except queue.Full:
            continue
    return False


def _enqueue_sentinel(q, stop_event, sentinel=None, timeout=0.1):
    """Put a sentinel value into *q*, retrying until success or stop."""
    _enqueue_with_retry(q, sentinel, stop_event, timeout)


class FramePrefetcher:
    """
    Asynchronous frame prefetcher for video processing.

    Uses a background thread to read frames ahead of time, reducing I/O blocking
    in the main processing loop. Maintains a small buffer of pre-read frames.

    Example:
        cap = cv2.VideoCapture("video.mp4")
        prefetcher = FramePrefetcher(cap, buffer_size=2)
        prefetcher.start()

        while True:
            ret, frame = prefetcher.read()
            if not ret:
                break
            # Process frame...

        prefetcher.stop()
    """

    def __init__(self, video_capture, buffer_size=2, read_timeout=30.0):
        """
        Initialize frame prefetcher.

        Args:
            video_capture: OpenCV VideoCapture object
            buffer_size (int): Number of frames to buffer (default: 2)
                              Higher = more memory, but better I/O tolerance
            read_timeout (float): Seconds to wait for a frame before declaring
                              a stall (default: 30).  Increase when the decode
                              backend shares resources with GPU inference.
        """
        self.cap = video_capture
        self.buffer_size = buffer_size
        self._read_timeout = read_timeout
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self._frames_read = 0  # frames successfully read by bg thread
        self._frames_consumed = 0  # frames returned to caller via read()
        self._last_read_ok = True  # last cap.read() return value
        self._stopped_reason = None  # why the bg thread exited
        self.stop_requested = threading.Event()
        self.exception = None  # Store exceptions from background thread
        self.thread = None
        self._started = False

    def start(self: object) -> object:
        """Start the background prefetching thread."""
        if self._started:
            logger.warning("FramePrefetcher already started")
            return

        self.stop_requested.clear()
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()
        self._started = True
        logger.debug("FramePrefetcher started with buffer_size=%d", self.buffer_size)

    def _prefetch_loop(self):
        """Background thread loop that reads frames ahead of time."""
        try:
            while not self.stop_requested.is_set():
                ret, frame = self.cap.read()
                self._last_read_ok = ret

                if not _enqueue_with_retry(
                    self.frame_queue, (ret, frame), self.stop_requested
                ):
                    self._stopped_reason = "stop_requested"
                    break

                if ret:
                    self._frames_read += 1
                else:
                    self._stopped_reason = "eof_or_read_error"
                    self._log_eof()
                    break

            if self._stopped_reason is None and self.stop_requested.is_set():
                self._stopped_reason = "stop_requested"

        except Exception as e:
            self.exception = e
            self._stopped_reason = f"exception: {e}"
            logger.error("Exception in prefetcher thread: %s", e, exc_info=True)
            try:
                self.frame_queue.put((False, None), block=False)
            except queue.Full:
                pass

    def _log_eof(self):
        """Log details when cap.read() returns False."""
        logger.warning(
            "Prefetcher bg-thread: cap.read() returned False after "
            "%d successful frames. cap.isOpened()=%s, "
            "CAP_PROP_POS_FRAMES=%.0f, CAP_PROP_FRAME_COUNT=%.0f",
            self._frames_read,
            self.cap.isOpened() if self.cap else "N/A",
            (
                _safe_cap_float(self.cap, 1)
                if self.cap and self.cap.isOpened()
                else -1.0
            ),
            (
                _safe_cap_float(self.cap, 7)
                if self.cap and self.cap.isOpened()
                else -1.0
            ),
        )

    def read(self: object) -> object:
        """
        Read the next frame (from prefetch buffer).

        Returns:
            tuple: (ret, frame) where ret is bool indicating success,
                   and frame is the numpy array or None
        """
        # Check for exceptions from background thread
        if self.exception is not None:
            raise RuntimeError("Prefetcher thread failed") from self.exception

        try:
            # Get frame from queue with timeout to detect stalls
            ret, frame = self.frame_queue.get(timeout=self._read_timeout)
            if ret:
                self._frames_consumed += 1
            return ret, frame
        except queue.Empty:
            # Queue is empty after timeout - likely stalled
            _thread_alive = self.thread.is_alive() if self.thread else False
            logger.error(
                "Frame prefetcher timeout (%.1fs) — no frames available. "
                "bg_thread_alive=%s, frames_read=%d, frames_consumed=%d, "
                "queue_size=%d/%d, stopped_reason=%s, last_read_ok=%s, "
                "cap_isOpened=%s, cap_pos=%.0f, exception=%s",
                self._read_timeout,
                _thread_alive,
                self._frames_read,
                self._frames_consumed,
                self.frame_queue.qsize(),
                self.buffer_size,
                self._stopped_reason,
                self._last_read_ok,
                self.cap.isOpened() if self.cap else "N/A",
                (
                    _safe_cap_float(self.cap, cv2.CAP_PROP_POS_FRAMES)
                    if self.cap and self.cap.isOpened()
                    else -1
                ),
                self.exception,
            )
            return False, None

    def stop(self: object) -> object:
        """Stop the prefetching thread and clean up."""
        if not self._started:
            return

        logger.debug("Stopping FramePrefetcher...")
        self.stop_requested.set()

        # Wait for thread to finish (with timeout)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Prefetcher thread did not stop cleanly")

        # Clear any remaining frames from queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self._started = False
        logger.debug("FramePrefetcher stopped")

    def __enter__(self):
        """Context manager support."""
        self.start()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager support."""
        self.stop()
        return False


class SparseFramePrefetcher:
    """
    Prefetcher for a pre-determined list of sparse frame indices.

    Reads frames in a background thread using seek-then-read, skipping the
    seek when frames are contiguous.  The main thread calls ``read()`` to
    get ``(frame_idx, ret, frame)`` tuples in the same order as the
    supplied *frame_indices* list.
    """

    def __init__(self, video_capture, frame_indices, buffer_size=4):
        self.cap = video_capture
        self.frame_indices = list(frame_indices)
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stop_requested = threading.Event()
        self.exception = None
        self.thread = None
        self._started = False

    def start(self):
        """Start the background thread that seeks to and reads each requested frame index."""
        if self._started:
            return
        self.stop_requested.clear()
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()
        self._started = True

    def _prefetch_loop(self):
        try:
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            for f in self.frame_indices:
                if self.stop_requested.is_set():
                    break
                if f != current_pos:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = self.cap.read()
                current_pos = f + 1
                _enqueue_with_retry(
                    self.frame_queue, (f, ret, frame), self.stop_requested
                )
            _enqueue_sentinel(self.frame_queue, self.stop_requested)
        except Exception as e:
            self.exception = e
            logger.error("SparseFramePrefetcher error: %s", e, exc_info=True)
            try:
                self.frame_queue.put(None, block=False)
            except queue.Full:
                pass

    def read(self):
        """Return ``(frame_idx, ret, frame)`` or *None* at end."""
        if self.exception is not None:
            raise RuntimeError("SparseFramePrefetcher failed") from self.exception
        try:
            item = self.frame_queue.get(timeout=10.0)
            return item
        except queue.Empty:
            return None

    def stop(self):
        """Signal the background thread to stop and drain the frame queue."""
        if not self._started:
            return
        self.stop_requested.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        return False


class SequentialScanPrefetcher:
    """
    Prefetcher that does a single sequential forward pass through a frame range.

    Instead of seeking to each needed frame individually (expensive with
    H.264/H.265 codecs), this reads every frame from ``min(frame_indices)``
    to ``max(frame_indices)`` sequentially and only queues frames that appear
    in the *frame_indices* set.  Frames not in the set are decoded but
    immediately discarded.

    This is dramatically faster than :class:`SparseFramePrefetcher` when the
    needed frames are spread across a large portion of the video range, because
    sequential ``cap.read()`` avoids the per-frame seek cost (~5–50 ms each on
    compressed codecs).
    """

    def __init__(self, video_capture, frame_indices, buffer_size=4):
        self.cap = video_capture
        self.needed = set(frame_indices)
        self.first_frame = min(frame_indices)
        self.last_frame = max(frame_indices)
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stop_requested = threading.Event()
        self.exception = None
        self.thread = None
        self._started = False

    def start(self):
        """Seek to the first needed frame and start the sequential scan thread."""
        if self._started:
            return
        self.stop_requested.clear()
        self.thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.thread.start()
        self._started = True

    def _scan_loop(self):
        try:
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_pos != self.first_frame:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.first_frame)

            for f in range(self.first_frame, self.last_frame + 1):
                if self.stop_requested.is_set():
                    break
                ret, frame = self.cap.read()
                if not ret:
                    break
                if f not in self.needed:
                    continue
                _enqueue_with_retry(
                    self.frame_queue, (f, ret, frame), self.stop_requested
                )
            _enqueue_sentinel(self.frame_queue, self.stop_requested)
        except Exception as e:
            self.exception = e
            logger.error("SequentialScanPrefetcher error: %s", e, exc_info=True)
            try:
                self.frame_queue.put(None, block=False)
            except queue.Full:
                pass

    def read(self):
        """Return ``(frame_idx, ret, frame)`` or *None* at end."""
        if self.exception is not None:
            raise RuntimeError("SequentialScanPrefetcher failed") from self.exception
        try:
            item = self.frame_queue.get(timeout=10.0)
            return item
        except queue.Empty:
            return None

    def stop(self):
        """Signal the scan thread to stop and drain any buffered frames."""
        if not self._started:
            return
        self.stop_requested.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        return False
