"""Frame-ordered cache writer for the inference pipeline.

Accepts ``FrameResult`` objects submitted in any order and writes them to the
per-type ``CacheHandle``s in strictly ascending frame-index order.  This is
needed for depth>1 pipelined runs where windows can complete out of order; for
depth=1 (synchronous) the ordering guarantee is free because frames already
arrive in order.

``close()`` stops the worker thread (async mode) and flushes the buffer, but
does **not** close the caller-owned handles.  The runner is responsible for
closing handles in its own ``finally`` block (see ``runner.run_batch_pass``).
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..result import FrameResult


class CacheWriter:
    """Ordered-buffer cache writer supporting sync and async (threaded) modes.

    Parameters
    ----------
    handles:
        Mapping from cache-type label to ``CacheHandle`` instance.  The writer
        understands the following keys: ``"detection"``, ``"headtail"``,
        ``"pose"``, ``"apriltag"``, and any label prefixed with ``"cnn_"``
        (e.g. ``"cnn_identity"``).
    cnn_configs:
        CNN phase config list in phase order.  Used to match CNN results to
        the correct per-phase handle (keyed by ``"cnn_<label>"``).
    async_mode:
        When *True* a single worker thread drains the ordered buffer; all
        ``write_frame`` calls on the underlying handles happen in that thread.
        When *False* draining happens inline in ``submit``/``flush``.
    start_frame:
        The first frame index expected.  Defaults to 0.  The cursor advances
        monotonically; if a submitted frame has a lower index than the cursor
        it is silently ignored (already written).
    """

    def __init__(
        self,
        handles: dict[str, Any],
        cnn_configs: list,
        *,
        async_mode: bool,
        start_frame: int = 0,
    ) -> None:
        self._handles = handles
        self._cnn_configs = cnn_configs
        self._async_mode = async_mode
        self._next_expected: int = start_frame
        self._buffer: dict[int, FrameResult] = {}
        self._closed = False

        if async_mode:
            self._queue: queue.Queue = queue.Queue()
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

    # --- public API --------------------------------------------------------

    def submit(self, frame_result: FrameResult) -> None:
        """Buffer ``frame_result`` and drain any contiguous-ready frames."""
        if self._closed:
            raise RuntimeError("CacheWriter is closed")
        if self._async_mode:
            self._queue.put(frame_result)
        else:
            self._buffer[frame_result.frame_idx] = frame_result
            self._drain_sync()

    def flush(self) -> None:
        """Drain everything currently buffered (in order)."""
        if self._async_mode:
            self._queue.join()
        else:
            self._drain_all_sync()

    def close(self) -> None:
        """Flush + stop the worker thread (async).  Does NOT close handles."""
        if self._closed:
            return
        self._closed = True
        if self._async_mode:
            self._queue.join()
            self._queue.put(None)  # sentinel
            self._worker.join()
        else:
            self._drain_all_sync()

    # --- write helpers used by both paths ----------------------------------

    def write_detection(self, frame_idx: int, obb_result: Any) -> None:
        """Write an OBB result directly (bypasses the ordered buffer).

        Detection writes come from the pipeline one-at-a-time in ascending
        frame order within a window, so no reordering is needed.
        """
        h = self._handles.get("detection")
        if h is not None:
            h.write_frame(frame_idx, result=obb_result)

    def write_downstream(
        self,
        frame_idx: int,
        *,
        det_indices: Any,
        headtail: Any | None,
        cnn_results: list,
        pose: Any | None,
        apriltag: Any | None,
    ) -> None:
        """Write downstream (non-detection) results directly.

        Like ``write_detection``, these come in ascending order within a
        window, so no reordering buffer is needed.
        """
        self._write_downstream_to_handles(
            frame_idx,
            det_indices=det_indices,
            headtail=headtail,
            cnn_results=cnn_results,
            pose=pose,
            apriltag=apriltag,
        )

    # --- internal ordering logic -------------------------------------------

    def _drain_sync(self) -> None:
        """Emit all frames >= next_expected that are contiguous."""
        while self._next_expected in self._buffer:
            fr = self._buffer.pop(self._next_expected)
            self._write_frame_result(fr)
            self._next_expected += 1

    def _drain_all_sync(self) -> None:
        """Emit all buffered frames in sorted order (ignoring gaps)."""
        for idx in sorted(self._buffer):
            self._write_frame_result(self._buffer[idx])
        self._buffer.clear()
        # Advance cursor past everything we just emitted.
        if self._buffer:
            self._next_expected = max(self._buffer) + 1

    def _write_frame_result(self, fr: FrameResult) -> None:
        """Map a FrameResult to handle write_frame calls."""
        h_det = self._handles.get("detection")
        if h_det is not None:
            h_det.write_frame(fr.frame_idx, result=fr.obb)

        det_indices = fr.filtered_indices

        h_ht = self._handles.get("headtail")
        if h_ht is not None and fr.headtail is not None:
            import numpy as np

            h_ht.write_frame(
                fr.frame_idx,
                det_indices=np.array(det_indices, dtype=np.int32),
                heading_hints=fr.headtail.heading_hints,
                heading_confidences=fr.headtail.heading_confidences,
                directed_mask=fr.headtail.directed_mask,
            )

        for cfg, cnn_result in zip(self._cnn_configs, fr.cnn):
            h_cnn = self._handles.get(f"cnn_{cfg.label}")
            if h_cnn is not None and cnn_result is not None:
                h_cnn.write_frame(fr.frame_idx, predictions=cnn_result.predictions)

        h_pose = self._handles.get("pose")
        if h_pose is not None and fr.pose is not None:
            import numpy as np

            h_pose.write_frame(
                fr.frame_idx,
                det_indices=np.array(det_indices, dtype=np.int32),
                keypoints=fr.pose.keypoints,
                valid_mask=fr.pose.valid_mask,
            )

        h_at = self._handles.get("apriltag")
        if h_at is not None and fr.apriltag is not None:
            h_at.write_frame(fr.frame_idx, result=fr.apriltag)

    def _write_downstream_to_handles(
        self,
        frame_idx: int,
        *,
        det_indices: Any,
        headtail: Any | None,
        cnn_results: list,
        pose: Any | None,
        apriltag: Any | None,
    ) -> None:
        h_ht = self._handles.get("headtail")
        if h_ht is not None and headtail is not None:
            h_ht.write_frame(
                frame_idx,
                det_indices=det_indices,
                heading_hints=headtail.heading_hints,
                heading_confidences=headtail.heading_confidences,
                directed_mask=headtail.directed_mask,
            )

        for cfg, cnn_result in zip(self._cnn_configs, cnn_results):
            h_cnn = self._handles.get(f"cnn_{cfg.label}")
            if h_cnn is not None and cnn_result is not None:
                h_cnn.write_frame(frame_idx, predictions=cnn_result.predictions)

        h_pose = self._handles.get("pose")
        if h_pose is not None and pose is not None:
            h_pose.write_frame(
                frame_idx,
                det_indices=det_indices,
                keypoints=pose.keypoints,
                valid_mask=pose.valid_mask,
            )

        h_at = self._handles.get("apriltag")
        if h_at is not None and apriltag is not None:
            h_at.write_frame(frame_idx, result=apriltag)

    # --- async worker thread -----------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:  # sentinel: time to exit
                self._queue.task_done()
                break
            self._buffer[item.frame_idx] = item
            self._drain_sync()
            self._queue.task_done()
