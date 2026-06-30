"""Cache writer for the inference pipeline.

The pipeline has a SINGLE in-order consumer that calls ``write_detection`` /
``write_downstream`` in strictly ascending window order.  Because writes are
already produced in order, no reordering buffer is needed — a FIFO is enough.

Two modes:

* ``async_mode=False`` — writes happen inline on the calling (consumer) thread.
* ``async_mode=True`` — a single worker thread drains a FIFO ``queue.Queue``;
  ``write_detection`` / ``write_downstream`` enqueue a write item and return
  immediately, so disk I/O never stalls compute (spec §7).  The worker writes
  to the handles in FIFO order (== window order, since the consumer enqueues in
  order), so the on-disk cache layout is byte-identical to sync mode.

``close()`` drains + joins the worker (async) and surfaces any worker exception,
but does **not** close the caller-owned handles.  The runner is responsible for
closing handles in its own ``finally`` block (see ``runner.run_batch_pass``).
"""

from __future__ import annotations

import queue
import threading
from typing import Any


class CacheWriter:
    """FIFO cache writer supporting sync (inline) and async (threaded) modes.

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
        When *True* a single worker thread drains a FIFO queue of write items;
        ``write_detection`` / ``write_downstream`` enqueue and return without
        blocking on disk I/O.  When *False* writes happen inline.
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
        self._closed = False
        self._worker_error: BaseException | None = None

        if async_mode:
            self._queue: queue.Queue = queue.Queue()
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

    # --- public write API --------------------------------------------------

    def write_detection(self, frame_idx: int, obb_result: Any) -> None:
        """Write (or enqueue) a detection OBB result for ``frame_idx``.

        Called by the single in-order consumer in ascending window order, so no
        reordering is needed.  In async mode the write is offloaded to the
        worker thread (FIFO order == window order); in sync mode it happens
        inline on the caller's thread.
        """
        self._enqueue_or_write(
            {"kind": "detection", "frame_idx": frame_idx, "obb": obb_result}
        )

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
        """Write (or enqueue) downstream (non-detection) results for ``frame_idx``.

        Like ``write_detection`` these arrive in ascending window order from the
        single consumer; async mode offloads to the worker, sync writes inline.
        """
        self._enqueue_or_write(
            {
                "kind": "downstream",
                "frame_idx": frame_idx,
                "det_indices": det_indices,
                "headtail": headtail,
                "cnn_results": cnn_results,
                "pose": pose,
                "apriltag": apriltag,
            }
        )

    def flush(self) -> None:
        """Block until all enqueued writes have landed (async); no-op (sync)."""
        if self._async_mode:
            self._queue.join()
            if self._worker_error is not None:
                raise self._worker_error

    def close(self) -> None:
        """Drain + stop the worker thread (async).  Does NOT close handles."""
        if self._closed:
            return
        self._closed = True
        if self._async_mode:
            self._queue.join()
            self._queue.put(None)  # sentinel
            self._worker.join()
            if self._worker_error is not None:
                raise self._worker_error

    # --- internal ----------------------------------------------------------

    def _enqueue_or_write(self, item: dict) -> None:
        if self._closed:
            raise RuntimeError("CacheWriter is closed")
        if self._async_mode:
            self._queue.put(item)
        else:
            self._apply(item)

    def _apply(self, item: dict) -> None:
        """Execute a single write item against the handles."""
        kind = item["kind"]
        if kind == "detection":
            h = self._handles.get("detection")
            if h is not None:
                h.write_frame(item["frame_idx"], result=item["obb"])
        else:  # downstream
            self._write_to_handles(
                item["frame_idx"],
                det_indices=item["det_indices"],
                headtail=item["headtail"],
                cnn_results=item["cnn_results"],
                pose=item["pose"],
                apriltag=item["apriltag"],
            )

    def _write_to_handles(
        self,
        frame_idx: int,
        *,
        det_indices: Any,
        headtail: Any | None,
        cnn_results: list,
        pose: Any | None,
        apriltag: Any | None,
    ) -> None:
        """Single implementation of the downstream FrameResult→handle mapping."""
        import numpy as np

        h_ht = self._handles.get("headtail")
        if h_ht is not None and headtail is not None:
            h_ht.write_frame(
                frame_idx,
                det_indices=np.asarray(det_indices, dtype=np.int32),
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
                det_indices=np.asarray(det_indices, dtype=np.int32),
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
            try:
                if item is None:  # sentinel: time to exit
                    break
                self._apply(item)
            except BaseException as exc:  # noqa: BLE001
                if self._worker_error is None:
                    self._worker_error = exc
            finally:
                self._queue.task_done()
