"""Cache writer for the inference pipeline.

The pipeline has a SINGLE in-order consumer that calls ``write_detection`` /
``write_downstream`` in strictly ascending window order.  Because writes are
already produced in order, no reordering buffer is needed — a FIFO is enough.

Two modes:

* ``async_mode=False`` — writes happen inline on the calling (consumer) thread.
* ``async_mode=True`` — a single worker thread drains a byte-bounded FIFO;
  producers apply backpressure before queued payloads exceed the configured
  budget. The worker preserves enqueue order, so sync and async cache layouts
  remain equivalent.

``close()`` drains + joins the worker (async) and surfaces any worker exception,
but does **not** close the caller-owned handles.  The runner is responsible for
closing handles in its own ``finally`` block (see ``runner.run_batch_pass``).
"""

from __future__ import annotations

import collections
import sys
import threading
import time
from typing import Any

from hydra_suite.utils import profiling_names as N
from hydra_suite.utils.profiling import bind_target, span


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
        When *True* a single worker thread drains a byte-bounded FIFO. When
        *False* writes happen inline.
    max_queue_bytes:
        Maximum retained bytes across queued and currently-writing payloads.
        A single larger payload is rejected instead of silently defeating the
        host-memory bound.
    """

    def __init__(
        self,
        handles: dict[str, Any],
        cnn_configs: list,
        *,
        async_mode: bool,
        start_frame: int = 0,
        max_queue_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self._handles = handles
        self._cnn_configs = cnn_configs
        self._async_mode = async_mode
        self._closed = False
        self._worker_error: BaseException | None = None
        self._error_reported = False

        if int(max_queue_bytes) < 1:
            raise ValueError("max_queue_bytes must be >= 1")
        self.max_retained_bytes = int(max_queue_bytes)
        bounded_handles = [
            handle
            for handle in handles.values()
            if callable(getattr(handle, "set_buffer_limit", None))
        ]
        handle_budget = (
            self.max_retained_bytes // 2 if async_mode else self.max_retained_bytes
        )
        if bounded_handles:
            per_handle = handle_budget // len(bounded_handles)
            if per_handle < 1:
                raise ValueError(
                    "max_queue_bytes is too small for cache handle buffers"
                )
            for handle in bounded_handles:
                handle.set_buffer_limit(per_handle)
            assigned_handle_bytes = per_handle * len(bounded_handles)
        else:
            assigned_handle_bytes = 0
        self._max_queue_bytes = self.max_retained_bytes - assigned_handle_bytes

        if async_mode:
            self._condition = threading.Condition()
            self._items: collections.deque[tuple[dict, int]] = collections.deque()
            # Includes the item currently being written. Keeping active bytes
            # reserved prevents a fast producer from filling a second complete
            # budget while the worker still owns the first.
            self._inflight_bytes = 0
            self._worker_active = False
            self._stop_requested = False
            self._worker = threading.Thread(
                target=bind_target(self._worker_loop), daemon=True
            )
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

    def flush(self, timeout: float = 30.0) -> None:
        """Block until all enqueued writes have landed (async); no-op (sync)."""
        if self._async_mode:
            with self._condition:
                completed = self._condition.wait_for(
                    lambda: self._worker_error is not None
                    or (not self._items and not self._worker_active),
                    timeout=max(0.0, float(timeout)),
                )
                if not completed:
                    raise TimeoutError("cache writer flush timed out")
                self._raise_worker_error_once_locked()

    def close(self, timeout: float = 30.0) -> None:
        """Drain + stop the worker thread (async).  Does NOT close handles."""
        if self._closed:
            if self.worker_alive:
                raise TimeoutError("cache writer worker is still running")
            return
        self._closed = True
        if self._async_mode:
            deadline = time.monotonic() + max(0.0, float(timeout))
            pending_error: BaseException | None = None
            try:
                self.flush(timeout=max(0.0, deadline - time.monotonic()))
            except BaseException as exc:  # noqa: BLE001,B036 - re-raised after join
                pending_error = exc
            with self._condition:
                self._stop_requested = True
                self._condition.notify_all()
            self._worker.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._worker.is_alive():
                raise TimeoutError(
                    "cache writer worker did not stop before the close deadline"
                ) from pending_error
            if pending_error is not None:
                raise pending_error
            with self._condition:
                self._raise_worker_error_once_locked()

    # --- internal ----------------------------------------------------------

    def _enqueue_or_write(self, item: dict) -> None:
        with span(N.ENQUEUE):
            if self._closed:
                raise RuntimeError("CacheWriter is closed")
            if self._async_mode:
                item_bytes = _payload_bytes(item)
                if item_bytes > self._max_queue_bytes:
                    raise ValueError(
                        "cache write payload exceeds max_queue_bytes: "
                        f"{item_bytes} > {self._max_queue_bytes}"
                    )
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._worker_error is not None
                        or self._closed
                        or self._inflight_bytes + item_bytes <= self._max_queue_bytes
                    )
                    self._raise_worker_error_once_locked(reject_after_failure=True)
                    if self._closed:
                        raise RuntimeError("CacheWriter is closed")
                    self._items.append((item, item_bytes))
                    self._inflight_bytes += item_bytes
                    self._condition.notify_all()
            else:
                item_bytes = _payload_bytes(item)
                if item_bytes > self.max_retained_bytes:
                    raise ValueError(
                        "cache write payload exceeds max_queue_bytes: "
                        f"{item_bytes} > {self.max_retained_bytes}"
                    )
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
        if h_ht is not None:
            count = len(det_indices)
            h_ht.write_frame(
                frame_idx,
                det_indices=np.asarray(det_indices, dtype=np.int32),
                heading_hints=(
                    headtail.heading_hints
                    if headtail is not None
                    else np.full(count, np.nan, dtype=np.float32)
                ),
                heading_confidences=(
                    headtail.heading_confidences
                    if headtail is not None
                    else np.zeros(count, dtype=np.float32)
                ),
                directed_mask=(
                    headtail.directed_mask
                    if headtail is not None
                    else np.zeros(count, dtype=np.uint8)
                ),
            )

        cnn_by_label = {
            cfg.label: result
            for cfg, result in zip(self._cnn_configs, cnn_results)
            if result is not None
        }
        for cfg in self._cnn_configs:
            h_cnn = self._handles.get(f"cnn_{cfg.label}")
            if h_cnn is not None:
                cnn_result = cnn_by_label.get(cfg.label)
                h_cnn.write_frame(
                    frame_idx,
                    predictions=(
                        cnn_result.predictions if cnn_result is not None else []
                    ),
                )

        h_pose = self._handles.get("pose")
        if h_pose is not None:
            h_pose.write_frame(
                frame_idx,
                det_indices=np.asarray(det_indices, dtype=np.int32),
                keypoints=(
                    pose.keypoints
                    if pose is not None
                    else np.zeros((len(det_indices), 0, 3), dtype=np.float32)
                ),
                valid_mask=(
                    pose.valid_mask
                    if pose is not None
                    else np.zeros(len(det_indices), dtype=bool)
                ),
            )

        h_at = self._handles.get("apriltag")
        if h_at is not None:
            if apriltag is None:
                from ..result import AprilTagResult

                apriltag = AprilTagResult(
                    tag_ids=[],
                    det_indices=[],
                    centers=np.zeros((0, 2), dtype=np.float32),
                    corners=np.zeros((0, 4, 2), dtype=np.float32),
                )
            h_at.write_frame(frame_idx, result=apriltag)

    # --- async worker thread -----------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._items
                    or self._stop_requested
                    or self._worker_error is not None
                )
                if self._worker_error is not None:
                    return
                if not self._items:
                    if self._stop_requested:
                        return
                    continue
                item, item_bytes = self._items.popleft()
                self._worker_active = True
            try:
                with span(N.FLUSH):
                    self._apply(item)
            except BaseException as exc:  # noqa: BLE001,B036
                with self._condition:
                    if self._worker_error is None:
                        self._worker_error = exc
                    # Pending work cannot succeed after a handle failure. Drop
                    # references now and wake every producer/flush waiter.
                    self._items.clear()
                    self._inflight_bytes = item_bytes
            finally:
                with self._condition:
                    self._inflight_bytes -= item_bytes
                    self._worker_active = False
                    self._condition.notify_all()

    @property
    def queued_bytes(self) -> int:
        """Bytes owned by queued and currently-writing payloads."""
        if not self._async_mode:
            return 0
        with self._condition:
            return self._inflight_bytes

    @property
    def retained_bytes(self) -> int:
        return self.queued_bytes + sum(
            int(getattr(handle, "buffered_bytes", 0))
            for handle in self._handles.values()
        )

    @property
    def worker_alive(self) -> bool:
        return bool(self._async_mode and self._worker.is_alive())

    def _raise_worker_error_once_locked(
        self, *, reject_after_failure: bool = False
    ) -> None:
        if self._worker_error is not None and not self._error_reported:
            self._error_reported = True
            raise self._worker_error
        if self._worker_error is not None and reject_after_failure:
            raise RuntimeError("CacheWriter is unavailable after worker failure") from (
                self._worker_error
            )


def _payload_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Conservatively estimate memory retained by one queued write payload."""
    import numpy as np

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        return int(value.nbytes) + sys.getsizeof(value)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(
            _payload_bytes(key, seen) + _payload_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(_payload_bytes(item, seen) for item in value)
    fields = getattr(value, "__dict__", None)
    if fields is not None:
        return sys.getsizeof(value) + _payload_bytes(fields, seen)
    return sys.getsizeof(value)
