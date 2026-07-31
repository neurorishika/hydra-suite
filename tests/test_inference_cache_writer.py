"""Tests for CacheWriter: FIFO sync and async (offload) modes.

The pipeline has a single in-order consumer, so the writer is a FIFO: sync mode
writes inline, async mode offloads to a worker thread that writes in enqueue
(== window) order.  These tests cover routing, in-order async offload, the
close()-joins-worker contract, handle-lifecycle, and worker-exception surfacing.
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.inference.cache.writer import CacheWriter
from hydra_suite.core.inference.result import (
    AprilTagResult,
    HeadTailResult,
    OBBResult,
)

# ---------------------------------------------------------------------------
# Fake handle helpers
# ---------------------------------------------------------------------------


class _RecordingHandle:
    """Records (frame_idx, kwargs) for each write_frame call."""

    def __init__(self):
        self.calls: list[tuple[int, dict]] = []
        self.close_count = 0

    def write_frame(self, frame_idx: int, **kw):
        self.calls.append((frame_idx, kw))

    def close(self):
        self.close_count += 1

    @property
    def frames(self) -> list[int]:
        return [c[0] for c in self.calls]


def _obb(frame_idx: int) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((0, 2), np.float32),
        angles=np.zeros(0, np.float32),
        sizes=np.zeros(0, np.float32),
        shapes=np.zeros((0, 2), np.float32),
        confidences=np.zeros(0, np.float32),
        corners=np.zeros((0, 4, 2), np.float32),
        detection_ids=np.zeros(0, np.int64),
    )


# ---------------------------------------------------------------------------
# Tests: sync mode (inline FIFO)
# ---------------------------------------------------------------------------


def test_sync_writes_inline_in_order():
    """Sync mode writes each detection inline in the order submitted."""
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=False)
    for i in range(5):
        w.write_detection(i, _obb(i))
    assert h.frames == [0, 1, 2, 3, 4]
    w.close()


# ---------------------------------------------------------------------------
# Tests: async mode (worker offload)
# ---------------------------------------------------------------------------


def test_async_writes_offloaded_and_in_order():
    """Async mode: writes land in enqueue order; close() joins the worker.

    The consumer enqueues in window order, so the FIFO worker writes in that
    same order. close() must drain + join the worker before returning.
    """
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=True)
    for i in range(5):
        w.write_detection(i, _obb(i))
    # close() drains the queue and joins the worker thread.
    w.close()
    assert h.frames == [0, 1, 2, 3, 4]
    assert not w._worker.is_alive(), "close() must join the worker thread"


def test_async_flush_blocks_until_all_written():
    """flush() blocks until all enqueued writes have landed (async)."""
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=True)
    for i in range(3):
        w.write_detection(i, _obb(i))
    w.flush()
    assert h.frames == [0, 1, 2]
    w.close()


# ---------------------------------------------------------------------------
# Tests: handle lifecycle — close() must NOT close handles
# ---------------------------------------------------------------------------


def test_close_does_not_close_handles():
    """close() stops the writer but must not call handle.close()."""
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=False)
    w.write_detection(0, _obb(0))
    w.close()
    assert h.close_count == 0, "CacheWriter.close() must not close caller-owned handles"


def test_close_does_not_close_handles_async():
    """Same lifecycle guarantee for async mode."""
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=True)
    w.write_detection(0, _obb(0))
    w.close()
    assert h.close_count == 0, "CacheWriter.close() must not close caller-owned handles"


def test_double_close_is_idempotent():
    """Calling close() twice must not raise."""
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=False)
    w.close()
    w.close()  # second call must be a no-op


# ---------------------------------------------------------------------------
# Tests: multi-handle routing
# ---------------------------------------------------------------------------


class _SimpleCNNConfig:
    def __init__(self, label: str):
        self.label = label


def test_detection_handle_receives_obb_result():
    """Detection handle write_frame receives result=<OBBResult>."""
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=False)
    w.write_detection(7, _obb(7))
    assert h.frames == [7]
    assert "result" in h.calls[0][1]


def test_write_downstream_routes_to_headtail_handle():
    """write_downstream routes headtail data to 'headtail' handle."""
    h_ht = _RecordingHandle()
    ht = HeadTailResult(
        heading_hints=np.array([0.0]),
        heading_confidences=np.array([1.0]),
        directed_mask=np.array([1], dtype=np.uint8),
        canonical_affines=None,
    )
    w = CacheWriter({"headtail": h_ht}, [], async_mode=False)
    w.write_downstream(
        0,
        det_indices=np.array([0], dtype=np.int32),
        headtail=ht,
        cnn_results=[],
        pose=None,
        apriltag=None,
    )
    assert h_ht.frames == [0]
    assert "heading_hints" in h_ht.calls[0][1]


def test_write_downstream_routes_to_apriltag_handle():
    """write_downstream routes apriltag data to 'apriltag' handle."""
    h_at = _RecordingHandle()
    at = AprilTagResult(
        tag_ids=[1],
        det_indices=[0],
        centers=np.zeros((1, 2), np.float32),
        corners=np.zeros((1, 4, 2), np.float32),
    )
    w = CacheWriter({"apriltag": h_at}, [], async_mode=False)
    w.write_downstream(
        3,
        det_indices=np.zeros(0, np.int32),
        headtail=None,
        cnn_results=[],
        pose=None,
        apriltag=at,
    )
    assert h_at.frames == [3]
    assert "result" in h_at.calls[0][1]


# ---------------------------------------------------------------------------
# Tests: async worker exception surfacing (Task-10 behavior preserved)
# ---------------------------------------------------------------------------


class _RaisingHandle:
    """A handle whose write_frame always raises RuntimeError."""

    def write_frame(self, frame_idx: int, **kw):
        raise RuntimeError(f"write_frame failed for frame {frame_idx}")

    def close(self):
        pass


def test_async_worker_exception_surfaces_on_close():
    """An exception in the async worker must surface from close(), not hang.

    Regression for the task_done() omission: if write_frame raises, close()
    must NOT deadlock and must re-raise the worker's exception.
    """
    import pytest

    h = _RaisingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=True)
    # Enqueue a write that will trigger write_frame, which raises.
    w.write_detection(0, _obb(0))
    with pytest.raises(RuntimeError, match="write_frame failed"):
        w.close()
