"""Tests for CacheWriter: frame-ordered, sync and async modes."""

from __future__ import annotations

import numpy as np

from hydra_suite.core.inference.cache.writer import CacheWriter
from hydra_suite.core.inference.result import (
    AprilTagResult,
    FrameResult,
    HeadTailResult,
    OBBResult,
)

# ---------------------------------------------------------------------------
# Fake handle and FrameResult helpers
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


def _fr(frame_idx: int) -> FrameResult:
    """Minimal FrameResult with the given frame index."""
    return FrameResult(
        frame_idx=frame_idx,
        obb=_obb(frame_idx),
        filtered_indices=[],
        headtail=None,
        cnn=[],
        pose=None,
        apriltag=None,
        resolved_headings=np.zeros(0, np.float32),
    )


# ---------------------------------------------------------------------------
# Tests: sync mode
# ---------------------------------------------------------------------------


def test_writes_in_frame_order_despite_out_of_order_submit():
    """Out-of-order submit (2, 0, 1) must write to handle in order [0, 1, 2]."""
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=False)
    for fr in [_fr(2), _fr(0), _fr(1)]:
        w.submit(fr)
    w.flush()
    w.close()
    assert h.frames == [0, 1, 2]


def test_sync_in_order_submit():
    """In-order submit flushes immediately at each step."""
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=False)
    for i in range(5):
        w.submit(_fr(i))
    assert h.frames == [0, 1, 2, 3, 4]
    w.close()


def test_flush_drains_gap_frames():
    """flush() emits everything in sorted order even if there are gaps."""
    h = _RecordingHandle()
    # start_frame=0 but submit frames 3, 1, 5 — gaps remain after contiguous drain
    w = CacheWriter({"detection": h}, [], async_mode=False)
    w.submit(_fr(3))
    w.submit(_fr(1))
    w.submit(_fr(5))
    # No contiguous drain possible from 0 since 0 was never submitted.
    assert h.frames == []
    w.flush()
    # flush() drains all buffered frames in sorted order.
    assert h.frames == [1, 3, 5]
    w.close()


# ---------------------------------------------------------------------------
# Tests: async mode
# ---------------------------------------------------------------------------


def test_async_writes_in_frame_order():
    """Async mode: out-of-order submit produces ordered writes after close()."""
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=True)
    for fr in [_fr(2), _fr(0), _fr(1)]:
        w.submit(fr)
    w.close()
    assert h.frames == [0, 1, 2]


def test_async_flush_blocks_until_all_submitted_written():
    """flush() blocks until all submitted FrameResults have been written."""
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=True)
    for fr in [_fr(1), _fr(0), _fr(2)]:
        w.submit(fr)
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
    w.submit(_fr(0))
    w.close()
    assert h.close_count == 0, "CacheWriter.close() must not close caller-owned handles"


def test_close_does_not_close_handles_async():
    """Same lifecycle guarantee for async mode."""
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=True)
    w.submit(_fr(0))
    w.close()
    assert h.close_count == 0, "CacheWriter.close() must not close caller-owned handles"


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


def test_double_close_is_idempotent():
    """Calling close() twice must not raise."""
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=False)
    w.close()
    w.close()  # second call must be a no-op


# ---------------------------------------------------------------------------
# Tests: Fix 2 — _drain_all_sync cursor advance
# ---------------------------------------------------------------------------


def test_flush_then_submit_no_duplicates():
    """submit after flush must not re-write already-flushed frames.

    Regression for the _drain_all_sync cursor bug: after flush() drains frames
    0 and 2 (with a gap at 1), submitting frame 3 must write [3] only — not
    re-emit 0 or 2 again.
    """
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, [], async_mode=False)
    # Submit 0 and 2; frame 1 is missing so contiguous drain stops at 0.
    w.submit(_fr(0))
    w.submit(_fr(2))
    # After this flush: frames 0 and 2 are written; cursor must advance to 3.
    w.flush()
    assert h.frames == [0, 2], "flush must drain 0 and 2 in sorted order"
    # Now submit frame 3; it should be written once, immediately.
    w.submit(_fr(3))
    w.close()
    assert h.frames == [0, 2, 3], "frame 3 must appear exactly once, no duplicates"
    # Verify no frame appears more than once.
    assert len(h.frames) == len(set(h.frames)), "duplicate writes detected"


# ---------------------------------------------------------------------------
# Tests: Fix 3 — async worker exception surfacing
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
    # Submit a frame that will trigger write_frame, which raises.
    w.submit(_fr(0))
    with pytest.raises(RuntimeError, match="write_frame failed"):
        w.close()
