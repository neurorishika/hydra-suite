"""Tests for CpuFrameReader and the FrameSource abstraction.

A tiny synthetic video is written with cv2.VideoWriter so these tests work
without any real video fixture.  The synthetic clip has a fixed number of
distinct solid-colour frames so we can assert exact frame indices and shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixture: tiny synthetic AVI written to a tmp path
# ---------------------------------------------------------------------------


def _write_tiny_video(path, n_frames: int = 5, width: int = 16, height: int = 12):
    """Write a tiny BGR video with ``n_frames`` solid-colour frames."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    assert writer.isOpened(), f"VideoWriter failed for {path}"
    for i in range(n_frames):
        # Each frame has a unique solid colour so we can verify identity.
        color = (i * 40 % 256, i * 20 % 256, i * 10 % 256)
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def tiny_video(tmp_path):
    p = tmp_path / "tiny.avi"
    _write_tiny_video(p, n_frames=5)
    return p


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------


def _import():
    from hydra_suite.core.inference.sources import CpuFrameReader, FrameSource

    return FrameSource, CpuFrameReader


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_import():
    """FrameSource and CpuFrameReader can be imported from sources module."""
    FrameSource, CpuFrameReader = _import()
    assert FrameSource is not None
    assert CpuFrameReader is not None


def test_cpu_frame_reader_yields_all_frames(tiny_video):
    """CpuFrameReader yields (index, ndarray) for all frames in ascending order."""
    _, CpuFrameReader = _import()

    reader = CpuFrameReader(tiny_video)
    pairs = list(reader)
    reader.close()

    assert len(pairs) == 5
    for expected_idx, (actual_idx, frame) in enumerate(pairs):
        assert (
            actual_idx == expected_idx
        ), f"expected idx {expected_idx}, got {actual_idx}"
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # BGR channels


def test_cpu_frame_reader_frame_count(tiny_video):
    """frame_count equals total frames in the default (full) range."""
    _, CpuFrameReader = _import()
    reader = CpuFrameReader(tiny_video)
    try:
        assert reader.frame_count == 5
    finally:
        reader.close()


def test_cpu_frame_reader_start_end_range(tiny_video):
    """CpuFrameReader(path, start_frame=1, end_frame=3) yields exactly frames 1,2,3."""
    _, CpuFrameReader = _import()

    reader = CpuFrameReader(tiny_video, start_frame=1, end_frame=3)
    pairs = list(reader)
    reader.close()

    assert len(pairs) == 3
    assert reader.frame_count == 3
    indices = [idx for idx, _ in pairs]
    assert indices == [1, 2, 3]


def test_cpu_frame_reader_single_frame(tiny_video):
    """A single-frame range yields exactly one pair."""
    _, CpuFrameReader = _import()

    reader = CpuFrameReader(tiny_video, start_frame=2, end_frame=2)
    pairs = list(reader)
    reader.close()

    assert len(pairs) == 1
    assert pairs[0][0] == 2
    assert reader.frame_count == 1


def test_cpu_frame_reader_end_clamped_to_video(tiny_video):
    """end_frame beyond the video length is clamped to the last valid frame."""
    _, CpuFrameReader = _import()

    reader = CpuFrameReader(tiny_video, start_frame=0, end_frame=9999)
    try:
        assert reader.frame_count == 5
        pairs = list(reader)
        assert len(pairs) == 5
    finally:
        reader.close()


def test_cpu_frame_reader_is_frame_source_subclass(tiny_video):
    """CpuFrameReader is a proper subclass of FrameSource."""
    from hydra_suite.core.inference.sources import CpuFrameReader, FrameSource

    assert issubclass(CpuFrameReader, FrameSource)
    reader = CpuFrameReader(tiny_video)
    assert isinstance(reader, FrameSource)
    reader.close()


def test_cpu_frame_reader_context_manager(tiny_video):
    """CpuFrameReader can be used as a context manager."""
    _, CpuFrameReader = _import()

    with CpuFrameReader(tiny_video, start_frame=0, end_frame=2) as src:
        pairs = list(src)

    assert len(pairs) == 3


def test_cpu_frame_reader_close_is_idempotent(tiny_video):
    """Calling close() more than once does not raise."""
    _, CpuFrameReader = _import()

    reader = CpuFrameReader(tiny_video)
    reader.close()
    reader.close()  # second call must not raise


def test_cpu_frame_reader_bad_path():
    """CpuFrameReader raises IOError for a non-existent video file."""
    _, CpuFrameReader = _import()

    with pytest.raises(IOError):
        CpuFrameReader("/nonexistent/no_such_file.avi")


def test_cpu_frame_reader_public_range_properties(tiny_video):
    """start_frame and end_frame are publicly accessible after clamping."""
    _, CpuFrameReader = _import()

    reader = CpuFrameReader(tiny_video, start_frame=1, end_frame=3)
    try:
        assert reader.start_frame == 1
        assert reader.end_frame == 3
        # Verify that accessing the properties doesn't require private attribute access.
        assert reader.frame_count == 3
    finally:
        reader.close()


def test_cpu_frame_reader_closed_iteration_yields_nothing(tiny_video):
    """Iterating after close() yields nothing instead of crashing."""
    _, CpuFrameReader = _import()

    reader = CpuFrameReader(tiny_video)
    reader.close()
    pairs = list(reader)  # Should yield nothing, not crash
    assert pairs == []
