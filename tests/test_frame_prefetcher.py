"""
Comprehensive tests for FramePrefetcher classes.

Tests cover:
- Forward frame prefetching
- Backward frame prefetching
- Buffer management
- Thread safety and cleanup
- Error handling
"""

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.utils.frame_prefetcher import (
    FramePrefetcher,
    SequentialScanPrefetcher,
    SparseFramePrefetcher,
)


def create_test_video(num_frames=30, width=640, height=480):
    """Create a temporary test video file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = temp_file.name
    temp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, 30.0, (width, height))

    for i in range(num_frames):
        # Create a frame with frame number drawn on it
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        cv2.putText(
            frame,
            str(i),
            (width // 2 - 50, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3,
        )
        writer.write(frame)

    writer.release()
    return temp_path


class TestFramePrefetcher:
    """Test suite for FramePrefetcher class."""

    def test_initialization(self):
        """Test that FramePrefetcher initializes correctly."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=2)

            assert prefetcher.cap == cap
            assert prefetcher.buffer_size == 2
            assert not prefetcher._started
            assert prefetcher.exception is None

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_start(self):
        """Test starting the prefetcher thread."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=2)

            prefetcher.start()
            assert prefetcher._started
            assert prefetcher.thread is not None
            assert prefetcher.thread.is_alive()

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_read_single_frame(self):
        """Test reading a single frame."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=2)
            prefetcher.start()

            ret, frame = prefetcher.read()
            assert ret is True
            assert frame is not None
            assert frame.shape == (480, 640, 3)

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_read_all_frames(self):
        """Test reading all frames from a video."""
        num_frames = 20
        video_path = create_test_video(num_frames)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=3)
            prefetcher.start()

            frames_read = 0
            while True:
                ret, frame = prefetcher.read()
                if not ret:
                    break
                assert frame is not None
                frames_read += 1

            assert frames_read == num_frames

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_slow_consumer_does_not_drop_frames(self):
        """A full queue must not cause already-read frames to be discarded."""
        num_frames = 20
        video_path = create_test_video(num_frames)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=1)
            prefetcher.start()

            # Give the producer time to fill the single-slot queue.
            time.sleep(0.25)

            frames_read = 0
            while True:
                ret, frame = prefetcher.read()
                if not ret:
                    break
                assert frame is not None
                frames_read += 1
                time.sleep(0.02)

            assert frames_read == num_frames

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_buffer_size_effect(self):
        """Test that different buffer sizes work correctly."""
        video_path = create_test_video(30)
        try:
            for buffer_size in [1, 2, 5]:
                cap = cv2.VideoCapture(video_path)
                prefetcher = FramePrefetcher(cap, buffer_size=buffer_size)
                prefetcher.start()

                # Read a few frames
                for _ in range(10):
                    ret, frame = prefetcher.read()
                    assert ret is True
                    assert frame is not None

                prefetcher.stop()
                cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_context_manager(self):
        """Test using FramePrefetcher as a context manager."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)

            with FramePrefetcher(cap, buffer_size=2) as prefetcher:
                assert prefetcher._started
                ret, frame = prefetcher.read()
                assert ret is True
                assert frame is not None

            # After exiting context, should be stopped
            assert not prefetcher._started

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_stop_clears_queue(self):
        """Test that stopping clears the frame queue."""
        video_path = create_test_video(20)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=5)
            prefetcher.start()

            # Read a few frames to populate the buffer
            time.sleep(0.2)

            # Stop should clear the queue
            prefetcher.stop()
            assert prefetcher.frame_queue.empty()

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_double_start_warning(self):
        """Test that starting an already started prefetcher logs a warning."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=2)

            prefetcher.start()
            # Second start should be a no-op with warning
            prefetcher.start()

            assert prefetcher._started

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_stop_without_start(self):
        """Test that stopping without starting is safe."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=2)

            # Should not crash
            prefetcher.stop()

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)


class TestFramePrefetcherRobustness:
    """Test robustness and error handling."""

    def test_empty_video(self):
        """Test handling of empty/corrupt video."""
        video_path = create_test_video(0)  # Create empty video
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = FramePrefetcher(cap, buffer_size=2)
            prefetcher.start()

            ret, frame = prefetcher.read()
            # Should return False for empty video
            assert ret is False

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_rapid_start_stop(self):
        """Test rapid start/stop cycles."""
        video_path = create_test_video(20)
        try:
            cap = cv2.VideoCapture(video_path)

            for _ in range(3):
                prefetcher = FramePrefetcher(cap, buffer_size=2)
                prefetcher.start()
                ret, frame = prefetcher.read()
                prefetcher.stop()

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)


class TestSparseFramePrefetcher:
    """Test suite for SparseFramePrefetcher class."""

    def test_read_specific_frames(self):
        """Test reading specific sparse frames."""
        num_frames = 30
        video_path = create_test_video(num_frames)
        try:
            cap = cv2.VideoCapture(video_path)
            indices = [0, 5, 10, 15, 20, 25]
            prefetcher = SparseFramePrefetcher(cap, indices, buffer_size=4)
            prefetcher.start()

            read_indices = []
            while True:
                item = prefetcher.read()
                if item is None:
                    break
                f, ret, frame = item
                assert ret is True
                assert frame is not None
                read_indices.append(f)

            assert read_indices == indices

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_contiguous_frames(self):
        """Test reading contiguous frames (should skip seeks)."""
        num_frames = 20
        video_path = create_test_video(num_frames)
        try:
            cap = cv2.VideoCapture(video_path)
            indices = list(range(5, 15))
            prefetcher = SparseFramePrefetcher(cap, indices, buffer_size=4)
            prefetcher.start()

            read_indices = []
            while True:
                item = prefetcher.read()
                if item is None:
                    break
                f, ret, frame = item
                assert ret is True
                read_indices.append(f)

            assert read_indices == indices

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_context_manager(self):
        """Test context manager usage."""
        video_path = create_test_video(20)
        try:
            cap = cv2.VideoCapture(video_path)
            with SparseFramePrefetcher(cap, [0, 5, 10], buffer_size=2) as pf:
                item = pf.read()
                assert item is not None
                f, ret, frame = item
                assert f == 0
                assert ret is True
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)


class TestSequentialScanPrefetcher:
    """Test suite for SequentialScanPrefetcher class."""

    def test_read_specific_frames(self):
        """Test that only needed frames are returned."""
        num_frames = 30
        video_path = create_test_video(num_frames)
        try:
            cap = cv2.VideoCapture(video_path)
            indices = [5, 10, 15, 20, 25]
            prefetcher = SequentialScanPrefetcher(cap, indices, buffer_size=4)
            prefetcher.start()

            read_indices = []
            while True:
                item = prefetcher.read()
                if item is None:
                    break
                f, ret, frame = item
                assert ret is True
                assert frame is not None
                read_indices.append(f)

            assert read_indices == indices

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_contiguous_frames(self):
        """Test reading contiguous frames."""
        num_frames = 20
        video_path = create_test_video(num_frames)
        try:
            cap = cv2.VideoCapture(video_path)
            indices = list(range(3, 13))
            prefetcher = SequentialScanPrefetcher(cap, indices, buffer_size=4)
            prefetcher.start()

            read_indices = []
            while True:
                item = prefetcher.read()
                if item is None:
                    break
                f, ret, frame = item
                assert ret is True
                read_indices.append(f)

            assert read_indices == indices

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_single_frame(self):
        """Test with a single needed frame."""
        video_path = create_test_video(20)
        try:
            cap = cv2.VideoCapture(video_path)
            prefetcher = SequentialScanPrefetcher(cap, [10], buffer_size=2)
            prefetcher.start()

            item = prefetcher.read()
            assert item is not None
            f, ret, frame = item
            assert f == 10
            assert ret is True

            # Should get sentinel next
            item = prefetcher.read()
            assert item is None

            prefetcher.stop()
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_context_manager(self):
        """Test context manager usage."""
        video_path = create_test_video(20)
        try:
            cap = cv2.VideoCapture(video_path)
            with SequentialScanPrefetcher(cap, [2, 8, 14], buffer_size=2) as pf:
                item = pf.read()
                assert item is not None
                f, ret, frame = item
                assert f == 2
                assert ret is True
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_returns_same_frames_as_sparse(self):
        """Verify sequential and sparse prefetchers return same frame indices."""
        num_frames = 30
        video_path = create_test_video(num_frames)
        try:
            indices = [2, 7, 12, 17, 22, 27]

            # Sparse
            cap1 = cv2.VideoCapture(video_path)
            pf1 = SparseFramePrefetcher(cap1, indices, buffer_size=4)
            pf1.start()
            sparse_indices = []
            while True:
                item = pf1.read()
                if item is None:
                    break
                sparse_indices.append(item[0])
            pf1.stop()
            cap1.release()

            # Sequential
            cap2 = cv2.VideoCapture(video_path)
            pf2 = SequentialScanPrefetcher(cap2, indices, buffer_size=4)
            pf2.start()
            seq_indices = []
            while True:
                item = pf2.read()
                if item is None:
                    break
                seq_indices.append(item[0])
            pf2.stop()
            cap2.release()

            assert sparse_indices == seq_indices == indices
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_stop_without_start(self):
        """Test that stopping without starting is safe."""
        video_path = create_test_video(10)
        try:
            cap = cv2.VideoCapture(video_path)
            pf = SequentialScanPrefetcher(cap, [0, 5], buffer_size=2)
            pf.stop()  # Should not crash
            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)
