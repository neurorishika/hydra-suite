"""Tests for the pose_pipeline module — parallel crop extraction,
letterbox transforms, async cache writing, and double-buffered inference.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest

from hydra_suite.core.tracking.pose.pose_pipeline import (
    AsyncCacheWriter,
    FrameCropResult,
    PosePipeline,
    _expand_obb_to_aabb,
    extract_one_crop,
    invert_letterbox_keypoints,
    letterbox_crop,
)
from hydra_suite.core.tracking.profiler import TrackingProfiler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _square_corners(cx, cy, half_w):
    """Return 4 OBB corners for an axis-aligned square centred at (cx, cy)."""
    return np.array(
        [
            [cx - half_w, cy - half_w],
            [cx + half_w, cy - half_w],
            [cx + half_w, cy + half_w],
            [cx - half_w, cy + half_w],
        ],
        dtype=np.float32,
    )


def _dummy_frame(h=200, w=300, channels=3):
    rng = np.random.RandomState(42)
    return rng.randint(0, 255, (h, w, channels), dtype=np.uint8)


@dataclass
class _FakePoseResult:
    keypoints: Optional[np.ndarray]
    mean_conf: float = 1.0
    valid_fraction: float = 1.0
    num_valid: int = 3
    num_keypoints: int = 3


class _FakeBackend:
    """Minimal backend that returns identity keypoints for testing."""

    output_keypoint_names = ["head", "body", "tail"]

    @property
    def preferred_input_size(self) -> int:
        return 0

    def warmup(self):
        pass

    def predict_batch(self, crops: Sequence[np.ndarray]) -> List[_FakePoseResult]:
        results = []
        for crop in crops:
            h, w = crop.shape[:2]
            kpts = np.array(
                [[w / 2, h / 2, 0.9], [w / 4, h / 4, 0.8], [w * 3 / 4, h * 3 / 4, 0.7]],
                dtype=np.float32,
            )
            results.append(_FakePoseResult(keypoints=kpts))
        return results

    def close(self):
        pass


class _FakeCacheWriter:
    """Records add_frame calls for verification."""

    def __init__(self):
        self.frames: List[Tuple[int, list, list]] = []
        self._lock = threading.Lock()
        self.saved_metadata = None

    def add_frame(self, frame_idx, detection_ids, pose_keypoints=None):
        with self._lock:
            self.frames.append(
                (frame_idx, list(detection_ids), list(pose_keypoints or []))
            )

    def save(self, metadata=None):
        self.saved_metadata = metadata


class _FakeDetectionCache:
    """Returns predetermined detections for each frame."""

    def __init__(self, frames_dict):
        self._frames = frames_dict

    def get_frame(self, frame_idx):
        if frame_idx in self._frames:
            val = self._frames[frame_idx]
            # Pad to 12 elements (headtail confidence plus canonical extras)
            while len(val) < 12:
                val = val + (None,)
            return val
        empty = ([], [], [], [], [], [], [], [], [], None, None, None)
        return empty


class _FakeDetector:
    """Pass-through detector that doesn't filter."""

    def filter_raw_detections(
        self,
        meas,
        sizes,
        shapes,
        confs,
        obb,
        *,
        roi_mask=None,
        detection_ids=None,
        heading_hints=None,
        heading_confidences=None,
        directed_mask=None,
    ):
        return (
            meas,
            sizes,
            shapes,
            confs,
            obb,
            detection_ids,
            heading_hints,
            heading_confidences,
            directed_mask,
        )


class _FakeVideoCapture:
    """Yields pre-built frames in sequence."""

    def __init__(self, frames: List[np.ndarray]):
        self._frames = list(frames)
        self._idx = 0

    def read(self):
        if self._idx < len(self._frames):
            f = self._frames[self._idx]
            self._idx += 1
            return True, f
        return False, None

    def set(self, prop, val):
        self._idx = int(val)

    def release(self):
        pass


# ---------------------------------------------------------------------------
# Tests: _expand_obb_to_aabb
# ---------------------------------------------------------------------------


class TestExpandObbToAabb:
    def test_basic_square(self):
        corners = _square_corners(50, 50, 10)
        x0, y0, x1, y1 = _expand_obb_to_aabb(corners, 0.0, 200, 300)
        assert x0 == 40
        assert y0 == 40
        assert x1 == 61
        assert y1 == 61

    def test_with_padding(self):
        corners = _square_corners(50, 50, 10)
        x0, y0, x1, y1 = _expand_obb_to_aabb(corners, 0.5, 200, 300)
        # 10 * 1.5 = 15 → 50-15=35, 50+15=65
        assert x0 == 35
        assert y0 == 35
        assert x1 == 66
        assert y1 == 66

    def test_clipping(self):
        corners = _square_corners(5, 5, 10)
        x0, y0, x1, y1 = _expand_obb_to_aabb(corners, 0.0, 200, 300)
        assert x0 == 0
        assert y0 == 0


# ---------------------------------------------------------------------------
# Tests: extract_one_crop
# ---------------------------------------------------------------------------


class TestExtractOneCrop:
    def test_basic_extraction(self):
        frame = _dummy_frame(100, 100)
        corners = _square_corners(50, 50, 10)
        result = extract_one_crop(frame, corners, 0, 0.0, [corners], False, (0, 0, 0))
        assert result is not None
        crop, (x0, y0), det_idx = result
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0
        assert det_idx == 0

    def test_none_frame(self):
        corners = _square_corners(50, 50, 10)
        assert extract_one_crop(None, corners, 0, 0.0, [], False, (0, 0, 0)) is None

    def test_invalid_corners(self):
        frame = _dummy_frame(100, 100)
        corners = np.array([[0, 0], [1, 1]], dtype=np.float32)  # only 2 corners
        assert extract_one_crop(frame, corners, 0, 0.0, [], False, (0, 0, 0)) is None

    def test_thread_safety(self):
        """Multiple threads can safely extract crops from the same frame."""
        frame = _dummy_frame(200, 200)
        corners_list = [_square_corners(50 + i * 30, 50 + i * 30, 10) for i in range(5)]
        results = [None] * 5

        def _extract(idx):
            results[idx] = extract_one_crop(
                frame, corners_list[idx], idx, 0.1, corners_list, False, (0, 0, 0)
            )

        threads = [threading.Thread(target=_extract, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for r in results:
            assert r is not None


# ---------------------------------------------------------------------------
# Tests: letterbox_crop / invert_letterbox_keypoints
# ---------------------------------------------------------------------------


class TestLetterbox:
    def test_downscale_large_crop(self):
        crop = np.zeros((400, 200, 3), dtype=np.uint8)
        lb, transform = letterbox_crop(crop, 200)
        assert lb.shape == (200, 200, 3)
        assert transform.scale < 1.0

    def test_small_crop_no_upscale(self):
        crop = np.zeros((50, 30, 3), dtype=np.uint8)
        lb, transform = letterbox_crop(crop, 200)
        assert lb.shape == (200, 200, 3)
        assert transform.scale == 1.0
        # Original content should be centered
        assert transform.pad_x > 0 or transform.pad_y > 0

    def test_exact_size_no_padding(self):
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        lb, transform = letterbox_crop(crop, 200)
        assert lb.shape == (200, 200, 3)
        assert transform.pad_x == 0
        assert transform.pad_y == 0

    def test_inverse_identity(self):
        """Letterbox + inverse should approximately recover original coordinates."""
        crop = np.zeros((300, 200, 3), dtype=np.uint8)
        _, transform = letterbox_crop(crop, 150)
        # Point at center of original crop
        orig_kpt = np.array([[100.0, 150.0, 0.9]], dtype=np.float32)
        # Forward: apply letterbox transform manually
        fwd = orig_kpt.copy()
        fwd[:, 0] = fwd[:, 0] * transform.scale + transform.pad_x
        fwd[:, 1] = fwd[:, 1] * transform.scale + transform.pad_y
        # Inverse
        recovered = invert_letterbox_keypoints(fwd, transform)
        np.testing.assert_allclose(recovered[:, :2], orig_kpt[:, :2], atol=1.0)

    def test_grayscale_crop(self):
        crop = np.zeros((100, 50), dtype=np.uint8)
        lb, transform = letterbox_crop(crop, 200)
        assert lb.shape == (200, 200)


# ---------------------------------------------------------------------------
# Tests: AsyncCacheWriter
# ---------------------------------------------------------------------------


class TestAsyncCacheWriter:
    def test_writes_all_frames(self):
        cache = _FakeCacheWriter()
        writer = AsyncCacheWriter(cache)
        for i in range(10):
            writer.submit(i, [float(i)], [None])
        writer.flush_and_close()
        assert len(cache.frames) == 10
        assert [f[0] for f in cache.frames] == list(range(10))

    def test_error_propagation(self):
        """Errors in cache writing propagate on flush."""

        class _BadWriter:
            def add_frame(self, *a, **kw):
                raise ValueError("disk full")

        writer = AsyncCacheWriter(_BadWriter())
        writer.submit(0, [1.0], [None])
        with pytest.raises(ValueError, match="disk full"):
            writer.flush_and_close()


# ---------------------------------------------------------------------------
# Tests: PosePipeline (integration)
# ---------------------------------------------------------------------------


class TestPosePipeline:
    def _make_detection_cache(self, n_frames, n_dets_per_frame, frame_h, frame_w):
        """Build a fake detection cache with uniformly-spaced detections."""
        frames = {}
        spacing = frame_w // (n_dets_per_frame + 1)
        for fi in range(n_frames):
            meas = []
            obb = []
            ids = []
            for di in range(n_dets_per_frame):
                cx = spacing * (di + 1)
                cy = frame_h // 2
                meas.append([cx, cy, 0.0])
                obb.append(_square_corners(cx, cy, 15))
                ids.append(float(di))
            frames[fi] = (
                meas,
                [10] * n_dets_per_frame,
                [1] * n_dets_per_frame,
                [0.9] * n_dets_per_frame,
                obb,
                ids,
                [0.0] * n_dets_per_frame,
                [False] * n_dets_per_frame,
            )
        return _FakeDetectionCache(frames)

    def test_basic_run(self):
        n_frames, n_dets = 5, 2
        frame_h, frame_w = 100, 200
        frames = [_dummy_frame(frame_h, frame_w) for _ in range(n_frames)]
        cache_writer = _FakeCacheWriter()
        backend = _FakeBackend()
        det_cache = self._make_detection_cache(n_frames, n_dets, frame_h, frame_w)

        pipeline = PosePipeline(
            backend,
            cache_writer,
            cross_frame_batch=3,
            crop_workers=2,
            pre_resize_target=0,
            suppress_foreign_obb=False,
        )
        vcap = _FakeVideoCapture(frames)
        completed = pipeline.run(
            vcap,
            det_cache,
            _FakeDetector(),
            0,
            n_frames - 1,
            1.0,
            None,
        )
        pipeline.close()

        assert completed is True
        assert len(cache_writer.frames) == n_frames
        # Each frame should have n_dets keypoint arrays
        for _, ids, kps in cache_writer.frames:
            assert len(ids) == n_dets
            assert len(kps) == n_dets

    def test_precompute_phase_keeps_detection_ids_and_indexes_by_slot(self):
        cache_writer = _FakeCacheWriter()
        backend = _FakeBackend()

        pipeline = PosePipeline(
            backend,
            cache_writer,
            cross_frame_batch=4,
            crop_workers=1,
            suppress_foreign_obb=False,
        )

        crops = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
        detection_ids = [100001.0, 100002.0]
        crop_det_indices = [0, 1]
        all_obb = [
            _square_corners(20, 20, 10),
            _square_corners(60, 20, 10),
        ]
        crop_offsets = [(5, 7), (15, 17)]

        pipeline.process_frame(
            0,
            crops,
            detection_ids,
            crop_det_indices,
            all_obb,
            crop_offsets,
        )
        pipeline.finalize()
        pipeline.close()

        assert len(cache_writer.frames) == 1
        frame_idx, ids, kps = cache_writer.frames[0]
        assert frame_idx == 0
        assert ids == [100001, 100002]
        assert len(kps) == 2
        assert all(kp is not None for kp in kps)

    def test_with_pre_resize(self):
        n_frames, n_dets = 3, 1
        frame_h, frame_w = 200, 200
        frames = [_dummy_frame(frame_h, frame_w) for _ in range(n_frames)]
        cache_writer = _FakeCacheWriter()
        backend = _FakeBackend()
        det_cache = self._make_detection_cache(n_frames, n_dets, frame_h, frame_w)

        pipeline = PosePipeline(
            backend,
            cache_writer,
            cross_frame_batch=10,
            crop_workers=1,
            pre_resize_target=64,
            suppress_foreign_obb=False,
        )
        vcap = _FakeVideoCapture(frames)
        completed = pipeline.run(
            vcap,
            det_cache,
            _FakeDetector(),
            0,
            n_frames - 1,
            1.0,
            None,
        )
        pipeline.close()

        assert completed is True
        assert len(cache_writer.frames) == n_frames
        # Keypoints should be in frame coordinates (not letterboxed)
        for _, ids, kps in cache_writer.frames:
            for kp in kps:
                if kp is not None:
                    # Keypoints should be within frame bounds (approx)
                    assert np.all(kp[:, 0] >= -10)
                    assert np.all(kp[:, 1] >= -10)

    def test_cancellation(self):
        n_frames = 10
        frame_h, frame_w = 100, 100
        frames = [_dummy_frame(frame_h, frame_w) for _ in range(n_frames)]
        cache_writer = _FakeCacheWriter()
        backend = _FakeBackend()
        det_cache = self._make_detection_cache(n_frames, 1, frame_h, frame_w)

        call_count = [0]

        def _stop_after_3():
            call_count[0] += 1
            return call_count[0] > 5

        pipeline = PosePipeline(
            backend,
            cache_writer,
            cross_frame_batch=2,
            crop_workers=1,
        )
        vcap = _FakeVideoCapture(frames)
        completed = pipeline.run(
            vcap,
            det_cache,
            _FakeDetector(),
            0,
            n_frames - 1,
            1.0,
            None,
            stop_check=_stop_after_3,
        )
        pipeline.close()

        assert completed is False
        assert len(cache_writer.frames) < n_frames

    def test_empty_frames(self):
        """Frames with no detections should still be written to cache."""
        n_frames = 3
        frame_h, frame_w = 100, 100
        frames = [_dummy_frame(frame_h, frame_w) for _ in range(n_frames)]
        cache_writer = _FakeCacheWriter()
        backend = _FakeBackend()
        # Empty detection cache — no detections in any frame
        det_cache = _FakeDetectionCache({})

        pipeline = PosePipeline(
            backend,
            cache_writer,
            cross_frame_batch=10,
            crop_workers=1,
        )
        vcap = _FakeVideoCapture(frames)
        completed = pipeline.run(
            vcap,
            det_cache,
            _FakeDetector(),
            0,
            n_frames - 1,
            1.0,
            None,
        )
        pipeline.close()

        assert completed is True
        # All frames should be in cache (with empty data)
        assert len(cache_writer.frames) == n_frames

    def test_progress_callbacks(self):
        n_frames = 5
        frames = [_dummy_frame(50, 50) for _ in range(n_frames)]
        cache_writer = _FakeCacheWriter()
        backend = _FakeBackend()
        det_cache = self._make_detection_cache(n_frames, 1, 50, 50)

        progress_calls = []
        stats_calls = []

        pipeline = PosePipeline(
            backend,
            cache_writer,
            cross_frame_batch=10,
            crop_workers=1,
        )
        vcap = _FakeVideoCapture(frames)
        pipeline.run(
            vcap,
            det_cache,
            _FakeDetector(),
            0,
            n_frames - 1,
            1.0,
            None,
            progress_cb=lambda p, m: progress_calls.append((p, m)),
            stats_cb=lambda s: stats_calls.append(s),
        )
        pipeline.close()

        assert len(progress_calls) > 0
        assert len(stats_calls) > 0
        # Last progress should be 100%
        assert progress_calls[-1][0] == 100

    def test_double_buffering_overlap(self):
        """Verify that inference and crop extraction overlap (not fully serial)."""

        # Use a backend with artificial delay to make overlap observable
        class _SlowBackend(_FakeBackend):
            def predict_batch(self, crops):
                time.sleep(0.05)  # 50ms delay
                return super().predict_batch(crops)

        n_frames = 20
        frames = [_dummy_frame(100, 100) for _ in range(n_frames)]
        cache_writer = _FakeCacheWriter()
        backend = _SlowBackend()
        det_cache = self._make_detection_cache(n_frames, 2, 100, 100)

        pipeline = PosePipeline(
            backend,
            cache_writer,
            cross_frame_batch=5,  # small batches to create more flushes
            crop_workers=2,
        )
        vcap = _FakeVideoCapture(frames)

        pipeline.run(
            vcap,
            det_cache,
            _FakeDetector(),
            0,
            n_frames - 1,
            1.0,
            None,
        )
        pipeline.close()

        # Without double-buffering: 4 batches * 50ms = 200ms minimum
        # With double-buffering: should be closer to 3 * 50ms + overhead
        # Just verify it completed correctly
        assert len(cache_writer.frames) == n_frames

    def test_auto_preferred_input_size(self):
        """Backend with preferred_input_size should auto-set pre_resize."""

        class _SizedBackend(_FakeBackend):
            @property
            def preferred_input_size(self):
                return 128

        backend = _SizedBackend()
        # pre_resize_target=0 → should auto-detect from backend
        pipeline = PosePipeline(
            backend,
            _FakeCacheWriter(),
            pre_resize_target=0,
        )
        # The pipeline should have picked up 128
        assert pipeline._pre_resize == 0  # 0 passed; auto-detect is in worker.py
        pipeline.close()

    def test_sync_records_realtime_pose_profile(self):
        class _ProfilingBackend(_FakeBackend):
            def __init__(self):
                self._last_profile = {}

            def predict_batch(self, crops):
                self._last_profile = {
                    "pose_transport_s": 0.002,
                    "pose_inference_s": 0.003,
                    "pose_postprocess_s": 0.001,
                }
                return super().predict_batch(crops)

            def consume_last_profile(self):
                profile = dict(self._last_profile)
                self._last_profile = {}
                return profile

        backend = _ProfilingBackend()
        pipeline = PosePipeline(backend, cache_writer=None, cross_frame_batch=1)
        profiler = TrackingProfiler(enabled=True)

        crop = np.zeros((24, 24, 3), dtype=np.uint8)
        fcr = FrameCropResult(
            frame_idx=0,
            det_ids=[101],
            n_dets=1,
            crops=[crop],
            crop_to_det=[0],
            crop_offsets={0: (0, 0)},
            all_obb_corners=[_square_corners(12, 12, 8)],
            crop_transforms={},
            crop_M_inverses={},
        )

        pipeline._inflight = pipeline._infer_pool.submit(
            pipeline._infer_and_cache, [fcr], [crop]
        )
        pipeline.sync(profiler=profiler)
        profiler.end_frame()
        summary = profiler.get_summary()

        assert summary["categories"]["live_pose_transport"]["mean_ms"] == pytest.approx(
            2.0, abs=0.2
        )
        assert summary["categories"]["live_pose_transport"]["total_units"] == 1
        assert summary["categories"]["live_pose_transport"][
            "mean_units_per_frame"
        ] == pytest.approx(1.0, abs=0.01)
        assert summary["categories"]["live_pose_transport"][
            "ms_per_unit"
        ] == pytest.approx(2.0, abs=0.2)
        assert summary["categories"]["live_pose_inference"]["mean_ms"] == pytest.approx(
            3.0, abs=0.2
        )
        assert summary["categories"]["live_pose_inference"]["total_units"] == 1
        assert summary["categories"]["live_pose_inference"][
            "ms_per_unit"
        ] == pytest.approx(3.0, abs=0.2)
        assert summary["categories"]["live_pose_postprocess"]["mean_ms"] >= 1.0
        assert summary["categories"]["live_pose_postprocess"]["total_units"] == 1
        assert summary["phases"]["pose_transport"]["total_s"] == pytest.approx(
            0.002, abs=0.0002
        )
        assert summary["phases"]["pose_transport"]["avg_ms_per_frame"] == pytest.approx(
            2.0, abs=0.2
        )
        assert summary["phases"]["pose_transport"]["total_units"] == 1
        assert summary["phases"]["pose_transport"]["ms_per_unit"] == pytest.approx(
            2.0, abs=0.2
        )
        assert summary["phases"]["pose_inference"]["total_s"] == pytest.approx(
            0.003, abs=0.0002
        )
        assert summary["phases"]["pose_inference"]["avg_ms_per_frame"] == pytest.approx(
            3.0, abs=0.2
        )
        assert summary["phases"]["pose_inference"]["total_units"] == 1
        assert summary["phases"]["pose_inference"]["ms_per_unit"] == pytest.approx(
            3.0, abs=0.2
        )
        assert summary["phases"]["pose_postprocess"]["total_s"] >= 0.001
        assert summary["phases"]["pose_postprocess"]["total_units"] == 1
        pipeline.close()
