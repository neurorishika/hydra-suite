"""Tests for the pose_pipeline module — crop extraction and letterbox
transform utilities shared by the crops worker's pose-precompute path.
"""

from __future__ import annotations

import threading

import numpy as np

from hydra_suite.core.tracking.pose.pose_pipeline import (
    _expand_obb_to_aabb,
    extract_one_crop,
    invert_letterbox_keypoints,
    letterbox_crop,
)

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
