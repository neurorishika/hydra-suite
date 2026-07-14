"""CPU unit tests for the cv2-free, GPU-native mask -> rotated-rect kernel."""

from __future__ import annotations

import math

import torch

from hydra_suite.utils.obb_from_mask import _letterbox_gain_pad, rotated_rect_from_masks


def _rasterize_rotated_rect(
    size: int, cx: float, cy: float, w: float, h: float, angle_deg: float
) -> torch.Tensor:
    """Build a (size, size) binary mask of a rotated rectangle, for ground truth."""
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    dx, dy = xs - cx, ys - cy
    theta = math.radians(angle_deg)
    # Rotate the query grid into the rectangle's own axis-aligned frame.
    u = dx * math.cos(theta) + dy * math.sin(theta)
    v = -dx * math.sin(theta) + dy * math.cos(theta)
    return ((u.abs() <= w / 2) & (v.abs() <= h / 2)).float()


def test_letterbox_gain_pad_matches_scale_boxes_formula():
    # Square mask canvas (160x160), non-square original frame (1080x1920) --
    # the exact scenario that breaks a naive per-axis ratio.
    gain, pad_x, pad_y = _letterbox_gain_pad((160, 160), (1080, 1920))
    expected_gain = min(160 / 1080, 160 / 1920)
    assert math.isclose(gain, expected_gain, rel_tol=1e-6)
    assert pad_x >= 0 and pad_y >= 0
    # The wider dimension (1920) should produce zero pad on that axis, all
    # the pad should land on the shorter (1080) axis.
    assert math.isclose(pad_x, 0.0, abs_tol=1e-3)
    assert pad_y > 0


def test_rotated_rect_from_masks_recovers_axis_aligned_rectangle():
    mask = _rasterize_rotated_rect(128, cx=64, cy=64, w=50, h=20, angle_deg=0.0)
    masks = mask.unsqueeze(0)  # (1, 128, 128)
    boxes = torch.tensor([[64 - 25 - 5, 64 - 10 - 5, 64 + 25 + 5, 64 + 10 + 5]])

    rect = rotated_rect_from_masks(masks, boxes, num_angles=24, crop_size=64)

    assert rect.shape == (1, 5)
    cx, cy, w, h, angle = rect[0].tolist()
    assert math.isclose(cx, 64, abs_tol=1.5)
    assert math.isclose(cy, 64, abs_tol=1.5)
    major, minor = max(w, h), min(w, h)
    assert math.isclose(major, 50, abs_tol=3.0)
    assert math.isclose(minor, 20, abs_tol=3.0)
    # Major axis along x -> angle ~ 0 (mod pi).
    assert min(angle % math.pi, math.pi - (angle % math.pi)) < math.radians(5)


def test_rotated_rect_from_masks_recovers_rotated_rectangle():
    mask = _rasterize_rotated_rect(128, cx=64, cy=64, w=50, h=20, angle_deg=35.0)
    masks = mask.unsqueeze(0)
    # Loose axis-aligned bbox covering the rotated rect, padded generously.
    boxes = torch.tensor([[14.0, 14.0, 114.0, 114.0]])

    rect = rotated_rect_from_masks(masks, boxes, num_angles=36, crop_size=96)

    _, _, w, h, angle = rect[0].tolist()
    major, minor = max(w, h), min(w, h)
    assert math.isclose(major, 50, abs_tol=4.0)
    assert math.isclose(minor, 20, abs_tol=4.0)
    expected_rad = math.radians(35.0)
    diff = min(
        abs((angle % math.pi) - expected_rad),
        math.pi - abs((angle % math.pi) - expected_rad),
    )
    assert diff < math.radians(6)


def test_rotated_rect_from_masks_empty_mask_yields_nan_row():
    masks = torch.zeros((1, 64, 64))
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    rect = rotated_rect_from_masks(masks, boxes)
    assert torch.isnan(rect[0]).all()


def test_rotated_rect_from_masks_handles_zero_detections():
    masks = torch.zeros((0, 64, 64))
    boxes = torch.zeros((0, 4))
    rect = rotated_rect_from_masks(masks, boxes)
    assert rect.shape == (0, 5)


def test_rotated_rect_from_masks_batched_multi_detection():
    """Verify per-row independence: two detections in a single call."""
    # First detection: axis-aligned rectangle at (64, 64), w=50, h=20, angle=0.
    mask1 = _rasterize_rotated_rect(128, cx=64, cy=64, w=50, h=20, angle_deg=0.0)
    # Second detection: rotated rectangle at (64, 64), w=50, h=20, angle=35.
    mask2 = _rasterize_rotated_rect(128, cx=64, cy=64, w=50, h=20, angle_deg=35.0)
    # Stack into (2, 128, 128) batch.
    masks = torch.stack([mask1, mask2], dim=0)
    # Provide matching bounding boxes (loose axis-aligned coverage).
    boxes = torch.tensor(
        [
            [14.0, 14.0, 114.0, 114.0],  # covers both
            [14.0, 14.0, 114.0, 114.0],
        ]
    )

    rect = rotated_rect_from_masks(masks, boxes, num_angles=36, crop_size=96)

    # Both rows should be returned with correct shape.
    assert rect.shape == (2, 5)

    # Row 0: axis-aligned rectangle (angle ~ 0 mod pi).
    cx0, cy0, w0, h0, angle0 = rect[0].tolist()
    assert math.isclose(cx0, 64, abs_tol=1.5)
    assert math.isclose(cy0, 64, abs_tol=1.5)
    major0, minor0 = max(w0, h0), min(w0, h0)
    assert math.isclose(major0, 50, abs_tol=3.0)
    assert math.isclose(minor0, 20, abs_tol=3.0)
    assert min(angle0 % math.pi, math.pi - (angle0 % math.pi)) < math.radians(5)

    # Row 1: rotated rectangle (angle ~ 35° mod pi).
    cx1, cy1, w1, h1, angle1 = rect[1].tolist()
    assert math.isclose(cx1, 64, abs_tol=1.5)
    assert math.isclose(cy1, 64, abs_tol=1.5)
    major1, minor1 = max(w1, h1), min(w1, h1)
    assert math.isclose(major1, 50, abs_tol=4.0)
    assert math.isclose(minor1, 20, abs_tol=4.0)
    expected_rad = math.radians(35.0)
    diff = min(
        abs((angle1 % math.pi) - expected_rad),
        math.pi - abs((angle1 % math.pi) - expected_rad),
    )
    assert diff < math.radians(6)
