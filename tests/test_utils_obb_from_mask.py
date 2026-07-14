"""CPU unit tests for the cv2-free, GPU-native mask -> rotated-rect kernel."""

from __future__ import annotations

import math

import torch

from hydra_suite.utils.obb_from_mask import letterbox_gain_pad, rotated_rect_from_masks


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
    gain, pad_x, pad_y = letterbox_gain_pad((160, 160), (1080, 1920))
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


def test_rotated_rect_from_masks_asymmetric_mask_reports_rect_center():
    """The returned center must be the RECTANGLE's center, not the mask's mass
    centroid: for an asymmetric wedge the two differ by many pixels."""
    size = 128
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    # Right triangle wedge: x in [40, 90], y in [50, 70]; the mask's mass
    # centroid sits well left/up of the bounding rectangle's center (65, 60).
    inside = (
        (xs >= 40)
        & (xs <= 90)
        & (ys >= 50)
        & (ys <= 70)
        & ((ys - 50) <= (xs - 40) * (20.0 / 50.0))
    )
    masks = inside.float().unsqueeze(0)
    boxes = torch.tensor([[38.0, 48.0, 92.0, 72.0]])

    rect = rotated_rect_from_masks(masks, boxes, num_angles=36, crop_size=96)
    cx, cy, _, _, angle = rect[0].tolist()

    # Ground truth, angle-agnostic: in the frame of the RETURNED angle, the
    # rectangle's center is the midpoint of the foreground's (min, max) extent
    # -- NOT the mask's mass centroid.
    fg = torch.nonzero(inside, as_tuple=False).float()
    px, py = fg[:, 1], fg[:, 0]
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    u = px * cos_t + py * sin_t
    v = -px * sin_t + py * cos_t
    mid_u = float((u.max() + u.min()) / 2)
    mid_v = float((v.max() + v.min()) / 2)
    exp_cx = mid_u * cos_t - mid_v * sin_t
    exp_cy = mid_u * sin_t + mid_v * cos_t

    # Sanity: the mass centroid is far from the rect center, so this test can
    # actually tell the two apart.
    mass_cx, mass_cy = float(px.mean()), float(py.mean())
    assert math.hypot(mass_cx - exp_cx, mass_cy - exp_cy) > 5.0

    assert math.isclose(cx, exp_cx, abs_tol=2.0), f"cx={cx} expected {exp_cx}"
    assert math.isclose(cy, exp_cy, abs_tol=2.0), f"cy={cy} expected {exp_cy}"


def test_rotated_rect_from_masks_size_not_inflated_by_grid_endpoints():
    """The local sampling grid must use roi_align bin CENTERS, not an
    endpoint-inclusive linspace (which inflates w/h by crop_size/(crop_size-1)).

    Fully-saturated crop: a 32x32 foreground square with the ROI box exactly on
    it and ``pad_ratio=0`` means EVERY one of the 16x16 roi_align bins samples
    foreground.  The bins sit at physical offsets (i + 0.5) * 2 px inside the
    32 px ROI, so the true extent of the sampled foreground is 15 * 2 = 30 px.
    The endpoint-inclusive grid instead reports the full 32 px ROI side.
    """
    canvas = torch.zeros((64, 64))
    canvas[16:48, 16:48] = 1.0
    rect = rotated_rect_from_masks(
        canvas.unsqueeze(0),
        torch.tensor([[16.0, 16.0, 48.0, 48.0]]),
        num_angles=24,
        crop_size=16,
        pad_ratio=0.0,
    )
    _, _, w, h, _ = rect[0].tolist()
    assert math.isclose(w, 30.0, abs_tol=0.05), f"w={w}"
    assert math.isclose(h, 30.0, abs_tol=0.05), f"h={h}"
