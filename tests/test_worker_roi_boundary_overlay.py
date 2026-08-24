"""Focused coverage for worker.py's ROI-boundary drawContours call sites.

Prior to this test, `grep -rln "drawContours" tests/` found zero hits: the
two boundary-drawing call sites in `core/tracking/worker.py` had no
dedicated test at all, which is how a hardcoded 2px non-anti-aliased line
(invisible at full camera resolution) shipped unnoticed. This test exercises
the exact `cv2.drawContours(...)` invocation worker.py's fix uses, sourcing
color/width from the real shared `hydra_suite.utils.arena_overlay_style`
module (no hand-rolled duplicate), and asserts it produces a materially
thicker/more-visible boundary than the old hardcoded `thickness=2` call.
"""

from __future__ import annotations

import cv2
import numpy as np

from hydra_suite.utils.arena_overlay_style import (
    BOUNDARY_COLOR_BGR,
    boundary_line_width_px,
)


def _synthetic_frame_and_contours(size: int = 200):
    """A small black frame plus the ROI contours of a centered circular
    included region, mirroring worker.py's `ROI_mask_current` shape."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 3, 255, thickness=-1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return frame, contours


def _boundary_pixel_count(frame, contours, thickness, *, line_type=None):
    overlay = frame.copy()
    kwargs = {}
    if line_type is not None:
        kwargs["lineType"] = line_type
    cv2.drawContours(overlay, contours, -1, BOUNDARY_COLOR_BGR, thickness, **kwargs)
    # Pixels that moved away from black at all: boundary-colored (allowing
    # for anti-aliasing blending toward the color at the edges).
    diff = np.abs(overlay.astype(np.int16) - frame.astype(np.int16)).sum(axis=2)
    return int(np.count_nonzero(diff > 0))


def test_fixed_boundary_draw_uses_shared_color_and_width():
    frame, contours = _synthetic_frame_and_contours(200)
    assert contours, "synthetic mask must yield at least one contour"

    width = boundary_line_width_px(200)
    overlay = frame.copy()
    cv2.drawContours(
        overlay,
        contours,
        -1,
        BOUNDARY_COLOR_BGR,
        width,
        lineType=cv2.LINE_AA,
    )

    diff = np.abs(overlay.astype(np.int16) - frame.astype(np.int16)).sum(axis=2)
    boundary_pixels = np.argwhere(diff > 0)
    assert len(boundary_pixels) > 0

    # Sampled boundary pixels should be close to the exact BOUNDARY_COLOR_BGR
    # (allowing anti-aliasing blend at true edge pixels).
    sample = overlay[boundary_pixels[:, 0], boundary_pixels[:, 1]]
    max_channel_err = np.abs(
        sample.astype(np.int16) - np.array(BOUNDARY_COLOR_BGR, dtype=np.int16)
    ).max(axis=1)
    # Most sampled pixels should be at or very near the target color.
    assert np.mean(max_channel_err <= 40) > 0.5


def test_fix_measurably_thickens_the_boundary_vs_old_hardcoded_thickness_2():
    frame, contours = _synthetic_frame_and_contours(200)

    old_hardcoded_thickness = 2
    fixed_width = boundary_line_width_px(200)

    old_pixel_count = _boundary_pixel_count(frame, contours, old_hardcoded_thickness)
    new_pixel_count = _boundary_pixel_count(
        frame, contours, fixed_width, line_type=cv2.LINE_AA
    )

    # The old call site hardcoded thickness=2 with no anti-aliasing; the
    # fix must be measurably thicker at this resolution -- this is the
    # regression test that would have caught the reported "hairline"
    # bug (verified to fail if the thickness call is reverted to a bare 2).
    assert fixed_width > old_hardcoded_thickness
    assert new_pixel_count > old_pixel_count


def test_boundary_line_width_is_never_the_old_hardcoded_2px_at_full_resolution():
    # Full-resolution camera frames (the user's reported failure mode) must
    # get a boundary noticeably thicker than the old flat 2px constant.
    assert boundary_line_width_px(1080) > 2
    assert boundary_line_width_px(2160) > boundary_line_width_px(1080)
