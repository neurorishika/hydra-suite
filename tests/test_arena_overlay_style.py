"""Shared arena-boundary overlay styling (utils/, Qt-free, cv2-consumable).

Both the arena-setup canvas (Qt) and the live tracking preview (cv2) derive
their boundary color/width from this one module -- see
``hydra_suite.utils.arena_overlay_style``.
"""

from hydra_suite.utils.arena_overlay_style import (
    BOUNDARY_COLOR_BGR,
    BOUNDARY_COLOR_RGB,
    boundary_line_width_px,
)


def test_bgr_is_the_reverse_of_rgb():
    assert BOUNDARY_COLOR_BGR == tuple(reversed(BOUNDARY_COLOR_RGB))


def test_boundary_line_width_matches_documented_divisor_values():
    assert boundary_line_width_px(520) == 4
    assert boundary_line_width_px(1080) == 8


def test_boundary_line_width_scales_up_with_min_dim():
    assert boundary_line_width_px(4000) > boundary_line_width_px(400)
    assert boundary_line_width_px(1080) > boundary_line_width_px(520)


def test_boundary_line_width_never_below_the_floor():
    assert boundary_line_width_px(1) >= 2
    assert boundary_line_width_px(200) >= 2
