"""Overlay styling rules. Qt-free: returns plain RGB tuples and ints."""

from hydra_suite.trackerkit.gui.widgets.arena_style import (
    CLICK_DRAG_THRESHOLD_PX,
    GLYPH_MAX_PX,
    GLYPH_MIN_PX,
    TEXT_ALPHA,
    VEIL_ALPHA,
    frame_palette,
    glyph_size_px,
    line_width_px,
)


def test_light_frame_gets_a_dark_veil():
    assert frame_palette(0.90).veil == (0, 0, 0)


def test_dark_frame_gets_a_light_veil():
    assert frame_palette(0.10).veil == (255, 255, 255)


def test_glyph_and_halo_are_opposite_poles_on_light_frames():
    palette = frame_palette(0.90)
    assert sum(palette.glyph) < sum(palette.halo)


def test_glyph_and_halo_are_opposite_poles_on_dark_frames():
    palette = frame_palette(0.10)
    assert sum(palette.glyph) > sum(palette.halo)


def test_role_hues_are_distinct():
    """Include, exclude and in-progress must never be confusable."""
    palette = frame_palette(0.50)
    hues = {palette.line_include, palette.line_exclude, palette.line_preview}
    assert len(hues) == 3


def test_exclude_stays_red_on_both_polarities():
    """Role meaning must be learnable, so hue cannot flip with the footage."""
    light = frame_palette(0.90).line_exclude
    dark = frame_palette(0.10).line_exclude
    assert light[0] == max(light) and dark[0] == max(dark)


def test_line_width_grows_with_viewport_but_never_vanishes():
    assert line_width_px(200) >= 2
    assert line_width_px(4000) > line_width_px(400)


def test_glyph_size_is_clamped_at_both_ends():
    assert glyph_size_px(1.0) == GLYPH_MIN_PX
    assert glyph_size_px(100000.0) == GLYPH_MAX_PX


def test_glyph_size_scales_between_the_clamps():
    small = glyph_size_px(30.0)
    large = glyph_size_px(60.0)
    assert GLYPH_MIN_PX <= small < large <= GLYPH_MAX_PX


def test_alpha_and_threshold_constants_match_the_spec():
    assert VEIL_ALPHA == 0.15
    assert TEXT_ALPHA == 0.70
    assert CLICK_DRAG_THRESHOLD_PX == 3
