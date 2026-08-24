"""Overlay styling rules. Qt-free: returns plain RGB tuples and ints."""

from hydra_suite.trackerkit.gui.widgets.arena_style import (
    CLICK_DRAG_THRESHOLD_PX,
    GLYPH_MAX_PX,
    GLYPH_MIN_PX,
    TEXT_ALPHA,
    VEIL_ALPHA,
    boundary_line_width_px,
    frame_palette,
    glyph_size_px,
    line_width_px,
    zone_line_width_px,
)


def test_veil_colour_is_fixed_blue_regardless_of_frame_polarity():
    """The veil no longer flips black/white with luminance -- it's the same
    fixed blue as the boundary stroke on both light and dark footage."""
    light = frame_palette(0.90).veil
    dark = frame_palette(0.10).veil
    assert light == dark == (20, 100, 220)


def test_glyph_and_halo_are_opposite_poles_on_light_frames():
    palette = frame_palette(0.90)
    assert sum(palette.glyph) < sum(palette.halo)


def test_glyph_and_halo_are_opposite_poles_on_dark_frames():
    palette = frame_palette(0.10)
    assert sum(palette.glyph) > sum(palette.halo)


def test_role_hues_are_distinct():
    """Include, exclude, in-progress preview and boundary must never be
    confusable with one another."""
    palette = frame_palette(0.50)
    hues = {
        palette.line_include,
        palette.line_exclude,
        palette.line_preview,
        palette.line_boundary,
    }
    assert len(hues) == 4


def test_exclude_stays_red_on_both_polarities():
    """Role meaning must be learnable, so hue cannot flip with the footage."""
    light = frame_palette(0.90).line_exclude
    dark = frame_palette(0.10).line_exclude
    assert light[0] == max(light) and dark[0] == max(dark)
    assert light == dark


def test_include_is_green_and_boundary_is_blue_fixed_across_polarities():
    """New fixed-colour roles: include is green-dominant, boundary/veil hue
    is blue-dominant, regardless of frame luminance."""
    for luminance in (0.10, 0.50, 0.90):
        palette = frame_palette(luminance)
        assert palette.line_include == (0, 160, 60)
        assert palette.line_include[1] == max(palette.line_include)
        assert palette.line_boundary == (20, 100, 220)
        assert palette.line_boundary[2] == max(palette.line_boundary)


def test_preview_is_orange_and_distinct_from_include_and_boundary():
    """The in-progress preview colour must not be confusable with either a
    committed include zone (green) or the combined-ROI boundary (blue)."""
    palette = frame_palette(0.50)
    preview = palette.line_preview
    include = palette.line_include
    boundary = palette.line_boundary
    assert preview != include
    assert preview != boundary
    assert include != boundary
    # Orange: red and green channels both elevated, unlike include's
    # green-only dominance.
    assert preview[0] > 100 and preview[1] > 100
    assert include[0] < 100


def test_line_width_grows_with_viewport_but_never_vanishes():
    assert line_width_px(200) >= 2
    assert line_width_px(4000) > line_width_px(400)


def test_zone_width_is_thinner_than_boundary_width_at_same_viewport():
    for dim in (260, 520, 1080, 4000):
        assert zone_line_width_px(dim) < boundary_line_width_px(dim)


def test_zone_width_derives_from_line_width_and_never_vanishes():
    assert zone_line_width_px(200) >= 1
    assert zone_line_width_px(4000) > zone_line_width_px(400)


def test_boundary_width_is_double_the_base_line_width():
    for dim in (260, 520, 1080):
        assert boundary_line_width_px(dim) == line_width_px(dim) * 2


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
