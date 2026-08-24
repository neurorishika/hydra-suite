"""Overlay painting: veil polarity, zoom-invariant weight, halo compositing."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.gui.widgets.arena_canvas import (  # noqa: E402
    ArenaCanvas,
    paint_arena_number,
)
from hydra_suite.trackerkit.gui.widgets.arena_style import (  # noqa: E402
    zone_line_width_px,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _frame(app, gray):
    image = QImage(200, 200, QImage.Format_RGB888)
    image.fill(QColor(gray, gray, gray))
    return image


def _canvas(app, gray=230):
    widget = ArenaCanvas()
    widget.set_frame(_frame(app, gray))
    widget.set_shapes(
        [
            {
                "type": "circle",
                "params": (100, 100, 40),
                "mode": "include",
                "arena_id": 0,
            }
        ]
    )
    return widget


def test_mean_luminance_reads_a_light_frame(app):
    assert _canvas(app, 230).mean_luminance() > 0.8


def test_mean_luminance_reads_a_dark_frame(app):
    assert _canvas(app, 20).mean_luminance() < 0.2


def _rendered(canvas):
    out = QImage(canvas.width(), canvas.height(), QImage.Format_RGB888)
    out.fill(QColor(255, 255, 255))
    painter = QPainter(out)
    painter.drawPixmap(0, 0, canvas._scaled)
    canvas.render_overlay(painter)
    painter.end()
    return out


def test_veil_colour_does_not_depend_on_frame_polarity(app):
    """The veil is now a fixed blue fill (matching the boundary) rather than
    a luminance-adaptive black/white -- so its RGB colour tuple is the same
    regardless of whether the footage is light or dark. (A dedicated pixel
    test of the blue tint itself lives in
    ``test_veil_fill_is_blue_tinted_relative_to_background`` below, sampled
    off the arena-number glyph.)"""
    from hydra_suite.trackerkit.gui.widgets.arena_style import frame_palette

    assert frame_palette(0.90).veil == frame_palette(0.10).veil


def test_exclude_hole_is_not_veiled(app):
    """An exclude zone is outside the ROI, so it must not carry the veil."""
    canvas = _canvas(app, 230)
    canvas.set_shapes(
        [
            {
                "type": "circle",
                "params": (100, 100, 40),
                "mode": "include",
                "arena_id": 0,
            },
            {
                "type": "circle",
                "params": (100, 100, 15),
                "mode": "exclude",
                "arena_id": 0,
            },
        ]
    )
    out = _rendered(canvas)
    # Sample off-centre (still within the r=15 exclude hole) to avoid the
    # arena-number glyph, which is drawn at the shape's exact centroid.
    in_hole = QColor(out.pixel(100, 88)).lightness()
    in_ring = QColor(out.pixel(130, 100)).lightness()
    assert in_hole > in_ring


def test_outline_width_is_independent_of_zoom(app):
    """The core requirement: constant APPARENT thickness at any zoom."""
    canvas = _canvas(app, 230)
    canvas.set_zoom(1.0)
    width_at_1x = canvas._line_width()
    canvas.set_zoom(4.0)
    assert canvas._line_width() == width_at_1x


def _canvas_with_include_and_exclude(app, gray=150):
    """Non-overlapping include/exclude circles so each shape's own outline
    stroke lands on unambiguous, non-shared pixels."""
    widget = ArenaCanvas()
    widget.set_frame(_frame(app, gray))
    widget.set_shapes(
        [
            {
                "type": "circle",
                "params": (60, 100, 30),
                "mode": "include",
                "arena_id": 0,
            },
            {
                "type": "circle",
                "params": (140, 100, 15),
                "mode": "exclude",
                "arena_id": 0,
            },
        ]
    )
    return widget


def test_include_outline_pixel_is_green_dominant(app):
    canvas = _canvas_with_include_and_exclude(app)
    out = _rendered(canvas)
    r, g, b = QColor(out.pixel(60, 70)).getRgb()[:3]
    assert g == max(r, g, b)


def test_exclude_outline_pixel_is_red_dominant(app):
    canvas = _canvas_with_include_and_exclude(app)
    out = _rendered(canvas)
    r, g, b = QColor(out.pixel(155, 100)).getRgb()[:3]
    assert r == max(r, g, b)


def test_veil_fill_is_blue_tinted_relative_to_background(app):
    """A pixel inside the net ROI, away from any stroke or the arena-number
    glyph, should show an elevated blue channel relative to the untouched
    background -- the fixed blue veil, not the old luminance-adaptive
    black/white fill."""
    canvas = _canvas(app, gray=100)
    out = _rendered(canvas)
    inside = QColor(out.pixel(75, 75))
    outside = QColor(out.pixel(5, 5))
    assert inside.blue() > outside.blue()


def test_boundary_stroke_pixel_is_blue_dominant_and_thicker_than_zone_outline(app):
    """A pixel that falls within the thicker boundary stroke's width but
    outside the thin zone outline's width (just inside the include shape's
    edge) should be blue-dominant -- proving the boundary is both a
    distinct colour and visibly wider than the zone outline."""
    canvas = _canvas(app, gray=150)
    out = _rendered(canvas)
    r, g, b = QColor(out.pixel(60, 68)).getRgb()[:3]
    assert b == max(r, g, b)

    # And directly: the boundary width is a measurably larger integer than
    # the zone outline width at the same viewport size.
    zone_width = zone_line_width_px(min(canvas.parentWidth(), canvas.parentHeight()))
    boundary_width = canvas._boundary_width_for(0)
    assert boundary_width > zone_width


def test_boundary_for_current_arena_is_thicker_than_non_current(app):
    canvas = _canvas(app, 230)
    canvas.set_shapes(
        [
            {
                "type": "circle",
                "params": (60, 100, 30),
                "mode": "include",
                "arena_id": 0,
            },
            {
                "type": "circle",
                "params": (140, 100, 20),
                "mode": "include",
                "arena_id": 1,
            },
        ]
    )
    canvas.set_current_arena(0)
    current_width = canvas._boundary_width_for(0)
    non_current_width = canvas._boundary_width_for(1)
    assert current_width > non_current_width


def test_paint_arena_number_draws_a_dark_glyph_over_a_light_halo(app):
    """Halo and glyph composite once, so their overlap is not double-darkened.

    A dark glyph on a white halo over mid grey must leave the glyph body
    DARKER than the surrounding halo ring, and the halo LIGHTER than the
    untouched background. Double-compositing would darken the halo/glyph
    overlap and invert the second relationship.
    """
    out = QImage(160, 160, QImage.Format_ARGB32)
    out.fill(QColor(128, 128, 128))
    painter = QPainter(out)
    paint_arena_number(
        painter, "1", QPointF(80, 80), 90, (20, 20, 20), (255, 255, 255), 0.70
    )
    painter.end()

    lightness = [
        [QColor(out.pixel(x, y)).lightness() for x in range(160)] for y in range(160)
    ]
    flat = [v for row in lightness for v in row]
    background = QColor(out.pixel(2, 2)).lightness()
    assert min(flat) < background, "no dark glyph body was drawn"
    assert max(flat) > background, "no light halo was drawn"


def test_paint_arena_number_respects_alpha(app):
    """At alpha 0 nothing is drawn; the whole layer is composited once."""
    out = QImage(160, 160, QImage.Format_ARGB32)
    out.fill(QColor(128, 128, 128))
    painter = QPainter(out)
    paint_arena_number(
        painter, "1", QPointF(80, 80), 90, (20, 20, 20), (255, 255, 255), 0.0
    )
    painter.end()
    assert QColor(out.pixel(80, 80)).lightness() == QColor(128, 128, 128).lightness()


def test_current_arena_boundary_is_heavier(app):
    canvas = _canvas(app, 230)
    canvas.set_current_arena(None)
    plain = canvas._boundary_width_for(0)
    canvas.set_current_arena(0)
    assert canvas._boundary_width_for(0) > plain


def test_arena_number_centres_on_net_region_not_raw_include_centroid(app, monkeypatch):
    """Fix Wave 14, Fix 1: the number must be placed at the centre of the NET
    (union-of-includes minus union-of-excludes) region -- not the average of
    the raw include shapes' own centroids, which ignores excludes entirely.

    Geometry: one large include circle centred at (100, 100) with an exclude
    circle offset to the left that eats roughly the left half of it. The net
    visible region is the right-hand crescent, so the glyph must land
    meaningfully to the RIGHT of the raw include circle's own centre (x=100
    in image space / viewport space, since zoom is 1.0 and there's no pan).
    """
    import hydra_suite.trackerkit.gui.widgets.arena_canvas as arena_canvas_module

    canvas = ArenaCanvas()
    canvas.set_frame(_frame(app, 200))
    canvas.set_shapes(
        [
            {
                "type": "circle",
                "params": (100, 100, 50),
                "mode": "include",
                "arena_id": 0,
            },
            {
                "type": "circle",
                "params": (60, 100, 50),
                "mode": "exclude",
                "arena_id": 0,
            },
        ]
    )

    captured_positions = []
    real_paint_arena_number = arena_canvas_module.paint_arena_number

    def _spy(painter, text, pos, *args, **kwargs):
        captured_positions.append(pos)
        return real_paint_arena_number(painter, text, pos, *args, **kwargs)

    monkeypatch.setattr(arena_canvas_module, "paint_arena_number", _spy)

    out = QImage(canvas.width(), canvas.height(), QImage.Format_RGB888)
    out.fill(QColor(255, 255, 255))
    painter = QPainter(out)
    painter.drawPixmap(0, 0, canvas._scaled)
    canvas.render_overlay(painter)
    painter.end()

    assert len(captured_positions) == 1
    glyph_pos = captured_positions[0]
    # The raw include circle's own centre is x=100. The net (post-exclusion)
    # region's centre must sit meaningfully to its right, not on top of it
    # and not to the left (which would put the number back inside the
    # excluded/removed area).
    assert glyph_pos.x() > 100 + 5, (
        f"glyph placed at x={glyph_pos.x()}, expected meaningfully right of "
        "the raw include circle's own centre (x=100)"
    )
