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


def test_veil_darkens_the_arena_interior_on_light_footage(app):
    """Veil goes INSIDE the ROI, per the design decision."""
    canvas = _canvas(app, 230)
    out = _rendered(canvas)
    inside = QColor(out.pixel(100, 100)).lightness()
    outside = QColor(out.pixel(5, 5)).lightness()
    assert inside < outside


def test_veil_lightens_the_arena_interior_on_dark_footage(app):
    canvas = _canvas(app, 20)
    out = _rendered(canvas)
    inside = QColor(out.pixel(100, 100)).lightness()
    outside = QColor(out.pixel(5, 5)).lightness()
    assert inside > outside


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
    in_hole = QColor(out.pixel(100, 100)).lightness()
    in_ring = QColor(out.pixel(130, 100)).lightness()
    assert in_hole > in_ring


def test_outline_width_is_independent_of_zoom(app):
    """The core requirement: constant APPARENT thickness at any zoom."""
    canvas = _canvas(app, 230)
    canvas.set_zoom(1.0)
    width_at_1x = canvas._line_width()
    canvas.set_zoom(4.0)
    assert canvas._line_width() == width_at_1x


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


def test_current_arena_outline_is_heavier(app):
    canvas = _canvas(app, 230)
    canvas.set_current_arena(None)
    plain = canvas._outline_width_for(0)
    canvas.set_current_arena(0)
    assert canvas._outline_width_for(0) > plain
