"""ArenaCanvas coordinate transform and click/drag disambiguation.

Constructed under offscreen Qt; never shown, never exec()'d.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.gui.widgets.arena_canvas import ArenaCanvas  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas(app):
    widget = ArenaCanvas()
    widget.set_frame(QImage(400, 300, QImage.Format_RGB888))
    return widget


@pytest.mark.parametrize("zoom", [0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
def test_image_viewport_round_trip_is_identity(canvas, zoom):
    """The regression test for the defect underlying both reported problems."""
    canvas.set_zoom(zoom)
    for point in [(0.0, 0.0), (123.0, 45.0), (399.0, 299.0)]:
        back = canvas.to_image(canvas.to_viewport(*point))
        assert back == pytest.approx(point, abs=1e-6)


def test_viewport_origin_maps_to_image_origin(canvas):
    canvas.set_zoom(3.0)
    assert canvas.to_image(QPointF(0.0, 0.0)) == pytest.approx((0.0, 0.0))


def test_zoom_scales_viewport_coordinates(canvas):
    canvas.set_zoom(2.0)
    point = canvas.to_viewport(50.0, 50.0)
    assert (point.x(), point.y()) == pytest.approx((100.0, 100.0))


def test_widget_size_tracks_zoomed_frame(canvas):
    canvas.set_zoom(2.0)
    assert (canvas.width(), canvas.height()) == (800, 600)


def test_small_left_movement_is_a_click(canvas):
    assert canvas._is_click(0, 0, 2, 1) is True


def test_large_left_movement_is_a_drag(canvas):
    assert canvas._is_click(0, 0, 40, 3) is False


def test_threshold_boundary_counts_as_a_drag(canvas):
    """At exactly the threshold the gesture is a drag, so a click is strictly under."""
    assert canvas._is_click(0, 0, 3, 0) is False
