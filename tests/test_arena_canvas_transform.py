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


def test_right_button_drag_while_drawing_does_not_remove_point(canvas, app):
    """Right-button drag should not emit point_removed (requires click, not drag).

    This test verifies the fix for the double-fire bug: mouseMoveEvent emits
    pan_delta for any button's drag past the threshold, so without gating
    on was_click, a right-button drag would both pan the view and delete
    the last point.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QMouseEvent

    canvas.set_drawing(True)

    # Track emissions of point_removed signal
    removed_count = []
    canvas.point_removed.connect(lambda: removed_count.append(1))

    # Simulate a right-button press
    press_event = QMouseEvent(
        QMouseEvent.MouseButtonPress,
        QPointF(100.0, 100.0),
        QPointF(100.0, 100.0),
        Qt.RightButton,
        Qt.RightButton,
        Qt.NoModifier,
    )
    canvas.mousePressEvent(press_event)

    # Simulate a move past the threshold (drag, not click)
    move_event = QMouseEvent(
        QMouseEvent.MouseMove,
        QPointF(150.0, 100.0),  # 50px right, well past CLICK_DRAG_THRESHOLD_PX (3px)
        QPointF(150.0, 100.0),
        Qt.NoButton,
        Qt.RightButton,
        Qt.NoModifier,
    )
    canvas.mouseMoveEvent(move_event)

    # Simulate the release
    release_event = QMouseEvent(
        QMouseEvent.MouseButtonRelease,
        QPointF(150.0, 100.0),
        QPointF(150.0, 100.0),
        Qt.RightButton,
        Qt.NoButton,
        Qt.NoModifier,
    )
    canvas.mouseReleaseEvent(release_event)

    # Assert point_removed was NOT emitted (list should be empty)
    assert len(removed_count) == 0, "Right-button drag should not emit point_removed"
