"""Regression guards for the Fix-Wave-1 zoom architecture bugs.

Covers:
- Fix 1: `_set_video_pixmap(..., already_scaled=True)` must not let the
  canvas ALSO scale a pixmap the caller already sized for display.
- Fix 2: `update_roi_preview()` must never change the zoom slider's value as
  a side effect (that used to be scheduled via an auto-refit on every call).
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def window():
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    return MainWindow()


def _solid_pixmap(width: int, height: int):
    from PySide6.QtGui import QColor, QPixmap

    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("black"))
    return pixmap


def test_already_scaled_pixmap_is_not_scaled_again(window):
    """Fix 1: a pre-scaled pixmap must land on the canvas at its own size."""
    # Leave some stale zoom on the canvas (as if a previous frame left it
    # non-1.0) to prove already_scaled=True actively resets it rather than
    # happening to already be 1.0.
    window.video_label.set_zoom(2.5)

    pixmap = _solid_pixmap(500, 360)
    window._set_video_pixmap(pixmap, already_scaled=True)

    assert window.video_label._zoom == 1.0
    assert window.video_label.width() == 500
    assert window.video_label.height() == 360


def test_non_scaled_pixmap_still_gets_canvas_zoom_applied(window):
    """Without already_scaled, the canvas must still scale by its own zoom."""
    window.video_label.set_zoom(2.0)

    pixmap = _solid_pixmap(100, 80)
    window._set_video_pixmap(pixmap)

    assert window.video_label.width() == int(100 * 2.0)
    assert window.video_label.height() == int(80 * 2.0)


def test_update_roi_preview_never_changes_zoom_slider(window):
    """Fix 2: adding/removing ROI points must not silently auto-refit.

    NOTE: the bug this guards (`QTimer.singleShot(10, self._mw._fit_image_to_screen)`
    scheduled from inside `update_roi_preview`) is a DEFERRED callback -- it only
    fires once Qt's event loop is pumped. Asserting the slider value synchronously
    right after calling `update_roi_preview()`, with no event loop pumped in
    between, would pass identically whether or not the buggy line is present.
    We therefore pump the event loop (`QTest.qWait`) after each call so a
    reintroduced singleShot auto-refit actually gets a chance to fire and flip
    the slider before we assert on it.
    """
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest

    window.roi_base_frame = QImage(640, 480, QImage.Format_RGB888)
    window.roi_selection_active = True
    window.roi_current_mode = "polygon"
    window.roi_points = []

    window.slider_zoom.setValue(137)
    initial_zoom = window.slider_zoom.value()

    # Simulate adding several points one at a time (each add_roi_point call
    # ends with update_roi_preview() in the real flow -- call it directly to
    # isolate the regression this guards).
    for i in range(5):
        window.roi_points.append((10.0 * i, 20.0 * i))
        window._session_orch.update_roi_preview()
        QTest.qWait(50)
        assert window.slider_zoom.value() == initial_zoom, (
            f"slider_zoom.value() changed after point {i}: "
            f"{window.slider_zoom.value()} != {initial_zoom}"
        )

    assert window.slider_zoom.value() == initial_zoom


def test_update_roi_preview_never_schedules_a_fit_to_screen():
    """update_roi_preview must never auto-refit -- zoom only changes on
    explicit user action. A reintroduced singleShot(..., _fit_image_to_screen)
    inside this method would silently reset the zoom slider on every point
    add/remove/finish/clear/undo, which is exactly the bug this guards."""
    import inspect

    from hydra_suite.trackerkit.gui.orchestrators import session

    source = inspect.getsource(session.SessionOrchestrator.update_roi_preview)
    assert "_fit_image_to_screen" not in source
