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


def test_start_roi_selection_applies_zoom_synchronously(window, tmp_path):
    """Fix Wave 3, Fix 2: start_roi_selection must apply the slider's zoom to
    the canvas SYNCHRONOUSLY, not rely on the update_roi_preview()'s deferred
    QTimer.singleShot(50, _display_roi_with_zoom) reapplication.

    Before the fix, the canvas's `_zoom` stayed at whatever it happened to be
    (e.g. left at 1.0 by an earlier already_scaled=True push) until that timer
    fired 50ms later. A point placed before the timer fires would convert
    through the WRONG zoom, permanently corrupting its recorded coordinates.
    This assertion is made with NO event loop pumped in between, so it can
    only pass if the zoom was applied synchronously inside the call itself.
    """
    import pytest
    from PySide6.QtGui import QImage

    # _ensure_roi_base_frame requires a non-empty video path on the setup
    # panel even when roi_base_frame is already loaded, else it shows a
    # blocking "No Video" QMessageBox instead of proceeding.
    window._setup_panel.file_line.setText(str(tmp_path / "dummy.mp4"))

    # Leave the canvas zoom stale at a value that disagrees with the slider,
    # as an earlier already_scaled=True push would.
    window.video_label.set_zoom(1.0)

    window.roi_base_frame = QImage(640, 480, QImage.Format_RGB888)
    window.roi_current_mode = "polygon"
    window.roi_current_zone_type = "include"
    window.slider_zoom.setValue(150)

    window._session_orch.start_roi_selection()

    assert window.video_label._zoom == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Fix Wave 10: the ArenaCanvas overlay must not survive into tracking/preview
# display modes with its raw image-space shapes, since the canvas's own
# `_zoom` gets reset to 1.0 by any already_scaled=True push -- rendering
# leftover shapes wildly oversized and misaligned. Each frame-push site must
# clear the canvas's shapes; `update_roi_preview()` restores them the moment
# ROI editing resumes.
# ---------------------------------------------------------------------------


def _fake_rgb(width: int = 64, height: int = 48):
    import numpy as np

    return np.zeros((height, width, 3), dtype=np.uint8)


def test_on_new_frame_clears_stale_arena_shapes(window):
    """Fix Wave 10: tracking's on_new_frame must clear leftover ROI shapes."""
    window.video_label.set_shapes(
        [{"type": "circle", "cx": 2256, "cy": 2256, "r": 2000}]
    )
    assert window.video_label._shapes != []

    window._tracking_orch.on_new_frame(_fake_rgb())

    assert window.video_label._shapes == []


def test_update_preview_display_clears_stale_arena_shapes(window):
    """Fix Wave 10: detection-preview's _update_preview_display must clear
    leftover ROI shapes even when there is no preview frame loaded yet."""
    window.video_label.set_shapes([{"type": "circle", "cx": 100, "cy": 100, "r": 50}])
    assert window.video_label._shapes != []

    window.preview_frame_original = None
    window._detection_panel._update_preview_display()

    assert window.video_label._shapes == []


def test_redisplay_detection_test_clears_stale_arena_shapes(window):
    """Fix Wave 10: _redisplay_detection_test must clear leftover shapes."""
    window.video_label.set_shapes([{"type": "circle", "cx": 100, "cy": 100, "r": 50}])
    assert window.video_label._shapes != []

    window.detection_test_result = None
    window._detection_panel._redisplay_detection_test()

    assert window.video_label._shapes == []


def test_update_roi_preview_restores_shapes_after_tracking_cleared_them(window):
    """Fix Wave 10, other direction: once ROI editing resumes after a
    tracking/preview run cleared the canvas's shapes, update_roi_preview()
    must restore them from window.roi_shapes."""
    window.roi_shapes = [{"type": "circle", "cx": 10, "cy": 10, "r": 5}]

    # Simulate a tracking frame having cleared the overlay.
    window._tracking_orch.on_new_frame(_fake_rgb())
    assert window.video_label._shapes == []

    window._session_orch.update_roi_preview()

    assert window.video_label._shapes == window.roi_shapes
