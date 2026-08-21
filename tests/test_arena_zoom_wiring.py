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


def test_update_preview_display_does_not_clear_shapes_during_plain_display(window):
    """Fix Wave 13: `_update_preview_display` is the shared "show the current
    frame" method used for BOTH normal video review/scrubbing AND real
    detection-test rendering. When there is no active detection test
    (`detection_test_result is None`), it must NOT clear pre-set ROI
    shapes -- Fix Wave 10 previously added an unconditional
    `set_shapes([])` at its top that fired on every normal frame display
    too, wiping a legitimately-loaded ROI overlay the instant a video with
    saved ROIs opened or the user scrubbed to any frame. The
    detection-test case remains covered separately by
    `_redisplay_detection_test`'s own clear (see the test below)."""
    import numpy as np

    shapes = [{"type": "circle", "cx": 100, "cy": 100, "r": 50}]
    window.video_label.set_shapes(shapes)
    assert window.video_label._shapes != []

    # A real (small, fake) frame so this actually reaches past the
    # `detection_test_result` check into the real rendering body below,
    # rather than only exercising the early `preview_frame_original is None`
    # return.
    window.preview_frame_original = np.zeros((48, 64, 3), dtype=np.uint8)
    window.detection_test_result = None
    window._detection_panel._update_preview_display()

    assert window.video_label._shapes == shapes


def test_update_preview_display_still_clears_shapes_when_redirected_to_detection_test(
    window,
):
    """Fix Wave 13 (no regression on the original Fix Wave 10 bug): when a
    real detection-test result IS present, `_update_preview_display` must
    still redirect into `_redisplay_detection_test`, whose own (untouched)
    clear still fires -- the original misaligned-overlay bug this whole
    class of fix targets must remain fixed."""
    import numpy as np

    window.video_label.set_shapes([{"type": "circle", "cx": 100, "cy": 100, "r": 50}])
    assert window.video_label._shapes != []

    fake_frame = np.zeros((48, 64, 3), dtype=np.uint8)
    window.preview_frame_original = fake_frame
    window.detection_test_result = (fake_frame, 1.0)

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


# ---------------------------------------------------------------------------
# Fix Wave 14, Fix 3: _on_zoom_changed must route through
# _display_roi_with_zoom whenever roi_shapes are present, even when
# roi_base_frame was never populated (e.g. ROIs restored from a loaded
# config, which never touches roi_base_frame at all).
# ---------------------------------------------------------------------------


def _write_test_video(path, width=64, height=48, n_frames=5):
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10, (width, height))
    for _ in range(n_frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


def test_zoom_changed_routes_through_roi_display_when_config_restored_rois_have_no_base_frame(
    tmp_path,
):
    """The exact user-reported regression: a video opens with a saved
    config's roi_shapes (roi_base_frame is NEVER set by config restoration),
    then the user moves the zoom slider. Before the fix, this fell through
    to `_update_preview_display`, which resets canvas._zoom to 1.0
    regardless of the slider -- producing the "ROI mispositioned with weird
    zoom scaling" symptom. After the fix, roi_shapes alone routes zoom
    changes through `_display_roi_with_zoom`, which keeps canvas._zoom in
    sync with the slider.
    """
    import json

    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    QApplication.instance() or QApplication([])

    video_path = tmp_path / "sample.mp4"
    _write_test_video(video_path)

    config_path = tmp_path / "sample_config.json"
    config_path.write_text(
        json.dumps(
            {
                "roi_shapes": [
                    {
                        "type": "circle",
                        "params": [20, 20, 10],
                        "mode": "include",
                        "arena_id": 0,
                    }
                ]
            }
        )
    )

    win = MainWindow()
    win._setup_video_file(str(video_path), skip_config_load=False)

    assert win.roi_shapes, "config-restored roi_shapes must be present"
    assert (
        getattr(win, "roi_base_frame", None) is None
    ), "roi_base_frame must NOT be set by config restoration -- that's the root cause"

    # _display_roi_with_zoom reads the slider's own value, not the signal's
    # `value` argument -- set both, matching what the real valueChanged
    # signal wiring does (the slider is already at its new value by the
    # time the connected slot runs).
    win.slider_zoom.setValue(150)
    win._detection_panel._on_zoom_changed(150)

    assert win.video_label._zoom == pytest.approx(1.5)


def test_zoom_changed_still_routes_through_roi_display_when_roi_base_frame_is_set(
    window,
):
    """No regression on the case that already worked: when roi_base_frame
    IS populated (e.g. via start_roi_selection during manual ROI drawing),
    zoom changes must still route through _display_roi_with_zoom."""
    from PySide6.QtGui import QImage

    window.roi_base_frame = QImage(640, 480, QImage.Format_RGB888)
    window.roi_shapes = [
        {"type": "circle", "params": [20, 20, 10], "mode": "include", "arena_id": 0}
    ]
    window.detection_test_result = None

    window._detection_panel._on_zoom_changed(150)

    assert window.video_label._zoom == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Fix Wave 15: _update_preview_display must not bake the current zoom into
# canvas._frame (via already_scaled=True) -- it must push the NATIVE frame
# and sync canvas zoom via set_zoom(), exactly like
# start_roi_selection/_display_roi_with_zoom already do. Otherwise, once it
# runs at a non-1.0 zoom, canvas._frame silently becomes pre-shrunk while
# canvas._zoom resets to 1.0, and any SUBSEQUENT zoom change (routed through
# _display_roi_with_zoom) compounds on top of the already-shrunk frame
# instead of replacing it.
# ---------------------------------------------------------------------------


def _fake_rgb_wave15(width: int = 300, height: int = 200):
    import numpy as np

    return np.zeros((height, width, 3), dtype=np.uint8)


def test_update_preview_display_does_not_compound_with_subsequent_roi_zoom(window):
    """Exact root-cause compounding scenario: _update_preview_display at 50%
    zoom, THEN a zoom-slider move to 150% routed through
    _display_roi_with_zoom, must land on canvas._scaled == native * 1.5
    (450x300), NOT the compounded 225x150 the broken code produced."""
    window.preview_frame_original = _fake_rgb_wave15(300, 200)
    window.roi_shapes = [
        {"type": "circle", "params": [20, 20, 10], "mode": "include", "arena_id": 0}
    ]
    window.detection_test_result = None
    dp = window._detection_panel

    window.slider_zoom.setValue(50)
    dp._update_preview_display()

    window.slider_zoom.setValue(150)
    dp._on_zoom_changed(150)

    assert window.video_label._zoom == pytest.approx(1.5)
    assert window.video_label._scaled.width() == 450
    assert window.video_label._scaled.height() == 300


def test_update_preview_display_is_idempotent_at_same_zoom_across_calls(window):
    """Most-likely real-world trigger: a zoom slider value left over from a
    previously opened video (still 50 here), then a fresh video loads and
    _update_preview_display runs again at the SAME nominal zoom. Repeated
    calls at an unchanged zoom value must not compound -- each call must
    independently produce native * zoom, not native * zoom^2 on the second
    call."""
    window.preview_frame_original = _fake_rgb_wave15(300, 200)
    window.roi_shapes = []
    window.detection_test_result = None
    dp = window._detection_panel

    window.slider_zoom.setValue(50)
    dp._update_preview_display()
    assert window.video_label._scaled.width() == 150
    assert window.video_label._scaled.height() == 100

    # Simulate a fresh video load at the same nominal slider value.
    dp._update_preview_display()
    assert window.video_label._scaled.width() == 150
    assert window.video_label._scaled.height() == 100
