"""The main window's preview is an ArenaCanvas and zoom survives ROI drawing."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.trackerkit.gui.widgets.arena_canvas import ArenaCanvas  # noqa: E402


@pytest.fixture(scope="module")
def window():
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    return MainWindow()


def test_preview_widget_is_an_arena_canvas(window):
    assert isinstance(window.video_label, ArenaCanvas)


def test_zoom_slider_stays_enabled_during_roi_drawing(window):
    """The bug: zoom used to be force-disabled because clicks were image coords."""
    window.roi_selection_active = True
    window._session_orch._sync_contextual_controls()
    assert window.slider_zoom.isEnabled() is True


def test_source_has_no_remaining_roi_zoom_locks():
    """Guards the three deletions -- a re-introduced lock silently regresses this."""
    import inspect

    from hydra_suite.trackerkit.gui.orchestrators import session

    source = inspect.getsource(session)
    assert "slider_zoom.setEnabled(False)" not in source
