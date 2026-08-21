"""The main window hosts an ArenaPanel and no longer has the old ROI toolbar."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.trackerkit.gui.panels.arena_panel import ArenaPanel  # noqa: E402


@pytest.fixture(scope="module")
def window():
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    return MainWindow()


def test_window_has_an_arena_panel(window):
    assert isinstance(window.arena_panel, ArenaPanel)


@pytest.mark.parametrize(
    "attr",
    [
        "combo_roi_mode",
        "combo_roi_zone",
        "btn_start_roi",
        "btn_finish_roi",
        "btn_undo_roi",
        "btn_new_arena",
        "btn_generate_grid",
        "btn_clear_roi",
        "btn_crop_video",
    ],
)
def test_old_roi_toolbar_widgets_are_gone(window, attr):
    assert not hasattr(window, attr)


def test_roi_efficiency_readout_survives(window):
    """Unrelated to arena editing; must not be collateral damage."""
    assert hasattr(window, "roi_status_label")
    assert hasattr(window, "roi_optimization_label")


def test_panel_starts_in_the_empty_state(window):
    window.roi_shapes = []
    window.arena_panel.set_shapes([])
    assert window.arena_panel.lbl_default.text() == (
        "By default, the whole video is used."
    )
