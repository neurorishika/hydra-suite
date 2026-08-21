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


# --- Fix wave 8, round 2, finding 2: Clear Arena must flip made_via_grid
# to False even if the arena set was previously grid-generated -- otherwise
# reopening the grid dialog to "modify" it would silently regenerate and
# resurrect the shapes that were just cleared. Driven against a REAL
# ArenaPanel (not a MagicMock stand-in) so the wiring is actually verified,
# per the reviewer's note that a mocked ``_panels.arena`` would hide this. ---


def test_clear_arena_marks_hand_drawn_even_if_previously_grid_generated(window):
    window.roi_base_frame = None
    window.roi_shapes = [
        {"type": "circle", "params": [1, 1, 1], "mode": "include", "arena_id": 0},
    ]
    window.arena_panel.set_shapes(window.roi_shapes)
    window.arena_panel.mark_grid_generated({"first_arena_id": 0})
    assert window.arena_panel.made_via_grid is True

    window._on_clear_arena(0)

    assert window.arena_panel.made_via_grid is False
