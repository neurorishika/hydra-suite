from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QApplication = QtWidgets.QApplication
Qt = QtCore.Qt

from hydra_suite.classkit.config.schemas import Factor, LabelingScheme  # noqa: E402
from hydra_suite.classkit.gui.main_window import MainWindow  # noqa: E402
from hydra_suite.classkit.gui.panels import labeling_options  # noqa: E402

COLORS = ["blue", "green", "orange", "pink", "yellow", "none", "purple", "white"]


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def _window(qapp, width, height):
    scheme = LabelingScheme(
        name="s",
        factors=[Factor("tag_1", COLORS + ["unknown"]), Factor("tag_2", COLORS)],
        training_modes=[],
    )
    win = MainWindow()
    win._load_label_scheme = lambda: (scheme, {})
    win.classes = COLORS
    win.resize(width, height)
    win._stacked.setCurrentIndex(1)
    win.rebuild_label_buttons()
    win.view_mode_combo.setCurrentIndex(win.view_mode_combo.findData("labeling"))
    win.set_explorer_mode("labeling")
    win.show()
    for _ in range(10):
        qapp.processEvents()
    return win


def _inside_window(win, widget) -> bool:
    top_left = widget.mapTo(win, QtCore.QPoint(0, 0))
    rect = QtCore.QRect(top_left, widget.size())
    return widget.isVisible() and win.rect().contains(rect)


def test_landscape_window_puts_options_beside_a_taller_plot(qapp):
    win = _window(qapp, 2000, 850)
    assert win.labeling_panel.isVisible() and win.labeling_panel.beside
    assert win.explorer_split.orientation() == Qt.Horizontal
    # The old band left the plot ~260 px tall at this size.
    assert win.explorer.height() > 500
    assert _inside_window(win, win.btn_start_labeling)
    assert _inside_window(win, win.history_scroll) or not win.history_scroll.isVisible()


def test_narrow_window_falls_back_to_band_below_plot(qapp):
    win = _window(qapp, 1280, 800)
    assert not win.labeling_panel.beside
    assert win.explorer_split.orientation() == Qt.Vertical
    assert _inside_window(win, win.btn_start_labeling)
    assert win.al_grid_balance_btn.isVisibleTo(win.labeling_panel)


def test_placement_has_hysteresis(qapp):
    win = _window(qapp, 2000, 850)
    ctrl = win._labeling_placement
    w, h = win.explorer_split.width(), win.explorer_split.height()
    beside, below = ctrl.scores(w, h)
    assert beside > below
    # Shrink just past the break-even point: the layout must not flip yet.
    narrower = 2000
    while True:
        narrower -= 10
        win.resize(narrower, 850)
        qapp.processEvents()
        b, l = ctrl.scores(win.explorer_split.width(), win.explorer_split.height())
        if l > b:
            break
    assert win.labeling_panel.beside  # within the 10% margin
    win.resize(1280, 850)
    qapp.processEvents()
    assert not win.labeling_panel.beside


def test_panel_hidden_outside_labeling_mode(qapp):
    win = _window(qapp, 2000, 850)
    win.set_explorer_mode("explore")
    qapp.processEvents()
    assert not win.labeling_panel.isVisible()


def test_widgets_are_shared_and_wired(qapp, monkeypatch):
    called = []
    monkeypatch.setattr(
        MainWindow, "_start_prepared_labeling_batch", lambda self: called.append(1)
    )
    win = _window(qapp, 2000, 850)
    for name in labeling_options.WIDGET_NAMES:
        assert getattr(win, name) is getattr(win.labeling_panel, name)
    win.btn_start_labeling.setEnabled(True)
    win.btn_start_labeling.click()
    assert called == [1]


def test_candidate_group_follows_table(qapp):
    win = _window(qapp, 2000, 850)
    assert not win.prepared_candidates_group.isVisibleTo(win.labeling_panel)
    win._prepared_candidate_indices = [0]
    win._prepared_candidate_source = "random"
    win.image_paths = [Path(__file__)]
    win.image_labels = [None]
    win._refresh_prepared_candidate_table("1 candidate")
    assert win.prepared_candidates_group.isVisibleTo(win.labeling_panel)


def test_factor_buttons_wrap_and_left_panel_never_scrolls_sideways(qapp):
    win = _window(qapp, 1440, 900)
    stepper = win._stepper
    assert stepper is not None
    grid = stepper._btn_row
    assert grid.columnCount() == 3 and grid.rowCount() >= 3
    assert not win.context_panel.horizontalScrollBar().isVisible()
