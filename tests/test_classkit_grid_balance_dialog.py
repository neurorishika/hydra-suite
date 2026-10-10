from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtCore = pytest.importorskip("PySide6.QtCore")
QtGui = pytest.importorskip("PySide6.QtGui")
QtTest = pytest.importorskip("PySide6.QtTest")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QApplication = QtWidgets.QApplication
Qt = QtCore.Qt
QTest = QtTest.QTest

from hydra_suite.classkit.core.al.grid_balance import GridBalanceSession  # noqa: E402
from hydra_suite.classkit.gui.dialogs.grid_balance import (  # noqa: E402
    GridBalanceDialog,
)

LABELS = [["a", "b"], ["x", "y"]]


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


@pytest.fixture(autouse=True)
def no_settle_delay(monkeypatch):
    # Input guards are exercised explicitly in their own tests.
    monkeypatch.setattr(GridBalanceDialog, "SETTLE_S", 0.0)
    monkeypatch.setattr(GridBalanceDialog, "ADVANCE_LOCK_S", 0.0)


def _images(tmp_path, n):
    paths = []
    for i in range(n):
        img = QtGui.QImage(40, 30, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor.fromHsv((i * 37) % 360, 200, 220))
        p = tmp_path / f"img_{i:03d}.png"
        img.save(str(p))
        paths.append(p)
    return paths


def _wait_loaded(qapp, dialog, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if all(t.pixmap is not None or t.empty for t in dialog._tiles):
            return True
        time.sleep(0.01)
    return False


def _dialog(tmp_path, n=60, labels=None, target_cap=None, paths=None):
    rng = np.random.default_rng(3)
    probs = [rng.dirichlet(np.ones(2), size=n), rng.dirichlet(np.ones(2), size=n)]
    if labels is None:
        labels = ["a_x"] * 3 + [None] * (n - 3)
    session = GridBalanceSession(
        probs, LABELS, labels, grid_size=9, target_cap=target_cap
    )
    dialog = GridBalanceDialog(session, paths or _images(tmp_path, n))
    applied = []
    dialog.label_applied.connect(lambda i, lab: applied.append((i, lab)))
    dialog.show()
    QTest.qWaitForWindowExposed(dialog)
    app = QApplication.instance()
    _wait_loaded(app, dialog)
    return dialog, session, applied


def _press(dialog, key, modifier=Qt.NoModifier):
    """Key press after the grid's images have loaded."""
    _wait_loaded(QApplication.instance(), dialog)
    QTest.keyClick(dialog, key, modifier)


def test_number_key_labels_tile_and_mouse_click_too(qapp, tmp_path):
    dialog, session, applied = _dialog(tmp_path)
    combo = session.current
    idx = session.grid[4]
    _press(dialog, Qt.Key_5)
    assert applied == [(idx, combo)]
    idx2 = session.grid[0]
    QTest.mouseClick(dialog._tiles[0], Qt.LeftButton)
    assert applied[-1] == (idx2, combo)
    assert session.counts[combo] == 2


def test_enter_refreshes_without_labeling_and_n_moves_on(qapp, tmp_path):
    dialog, session, applied = _dialog(tmp_path)
    first_combo, first_grid = session.current, list(session.grid)
    _press(dialog, Qt.Key_Return)
    assert applied == [] and set(first_grid).isdisjoint(session.grid)
    assert dialog.isVisible()  # Enter must not hit the Close button
    _press(dialog, Qt.Key_N)
    assert session.current != first_combo and applied == []


def test_backspace_and_ctrl_z_undo(qapp, tmp_path):
    dialog, session, applied = _dialog(tmp_path)
    idx = session.grid[2]
    _press(dialog, Qt.Key_3)
    _press(dialog, Qt.Key_Backspace)
    assert applied == [(idx, session.current), (idx, None)]
    assert session.grid[2] == idx
    _press(dialog, Qt.Key_1)
    _press(dialog, Qt.Key_Z, Qt.ControlModifier)
    assert applied[-1][1] is None and not session.hits
    assert not dialog.undo_btn.isEnabled()


def test_reaching_target_advances_and_all_done(qapp, tmp_path):
    n = 60
    dialog, session, applied = _dialog(tmp_path, n=n, target_cap=1)
    n_combos = len(session.combos)
    for _ in range(n_combos):
        _press(dialog, Qt.Key_1)
    assert session.finished and len(applied) == n_combos
    assert "All done" in dialog.combo_label.text()
    assert not dialog.none_btn.isEnabled()
    # Undo still works from the finished state.
    _press(dialog, Qt.Key_Backspace)
    assert not session.finished and applied[-1][1] is None


def test_switch_to_4x4_uses_letter_keys(qapp, tmp_path):
    dialog, session, applied = _dialog(tmp_path)
    dialog.size_combo.setCurrentText("4 × 4")
    assert session.grid_size == 16 and len(dialog._tiles) == 16
    idx = session.grid[5]  # second row, second key: W
    _press(dialog, Qt.Key_W)
    assert applied == [(idx, session.current)]
    # Digits 5-9 belong to the old 3x3 grid only.
    _press(dialog, Qt.Key_9)
    assert len(applied) == 1


def test_screenshot_renders(qapp, tmp_path):
    dialog, session, _ = _dialog(tmp_path)
    assert _wait_loaded(qapp, dialog)
    pm = dialog.grab()
    assert not pm.isNull() and pm.width() > 300
    out = os.environ.get("GRID_BALANCE_SHOT")
    if out:
        pm.save(out)


def test_main_window_grid_balance_writes_human_labels_to_db(
    qapp, tmp_path, monkeypatch
):
    from hydra_suite.classkit.config.schemas import Factor, LabelingScheme
    from hydra_suite.classkit.core.store.db import ClassKitDB
    from hydra_suite.classkit.gui.main_window import MainWindow

    n = 40
    paths = [p.resolve() for p in _images(tmp_path, n)]
    db_path = tmp_path / "classkit.db"
    db = ClassKitDB(db_path)
    db.add_images(paths)
    db.update_labels_batch({str(paths[i]): "a_x" for i in range(3)})

    window = MainWindow()
    window.db_path = db_path
    window.image_paths = list(paths)
    window.image_labels = db.get_all_labels()
    window._image_review_status = db.get_label_review_status_by_path()
    rng = np.random.default_rng(0)
    window._model_probs = rng.dirichlet(np.ones(4), size=n)
    window._model_class_names = ["a_x", "a_y", "b_x", "b_y"]
    scheme = LabelingScheme(
        name="s",
        factors=[Factor("color", ["a", "b"]), Factor("side", ["x", "y"])],
        training_modes=[],
    )
    monkeypatch.setattr(window, "_resolve_training_scheme", lambda: scheme)
    window._update_al_status()
    assert window.al_grid_balance_btn.isEnabled()

    seen = {}

    def fake_exec(dialog):
        dialog.show()
        QTest.qWaitForWindowExposed(dialog)
        dialog.activateWindow()
        qapp.processEvents()
        seen["active"] = QApplication.activeWindow() is dialog
        seen["combo"] = dialog.session.current
        seen["first"] = dialog.session.grid[0]
        seen["second"] = dialog.session.grid[1]
        _press(dialog, Qt.Key_1)
        _press(dialog, Qt.Key_2)
        _press(dialog, Qt.Key_Backspace)  # undo the second click
        return 0

    monkeypatch.setattr(GridBalanceDialog, "exec", fake_exec)
    window._open_grid_balance()

    combo, first, second = seen["combo"], seen["first"], seen["second"]
    assert seen["active"], "dialog never became the active window"
    assert combo != "a_x"
    assert window.image_labels[first] == combo
    assert not window.image_labels[second]
    stored = ClassKitDB(db_path).get_label_review_status_by_path()
    assert stored[str(paths[first])]["label"] == combo
    assert stored[str(paths[first])]["verified"] is True
    assert stored[str(paths[first])]["label_source"] == "human"
    assert not stored[str(paths[second])]["label"]
    window.close()


def test_grid_balance_button_disabled_without_predictions(qapp):
    from hydra_suite.classkit.gui.main_window import MainWindow

    window = MainWindow()
    window._update_al_status()
    assert not window.al_grid_balance_btn.isEnabled()
    window.close()


def test_held_key_autorepeat_labels_only_once(qapp, tmp_path):
    dialog, session, applied = _dialog(tmp_path)
    QTest.keyClick(dialog, Qt.Key_1)
    for _ in range(3):
        _wait_loaded(qapp, dialog)
        ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_1, Qt.NoModifier, "1", True)
        QApplication.sendEvent(dialog, ev)
        qapp.processEvents()
    assert len(applied) == 1


def test_double_click_and_fast_reclick_do_not_label_replacement(
    qapp, tmp_path, monkeypatch
):
    monkeypatch.setattr(GridBalanceDialog, "SETTLE_S", 5.0)
    dialog, session, applied = _dialog(tmp_path)
    dialog._ready_at = [0.0] * len(dialog._ready_at)
    QTest.mouseClick(dialog._tiles[0], Qt.LeftButton)
    _wait_loaded(qapp, dialog)  # replacement is visible: only the settle guard
    QTest.mouseDClick(dialog._tiles[0], Qt.LeftButton)
    QTest.mouseClick(dialog._tiles[0], Qt.LeftButton)
    QTest.keyClick(dialog, Qt.Key_1)
    assert len(applied) == 1
    QTest.keyClick(dialog, Qt.Key_2)  # other tiles are unaffected
    assert len(applied) == 2


def test_loading_tile_is_not_clickable(qapp, tmp_path):
    dialog, session, applied = _dialog(tmp_path)
    dialog._pixmaps.clear()
    dialog._prefetcher.shutdown()
    dialog._prefetcher.request = lambda *a, **k: 0  # images never arrive
    dialog._render_tiles()
    QTest.keyClick(dialog, Qt.Key_1)
    QTest.mouseClick(dialog._tiles[1], Qt.LeftButton)
    assert applied == []


def test_input_locked_right_after_auto_advance(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(GridBalanceDialog, "ADVANCE_LOCK_S", 5.0)
    dialog, session, applied = _dialog(tmp_path, target_cap=1)
    first = session.current
    _press(dialog, Qt.Key_1)
    assert session.current != first
    _wait_loaded(qapp, dialog)
    QTest.keyClick(dialog, Qt.Key_2)  # muscle memory for the old grid
    assert len(applied) == 1
    assert "balanced" in dialog.status_label.text()


def test_unreadable_images_are_replaced_not_stuck(qapp, tmp_path):
    n = 60
    paths = _images(tmp_path, n)
    for p in paths[::3]:
        p.write_bytes(b"not an image")
    dialog, session, applied = _dialog(tmp_path, n=n, paths=paths)
    end = time.time() + 3
    while time.time() < end and not _wait_loaded(qapp, dialog, 0.05):
        pass
    bad = {str(p) for p in paths[::3]}
    shown = [dialog._paths[i] for i in session.grid if i is not None]
    assert shown and not bad & set(shown)
    assert all(t.pixmap is not None for t in dialog._tiles if not t.empty)
    assert applied == []  # replaced tiles are never labeled


def test_space_and_enter_do_not_close(qapp, tmp_path):
    dialog, session, applied = _dialog(tmp_path)
    assert dialog.focusWidget() not in dialog._buttons.buttons()
    QTest.keyClick(dialog, Qt.Key_Space)
    QTest.keyClick(dialog, Qt.Key_Enter)
    assert dialog.isVisible() and applied == []


def test_dialog_shrinks_back_after_4x4(qapp, tmp_path):
    dialog, session, _ = _dialog(tmp_path)
    h3 = dialog.height()
    dialog.size_combo.setCurrentText("4 × 4")
    dialog.size_combo.setCurrentText("3 × 3")
    qapp.processEvents()
    assert dialog.height() <= h3 + 2


def test_unreadable_images_never_return_in_later_combinations(qapp, tmp_path):
    n = 60
    paths = _images(tmp_path, n)
    for p in paths[::3]:
        p.write_bytes(b"not an image")
    bad = {str(p) for p in paths[::3]}
    dialog, session, applied = _dialog(tmp_path, n=n, paths=paths)
    for _ in range(len(session.combos)):
        end = time.time() + 3
        while time.time() < end and not _wait_loaded(qapp, dialog, 0.05):
            pass
        shown = {dialog._paths[i] for i in session.grid if i is not None}
        assert not shown & bad
        assert all(t.pixmap is not None for t in dialog._tiles if not t.empty)
        if not session.next_combo():
            break
        dialog._render()
    assert applied == []
    assert all(session.labels[i] is None for i in range(3, n, 3))
