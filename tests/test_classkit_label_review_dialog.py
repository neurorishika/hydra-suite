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

from hydra_suite.classkit.core.al.label_review import LabelReviewModel  # noqa: E402
from hydra_suite.classkit.gui.dialogs.label_review import (  # noqa: E402
    LabelReviewDialog,
)


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
        page = dialog._page_items()
        if all(dialog._visible(i) for i in page):
            return True
        time.sleep(0.01)
    return False


def _press(dialog, key, modifier=Qt.NoModifier):
    QTest.keyClick(dialog, key, modifier)
    QApplication.processEvents()


def _dialog(tmp_path, labels, *, machine=None, agreement=None, predicted=None):
    paths = _images(tmp_path, len(labels))
    model = LabelReviewModel(labels, machine, agreement, predicted)
    dialog = LabelReviewDialog(model, paths)
    removed, restored = [], []
    dialog.labels_removed.connect(lambda idx: removed.append(list(idx)))
    dialog.labels_restored.connect(lambda e: restored.append(list(e)))
    dialog.show()
    QTest.qWaitForWindowExposed(dialog)
    dialog.activateWindow()
    _wait_loaded(QApplication.instance(), dialog)
    return dialog, model, removed, restored


def test_select_with_keys_remove_and_undo(qapp, tmp_path):
    labels = ["a_x"] * 20 + ["b_y"] * 5
    agreement = np.linspace(0.0, 1.0, 25)
    dialog, model, removed, restored = _dialog(tmp_path, labels, agreement=agreement)
    assert dialog._label == "a_x" and dialog._page_items()[:3] == [0, 1, 2]
    _press(dialog, Qt.Key_1)
    _press(dialog, Qt.Key_Q)  # 4x4: second row starts at Q
    QTest.mouseClick(dialog._tiles[2], Qt.LeftButton)
    _press(dialog, Qt.Key_2)
    _press(dialog, Qt.Key_2)  # toggled twice: deselected
    assert dialog._selected == {0, 4, 2}
    _press(dialog, Qt.Key_Delete)
    assert removed == [[0, 2, 4]]
    assert model.labels[0] is None and model.labels[4] is None
    assert dialog._page_items()[:3] == [1, 3, 5]  # page reflows
    assert not dialog._selected
    _press(dialog, Qt.Key_Z, Qt.ControlModifier)
    assert restored == [[(0, "a_x"), (2, "a_x"), (4, "a_x")]]
    assert model.labels[:5] == ["a_x"] * 5
    assert dialog._page_items()[:3] == [0, 1, 2]


def test_nothing_unseen_is_selected(qapp, tmp_path, monkeypatch):
    labels = ["a_x"] * 10
    dialog, model, removed, _ = _dialog(tmp_path, labels)
    # Pretend tile 0's image is still loading.
    dialog._pixmaps.pop(dialog._paths[0], None)
    monkeypatch.setattr(
        dialog, "_pixmap", lambda idx: dialog._pixmaps.get(dialog._paths[idx])
    )
    _press(dialog, Qt.Key_1)
    assert not dialog._selected
    _press(dialog, Qt.Key_Delete)
    assert removed == [] and model.labels[0] == "a_x"


def test_paging_select_page_and_held_keys(qapp, tmp_path):
    labels = ["a_x"] * 40
    dialog, model, removed, _ = _dialog(tmp_path, labels)
    assert dialog.page_label.text() == "Page 1 / 3"
    _press(dialog, Qt.Key_PageDown)
    _wait_loaded(qapp, dialog)
    assert dialog._page_items()[0] == 16
    # A held key (autorepeat) must not flip pages again.
    ev = QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress, Qt.Key_PageDown, Qt.NoModifier, "", True
    )
    QApplication.sendEvent(dialog, ev)
    qapp.processEvents()
    assert dialog._page == 1
    _press(dialog, Qt.Key_A, Qt.ControlModifier)
    assert dialog._selected == set(range(16, 32))
    _press(dialog, Qt.Key_A, Qt.ControlModifier)  # again: deselect page
    assert not dialog._selected
    _press(dialog, Qt.Key_A, Qt.ControlModifier)
    _press(dialog, Qt.Key_Delete)
    assert removed == [list(range(16, 32))]
    assert dialog.page_label.text() == "Page 2 / 2"
    assert dialog._page_items() == list(range(32, 40))


def test_filters_source_and_disagreement(qapp, tmp_path):
    labels = ["a_x", "a_x", "a_x", "b_y"]
    machine = [False, True, False, False]
    predicted = ["a_x", "b_y", "b_y", "b_y"]
    dialog, _, _, _ = _dialog(tmp_path, labels, machine=machine, predicted=predicted)
    assert dialog.label_list.item(0).text() == "a_x   3   (2 ⚠)"
    dialog.source_combo.setCurrentText("Machine (unverified)")
    assert dialog._items == [1] and dialog.label_list.count() == 1
    dialog.source_combo.setCurrentText("All labels")
    dialog.disagree_check.setChecked(True)
    assert dialog._items == [1, 2]
    assert dialog._tiles[0].tag == "M" and "model: b_y" in dialog._tiles[0].caption


def test_switch_label_with_arrow_keys_clears_selection(qapp, tmp_path):
    dialog, _, _, _ = _dialog(tmp_path, ["a_x", "a_x", "b_y"])
    _press(dialog, Qt.Key_1)
    assert dialog._selected == {0}
    _press(dialog, Qt.Key_Down)
    assert dialog._label == "b_y" and not dialog._selected


def test_space_and_enter_do_not_close(qapp, tmp_path):
    dialog, _, _, _ = _dialog(tmp_path, ["a_x"] * 3)
    for key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
        _press(dialog, key)
    assert dialog.isVisible()


def test_unreadable_image_marked_and_removable(qapp, tmp_path):
    paths = _images(tmp_path, 3)
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    paths[1] = bad
    model = LabelReviewModel(["a_x"] * 3)
    dialog = LabelReviewDialog(model, paths)
    dialog.show()
    end = time.time() + 3
    while time.time() < end and str(bad) not in dialog._failed:
        qapp.processEvents()
        time.sleep(0.01)
    assert dialog._tiles[1].note == "✕ unreadable"
    dialog.toggle_slot(1)
    assert dialog._selected == {1}


def test_screenshot(qapp, tmp_path):
    labels = ["blue_pink"] * 30 + ["blue_red"] * 6
    rng = np.random.default_rng(0)
    agreement = rng.random(36)
    predicted = [
        "blue_red" if (a < 0.2 and i < 30) else lab
        for i, (a, lab) in enumerate(zip(agreement, labels))
    ]
    machine = [i % 7 == 0 for i in range(36)]
    dialog, _, _, _ = _dialog(
        tmp_path, labels, machine=machine, agreement=agreement, predicted=predicted
    )
    for key in (Qt.Key_1, Qt.Key_3):
        _press(dialog, key)
    qapp.processEvents()
    pm = dialog.grab()
    assert not pm.isNull()
    out = os.environ.get("LABEL_REVIEW_SHOT")
    if out:
        pm.save(out)


def _project(tmp_path, n=12):
    from hydra_suite.classkit.core.store.db import ClassKitDB

    paths = [p.resolve() for p in _images(tmp_path, n)]
    db_path = tmp_path / "classkit.db"
    db = ClassKitDB(db_path)
    db.add_images(paths)
    db.update_labels_batch({str(paths[i]): "a_x" for i in range(4)})
    db.update_labels_with_confidence_batch(
        {str(paths[4]): ("a_x", 0.77)},
        label_source="auto",
        verified=False,
        metadata_by_path={str(paths[4]): {"model": "m1"}},
    )
    return paths, db_path


def test_main_window_removes_and_restores_with_provenance(qapp, tmp_path, monkeypatch):
    from hydra_suite.classkit.core.store.db import ClassKitDB
    from hydra_suite.classkit.gui.main_window import MainWindow

    paths, db_path = _project(tmp_path)
    before = ClassKitDB(db_path).get_label_review_status_by_path()

    window = MainWindow()
    window.db_path = db_path
    window.image_paths = list(paths)
    window.image_labels = ClassKitDB(db_path).get_all_labels()
    window._image_review_status = before.copy()
    seen = {}

    def fake_exec(dialog):
        dialog.show()
        QTest.qWaitForWindowExposed(dialog)
        dialog.activateWindow()
        _wait_loaded(qapp, dialog)
        seen["items"] = list(dialog._items)
        seen["machine"] = bool(dialog.model.machine[4])
        dialog._selected = {0, 4}
        _press(dialog, Qt.Key_Delete)
        seen["after_remove"] = ClassKitDB(db_path).get_label_review_status_by_path()
        _press(dialog, Qt.Key_Backspace)  # undo restores both
        dialog._selected = {1}
        _press(dialog, Qt.Key_Delete)
        return 0

    monkeypatch.setattr(LabelReviewDialog, "exec", fake_exec)
    window._open_label_review()

    assert seen["items"] == [0, 1, 2, 3, 4] and seen["machine"]
    removed = seen["after_remove"]
    assert removed[str(paths[0])]["label"] is None
    assert removed[str(paths[4])]["label"] is None
    after = ClassKitDB(db_path).get_label_review_status_by_path()
    for i in (0, 4):  # restored exactly, machine label stays unverified
        assert after[str(paths[i])] == before[str(paths[i])]
    assert after[str(paths[4])]["label_source"] == "auto"
    assert after[str(paths[4])]["verified"] is False
    assert after[str(paths[4])]["confidence"] == pytest.approx(0.77)
    assert after[str(paths[1])]["label"] is None
    assert window.image_labels[1] is None and window.image_labels[4] == "a_x"
    assert window._pending_machine_review_count() == 1
    window.close()


def test_main_window_no_labels_shows_message(qapp, monkeypatch):
    from hydra_suite.classkit.gui.main_window import MainWindow

    window = MainWindow()
    shown = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *a, **k: shown.append(a[2])
    )
    window._open_label_review()
    assert shown and "no labeled images" in shown[0]
    window.close()


def test_main_window_ranks_with_model(qapp, tmp_path, monkeypatch):
    from hydra_suite.classkit.config.schemas import Factor, LabelingScheme
    from hydra_suite.classkit.core.store.db import ClassKitDB
    from hydra_suite.classkit.gui.main_window import MainWindow

    paths, db_path = _project(tmp_path)
    window = MainWindow()
    window.db_path = db_path
    window.image_paths = list(paths)
    window.image_labels = ClassKitDB(db_path).get_all_labels()
    probs = np.full((len(paths), 4), 0.25)
    probs[2] = [0.01, 0.01, 0.01, 0.97]  # labeled a_x, model says b_y
    window._model_probs = probs
    window._model_class_names = ["a_x", "a_y", "b_x", "b_y"]
    scheme = LabelingScheme(
        name="s",
        factors=[Factor("color", ["a", "b"]), Factor("side", ["x", "y"])],
        training_modes=[],
    )
    monkeypatch.setattr(window, "_resolve_training_scheme", lambda: scheme)
    seen = {}

    def fake_exec(dialog):
        seen["items"] = list(dialog._items)
        seen["disagree"] = dialog.model.disagrees(2)
        return 0

    monkeypatch.setattr(LabelReviewDialog, "exec", fake_exec)
    window._open_label_review()
    assert seen["items"][0] == 2 and seen["disagree"]
    window.close()
