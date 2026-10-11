"""Smart Select batch flow in a real PoseKit main window.

Covers adding a batch in pick order, Start Labeling (sort + first frame),
navigation following the batch, and persisting the batch with the project.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

from hydra_suite.posekit.gui import utils as gui_utils  # noqa: E402
from hydra_suite.posekit.gui.constants import SORT_SMART_BATCH  # noqa: E402
from hydra_suite.posekit.gui.models import Project  # noqa: E402
from hydra_suite.posekit.gui.project import (  # noqa: E402
    build_image_list,
    canonical_project_path,
    save_project_state,
)


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def isolate(qapp, tmp_path, monkeypatch):
    # Never read or write the user's ~/posekit/ui_settings.json.
    monkeypatch.setattr(
        gui_utils, "get_ui_settings_path", lambda: tmp_path / "ui_settings.json"
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def _project(tmp_path, n=12, labeled=(1,)):
    from PIL import Image

    images = tmp_path / "crops" / "images"
    images.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (32, 32)).save(images / f"img{i:02d}.png")
    root = tmp_path / "pose_project"
    labels = root / "labels"
    labels.mkdir(parents=True)
    for i in labeled:
        (labels / f"img{i:02d}.txt").write_text("0 0.5 0.5 1 1 0.2 0.2 2 0.8 0.8 2\n")
    proj = Project(
        images_dir=images,
        out_root=root,
        labels_dir=labels,
        project_path=canonical_project_path(root),
        class_names=["ant"],
        keypoint_names=["head", "tail"],
        skeleton_edges=[(0, 1)],
    )
    save_project_state(proj)
    return proj


def _window(proj):
    from hydra_suite.posekit.gui.main_window import MainWindow

    win = MainWindow(proj, build_image_list(proj))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    return win


class _FakeDialog:
    """Stands in for SmartSelectDialog: returns a fixed batch."""

    batch = [7, 3, 9]
    start = True
    kwargs = None

    def __init__(self, *args, **kwargs):
        _FakeDialog.kwargs = kwargs
        self.selected_indices = list(_FakeDialog.batch)
        self.start_labeling = _FakeDialog.start
        self._did_add = True

    def exec(self):
        return QDialog.Accepted

    def deleteLater(self):
        pass


def test_start_labeling_sorts_by_batch_and_navigates_in_order(
    qapp, tmp_path, monkeypatch
):
    proj = _project(tmp_path)
    win = _window(proj)
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.SmartSelectDialog", _FakeDialog
    )
    _FakeDialog.batch, _FakeDialog.start = [7, 3, 9], True
    win.open_smart_select()
    qapp.processEvents()

    assert {7, 3, 9} <= win.labeling_frames
    assert win.sort_combo.currentText() == SORT_SMART_BATCH
    assert win.current_index == 7
    order = [
        win.labeling_list.item(i).data(0x0100) for i in range(win.labeling_list.count())
    ]
    assert order[:3] == [7, 3, 9]  # batch first, in pick order
    assert win._labeling_order()[:3] == [7, 3, 9]

    win.next_frame()
    assert win.current_index == 3
    win.next_frame()
    assert win.current_index == 9
    win.prev_frame()
    assert win.current_index == 3

    # No model selected: the dialog was told why model signals are off.
    assert _FakeDialog.kwargs["prediction_source"] is None
    assert "No pose model" in _FakeDialog.kwargs["no_prediction_reason"]
    assert _FakeDialog.kwargs["label_path_fn"] == win._label_path_for


def test_add_without_start_keeps_sort_and_persists_batch(qapp, tmp_path, monkeypatch):
    proj = _project(tmp_path)
    win = _window(proj)
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.SmartSelectDialog", _FakeDialog
    )
    _FakeDialog.batch, _FakeDialog.start = [5, 2], False
    before = win.sort_combo.currentText()
    win.open_smart_select()
    assert win.sort_combo.currentText() == before
    assert {5, 2} <= win.labeling_frames
    # Default sort navigates by index, not batch order.
    assert win._labeling_order() == sorted(win.labeling_frames)

    win.save_project()
    reloaded = Project.from_json(canonical_project_path(proj.out_root))
    assert reloaded.smart_batch == [
        str(win.image_paths[5]),
        str(win.image_paths[2]),
    ]
    raw = (canonical_project_path(proj.out_root)).read_text()
    assert "../crops/images/img05.png" in raw  # stored relative to the project


def test_batch_order_survives_reopen(qapp, tmp_path, monkeypatch):
    proj = _project(tmp_path)
    win = _window(proj)
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.SmartSelectDialog", _FakeDialog
    )
    _FakeDialog.batch, _FakeDialog.start = [8, 4], True
    win.open_smart_select()
    win.save_project()
    win.close()

    reloaded = Project.from_json(canonical_project_path(proj.out_root))
    win2 = _window(reloaded)
    assert win2._smart_batch_indices() == [8, 4]
    win2.sort_combo.setCurrentText(SORT_SMART_BATCH)
    assert win2._labeling_order()[:2] == [8, 4]


def test_canceled_dialog_changes_nothing(qapp, tmp_path, monkeypatch):
    proj = _project(tmp_path)
    win = _window(proj)

    class _Cancel(_FakeDialog):
        def exec(self):
            return QDialog.Rejected

    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.SmartSelectDialog", _Cancel
    )
    before = set(win.labeling_frames)
    win.open_smart_select()
    assert win.labeling_frames == before
    assert win._smart_batch_paths == []


def test_labeling_set_and_batch_survive_images_added_between_sessions(
    qapp, tmp_path, monkeypatch
):
    from PIL import Image

    proj = _project(tmp_path)
    win = _window(proj)
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.SmartSelectDialog", _FakeDialog
    )
    _FakeDialog.batch, _FakeDialog.start = [7, 3, 9], True
    win.open_smart_select()
    win.save_project()
    names_before = {win.image_paths[i].name for i in win.labeling_frames}
    win.close()

    # A new image that sorts first shifts every index by one.
    Image.new("RGB", (32, 32)).save(proj.images_dir / "img00a.png")
    reloaded = Project.from_json(canonical_project_path(proj.out_root))
    win2 = _window(reloaded)
    names_after = {win2.image_paths[i].name for i in win2.labeling_frames}
    assert names_after == names_before
    win2.sort_combo.setCurrentText(SORT_SMART_BATCH)
    order = [win2.image_paths[i].name for i in win2._labeling_order()[:3]]
    assert order == ["img07.png", "img03.png", "img09.png"]


def test_next_loads_when_list_row_is_already_current(qapp, tmp_path, monkeypatch):
    proj = _project(tmp_path)
    win = _window(proj)
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.SmartSelectDialog", _FakeDialog
    )
    _FakeDialog.batch, _FakeDialog.start = [7, 3, 9], True
    win.open_smart_select()
    assert win.current_index == 7
    # Open a frame from the All-frames list; the labeling list keeps row 0.
    win._select_frame_in_list(11)
    assert win.current_index == 11
    win._select_frame_in_list(7)
    assert win.current_index == 7


def test_re_adding_existing_frames_reports_order_update(qapp, tmp_path, monkeypatch):
    proj = _project(tmp_path)
    win = _window(proj)
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.SmartSelectDialog", _FakeDialog
    )
    _FakeDialog.batch, _FakeDialog.start = [7, 3], False
    win.open_smart_select()
    _FakeDialog.batch = [3, 7]
    win.open_smart_select()
    assert "Batch order updated" in win.statusBar().currentMessage()
    assert win._smart_batch_indices() == [3, 7]
