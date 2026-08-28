"""Tests for DatasetPanel widget refactor (source combo + manage signal)."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from pathlib import Path  # noqa: E402

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_dataset_panel_has_source_combo(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    assert hasattr(panel, "source_combo")


def test_dataset_panel_has_manage_btn(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    assert hasattr(panel, "btn_manage_sources")


def test_dataset_panel_manage_signal(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    assert hasattr(panel, "manage_sources_requested")


def test_dataset_panel_refresh_sources(qapp, tmp_path):
    from hydra_suite.detectkit.gui.models import OBBSource
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()

    class FakeProj:
        sources = [OBBSource(path=str(tmp_path), name="ds1")]

    panel.refresh_sources(FakeProj())
    assert panel.source_combo.count() == 1


def test_dataset_panel_no_source_list(qapp):
    """Old QListWidget-based source_list must be gone."""
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    assert not hasattr(panel, "source_list"), "old source_list widget must not exist"


def test_dataset_panel_navigate_prev_next(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    # Methods should exist without raising
    panel.navigate_prev()
    panel.navigate_next()


def test_dataset_panel_xany_stage_copies_source_into_app_data(
    qapp, tmp_path, monkeypatch
):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    source_dir = tmp_path / "portable_source"
    (source_dir / "images").mkdir(parents=True)
    (source_dir / "labels").mkdir(parents=True)
    (source_dir / "images" / "frame.png").write_bytes(b"png")
    (source_dir / "labels" / "frame.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    (source_dir / "classes.txt").write_text("ant\n", encoding="utf-8")

    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path / "hydra_data"))

    panel = DatasetPanel()
    stage_dir = panel._prepare_xal_stage(source_dir)

    assert stage_dir.exists()
    assert stage_dir != source_dir
    assert (stage_dir / "classes.txt").read_text(encoding="utf-8") == "ant\n"
    assert (stage_dir / "images" / "frame.png").exists()
    assert (stage_dir / "labels" / "frame.txt").exists()


def _make_panel_with_source(qapp, tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    source_root = tmp_path / "src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    (source_root / "images" / "a.jpg").write_bytes(b"fake")
    (source_root / "labels" / "a.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    (source_root / "classes.txt").write_text("ant\n")

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    proj.sources = [OBBSource(path=str(source_root), name="src", level="obb")]

    panel = DatasetPanel()
    panel.set_project(proj, main_window=None)
    return panel, source_root


def test_clear_labels_from_frame_requires_confirmation(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    panel, source_root = _make_panel_with_source(qapp, tmp_path)
    panel.image_list.setCurrentRow(0)
    panel.image_list.item(0).setSelected(True)

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: QMessageBox.StandardButton.Cancel,
    )

    panel._clear_labels_from_frame()

    assert (
        source_root / "labels" / "a.txt"
    ).read_text() != ""  # untouched, confirm declined


def test_clear_labels_from_frame_clears_on_confirm(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    panel, source_root = _make_panel_with_source(qapp, tmp_path)
    panel.image_list.setCurrentRow(0)
    panel.image_list.item(0).setSelected(True)

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )

    panel._clear_labels_from_frame()

    assert (source_root / "labels" / "a.txt").read_text() == ""


def test_clear_labels_from_frame_confirmation_names_frame_count(
    qapp, tmp_path, monkeypatch
):
    """The confirmation dialog text must actually name what's about to be
    cleared -- not just exist. Captures the call instead of asserting
    nothing about its content."""
    from PySide6.QtWidgets import QMessageBox

    panel, _source_root = _make_panel_with_source(qapp, tmp_path)
    panel.image_list.setCurrentRow(0)
    panel.image_list.item(0).setSelected(True)

    captured = {}

    def _capture_warning(self, title, text, *a, **k):
        captured["title"] = title
        captured["text"] = text
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        _capture_warning,
    )

    panel._clear_labels_from_frame()

    assert "1 frame" in captured["text"] or "a.jpg" in captured["text"]


def test_clear_labels_from_frame_reselects_same_row_not_row_zero(
    qapp, tmp_path, monkeypatch
):
    """Regression: clearing a frame's labels must re-render that frame in
    place, not rebuild the image list and jump back to row 0 -- nothing
    about which images exist has changed."""
    from PySide6.QtWidgets import QMessageBox

    from hydra_suite.detectkit.gui.models import OBBSource

    panel, source_root = _make_panel_with_source(qapp, tmp_path)
    # Add a second image so row 0 vs row 1 is a meaningful distinction.
    (source_root / "images" / "b.jpg").write_bytes(b"fake")
    (source_root / "labels" / "b.txt").write_text("0 0.5 0.5 0.4 0.2\n")
    panel._project.sources = [OBBSource(path=str(source_root), name="src", level="obb")]
    panel.refresh_sources(panel._project)

    # Select whichever row corresponds to b.jpg.
    b_row = next(
        i
        for i in range(panel.image_list.count())
        if panel.image_list.item(i).data(Qt.UserRole)
        and Path(str(panel.image_list.item(i).data(Qt.UserRole))).name == "b.jpg"
    )
    panel.image_list.setCurrentRow(b_row)
    panel.image_list.item(b_row).setSelected(True)

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )

    panel._clear_labels_from_frame()

    assert panel.image_list.currentRow() == b_row


def test_dataset_panel_has_clear_source_labels_button(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    assert hasattr(panel, "btn_clear_source_labels")


def test_clear_labels_from_source_requires_confirmation(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    panel, source_root = _make_panel_with_source(qapp, tmp_path)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: QMessageBox.StandardButton.Cancel,
    )

    panel._clear_labels_from_source()

    assert (source_root / "labels" / "a.txt").read_text() != ""


def test_clear_labels_from_source_confirmation_names_source_and_count(
    qapp, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox

    panel, _source_root = _make_panel_with_source(qapp, tmp_path)
    captured = {}

    def _capture_warning(self, title, text, *a, **k):
        captured["text"] = text
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        _capture_warning,
    )

    panel._clear_labels_from_source()

    assert "src" in captured["text"]
    assert "1" in captured["text"]  # 1 label file


def test_clear_labels_from_source_confirmation_excludes_classes_txt_from_count(
    qapp, tmp_path, monkeypatch
):
    """A stray classes.txt under labels/ (unusual, but defensively handled by
    clear_labels_for_source, which excludes it by name) must not inflate the
    confirmation dialog's count -- the count must match what actually gets
    cleared."""
    from PySide6.QtWidgets import QMessageBox

    panel, source_root = _make_panel_with_source(qapp, tmp_path)
    # Add a stray classes.txt under labels/ alongside the real label file.
    (source_root / "labels" / "classes.txt").write_text("ant\n")

    captured = {}

    def _capture_warning(self, title, text, *a, **k):
        captured["text"] = text
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        _capture_warning,
    )

    panel._clear_labels_from_source()

    # Only a.txt should be counted (1), not a.txt + labels/classes.txt (2).
    assert "1" in captured["text"]
    assert "2" not in captured["text"]
    # The stray classes.txt under labels/ is untouched by the clear.
    assert (source_root / "labels" / "classes.txt").read_text() == "ant\n"


def test_clear_labels_from_source_clears_all_on_confirm(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    panel, source_root = _make_panel_with_source(qapp, tmp_path)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )

    panel._clear_labels_from_source()

    assert (source_root / "labels" / "a.txt").read_text() == ""


def test_clear_labels_from_source_no_op_when_no_labels(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    source_root = tmp_path / "empty_src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    (source_root / "classes.txt").write_text("ant\n")
    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    proj.sources = [OBBSource(path=str(source_root), name="empty", level="obb")]
    panel = DatasetPanel()
    panel.set_project(proj, main_window=None)

    warn_called = []
    info_called = []
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: warn_called.append(True) or QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.information",
        lambda *a, **k: info_called.append(True),
    )

    panel._clear_labels_from_source()

    assert not warn_called  # no pointless confirm
    assert info_called  # info dialog shown instead


def _make_panel_with_two_sources(qapp, tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    proj_dir = tmp_path / "myproject"
    proj_dir.mkdir()
    sources = []
    for name, is_linked in (("a", False), ("b", True)):
        # One source lives inside the project dir, one outside it (a
        # "linked" source, per DetectKitProject's portability concept) --
        # exercises the project-scope warning about out-of-project blast
        # radius.
        root = (proj_dir if not is_linked else tmp_path / "elsewhere") / f"src_{name}"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir(parents=True)
        (root / "images" / "f.jpg").write_bytes(b"fake")
        (root / "labels" / "f.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
        (root / "classes.txt").write_text("ant\n")
        sources.append(OBBSource(path=str(root), name=name, level="obb"))

    proj = DetectKitProject(project_dir=proj_dir, class_names=["ant"])
    proj.sources = sources
    panel = DatasetPanel()
    panel.set_project(proj, main_window=None)
    return panel, sources


def test_clear_all_labels_rejects_wrong_typed_name(qapp, tmp_path, monkeypatch):
    panel, sources = _make_panel_with_two_sources(qapp, tmp_path)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QInputDialog.getText",
        lambda *a, **k: ("not-the-project-name", True),
    )

    panel._clear_labels_from_all_sources()

    for src in sources:
        assert (Path(src.path) / "labels" / "f.txt").read_text() != ""


def test_clear_all_labels_rejects_empty_typed_name(qapp, tmp_path, monkeypatch):
    """Regression: an empty typed value must never satisfy the confirmation,
    even in the (non-live-reachable-today, but cheap-to-guard) edge case of
    an empty project folder name."""
    panel, sources = _make_panel_with_two_sources(qapp, tmp_path)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QInputDialog.getText",
        lambda *a, **k: ("", True),
    )

    panel._clear_labels_from_all_sources()

    for src in sources:
        assert (Path(src.path) / "labels" / "f.txt").read_text() != ""


def test_clear_all_labels_accepts_correct_typed_name(qapp, tmp_path, monkeypatch):
    panel, sources = _make_panel_with_two_sources(qapp, tmp_path)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QInputDialog.getText",
        lambda *a, **k: ("myproject", True),
    )

    panel._clear_labels_from_all_sources()

    for src in sources:
        assert (Path(src.path) / "labels" / "f.txt").read_text() == ""


def test_clear_all_labels_prompt_warns_about_linked_sources(
    qapp, tmp_path, monkeypatch
):
    """The confirmation prompt must call out that some sources live outside
    the project folder -- "type the project name" implies project-bounded
    blast radius, but a linked source can be anywhere on disk."""
    panel, _sources = _make_panel_with_two_sources(qapp, tmp_path)
    captured = {}

    def _capture_get_text(parent, title, text, *a, **k):
        captured["text"] = text
        return "", False

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QInputDialog.getText",
        _capture_get_text,
    )

    panel._clear_labels_from_all_sources()

    assert "outside" in captured["text"].lower()


def test_clear_all_labels_no_op_when_project_empty(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.models import DetectKitProject
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    panel = DatasetPanel()
    panel.set_project(proj, main_window=None)

    prompted = []
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QInputDialog.getText",
        lambda *a, **k: (prompted.append(True), ("", False))[1],
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.information",
        lambda *a, **k: None,
    )

    panel._clear_labels_from_all_sources()

    assert (
        not prompted
    )  # info dialog shown instead of a pointless type-to-confirm prompt
