"""Tests for DetectKit SourceManagerDialog."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialogButtonBox  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _make_proj(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    return DetectKitProject(project_dir=tmp_path, class_names=["ant"])


def test_source_manager_dialog_imports(qapp):
    pass  # noqa: F401


def test_source_manager_is_base_dialog(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = SourceManagerDialog(_make_proj(tmp_path))
    assert isinstance(dlg, BaseDialog)


def test_source_manager_has_close_button(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog

    dlg = SourceManagerDialog(_make_proj(tmp_path))
    # Should have a Close button, not Ok/Cancel
    close_btn = dlg._buttons.button(QDialogButtonBox.StandardButton.Close)
    assert close_btn is not None


def test_source_manager_shows_existing_sources(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.models import OBBSource

    proj = _make_proj(tmp_path)
    proj.sources = [OBBSource(path=str(tmp_path), name="ds1")]
    dlg = SourceManagerDialog(proj)
    assert dlg._source_list.count() == 1


def test_source_manager_remove_selected(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.models import OBBSource

    proj = _make_proj(tmp_path)
    proj.sources = [
        OBBSource(path=str(tmp_path / "a"), name="a"),
        OBBSource(path=str(tmp_path / "b"), name="b"),
    ]
    dlg = SourceManagerDialog(proj)
    dlg._source_list.setCurrentRow(0)
    dlg._remove_selected()
    assert len(proj.sources) == 1
    assert dlg._source_list.count() == 1


def test_source_manager_has_add_remove_buttons(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog

    dlg = SourceManagerDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "btn_add")
    assert hasattr(dlg, "btn_remove")


def test_source_manager_adds_imported_yolo_detect_source(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog

    source_root = tmp_path / "external_detect"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    (source_root / "images" / "sample.jpg").write_text("fake", encoding="utf-8")
    (source_root / "labels" / "sample.txt").write_text(
        "0 0.5 0.5 0.4 0.2\n",
        encoding="utf-8",
    )
    (source_root / "dataset.yaml").write_text(
        "train: images\nnames:\n  0: ant\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(source_root),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.confirm_detectkit_source_addition",
        lambda *args, **kwargs: True,
    )

    proj = _make_proj(tmp_path)
    dlg = SourceManagerDialog(proj)
    dlg._add_source()

    assert len(proj.sources) == 1
    added = proj.sources[0]
    assert added.original_path == str(source_root)
    assert added.source_kind == "yolo_detect"
    assert added.imported is True
    assert Path(added.path).is_dir()
    assert (Path(added.path) / "classes.txt").exists()
    assert (Path(added.path) / "labels" / "sample.txt").exists()


def test_source_manager_does_not_add_source_when_validation_cancelled(
    qapp, tmp_path, monkeypatch
):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog

    source_root = tmp_path / "external_detect"
    source_root.mkdir(parents=True)

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(source_root),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.inspect_detectkit_source",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.confirm_detectkit_source_addition",
        lambda *args, **kwargs: False,
    )

    def _should_not_materialize(*args, **kwargs):
        raise AssertionError("materialize_detectkit_source should not be called")

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.materialize_detectkit_source",
        _should_not_materialize,
    )

    proj = _make_proj(tmp_path)
    dlg = SourceManagerDialog(proj)
    dlg._add_source()

    assert proj.sources == []
    assert dlg._source_list.count() == 0
