"""Tests for DetectKit SourceManagerDialog."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Event
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _make_proj(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    return DetectKitProject(project_dir=tmp_path, class_names=["ant"])


def _add_source_and_wait(qapp, dlg):
    dlg._add_source()
    deadline = monotonic() + 10
    while dlg._inspection_worker is not None or dlg._import_worker is not None:
        assert monotonic() < deadline
        QTest.qWait(10)
    qapp.processEvents()


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
    assert hasattr(dlg, "btn_add_project")


def test_source_manager_imports_all_sources_from_project(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.models import OBBSource
    from hydra_suite.detectkit.gui.project import create_project, save_project

    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path / "user-data"))
    origin = create_project(tmp_path / "origin", class_names=["ant"])
    for index in range(2):
        root = tmp_path / f"dataset-{index}"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "images" / "frame.jpg").write_bytes(b"image")
        (root / "labels" / "frame.txt").write_text("0 0.5 0.5 0.4 0.2\n")
        (root / "classes.txt").write_text("ant\n")
        origin.sources.append(OBBSource(path=str(root), name=root.name))
    save_project(origin)
    destination = create_project(tmp_path / "destination", class_names=["ant"])
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(origin.project_dir),
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)
    dialog = SourceManagerDialog(destination)
    dialog._add_project_sources()
    deadline = monotonic() + 30
    while dialog._project_import_worker is not None:
        assert monotonic() < deadline
        QTest.qWait(10)
    qapp.processEvents()

    assert len(destination.sources) == 2
    assert all(source.imported for source in destination.sources)
    assert all(
        Path(source.path, "classes.txt").read_text() == "ant\n"
        for source in destination.sources
    )


def test_source_manager_adds_imported_yolo_detect_source(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.dialogs.source_validation import (
        SOURCE_ADD_MODE_PORTABLE,
        DetectKitSourceAdditionChoice,
    )

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
        lambda *args, **kwargs: DetectKitSourceAdditionChoice(
            mode=SOURCE_ADD_MODE_PORTABLE
        ),
    )

    proj = _make_proj(tmp_path)
    dlg = SourceManagerDialog(proj)
    _add_source_and_wait(qapp, dlg)

    assert len(proj.sources) == 1
    added = proj.sources[0]
    assert added.original_path == str(source_root)
    assert added.source_kind == "yolo_detect"
    assert added.imported is True
    assert Path(added.path).is_dir()
    assert (Path(added.path) / "classes.txt").exists()
    assert (Path(added.path) / "labels" / "sample.txt").exists()


def test_source_manager_add_source_collapses_al_round_to_one_source(
    qapp, tmp_path, monkeypatch
):
    """Picking an AL round container registers exactly ONE source -- the
    authoritative root -- named after the round folder itself, not the
    resolved level subfolder ("obb") and not one entry per sibling level."""
    import json

    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.dialogs.source_validation import (
        SOURCE_ADD_MODE_PORTABLE,
        DetectKitSourceAdditionChoice,
    )

    round_dir = tmp_path / "active_learning" / "20260827_172624"
    for level in ("obb", "aabb"):
        level_dir = round_dir / level
        (level_dir / "images").mkdir(parents=True)
        (level_dir / "labels").mkdir(parents=True)
        (level_dir / "images" / "f001.jpg").write_bytes(b"fake-image")
        (level_dir / "labels" / "f001.txt").write_text(
            "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8"
        )
        (level_dir / "classes.txt").write_text("ant\n", encoding="utf-8")

    (round_dir / "manifest.json").write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "level": "obb",
                        "authoritative": True,
                        "reviewed": True,
                        "path": str(round_dir / "obb"),
                    },
                    {
                        "level": "aabb",
                        "authoritative": False,
                        "reviewed": False,
                        "path": str(round_dir / "aabb"),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(round_dir),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.confirm_detectkit_source_addition",
        lambda *args, **kwargs: DetectKitSourceAdditionChoice(
            mode=SOURCE_ADD_MODE_PORTABLE
        ),
    )

    proj = _make_proj(tmp_path)
    dlg = SourceManagerDialog(proj)
    _add_source_and_wait(qapp, dlg)

    assert len(proj.sources) == 1
    added = proj.sources[0]
    assert added.name == "20260827_172624"
    assert added.level == "obb"
    assert added.source_kind == "detectkit_al"
    assert added.reviewed is True
    assert added.derived_from is None
    assert Path(added.path).is_dir()
    assert dlg._source_list.count() == 1


def test_source_manager_add_source_trusts_manifest_level_for_aabb_round(
    qapp, tmp_path, monkeypatch
):
    """An AL round whose AUTHORITATIVE level is aabb must be registered as
    level='aabb', not 'obb' -- re-scanning the label files can't tell them
    apart (both are 9-field quads), so the manifest's declared level must be
    trusted, not the geometry-scan result."""
    import json

    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.dialogs.source_validation import (
        SOURCE_ADD_MODE_PORTABLE,
        DetectKitSourceAdditionChoice,
    )

    round_dir = tmp_path / "active_learning" / "20260827_180000"
    level_dir = round_dir / "aabb"
    (level_dir / "images").mkdir(parents=True)
    (level_dir / "labels").mkdir(parents=True)
    (level_dir / "images" / "f001.jpg").write_bytes(b"fake-image")
    (level_dir / "labels" / "f001.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8"
    )
    (level_dir / "classes.txt").write_text("ant\n", encoding="utf-8")

    (round_dir / "manifest.json").write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "level": "aabb",
                        "authoritative": True,
                        "reviewed": True,
                        "path": str(round_dir / "aabb"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(round_dir),
    )
    # DetectKitSourceAdditionChoice defaults to level="obb" -- this is the
    # dialog's own re-scanned guess, which is exactly the wrong value this
    # test must NOT see land on the registered source.
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.confirm_detectkit_source_addition",
        lambda *args, **kwargs: DetectKitSourceAdditionChoice(
            mode=SOURCE_ADD_MODE_PORTABLE
        ),
    )

    proj = _make_proj(tmp_path)
    dlg = SourceManagerDialog(proj)
    _add_source_and_wait(qapp, dlg)

    assert len(proj.sources) == 1
    assert proj.sources[0].level == "aabb"


def test_remove_selected_deletes_pending_escalation_staging_dir(qapp, tmp_path):
    """Removing a source with an unreviewed pending escalation must not leak
    its staging directory under artifacts/pending_escalations/."""
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.models import OBBSource, StagedReview

    staged_dir = tmp_path / "artifacts" / "pending_escalations" / "orig-variant-abc123"
    staged_dir.mkdir(parents=True)
    (staged_dir / "labels").mkdir()

    proj = _make_proj(tmp_path)
    proj.sources = [
        OBBSource(
            path=str(tmp_path / "orig"),
            name="orig",
            staged_review=StagedReview(
                staged_path=str(staged_dir),
                target_level="polygon",
                producer="sam2",
                producer_variant="sam2.1-hiera-base_plus",
                created_at="2026-08-27T00:00:00",
            ),
        )
    ]
    dlg = SourceManagerDialog(proj)
    dlg._source_list.setCurrentRow(0)
    dlg._remove_selected()

    assert proj.sources == []
    assert not staged_dir.exists()


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
    from types import SimpleNamespace

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.source_workers.inspect_detectkit_source",
        lambda *args, **kwargs: SimpleNamespace(
            dataset_root=source_root, source_kind="detectkit"
        ),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.confirm_detectkit_source_addition",
        lambda *args, **kwargs: False,
    )

    def _should_not_materialize(*args, **kwargs):
        raise AssertionError("materialize_detectkit_source should not be called")

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.source_workers.materialize_detectkit_source",
        _should_not_materialize,
    )

    proj = _make_proj(tmp_path)
    dlg = SourceManagerDialog(proj)
    _add_source_and_wait(qapp, dlg)

    assert proj.sources == []
    assert dlg._source_list.count() == 0


def test_source_manager_adds_linked_source_in_place(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.dialogs.source_validation import (
        SOURCE_ADD_MODE_LINKED,
        DetectKitSourceAdditionChoice,
    )

    source_root = tmp_path / "linked_detect"
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
        lambda *args, **kwargs: DetectKitSourceAdditionChoice(
            mode=SOURCE_ADD_MODE_LINKED
        ),
    )

    proj = _make_proj(tmp_path)
    dlg = SourceManagerDialog(proj)
    _add_source_and_wait(qapp, dlg)

    assert len(proj.sources) == 1
    added = proj.sources[0]
    assert added.path == str(source_root)
    assert added.original_path == str(source_root)
    assert added.imported is False
    assert (source_root / "classes.txt").exists()


def test_source_import_keeps_event_loop_responsive_and_reports_progress(
    qapp, tmp_path, monkeypatch
):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.dialogs.source_validation import (
        DetectKitSourceAdditionChoice,
    )
    from hydra_suite.detectkit.gui.source_import import materialize_detectkit_source

    source_root = tmp_path / "source"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir()
    (source_root / "classes.txt").write_text("ant\n", encoding="utf-8")
    for index in range(3):
        (source_root / "images" / f"frame{index}.jpg").write_bytes(b"fake")

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(source_root),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.confirm_detectkit_source_addition",
        lambda *args, **kwargs: DetectKitSourceAdditionChoice(mode="portable"),
    )
    release = Event()

    def slow_materialize(*args, **kwargs):
        assert release.wait(5)
        return materialize_detectkit_source(*args, **kwargs)

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.source_workers.materialize_detectkit_source",
        slow_materialize,
    )
    proj = _make_proj(tmp_path)
    dlg = SourceManagerDialog(proj)
    values = []
    dlg._import_progress.valueChanged.connect(values.append)
    dlg.show()
    dlg._add_source()
    deadline = monotonic() + 5
    while dlg._import_worker is None:
        assert monotonic() < deadline
        QTest.qWait(10)
    assert dlg._import_worker is not None
    assert not dlg.btn_add.isEnabled()
    assert dlg._import_progress.isVisible()

    timer_observed_running_worker = []

    def on_timer():
        timer_observed_running_worker.append(dlg._import_worker.isRunning())
        release.set()

    QTimer.singleShot(50, on_timer)
    QTest.qWait(100)
    assert timer_observed_running_worker == [True]
    assert dlg._import_worker.wait(5000)
    qapp.processEvents()
    assert len(proj.sources) == 1
    assert any(0 < value < 100 for value in values)
    assert dlg.btn_add.isEnabled()
    dlg.close()


def test_source_inspection_keeps_event_loop_responsive(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.source_import import inspect_detectkit_source

    source_root = tmp_path / "source"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir()
    (source_root / "classes.txt").write_text("ant\n", encoding="utf-8")
    (source_root / "images" / "frame.jpg").write_bytes(b"fake")
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(source_root),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.confirm_detectkit_source_addition",
        lambda *args, **kwargs: None,
    )
    release = Event()

    def slow_inspect(*args, **kwargs):
        assert release.wait(5)
        return inspect_detectkit_source(*args, **kwargs)

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.source_workers.inspect_detectkit_source",
        slow_inspect,
    )
    dlg = SourceManagerDialog(_make_proj(tmp_path))
    dlg.show()
    dlg._add_source()
    assert dlg._inspection_worker is not None
    assert dlg._import_progress.maximum() == 0
    dlg.reject()
    dlg.close()
    assert dlg.isVisible()
    observed_running = []

    def on_timer():
        observed_running.append(dlg._inspection_worker.isRunning())
        release.set()

    QTimer.singleShot(50, on_timer)
    QTest.qWait(100)
    assert observed_running == [True]
    deadline = monotonic() + 5
    while dlg._inspection_worker is not None:
        assert monotonic() < deadline
        QTest.qWait(10)
    assert dlg._import_worker is None
    assert dlg.btn_add.isEnabled()
    dlg.close()


def test_failed_source_import_restores_controls(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.dialogs.source_validation import (
        DetectKitSourceAdditionChoice,
    )

    source_root = tmp_path / "source"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir()
    (source_root / "classes.txt").write_text("ant\n", encoding="utf-8")
    (source_root / "images" / "frame.jpg").write_bytes(b"fake")
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(source_root),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.confirm_detectkit_source_addition",
        lambda *args, **kwargs: DetectKitSourceAdditionChoice(mode="portable"),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.source_workers.materialize_detectkit_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))

    proj = _make_proj(tmp_path)
    dlg = SourceManagerDialog(proj)
    _add_source_and_wait(qapp, dlg)

    assert proj.sources == []
    assert dlg.btn_add.isEnabled()
    assert warnings == ["disk full"]


def test_source_import_remaps_classes_before_registration(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.dialogs.source_validation import (
        DetectKitSourceAdditionChoice,
    )

    source_root = tmp_path / "source"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir()
    (source_root / "classes.txt").write_text("bee\nant\n", encoding="utf-8")
    (source_root / "images" / "frame.jpg").write_bytes(b"fake")
    (source_root / "labels" / "frame.txt").write_text(
        "1 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(source_root),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.confirm_detectkit_source_addition",
        lambda *args, **kwargs: DetectKitSourceAdditionChoice(mode="portable"),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    proj = _make_proj(tmp_path)
    dlg = SourceManagerDialog(proj)
    _add_source_and_wait(qapp, dlg)

    assert len(proj.sources) == 1
    dest_root = Path(proj.sources[0].path)
    assert (dest_root / "classes.txt").read_text(encoding="utf-8") == "ant\n"
    assert (
        (dest_root / "labels" / "frame.txt")
        .read_text(encoding="utf-8")
        .startswith("0 ")
    )


def _imported_source(tmp_path, name):
    from hydra_suite.detectkit.gui.models import OBBSource

    path = tmp_path / "artifacts" / "imported_sources" / name
    (path / "images").mkdir(parents=True)
    (path / "images" / "a.jpg").write_bytes(b"x")
    return path, OBBSource(path=str(path), name=name)


def test_remove_imported_source_deletes_copy_after_saving(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.project import open_project

    copy, source = _imported_source(tmp_path, "ds-abc")
    keep, other = _imported_source(tmp_path, "ds-keep")
    proj = _make_proj(tmp_path)
    proj.sources = [source, other]
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    dlg = SourceManagerDialog(proj)
    dlg._source_list.setCurrentRow(0)
    dlg._remove_selected()

    assert [s.name for s in proj.sources] == ["ds-keep"]
    assert not copy.exists()
    assert keep.exists()
    assert [s.name for s in open_project(tmp_path).sources] == ["ds-keep"]


def test_remove_imported_source_cancel_keeps_everything(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog

    copy, source = _imported_source(tmp_path, "ds-abc")
    proj = _make_proj(tmp_path)
    proj.sources = [source]
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel
    )
    dlg = SourceManagerDialog(proj)
    dlg._source_list.setCurrentRow(0)
    dlg._remove_selected()

    assert len(proj.sources) == 1
    assert copy.exists()
