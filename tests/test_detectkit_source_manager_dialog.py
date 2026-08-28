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
    dlg._add_source()

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
    dlg._add_source()

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
    dlg._add_source()

    assert len(proj.sources) == 1
    assert proj.sources[0].level == "aabb"


def test_remove_selected_deletes_pending_escalation_staging_dir(qapp, tmp_path):
    """Removing a source with an unreviewed pending escalation must not leak
    its staging directory under artifacts/pending_escalations/."""
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.models import OBBSource, PendingEscalation

    staged_dir = tmp_path / "artifacts" / "pending_escalations" / "orig-variant-abc123"
    staged_dir.mkdir(parents=True)
    (staged_dir / "labels").mkdir()

    proj = _make_proj(tmp_path)
    proj.sources = [
        OBBSource(
            path=str(tmp_path / "orig"),
            name="orig",
            pending_escalation=PendingEscalation(
                staged_path=str(staged_dir),
                target_level="polygon",
                sam2_variant="sam2.1-hiera-base_plus",
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
        "hydra_suite.detectkit.gui.dialogs.source_manager.inspect_detectkit_source",
        lambda *args, **kwargs: SimpleNamespace(dataset_root=source_root),
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
    dlg._add_source()

    assert len(proj.sources) == 1
    added = proj.sources[0]
    assert added.path == str(source_root)
    assert added.original_path == str(source_root)
    assert added.imported is False
    assert (source_root / "classes.txt").exists()
