"""Renaming a class in the scheme editor migrates the labels already stored."""

from __future__ import annotations

import gc
import os
import sqlite3
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.classkit.config.schemas import (  # noqa: E402
    LabelingScheme,
    build_composite_rename_map,
)
from hydra_suite.classkit.core.store.db import ClassKitDB  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.classkit.gui.dialogs.class_editor import (  # noqa: E402
    ClassEditorDialog,
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


def _scheme(*factors: tuple[str, list[str]]) -> LabelingScheme:
    return LabelingScheme.from_dict(
        {
            "name": "s",
            "factors": [{"name": name, "labels": labels} for name, labels in factors],
            "training_modes": [],
        }
    )


# ── rename-map construction ───────────────────────────────────────────────


def test_rename_map_single_factor():
    scheme = _scheme(("class", ["worker", "queen"]))
    assert build_composite_rename_map(scheme, [{"worker": "forager"}]) == {
        "worker": "forager"
    }


def test_rename_map_covers_every_composite_containing_the_renamed_label():
    scheme = _scheme(("caste", ["worker", "queen"]), ("state", ["idle", "moving"]))
    mapping = build_composite_rename_map(scheme, [{"worker": "forager"}, {}])
    assert mapping["worker_idle"] == "forager_idle"
    assert mapping["worker_moving"] == "forager_moving"
    # 'unknown' is injected into every factor of a multi-factor scheme
    assert mapping["worker_unknown"] == "forager_unknown"
    assert not any(key.startswith("queen") for key in mapping)


def test_rename_map_is_empty_without_renames():
    scheme = _scheme(("class", ["a", "b"]))
    assert build_composite_rename_map(scheme, []) == {}
    assert build_composite_rename_map(scheme, [{}]) == {}
    assert build_composite_rename_map(scheme, [{"a": "a"}]) == {}


# ── database rewrite ──────────────────────────────────────────────────────


def _seed_db(tmp_path: Path, rows: list[tuple[str, str]]) -> ClassKitDB:
    db = ClassKitDB(tmp_path / "project.db")
    with sqlite3.connect(db.db_path) as conn:
        for path, label in rows:
            conn.execute(
                "INSERT INTO images (file_path, label, confidence, label_source,"
                " verified) VALUES (?, ?, ?, ?, ?)",
                (path, label, 0.9, "human", 1),
            )
        conn.commit()
    return db


def test_rename_labels_rewrites_and_preserves_provenance(tmp_path: Path):
    db = _seed_db(tmp_path, [("a.png", "worker"), ("b.png", "queen")])

    assert db.rename_labels({"worker": "forager"}) == 1

    with sqlite3.connect(db.db_path) as conn:
        rows = dict(conn.execute("SELECT file_path, label FROM images").fetchall())
        meta = conn.execute(
            "SELECT confidence, label_source, verified FROM images"
            " WHERE file_path = 'a.png'"
        ).fetchone()
    assert rows == {"a.png": "forager", "b.png": "queen"}
    assert meta == (0.9, "human", 1)


def test_rename_labels_handles_a_swap(tmp_path: Path):
    db = _seed_db(tmp_path, [("a.png", "left"), ("b.png", "right")])

    assert db.rename_labels({"left": "right", "right": "left"}) == 2

    with sqlite3.connect(db.db_path) as conn:
        rows = dict(conn.execute("SELECT file_path, label FROM images").fetchall())
    assert rows == {"a.png": "right", "b.png": "left"}


def test_rename_labels_ignores_empty_and_identity_mappings(tmp_path: Path):
    db = _seed_db(tmp_path, [("a.png", "worker")])
    assert db.rename_labels({}) == 0
    assert db.rename_labels({"worker": "worker"}) == 0
    assert db.rename_labels({"worker": ""}) == 0


# ── dialog reports the renames ────────────────────────────────────────────


def _open_dialog(**kwargs) -> ClassEditorDialog:
    return ClassEditorDialog(**kwargs)


def test_dialog_reports_edited_label_as_a_rename(qapp):
    dlg = _open_dialog(
        scheme_dict={
            "name": "caste",
            "factors": [{"name": "caste", "labels": ["worker", "queen"]}],
            "training_modes": [],
        }
    )
    dlg._label_rows[0]._name_edit.setText("forager")
    dlg._flush_current_factor()

    assert dlg.get_label_renames() == [{"worker": "forager"}]
    assert dlg.get_scheme_dict()["factors"][0]["labels"] == ["forager", "queen"]


def test_dialog_survives_a_rename_across_factor_switches(qapp):
    dlg = _open_dialog(
        scheme_dict={
            "name": "s",
            "factors": [
                {"name": "caste", "labels": ["worker", "queen"]},
                {"name": "state", "labels": ["idle", "moving"]},
            ],
            "training_modes": [],
        }
    )
    dlg._label_rows[0]._name_edit.setText("forager")
    dlg._factor_list.setCurrentRow(1)
    dlg._label_rows[1]._name_edit.setText("running")
    dlg._factor_list.setCurrentRow(0)
    dlg._flush_current_factor()

    assert dlg.get_label_renames() == [{"worker": "forager"}, {"moving": "running"}]


def test_dialog_reports_no_rename_for_added_or_removed_labels(qapp):
    dlg = _open_dialog(
        scheme_dict={
            "name": "caste",
            "factors": [{"name": "caste", "labels": ["worker", "queen"]}],
            "training_modes": [],
        }
    )
    dlg._add_label_row()
    dlg._label_rows[-1]._name_edit.setText("male")
    dlg._remove_label_row(dlg._label_rows[1])
    dlg._flush_current_factor()

    assert dlg.get_label_renames() == [{}]


def test_dialog_reports_no_rename_after_applying_a_preset(qapp):
    dlg = _open_dialog(classes=["worker", "queen"])
    dlg._factors = [
        {
            "name": "caste",
            "labels": ["forager", "queen"],
            "shortcuts": [],
            "origins": [None, None],
            "origin_index": None,
        }
    ]
    assert dlg.get_label_renames() == [{}]


# ── main-window helper end to end ─────────────────────────────────────────


def test_main_window_helper_migrates_stored_labels(tmp_path: Path, qapp):
    from hydra_suite.classkit.gui.main_window import MainWindow

    db = _seed_db(tmp_path, [("a.png", "worker_idle"), ("b.png", "queen_idle")])
    stub = types.SimpleNamespace(
        db_path=db.db_path,
        classes=["worker", "queen"],
        _reload_label_state_from_db=lambda _db=None: None,
    )
    old_scheme_dict = {
        "name": "s",
        "factors": [
            {"name": "caste", "labels": ["worker", "queen"]},
            {"name": "state", "labels": ["idle", "moving"]},
        ],
        "training_modes": [],
    }

    migrated = MainWindow._migrate_renamed_project_labels(
        stub, old_scheme_dict, [{"worker": "forager"}, {}]
    )

    assert migrated == 1
    with sqlite3.connect(db.db_path) as conn:
        rows = dict(conn.execute("SELECT file_path, label FROM images").fetchall())
    assert rows == {"a.png": "forager_idle", "b.png": "queen_idle"}


def test_main_window_helper_is_a_no_op_without_renames(tmp_path: Path, qapp):
    from hydra_suite.classkit.gui.main_window import MainWindow

    db = _seed_db(tmp_path, [("a.png", "worker")])
    stub = types.SimpleNamespace(
        db_path=db.db_path,
        classes=["worker"],
        _reload_label_state_from_db=lambda _db=None: None,
    )
    assert MainWindow._migrate_renamed_project_labels(stub, None, [{"a": "b"}]) == 0
    assert (
        MainWindow._migrate_renamed_project_labels(
            stub, {"name": "s", "factors": [], "training_modes": []}, [{}]
        )
        == 0
    )
