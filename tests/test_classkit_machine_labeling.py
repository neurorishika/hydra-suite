from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QApplication = QtWidgets.QApplication

MachineLabelingDialog = pytest.importorskip(
    "hydra_suite.classkit.gui.dialogs.machine_labeling"
).MachineLabelingDialog
MainWindow = pytest.importorskip("hydra_suite.classkit.gui.main_window").MainWindow


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


def test_machine_labeling_dialog_defaults_to_history_source_when_available(
    qapp, tmp_path: Path
) -> None:
    entry = {
        "display_name": "Past Model",
        "artifact_paths": [str(tmp_path / "history_model.pt")],
        "mode": "yolo",
    }

    dlg = MachineLabelingDialog(
        scope_options=[("All images", [0, 1])],
        predictions_available=False,
        image_count=2,
        model_history_entries=[entry],
        project_path=tmp_path,
        db_path=tmp_path / "classkit.db",
    )

    assert dlg.selected_model_source() == MachineLabelingDialog.MODEL_SOURCE_HISTORY


def test_machine_labeling_dialog_returns_selected_model_source_settings(
    qapp, tmp_path: Path
) -> None:
    entry = {
        "display_name": "Transferred Model",
        "artifact_paths": [str(tmp_path / "transferred_model.pt")],
        "mode": "tiny",
    }
    checkpoint = tmp_path / "foreign_project_model.pth"

    dlg = MachineLabelingDialog(
        scope_options=[("All images", [0, 1, 2])],
        predictions_available=True,
        image_count=3,
        model_history_entries=[entry],
        project_path=tmp_path,
        db_path=tmp_path / "classkit.db",
    )
    history_index = dlg.model_source_combo.findData(
        MachineLabelingDialog.MODEL_SOURCE_HISTORY
    )
    dlg.model_source_combo.setCurrentIndex(history_index)
    dlg._selected_model_entry = entry

    history_settings = dlg.get_settings()

    assert (
        history_settings["model_source"] == MachineLabelingDialog.MODEL_SOURCE_HISTORY
    )
    assert history_settings["model_entry"] == entry

    file_index = dlg.model_source_combo.findData(
        MachineLabelingDialog.MODEL_SOURCE_FILE
    )
    dlg.model_source_combo.setCurrentIndex(file_index)
    dlg._selected_checkpoint_path = str(checkpoint)

    file_settings = dlg.get_settings()

    assert file_settings["model_source"] == MachineLabelingDialog.MODEL_SOURCE_FILE
    assert file_settings["checkpoint_path"] == str(checkpoint)


def test_run_machine_labeling_model_source_uses_history_entry(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    entry = {
        "display_name": "Past Model",
        "artifact_paths": [str(tmp_path / "history_model.pt")],
        "mode": "tiny",
    }
    calls: dict[str, object] = {}

    def fake_load(model_entry, on_success=None):
        calls["entry"] = model_entry
        calls["has_callback"] = callable(on_success)
        if on_success is not None:
            on_success()

    def fake_apply(**kwargs):
        calls["apply"] = kwargs

    monkeypatch.setattr(window, "_load_model_from_cache_entry", fake_load)
    monkeypatch.setattr(window, "apply_model_predictions_as_review_labels", fake_apply)

    window._run_machine_labeling_model_source(
        {
            "scope_indices": [1, 3],
            "scope_label": "Selected",
            "skip_verified": False,
            "model_source": MachineLabelingDialog.MODEL_SOURCE_HISTORY,
            "model_entry": entry,
        }
    )

    assert calls["entry"] == entry
    assert calls["has_callback"] is True
    assert calls["apply"] == {
        "indices": [1, 3],
        "scope_label": "Selected",
        "skip_verified": False,
        "model_provider": "project_history_model",
        "model_metadata": {
            "model_name": "Past Model",
            "model_path": str(tmp_path / "history_model.pt"),
        },
    }


def test_machine_labeling_dialog_returns_other_project_source_settings(
    qapp, tmp_path: Path
) -> None:
    other_project = tmp_path / "other_project"
    other_project.mkdir()
    other_artifact = other_project / "history_model.pt"
    entry = {
        "display_name": "External Model",
        "artifact_paths": [str(other_artifact)],
        "mode": "tiny",
    }

    dlg = MachineLabelingDialog(
        scope_options=[("All images", [0, 1])],
        predictions_available=False,
        image_count=2,
        model_history_entries=[],
        project_path=tmp_path,
        db_path=tmp_path / "classkit.db",
    )

    other_index = dlg.model_source_combo.findData(
        MachineLabelingDialog.MODEL_SOURCE_OTHER_PROJECT
    )
    assert other_index >= 0
    dlg.model_source_combo.setCurrentIndex(other_index)
    dlg._selected_model_entry = entry
    dlg._other_project_path = other_project

    settings = dlg.get_settings()

    assert settings["model_source"] == MachineLabelingDialog.MODEL_SOURCE_OTHER_PROJECT
    assert settings["model_entry"] == entry
    assert settings["other_project_path"] == str(other_project)


def test_run_machine_labeling_model_source_uses_other_project_entry(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    other_project = tmp_path / "other_project"
    other_project.mkdir()
    other_artifact = other_project / "history_model.pt"
    other_artifact.touch()
    entry = {
        "display_name": "External Model",
        "artifact_paths": [str(other_artifact)],
        "mode": "tiny",
    }
    calls: dict[str, object] = {}

    def fake_load(model_entry, on_success=None):
        calls["entry"] = model_entry
        if on_success is not None:
            on_success()

    def fake_apply(**kwargs):
        calls["apply"] = kwargs

    monkeypatch.setattr(window, "_load_model_from_cache_entry", fake_load)
    monkeypatch.setattr(window, "apply_model_predictions_as_review_labels", fake_apply)

    window._run_machine_labeling_model_source(
        {
            "scope_indices": [0],
            "scope_label": "All images",
            "skip_verified": True,
            "model_source": MachineLabelingDialog.MODEL_SOURCE_OTHER_PROJECT,
            "model_entry": entry,
            "other_project_path": str(other_project),
        }
    )

    assert calls["entry"] == entry
    assert calls["apply"] == {
        "indices": [0],
        "scope_label": "All images",
        "skip_verified": True,
        "model_provider": "external_project_history_model",
        "model_metadata": {
            "model_name": "External Model",
            "model_path": str(other_artifact),
            "source_project_path": str(other_project),
        },
    }


def test_run_machine_labeling_model_source_uses_external_checkpoint(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    checkpoint = tmp_path / "foreign_model.pth"
    calls: dict[str, object] = {}

    def fake_load(path, on_success=None, show_message_box=True):
        calls["path"] = path
        calls["show_message_box"] = show_message_box
        calls["has_callback"] = callable(on_success)
        if on_success is not None:
            on_success()

    def fake_apply(**kwargs):
        calls["apply"] = kwargs

    monkeypatch.setattr(window, "_load_checkpoint_from_path", fake_load)
    monkeypatch.setattr(window, "apply_model_predictions_as_review_labels", fake_apply)

    window._run_machine_labeling_model_source(
        {
            "scope_indices": [0, 2],
            "scope_label": "All images",
            "skip_verified": True,
            "model_source": MachineLabelingDialog.MODEL_SOURCE_FILE,
            "checkpoint_path": str(checkpoint),
        }
    )

    assert calls["path"] == checkpoint
    assert calls["show_message_box"] is False
    assert calls["has_callback"] is True
    assert calls["apply"] == {
        "indices": [0, 2],
        "scope_label": "All images",
        "skip_verified": True,
        "model_provider": "external_checkpoint",
        "model_metadata": {
            "model_name": "foreign_model",
            "model_path": str(checkpoint),
        },
    }


def test_run_machine_labeling_model_source_uses_loaded_predictions(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        window,
        "apply_model_predictions_as_review_labels",
        lambda **kwargs: calls.setdefault("apply", kwargs),
    )

    window._run_machine_labeling_model_source(
        {
            "scope_indices": [4],
            "scope_label": "Pending review only",
            "skip_verified": True,
            "model_source": MachineLabelingDialog.MODEL_SOURCE_LOADED,
        }
    )

    assert calls["apply"] == {
        "indices": [4],
        "scope_label": "Pending review only",
        "skip_verified": True,
        "model_provider": "loaded_model",
    }
