"""Tests for the refactored DetectKit MainWindow."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QTabWidget, QToolBar  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture(scope="module")
def main_win(qapp):
    """Single shared DetectKitMainWindow — avoids per-test SVG GC crash."""
    from hydra_suite.detectkit.gui.main_window import DetectKitMainWindow

    win = DetectKitMainWindow()
    yield win


def test_main_window_has_toolbar(main_win):
    toolbars = main_win.findChildren(QToolBar)
    assert toolbars, "MainWindow must have at least one QToolBar"


def test_main_window_no_tab_widget(main_win):
    tabs = main_win.findChildren(QTabWidget)
    assert not tabs, "QTabWidget must be removed from MainWindow"


def test_main_window_has_tools_panel(main_win):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panels = main_win.findChildren(ToolsPanel)
    assert panels, "MainWindow must contain a ToolsPanel"


def test_main_window_tools_panel_fixed_width(main_win):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = main_win.findChildren(ToolsPanel)[0]
    assert panel.maximumWidth() == 280
    assert panel.minimumWidth() == 280


def test_main_window_has_open_source_manager(main_win):
    assert hasattr(main_win, "_open_source_manager")


def test_main_window_has_open_training_dialog(main_win):
    assert hasattr(main_win, "_open_training_dialog")


def test_main_window_has_open_evaluation_dialog(main_win):
    assert hasattr(main_win, "_open_evaluation_dialog")


def test_main_window_has_open_history_dialog(main_win):
    assert hasattr(main_win, "_open_history_dialog")


def test_main_window_has_open_active_learning_dialog(main_win):
    assert hasattr(main_win, "_open_active_learning_dialog")


def test_main_window_toolbar_hidden_on_welcome(qapp):
    """Fresh window (welcome screen) must have toolbar explicitly hidden."""
    from hydra_suite.detectkit.gui.main_window import DetectKitMainWindow

    fresh = DetectKitMainWindow()
    # _toolbar must be set explicitly invisible (not just hidden by parent)
    assert not fresh._toolbar.isVisibleTo(
        fresh
    ), "Toolbar should be hidden on welcome screen"


def test_main_window_toolbar_visible_after_project_load(qapp, main_win, tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    main_win._load_project(proj)
    # isVisibleTo(parent) returns True if widget would be visible when parent is shown
    assert main_win._toolbar.isVisibleTo(
        main_win
    ), "Toolbar should be visible after loading a project"


def test_save_project_no_deleted_panels(qapp, main_win, tmp_path):
    """_save_current_project must not raise (no old training/eval/history panels)."""
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject(project_dir=tmp_path / "proj2", class_names=["ant"])
    main_win._load_project(proj)
    main_win._save_current_project()  # Must not raise


def test_dialogs_init_exports(qapp):
    from hydra_suite.detectkit.gui import dialogs

    assert hasattr(dialogs, "NewProjectDialog")
    assert hasattr(dialogs, "SourceManagerDialog")
    assert hasattr(dialogs, "TrainingDialog")
    assert hasattr(dialogs, "EvaluationDialog")
    assert hasattr(dialogs, "HistoryDialog")
    assert hasattr(dialogs, "ActiveLearningDialog")


def test_load_project_populates_model_selector_from_history(qapp, main_win, tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    model_path = tmp_path / "models" / "best.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"weights")
    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    proj.training_history = [{"run_id": "run_1", "project_model_path": str(model_path)}]

    main_win._load_project(proj)

    # Model selector is now a read-only display label, not a combo box.
    assert main_win._tools_panel._active_model_path == str(model_path)
    assert model_path.name in main_win._tools_panel._model_display.text()


def test_load_project_filters_non_preview_models(qapp, main_win, tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    obb_model = tmp_path / "models" / "obb.pt"
    seq_model = tmp_path / "models" / "seq.pt"
    obb_model.parent.mkdir(parents=True)
    obb_model.write_bytes(b"obb")
    seq_model.write_bytes(b"seq")

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    proj.training_history = [
        {"run_id": "run_1", "role": "obb_direct", "project_model_path": str(obb_model)},
        {"run_id": "run_2", "role": "seq_detect", "project_model_path": str(seq_model)},
    ]

    main_win._load_project(proj)

    # Only the obb_direct model should be auto-selected (seq_detect has no counterpart).
    assert main_win._tools_panel._active_model_path == str(obb_model)


def test_show_image_does_not_auto_run_prediction_overlay(
    qapp, main_win, tmp_path, monkeypatch
):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    model_path = tmp_path / "models" / "best.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"weights")

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    proj.active_model_path = str(model_path)
    proj.training_history = [
        {
            "run_id": "run_1",
            "role": "obb_direct",
            "project_model_path": str(model_path),
        }
    ]

    main_win._load_project(proj)

    called: list[tuple[tuple, dict]] = []

    monkeypatch.setattr(main_win._canvas, "load_image", lambda _path: True)
    monkeypatch.setattr(main_win._canvas, "fit_in_view", lambda: None)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.main_window.find_label_for_image",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.main_window.predict_preview_detections",
        lambda *args, **kwargs: called.append((args, kwargs)) or [],
    )

    main_win.show_image(str(tmp_path), str(tmp_path / "sample.png"))

    assert called == []


def test_main_window_al_action_disabled_without_active_model(qapp):
    """AL toolbar action must be disabled on a fresh window (no project, no model)."""
    from hydra_suite.detectkit.gui.main_window import DetectKitMainWindow

    win = DetectKitMainWindow()
    assert not win._al_action.isEnabled()


def test_run_inference_overlay_populates_prediction_overlay(
    qapp, main_win, tmp_path, monkeypatch
):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    model_path = tmp_path / "models" / "best.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"weights")

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    proj.active_model_path = str(model_path)
    proj.training_history = [
        {
            "run_id": "run_1",
            "role": "obb_direct",
            "project_model_path": str(model_path),
        }
    ]

    main_win._load_project(proj)

    captured: dict[str, object] = {}

    monkeypatch.setattr(main_win._canvas, "load_image", lambda _path: True)
    monkeypatch.setattr(main_win._canvas, "fit_in_view", lambda: None)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.main_window.find_label_for_image",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.main_window.predict_preview_detections",
        lambda *args, **kwargs: [
            {
                "class_id": 0,
                "polygon_px": [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)],
                "confidence": 0.91,
            }
        ],
    )

    def _capture_pred(detections, class_names=None):
        captured["detections"] = detections
        captured["class_names"] = class_names

    monkeypatch.setattr(main_win._canvas, "set_pred_detections", _capture_pred)

    main_win.show_image(str(tmp_path), str(tmp_path / "sample.png"))
    main_win._run_inference_overlay()

    assert captured["class_names"] == ["ant"]
    assert len(captured["detections"]) == 1
