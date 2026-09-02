"""Tests for DetectKit EvaluationDialog."""

from __future__ import annotations

import os
import sys

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


def test_evaluation_dialog_is_base_dialog(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog
    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = EvaluationDialog(_make_proj(tmp_path))
    assert isinstance(dlg, BaseDialog)


def test_evaluation_dialog_has_close_button(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    dlg = EvaluationDialog(_make_proj(tmp_path))
    close_btn = dlg._buttons.button(QDialogButtonBox.StandardButton.Close)
    assert close_btn is not None


def test_evaluation_dialog_no_sources_message(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    dlg = EvaluationDialog(proj)
    dlg._run_dataset_analysis()
    assert "No dataset sources" in dlg._analysis_view.toPlainText()


def test_evaluation_dialog_has_quick_test_button(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    dlg = EvaluationDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "btn_quick_test")
    assert dlg.btn_quick_test.isEnabled()


def test_evaluation_dialog_exposes_validation_and_comparison_controls(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    dlg = EvaluationDialog(_make_proj(tmp_path))

    assert hasattr(dlg, "candidate_table")
    assert hasattr(dlg, "results_table")
    assert hasattr(dlg, "btn_evaluate_selected")
    assert hasattr(dlg, "btn_evaluate_all")
    result_headers = [
        dlg.results_table.horizontalHeaderItem(column).text()
        for column in range(dlg.results_table.columnCount())
    ]
    assert "Precision" in result_headers
    assert "Recall" in result_headers
    assert "mAP50" in result_headers
    assert "mAP50-95" in result_headers


def test_evaluation_dialog_displays_persisted_runs_side_by_side(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    project = _make_proj(tmp_path)
    project.evaluation_history = [
        {
            "evaluation_id": "eval-1",
            "run_id": "run-1",
            "role": "obb_direct",
            "model_name": "first.pt",
            "dataset_name": "derived-1",
            "precision": 0.8,
            "recall": 0.7,
            "map50": 0.75,
            "map50_95": 0.5,
            "elapsed_seconds": 3.0,
            "inference_ms": 2.0,
            "evaluated_at": "2026-09-02T12:00:00",
            "error": "",
        },
        {
            "evaluation_id": "eval-2",
            "run_id": "run-2",
            "role": "detect_direct",
            "model_name": "second.pt",
            "dataset_name": "derived-2",
            "precision": 0.6,
            "recall": 0.9,
            "map50": 0.7,
            "map50_95": 0.4,
            "elapsed_seconds": 4.0,
            "inference_ms": 3.0,
            "evaluated_at": "2026-09-02T13:00:00",
            "error": "",
        },
    ]

    dlg = EvaluationDialog(project)

    assert dlg.results_table.rowCount() == 2
    displayed_run_ids = {
        dlg.results_table.item(row, 1).text()
        for row in range(dlg.results_table.rowCount())
    }
    assert displayed_run_ids == {"run-1", "run-2"}


def test_evaluation_dialog_records_completed_metrics(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.evaluation import EvaluationResult
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    project = _make_proj(tmp_path)
    dlg = EvaluationDialog(project)
    saved = []
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.project.save_project",
        lambda candidate: saved.append(candidate),
    )
    result = EvaluationResult(
        evaluation_id="eval-1",
        run_id="run-1",
        role="obb_direct",
        model_path="/external/best.pt",
        dataset_dir="/external/derived",
        precision=0.8,
        recall=0.7,
        map50=0.75,
        map50_95=0.5,
        evaluated_at="2026-09-02T12:00:00",
    )

    dlg._on_evaluation_result(result)

    assert project.evaluation_history[0]["precision"] == 0.8
    assert "model_path" not in project.evaluation_history[0]
    assert dlg.results_table.rowCount() == 1
    assert saved == [project]


def test_evaluation_worker_is_base_worker():
    from hydra_suite.detectkit.jobs.evaluation import EvaluationWorker
    from hydra_suite.widgets.workers import BaseWorker

    assert issubclass(EvaluationWorker, BaseWorker)


def test_evaluation_worker_executes_evaluation_off_gui_thread(
    qapp, tmp_path, monkeypatch
):
    from PySide6.QtCore import QThread

    from hydra_suite.detectkit.evaluation import EvaluationCandidate, EvaluationResult
    from hydra_suite.detectkit.jobs import evaluation as evaluation_job

    called_from = []
    candidate = EvaluationCandidate(
        run_id="run-1",
        role="obb_direct",
        task="obb",
        model_path="/tmp/model.pt",
        dataset_dir="/tmp/dataset",
        dataset_yaml="/tmp/dataset/dataset.yaml",
        imgsz=640,
        available=True,
    )

    def _evaluate(candidate, **_kwargs):
        called_from.append(QThread.currentThread())
        return EvaluationResult.failed(candidate, "eval-1", "test result")

    monkeypatch.setattr(evaluation_job, "evaluate_candidate", _evaluate)
    worker = evaluation_job.EvaluationWorker(
        [candidate],
        output_root=tmp_path,
        device="cpu",
        batch=1,
    )

    worker.start()
    assert worker.wait(5000)
    qapp.processEvents()

    assert len(called_from) == 1
    assert called_from[0] is not qapp.thread()


def test_dataset_panel_emits_evaluate_request(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    emitted = []
    panel.evaluate_requested.connect(lambda: emitted.append(True))

    panel.btn_evaluate.click()

    assert emitted == [True]


def test_main_window_toolbar_opens_evaluation_dialog(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs import evaluation_dialog as dialog_module
    from hydra_suite.detectkit.gui.main_window import MainWindow

    window = MainWindow()
    window._project = _make_proj(tmp_path)
    captured = {}

    class _FakeDialog:
        def __init__(self, project, parent=None):
            captured["project"] = project
            captured["parent"] = parent

        def exec(self):
            captured["executed"] = True

    monkeypatch.setattr(dialog_module, "EvaluationDialog", _FakeDialog)
    actions = {action.text(): action for action in window._toolbar.actions()}

    actions["Evaluate"].trigger()

    assert captured["project"] is window._project
    assert captured["parent"] is window
    assert captured["executed"] is True
    window.close()


def test_evaluation_dialog_quick_test_no_model_shows_message(
    qapp, tmp_path, monkeypatch
):
    """Quick test with no active_model_path shows an informative message (not a crash)."""
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    proj = _make_proj(tmp_path)
    proj.active_model_path = ""  # no active model
    dlg = EvaluationDialog(proj)

    shown = []
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.evaluation_dialog.QMessageBox.information",
        lambda *a, **kw: shown.append(a[2]),
    )
    dlg._quick_test()
    assert shown, "Expected an informative message when no model is active"


def test_evaluation_dialog_quick_test_passes_role_specific_settings(
    qapp, tmp_path, monkeypatch
):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog
    from hydra_suite.detectkit.gui.models import OBBSource

    model_path = tmp_path / "models" / "best.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"weights")

    proj = _make_proj(tmp_path)
    proj.active_model_path = str(model_path)
    proj.imgsz_obb_direct = 896
    proj.crop_pad_ratio = 0.2
    proj.min_crop_size_px = 96
    proj.enforce_square = False
    proj.sources = [OBBSource(path=str(tmp_path / "dataset"), name="dataset")]
    proj.training_history = [
        {
            "run_id": "run_1",
            "role": "obb_direct",
            "project_model_path": str(model_path),
        }
    ]
    dlg = EvaluationDialog(proj)

    captured: dict[str, object] = {}

    class FakeDialog:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def open(self):
            captured["opened"] = True

    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.dialogs.model_test_dialog.ModelTestDialog",
        FakeDialog,
    )

    dlg._quick_test()

    assert captured["role"] == "obb_direct"
    assert captured["imgsz"] == 896
    assert captured["dataset_dir"] == str(tmp_path / "dataset")
    assert captured["crop_pad_ratio"] == 0.2
    assert captured["min_crop_size_px"] == 96
    assert captured["enforce_square"] is False
    assert captured["opened"] is True
