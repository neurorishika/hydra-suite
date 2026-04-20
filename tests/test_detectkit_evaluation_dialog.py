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


def test_evaluation_dialog_imports(qapp):
    pass  # noqa: F401


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


def test_evaluation_dialog_has_analyze_button(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    dlg = EvaluationDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "btn_analyze")


def test_evaluation_dialog_has_analysis_view(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    dlg = EvaluationDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "_analysis_view")


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
