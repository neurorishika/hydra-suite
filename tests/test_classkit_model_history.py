from __future__ import annotations

import os
import sys

import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QApplication = QtWidgets.QApplication


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_model_history_dialog_sets_high_contrast_alternating_rows(qapp, tmp_path):
    from hydra_suite.classkit.gui.dialogs.model_history import ModelHistoryDialog

    dialog = ModelHistoryDialog([], project_path=tmp_path)

    assert dialog.table.alternatingRowColors() is True
    assert "alternate-background-color: #2d2d30" in dialog.table.styleSheet()


def test_model_history_export_upgrades_legacy_flat_checkpoint_to_v2(
    qapp, tmp_path, legacy_torchvision_flat_headtail, monkeypatch
):
    from hydra_suite.classkit.gui.dialogs.model_history import ModelHistoryDialog

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.model_history.QMessageBox.information",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.model_history.QMessageBox.warning",
        lambda *_args, **_kwargs: None,
    )

    entry = {
        "id": 1,
        "display_name": "legacy headtail",
        "timestamp": "2026-04-18T01:19:13",
        "mode": "flat_custom",
        "class_names": ["left", "right", "unknown", "up"],
        "artifact_paths": [str(legacy_torchvision_flat_headtail)],
        "meta": {
            "training_settings": {
                "custom_backbone": "efficientnet_b0",
                "custom_input_size": 96,
                "monochrome": False,
            }
        },
    }

    dialog = ModelHistoryDialog([entry], project_path=tmp_path)
    dialog._export_selected()

    exported = next((tmp_path / "models").glob("*.pth"))
    ckpt = torch.load(str(exported), map_location="cpu", weights_only=False)
    assert "schema_version" not in ckpt
    assert ckpt["arch"] == "resnet18"
    assert ckpt["class_names"] == ["left", "right", "unknown", "up"]
    assert ckpt["input_size"] == (96, 80)
