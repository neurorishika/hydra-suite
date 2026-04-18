"""Tests for DetectKit source review dialog."""

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


def test_detectkit_source_validation_dialog_describes_import(qapp, tmp_path: Path):
    from hydra_suite.detectkit.gui.dialogs.source_validation import (
        DetectKitSourceValidationDialog,
    )
    from hydra_suite.detectkit.gui.source_import import DetectKitSourceInspection

    inspection = DetectKitSourceInspection(
        dataset_root=tmp_path,
        source_kind="coco",
        images_count=12,
        annotation_count=34,
        discovered_labels=["ant", "bee"],
        requires_import=True,
    )

    dialog = DetectKitSourceValidationDialog(tmp_path, inspection)
    ok_button = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)

    assert ok_button is not None
    assert ok_button.text() == "Import and Add"
    assert dialog._kind_value.text() == "COCO annotations dataset"
    assert dialog._images_value.text() == "12"
    assert dialog._annotations_value.text() == "34"
    assert dialog._class_names_value.text() == "ant, bee"
    assert "normalized to DetectKit's canonical" in dialog._action_value.text()
