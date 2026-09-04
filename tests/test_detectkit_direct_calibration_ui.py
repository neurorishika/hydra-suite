import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

LABEL_LINE = "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"


def _dataset(tmp_path: Path, split: str, names: list[str]) -> Path:
    images = tmp_path / "images" / split
    labels = tmp_path / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for name in names:
        cv2.imwrite(str(images / f"{name}.png"), np.zeros((200, 300, 3), np.uint8))
        (labels / f"{name}.txt").write_text(LABEL_LINE)
    yaml = tmp_path / "data.yaml"
    yaml.write_text(
        f"path: {tmp_path}\ntrain: images/train\nval: images/val\nnames:\n  0: ant\n"
    )
    return yaml


@pytest.fixture
def calibration_wizard(tmp_path):
    from hydra_suite.detectkit.gui.dialogs.direct_calibration_wizard import (
        DirectCalibrationWizard,
    )

    yaml = _dataset(tmp_path, "val", ["rec1_000", "rec1_001"])
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    wizard = DirectCalibrationWizard(
        None,
        model_path=model,
        task="obb",
        dataset_yaml=yaml,
        sources=[],
        training_geometry={
            "geometry_mode": "auto_object",
            "imgsz": 640,
            "object_tile_fraction": 0.4,
            "overlap": 0.2,
        },
        evidence_dir=tmp_path / "evidence",
    )
    yield wizard
    wizard.close()


def test_run_is_blocked_until_exhaustive_labels_are_acknowledged(calibration_wizard):
    assert calibration_wizard.chk_exhaustive.isChecked() is False
    assert calibration_wizard.btn_run.isEnabled() is False
    calibration_wizard.chk_exhaustive.setChecked(True)
    assert calibration_wizard.btn_run.isEnabled() is True


def test_summary_states_frames_instances_sizes_and_split(calibration_wizard):
    text = calibration_wizard.lbl_evidence_summary.text()
    assert "2 frames" in text and "2 instances" in text
    assert "val" in text and "200" in text


def test_candidate_table_lists_every_candidate_with_its_tile_cost(calibration_wizard):
    table = calibration_wizard.table_candidates
    assert table.rowCount() == len(calibration_wizard.candidates)
    assert table.item(0, 0).text() == "Full frame (no SAHI)"
    assert table.item(0, 1).text() == "1"


def test_detection_cap_is_user_visible_and_reaches_the_request(calibration_wizard):
    calibration_wizard.chk_exhaustive.setChecked(True)
    calibration_wizard.spin_max_targets.setValue(120)
    assert calibration_wizard.request().max_targets == 120


def test_experimental_label_is_shown(calibration_wizard):
    assert "Experimental calibration" in calibration_wizard.windowTitle()


def test_confirm_broad_sweep_reenables_run_when_a_candidate_is_over_budget(tmp_path):
    from hydra_suite.detectkit.gui.dialogs.direct_calibration_wizard import (
        DirectCalibrationWizard,
    )

    yaml = _dataset(tmp_path, "val", ["rec1_000", "rec1_001"])
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    wizard = DirectCalibrationWizard(
        None,
        model_path=model,
        task="obb",
        dataset_yaml=yaml,
        sources=[],
        training_geometry={
            "geometry_mode": "auto_object",
            "imgsz": 640,
            "object_tile_fraction": 0.4,
            "overlap": 0.2,
        },
        evidence_dir=tmp_path / "evidence",
        max_total_tiles=1,
    )
    try:
        wizard.chk_exhaustive.setChecked(True)
        assert any(estimate.failed_reason for estimate in wizard.estimates)
        assert wizard.btn_run.isEnabled() is False
        wizard.chk_confirm_broad_sweep.setChecked(True)
        assert wizard.btn_run.isEnabled() is True
    finally:
        wizard.close()
