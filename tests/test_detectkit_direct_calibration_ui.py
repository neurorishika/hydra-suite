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


def test_request_confidences_are_the_full_grid(calibration_wizard):
    confidences = calibration_wizard.request().confidences
    assert len(confidences) == 19
    assert confidences[0] == 0.05
    assert confidences[-1] == 0.95


def test_request_merge_settings_sweep_three_thresholds(calibration_wizard):
    merge_settings = calibration_wizard.request().merge_settings
    assert len(merge_settings) == 3
    assert {m.threshold for m in merge_settings} == {0.3, 0.5, 0.7}
    assert len({m.policy for m in merge_settings}) == 1
    assert len({m.metric for m in merge_settings}) == 1


def test_request_runtime_tier_reflects_combo_and_constructor_default(tmp_path):
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
        runtime_tier="gpu_fast",
    )
    try:
        assert wizard.combo_runtime_tier.currentText() == "gpu_fast"
        assert wizard.request().runtime_tier == "gpu_fast"
        wizard.combo_runtime_tier.setCurrentText("cpu")
        assert wizard.request().runtime_tier == "cpu"
    finally:
        wizard.close()


def test_zero_evidence_frames_leaves_run_disabled_even_when_acknowledged(tmp_path):
    from hydra_suite.detectkit.gui.dialogs.direct_calibration_wizard import (
        DirectCalibrationWizard,
    )

    # Empty dataset: no images/labels for either split, and no sources.
    empty_dir = tmp_path / "empty"
    (empty_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (empty_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    yaml = empty_dir / "data.yaml"
    yaml.write_text(
        f"path: {empty_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: ant\n"
    )
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
    try:
        assert wizard.evidence().frames == []
        wizard.chk_exhaustive.setChecked(True)
        assert wizard.btn_run.isEnabled() is False
        assert wizard.btn_run.toolTip() != ""
    finally:
        wizard.close()
