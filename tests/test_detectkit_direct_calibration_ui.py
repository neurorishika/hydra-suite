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


@pytest.fixture
def results_dialog(tmp_path):
    from hydra_suite.core.inference.direct_calibration import (
        CalibrationScore,
        DirectCalibrationPoint,
    )
    from hydra_suite.detectkit.gui.dialogs.direct_calibration_results import (
        DirectCalibrationResultsDialog,
    )
    from hydra_suite.detectkit.jobs.direct_calibration import DirectCalibrationOutcome

    def point(label, f1, failed=""):
        return DirectCalibrationPoint(
            label=label,
            enabled=label != "Full frame (no SAHI)",
            geometry_mode="auto_object",
            tile_width=640,
            tile_height=640,
            overlap=0.2,
            object_tile_fraction=0.4,
            max_detections=64,
            tiles_per_frame=9,
            seconds_per_frame=0.4,
            confidence=0.35,
            merge_policy="greedy_nmm",
            merge_metric="ios",
            merge_threshold=0.5,
            merge_backend="cv2",
            failed_reason=failed,
            score=CalibrationScore(
                frames=20,
                matched=200,
                missed=10,
                extra=10,
                duplicate=1,
                precision=f1,
                recall=f1,
                f1=f1,
                mean_iou=0.8,
            ),
        )

    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    outcome = DirectCalibrationOutcome(
        points=[
            point("Full frame (no SAHI)", 0.70),
            point("Training geometry", 0.92),
            point("fraction x1.5, overlap 0.1", 0.90),
            point("Custom 1x1", 0.0, failed="tile budget exceeded"),
        ]
    )
    dialog = DirectCalibrationResultsDialog(
        None,
        model_path=model,
        outcome=outcome,
        training_geometry={"geometry_mode": "auto_object", "imgsz": 640},
        previews=[],
    )
    yield dialog
    dialog.close()


def test_every_measured_row_including_failures_is_listed(results_dialog):
    dialog = results_dialog
    assert dialog.table_rows.rowCount() == len(dialog.outcome.points)
    statuses = [
        dialog.table_rows.item(i, dialog.COL_STATUS).text()
        for i in range(dialog.table_rows.rowCount())
    ]
    assert any("tile budget" in text for text in statuses)
    assert any("Recommended" in text for text in statuses)


def test_changing_rows_never_runs_the_model(results_dialog, monkeypatch):
    import hydra_suite.core.inference.stages.obb as obb_stage

    def explode(*_a, **_k):
        raise AssertionError("selecting a row must not run the model")

    monkeypatch.setattr(obb_stage, "run_obb", explode)
    monkeypatch.setattr(obb_stage, "collect_obb_parts_by_frame", explode)
    results_dialog.table_rows.setCurrentCell(1, 0)
    results_dialog._render_preview()


def test_nothing_touches_the_sidecar_until_accept(results_dialog, tmp_path):
    from hydra_suite.core.inference.slice_meta import sidecar_path

    results_dialog.table_rows.setCurrentCell(1, 0)
    results_dialog.save_profile("Balanced", note="Routine tracking", primary=True)
    assert results_dialog.staged_profiles(), "profile should be staged in memory"
    assert not sidecar_path(tmp_path / "m.pt").exists(), "sidecar written too early"
    results_dialog.accept()
    assert sidecar_path(tmp_path / "m.pt").exists()


def test_rejecting_the_dialog_saves_nothing(results_dialog, tmp_path):
    from hydra_suite.core.inference.slice_meta import sidecar_path

    results_dialog.table_rows.setCurrentCell(1, 0)
    results_dialog.save_profile("Balanced")
    results_dialog.reject()
    assert not sidecar_path(tmp_path / "m.pt").exists()


def test_several_profiles_from_one_run_live_on_one_artifact(results_dialog):
    results_dialog.table_rows.setCurrentCell(1, 0)
    results_dialog.save_profile("Balanced", primary=True)
    results_dialog.table_rows.setCurrentCell(2, 0)
    results_dialog.save_profile("High recall")
    assert [p["name"] for p in results_dialog.staged_profiles()] == [
        "Balanced",
        "High recall",
    ]
    meta = results_dialog.staged_meta()
    assert meta["primary_profile_id"] == meta["profiles"][0]["id"]


def test_duplicate_profile_names_are_rejected(results_dialog):
    results_dialog.table_rows.setCurrentCell(1, 0)
    results_dialog.save_profile("Balanced")
    with pytest.raises(ValueError):
        results_dialog.save_profile("balanced")


def test_a_failed_row_cannot_become_a_profile(results_dialog):
    failed_row = next(
        i for i, p in enumerate(results_dialog.outcome.points) if p.failed_reason
    )
    results_dialog.table_rows.setCurrentCell(failed_row, 0)
    with pytest.raises(ValueError, match="failed"):
        results_dialog.save_profile("Broken")


def test_settings_payload_is_complete_and_omits_reference_body_size(results_dialog):
    settings = results_dialog.settings_for_row(1)
    assert "REFERENCE_BODY_SIZE" not in settings
    for key in (
        "enabled",
        "geometry_mode",
        "slice_width",
        "slice_height",
        "overlap",
        "object_tile_fraction",
        "trained_body_px",
        "confidence_threshold",
        "merge_policy",
        "merge_metric",
        "merge_threshold",
        "merge_backend",
        "max_detections",
    ):
        assert key in settings


def test_measurement_records_provenance(results_dialog):
    measurement = results_dialog.measurement_for_row(1)
    for key in (
        "created_at",
        "checkpoint_fingerprint",
        "task",
        "frames",
        "instances",
        "runtime",
        "seconds_per_frame",
        "precision",
        "recall",
        "f1",
        "localization_quality",
        "max_detections",
    ):
        assert key in measurement
    assert measurement["checkpoint_fingerprint"].startswith("sha256:")


def test_removing_the_primary_profile_prompts_for_a_replacement(results_dialog):
    dialog = results_dialog
    dialog.table_rows.setCurrentCell(1, 0)
    dialog.save_profile("Balanced", primary=True)
    dialog.table_rows.setCurrentCell(2, 0)
    dialog.save_profile("High recall")
    staged = dialog.staged_profiles()
    with pytest.raises(ValueError, match="replacement"):
        dialog.remove_profile(staged[0]["id"])
    dialog.remove_profile(staged[0]["id"], new_primary_id=staged[1]["id"])
    assert [p["name"] for p in dialog.staged_profiles()] == ["High recall"]
