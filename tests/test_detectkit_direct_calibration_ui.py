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


def test_typing_a_name_and_clicking_save_stages_a_profile(results_dialog, tmp_path):
    from hydra_suite.core.inference.slice_meta import sidecar_path

    dialog = results_dialog
    dialog.table_rows.setCurrentCell(1, 0)
    dialog.edit_profile_name.setText("Balanced")
    assert dialog.btn_save_profile.isEnabled()
    dialog.btn_save_profile.click()
    names = [p["name"] for p in dialog.staged_profiles()]
    assert names == ["Balanced"]
    assert not sidecar_path(tmp_path / "m.pt").exists()


def test_save_button_disabled_for_empty_name_and_failed_row(results_dialog):
    dialog = results_dialog
    dialog.table_rows.setCurrentCell(1, 0)
    dialog.edit_profile_name.setText("")
    assert not dialog.btn_save_profile.isEnabled()
    dialog.edit_profile_name.setText("Something")
    assert dialog.btn_save_profile.isEnabled()
    failed_row = next(i for i, p in enumerate(dialog.outcome.points) if p.failed_reason)
    dialog.table_rows.setCurrentCell(failed_row, 0)
    assert not dialog.btn_save_profile.isEnabled()


def test_duplicate_name_save_click_warns_and_does_not_add_a_second_profile(
    results_dialog, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox

    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args) or QMessageBox.Ok,
    )

    dialog = results_dialog
    dialog.table_rows.setCurrentCell(1, 0)
    dialog.edit_profile_name.setText("Balanced")
    dialog.btn_save_profile.click()
    dialog.table_rows.setCurrentCell(2, 0)
    dialog.edit_profile_name.setText("balanced")
    dialog.btn_save_profile.click()

    assert len(warnings) == 1
    assert [p["name"] for p in dialog.staged_profiles()] == ["Balanced"]


def _two_staged_profiles(dialog):
    dialog.table_rows.setCurrentCell(1, 0)
    dialog.edit_profile_name.setText("Balanced")
    dialog.chk_make_primary.setChecked(True)
    dialog.btn_save_profile.click()
    dialog.table_rows.setCurrentCell(2, 0)
    dialog.edit_profile_name.setText("High recall")
    dialog.btn_save_profile.click()
    staged = dialog.staged_profiles()
    return (
        next(p["id"] for p in staged if p["name"] == "Balanced"),
        next(p["id"] for p in staged if p["name"] == "High recall"),
    )


def test_selecting_a_combo_entry_never_redesignates_the_primary(results_dialog):
    """Important 5: the combo is a SELECTOR. Merely picking a profile in
    order to remove it must not silently make it primary."""
    dialog = results_dialog
    balanced_id, high_recall_id = _two_staged_profiles(dialog)
    assert dialog.staged_meta()["primary_profile_id"] == balanced_id

    index = dialog.combo_primary.findData(high_recall_id)
    assert index >= 0
    dialog.combo_primary.setCurrentIndex(index)

    assert dialog.staged_meta()["primary_profile_id"] == balanced_id


def test_primary_is_designated_only_by_the_explicit_button(results_dialog):
    dialog = results_dialog
    _balanced_id, high_recall_id = _two_staged_profiles(dialog)
    dialog.combo_primary.setCurrentIndex(dialog.combo_primary.findData(high_recall_id))
    dialog.btn_set_primary.click()
    assert dialog.staged_meta()["primary_profile_id"] == high_recall_id


def test_removing_a_non_primary_profile_never_prompts(results_dialog, monkeypatch):
    """Follows from Important 5: selecting B to remove it used to make B
    primary, so Remove always hit the replacement prompt."""
    from PySide6.QtWidgets import QInputDialog

    dialog = results_dialog
    balanced_id, high_recall_id = _two_staged_profiles(dialog)
    prompts = []
    monkeypatch.setattr(
        QInputDialog,
        "getItem",
        staticmethod(lambda *a, **k: prompts.append(a) or ("", False)),
    )
    dialog.combo_primary.setCurrentIndex(dialog.combo_primary.findData(high_recall_id))
    dialog.btn_remove_profile.click()

    assert prompts == []
    assert [p["id"] for p in dialog.staged_profiles()] == [balanced_id]
    assert dialog.staged_meta()["primary_profile_id"] == balanced_id


def test_removing_the_staged_primary_prompts_and_honours_the_choice(
    results_dialog, monkeypatch
):
    dialog = results_dialog
    dialog.table_rows.setCurrentCell(1, 0)
    dialog.edit_profile_name.setText("Balanced")
    dialog.chk_make_primary.setChecked(True)
    dialog.btn_save_profile.click()
    dialog.table_rows.setCurrentCell(2, 0)
    dialog.edit_profile_name.setText("High recall")
    dialog.btn_save_profile.click()

    staged = dialog.staged_profiles()
    balanced_id = next(p["id"] for p in staged if p["name"] == "Balanced")
    high_recall_id = next(p["id"] for p in staged if p["name"] == "High recall")

    prompted = {}

    def fake_prompt(candidates):
        prompted["candidates"] = [c["id"] for c in candidates]
        return high_recall_id

    monkeypatch.setattr(dialog, "_prompt_replacement_primary", fake_prompt)

    index = dialog.combo_primary.findData(balanced_id)
    dialog.combo_primary.setCurrentIndex(index)
    dialog.btn_remove_profile.click()

    assert prompted["candidates"] == [high_recall_id]
    assert [p["name"] for p in dialog.staged_profiles()] == ["High recall"]
    assert dialog.staged_meta()["primary_profile_id"] == high_recall_id


def test_accept_with_nothing_staged_does_not_create_the_sidecar(
    results_dialog, tmp_path
):
    from hydra_suite.core.inference.slice_meta import sidecar_path

    results_dialog.accept()
    assert not sidecar_path(tmp_path / "m.pt").exists()


def _make_training_dialog(tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    proj.sources = [OBBSource(path=str(tmp_path / "ds1"), name="ds1")]
    return TrainingDialog(proj)


@pytest.fixture
def training_dialog(tmp_path):
    """A dialog whose session already has one completed, published
    obb_direct run -- exercises register/calibrate against the SAME
    already-published artifact publish_trained_model would have created,
    without actually running training or model_publish in a test."""
    dlg = _make_training_dialog(tmp_path)
    published = tmp_path / "published" / "m.pt"
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_bytes(b"weights")
    dlg._last_training_results = [
        {
            "role": "obb_direct",
            "success": True,
            "published_model_path": str(published),
        }
    ]
    dlg._refresh_register_controls()
    yield dlg
    dlg.close()


@pytest.fixture
def training_dialog_no_run(tmp_path):
    """A fresh dialog with no completed run -- the "nothing to act on yet"
    case both buttons must disable with a stated reason for."""
    dlg = _make_training_dialog(tmp_path)
    yield dlg
    dlg.close()


def test_register_with_training_geometry_skips_calibration(
    monkeypatch, training_dialog
):
    calls = []
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.training_dialog.open_direct_calibration",
        lambda *a, **k: (calls.append(k), [])[1],
    )
    training_dialog.register_with_training_geometry()
    assert calls == []
    assert len(training_dialog.registered_model_paths) == 1


def test_calibrate_then_register_produces_one_artifact(monkeypatch, training_dialog):
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.training_dialog.open_direct_calibration",
        lambda *a, **k: [{"id": "balanced-1", "name": "Balanced"}],
    )
    training_dialog.calibrate_then_register()
    assert len(training_dialog.registered_model_paths) == 1


def test_calibration_is_disabled_with_a_reason_when_labels_are_missing(training_dialog):
    training_dialog.set_calibration_enabled(False, "no labelled val split")
    assert training_dialog.btn_calibrate.isEnabled() is False
    assert "no labelled val split" in training_dialog.btn_calibrate.toolTip()


def test_register_with_training_geometry_does_not_touch_the_sidecar(
    training_dialog,
):
    from hydra_suite.core.inference.slice_meta import sidecar_path

    context = training_dialog._published_run_context()
    sidecar = sidecar_path(context[0])
    before_exists = sidecar.exists()
    before_bytes = sidecar.read_bytes() if before_exists else None

    training_dialog.register_with_training_geometry()

    after_exists = sidecar.exists()
    assert after_exists == before_exists
    if before_exists:
        assert sidecar.read_bytes() == before_bytes


def test_calibrate_then_register_never_publishes_again(monkeypatch, training_dialog):
    def _boom(*args, **kwargs):
        raise AssertionError("publish_trained_model must not be called here")

    monkeypatch.setattr(
        "hydra_suite.training.model_publish.publish_trained_model", _boom
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.training_dialog.open_direct_calibration",
        lambda *a, **k: [],
    )
    training_dialog.calibrate_then_register()
    assert len(training_dialog.registered_model_paths) == 1


def test_no_completed_run_disables_both_buttons_with_a_reason(
    training_dialog_no_run,
):
    assert training_dialog_no_run.btn_register.isEnabled() is False
    assert training_dialog_no_run.btn_calibrate.isEnabled() is False
    assert training_dialog_no_run.btn_register.toolTip()
    assert training_dialog_no_run.btn_calibrate.toolTip()


# ------------------------------------------------------------------
# History dialog and Tools-menu entry points: cancel must be side-effect
# free (open_direct_calibration owns the ONLY sidecar write, inside
# DirectCalibrationResultsDialog.accept()).
# ------------------------------------------------------------------


def _history_dialog_with_calibratable_run(tmp_path):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd
    from hydra_suite.detectkit.gui.models import DetectKitProject

    model_path = tmp_path / "published" / "m.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"weights")

    fake_run = {
        "run_id": "run_001",
        "role": "obb_direct",
        "status": "completed",
        "started_at": "2026-04-01T10:00:00",
        "spec": {"base_model": "yolo26s-obb.pt", "hyperparams": {"epochs": 50}},
        "artifact_paths": [str(model_path)],
        "published_model_path": str(model_path),
    }

    orig_load_runs = hd._load_runs
    hd._load_runs = lambda proj: [fake_run]
    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    dlg = hd.HistoryDialog(proj)
    dlg.table.selectRow(0)
    hd._load_runs = orig_load_runs
    return dlg, model_path


def test_history_calibrate_cancel_does_not_touch_the_sidecar(monkeypatch, tmp_path):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd
    from hydra_suite.core.inference.slice_meta import sidecar_path

    dlg, model_path = _history_dialog_with_calibratable_run(tmp_path)
    sidecar = sidecar_path(model_path)
    assert not sidecar.exists()

    monkeypatch.setattr(hd, "open_direct_calibration", lambda *a, **k: [])
    dlg._calibrate_selected()

    assert not sidecar.exists()
    dlg.close()


def test_history_calibrate_with_saved_profiles_still_writes_nothing_itself(
    monkeypatch, tmp_path
):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd
    from hydra_suite.core.inference.slice_meta import sidecar_path

    dlg, model_path = _history_dialog_with_calibratable_run(tmp_path)
    sidecar = sidecar_path(model_path)
    assert not sidecar.exists()

    monkeypatch.setattr(
        hd,
        "open_direct_calibration",
        lambda *a, **k: [{"id": "balanced-1", "name": "Balanced"}],
    )
    dlg._calibrate_selected()

    # The action itself never writes -- only the results dialog's accept()
    # (already-mocked out here) does. No sidecar should appear from this
    # call alone.
    assert not sidecar.exists()
    dlg.close()


def _main_window_with_project(tmp_path):
    from hydra_suite.detectkit.gui import main_window as mw
    from hydra_suite.detectkit.gui.models import DetectKitProject

    win = mw.DetectKitMainWindow()
    win._project = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    return win, mw


def test_tools_menu_calibrate_cancel_does_not_touch_the_sidecar(monkeypatch, tmp_path):
    from hydra_suite.core.inference.slice_meta import sidecar_path

    model_path = tmp_path / "models" / "m.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"weights")
    sidecar = sidecar_path(model_path)
    assert not sidecar.exists()

    # The Tools menu refuses to sweep a model whose task it cannot determine
    # (it must never default to "obb"), so give this one a resolvable task.
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.main_window._resolve_calibration_task",
        lambda *a, **k: "obb",
    )
    win, mw = _main_window_with_project(tmp_path)
    monkeypatch.setattr(
        mw.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(model_path), "")),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.direct_calibration_wizard.open_direct_calibration",
        lambda *a, **k: [],
    )

    win.calibrate_registered_model()

    assert not sidecar.exists()
    win.deleteLater()


def test_tools_menu_calibrate_with_saved_profiles_still_writes_nothing_itself(
    monkeypatch, tmp_path
):
    from hydra_suite.core.inference.slice_meta import sidecar_path

    model_path = tmp_path / "models" / "m.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"weights")
    sidecar = sidecar_path(model_path)
    assert not sidecar.exists()

    # The Tools menu refuses to sweep a model whose task it cannot determine
    # (it must never default to "obb"), so give this one a resolvable task.
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.main_window._resolve_calibration_task",
        lambda *a, **k: "obb",
    )
    win, mw = _main_window_with_project(tmp_path)
    monkeypatch.setattr(
        mw.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(model_path), "")),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.direct_calibration_wizard.open_direct_calibration",
        lambda *a, **k: [{"id": "balanced-1", "name": "Balanced"}],
    )
    # A real QMessageBox.information() is a blocking modal even under
    # offscreen Qt -- it never returns without a click, which is exactly
    # what the reported "assertion" reason for a hang would be. Confirm the
    # summary is shown, without letting the test block forever.
    info_calls = []
    monkeypatch.setattr(
        mw.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: info_calls.append((a, k))),
    )

    win.calibrate_registered_model()

    assert info_calls

    assert not sidecar.exists()
    win.deleteLater()


# ---------------------------------------------------------------------------
# Critical 1 -- the overlay must depict the SELECTED row, not the geometry's
# most permissive row.
# ---------------------------------------------------------------------------


def _overlay_dialog(tmp_path, points, previews):
    from hydra_suite.detectkit.gui.dialogs.direct_calibration_results import (
        DirectCalibrationResultsDialog,
    )
    from hydra_suite.detectkit.jobs.direct_calibration import DirectCalibrationOutcome

    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    image = tmp_path / "frame.png"
    cv2.imwrite(str(image), np.zeros((200, 300, 3), np.uint8))
    return DirectCalibrationResultsDialog(
        None,
        model_path=model,
        outcome=DirectCalibrationOutcome(points=list(points)),
        training_geometry={"geometry_mode": "auto_object", "imgsz": 640},
        previews=list(previews),
    )


def _overlay_point(*, confidence, merge_threshold, candidate_index, label, cap=64):
    from hydra_suite.core.inference.direct_calibration import (
        CalibrationScore,
        DirectCalibrationPoint,
    )

    return DirectCalibrationPoint(
        label=label,
        enabled=True,
        geometry_mode="auto_object",
        tile_width=640,
        tile_height=640,
        overlap=0.2,
        object_tile_fraction=0.4,
        max_detections=cap,
        tiles_per_frame=9,
        seconds_per_frame=0.4,
        confidence=confidence,
        merge_policy="greedy_nmm",
        merge_metric="ios",
        merge_threshold=merge_threshold,
        merge_backend="cv2",
        candidate_index=candidate_index,
        score=CalibrationScore(
            frames=1,
            matched=1,
            missed=0,
            extra=0,
            duplicate=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            mean_iou=0.8,
        ),
    )


def _square(x):
    return np.array(
        [[x, 0.0], [x + 10.0, 0.0], [x + 10.0, 10.0], [x, 10.0]], dtype=np.float32
    )


def _overlay_preview(tmp_path, *, candidate_index, merge_threshold, confidences):
    from hydra_suite.detectkit.jobs.direct_calibration import CalibrationPreview

    polygons = [_square(10.0 * i) for i in range(len(confidences))]
    return CalibrationPreview(
        candidate_label="Training geometry",
        frames=[(tmp_path / "frame.png", [], polygons)],
        candidate_index=candidate_index,
        merge_threshold=merge_threshold,
        pred_confidences=[list(confidences)],
        pred_sizes=[[100.0] * len(confidences)],
    )


def test_overlay_shows_only_detections_the_selected_row_emits(tmp_path):
    """Selecting a strict-confidence row must not render the 0.05 overlay."""
    low = _overlay_point(
        confidence=0.05, merge_threshold=0.5, candidate_index=0, label="Training"
    )
    high = _overlay_point(
        confidence=0.65, merge_threshold=0.5, candidate_index=0, label="Training"
    )
    preview = _overlay_preview(
        tmp_path,
        candidate_index=0,
        merge_threshold=0.5,
        confidences=[0.1, 0.7, 0.9],
    )
    dialog = _overlay_dialog(tmp_path, [low, high], [preview])
    try:
        assert len(dialog._row_predictions(preview, low, 0)) == 3
        assert len(dialog._row_predictions(preview, high, 0)) == 2
        dialog.table_rows.selectRow(1)
        assert "0.65" in dialog.lbl_overlay_caption.text()
    finally:
        dialog.close()


def test_overlay_reapplies_the_rows_detection_cap_largest_first(tmp_path):
    from hydra_suite.detectkit.jobs.direct_calibration import CalibrationPreview

    point = _overlay_point(
        confidence=0.1, merge_threshold=0.5, candidate_index=0, label="T", cap=2
    )
    preview = CalibrationPreview(
        candidate_label="T",
        frames=[(tmp_path / "frame.png", [], [_square(0), _square(20), _square(40)])],
        candidate_index=0,
        merge_threshold=0.5,
        pred_confidences=[[0.9, 0.9, 0.9]],
        pred_sizes=[[1.0, 300.0, 200.0]],
    )
    dialog = _overlay_dialog(tmp_path, [point], [preview])
    try:
        kept = dialog._row_predictions(preview, point, 0)
        assert len(kept) == 2
        assert [float(p[0][0]) for p in kept] == [20.0, 40.0]
    finally:
        dialog.close()


def test_rows_are_matched_to_previews_by_identity_not_label(tmp_path):
    """Two geometries sharing a label must not share an overlay."""
    a = _overlay_point(
        confidence=0.3, merge_threshold=0.5, candidate_index=0, label="same"
    )
    b = _overlay_point(
        confidence=0.3, merge_threshold=0.5, candidate_index=1, label="same"
    )
    preview_a = _overlay_preview(
        tmp_path, candidate_index=0, merge_threshold=0.5, confidences=[0.9]
    )
    preview_b = _overlay_preview(
        tmp_path, candidate_index=1, merge_threshold=0.5, confidences=[0.9, 0.9]
    )
    dialog = _overlay_dialog(tmp_path, [a, b], [preview_a, preview_b])
    try:
        assert dialog._preview_for_point(a) is preview_a
        assert dialog._preview_for_point(b) is preview_b
    finally:
        dialog.close()


def test_each_merge_setting_has_its_own_overlay(tmp_path):
    strict = _overlay_point(
        confidence=0.3, merge_threshold=0.3, candidate_index=0, label="T"
    )
    loose = _overlay_point(
        confidence=0.3, merge_threshold=0.7, candidate_index=0, label="T"
    )
    p3 = _overlay_preview(
        tmp_path, candidate_index=0, merge_threshold=0.3, confidences=[0.9]
    )
    p7 = _overlay_preview(
        tmp_path, candidate_index=0, merge_threshold=0.7, confidences=[0.9, 0.9]
    )
    dialog = _overlay_dialog(tmp_path, [strict, loose], [p3, p7])
    try:
        assert dialog._preview_for_point(strict) is p3
        assert dialog._preview_for_point(loose) is p7
        dialog.table_rows.selectRow(1)
        assert "0.7" in dialog.lbl_overlay_caption.text()
    finally:
        dialog.close()


# ---------------------------------------------------------------------------
# Critical 2 / 3, Important 6 / 7 -- what the GUI entry points actually pass.
# ---------------------------------------------------------------------------


def _run_dataset(tmp_path: Path) -> Path:
    """A prepared run dataset with a labelled val split (full resolution)."""
    root = tmp_path / "dataset"
    _dataset(root, "val", ["rec1_000", "rec1_001"])
    _dataset(root, "train", ["rec2_000"])
    (root / "dataset.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: ant\n"
    )
    return root


def _stamp_sidecar(model_path: Path, geometry: dict) -> None:
    from hydra_suite.core.inference.slice_meta import write_slice_meta

    write_slice_meta(model_path, geometry)


def test_history_calibration_uses_the_runs_val_split(monkeypatch, tmp_path):
    """Critical 2: the val-split default must actually fire from the GUI."""
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd
    from hydra_suite.detectkit.jobs.direct_calibration import collect_evidence

    dataset = _run_dataset(tmp_path)
    dlg, model_path = _history_dialog_with_calibratable_run(tmp_path)
    dlg._runs[0]["spec"]["derived_dataset_dir"] = str(dataset)
    seen = {}
    monkeypatch.setattr(
        hd, "open_direct_calibration", lambda *a, **k: seen.update(k) or []
    )
    dlg._calibrate_selected()
    dlg.close()

    assert seen["dataset_yaml"] is not None
    evidence = collect_evidence(dataset_yaml=seen["dataset_yaml"], sources=[])
    assert evidence.split == "val"
    assert evidence.frames


def test_history_calibration_follows_a_sliced_dataset_back_to_full_frames(tmp_path):
    """A sliced run's derived dataset is TILES; calibrating on it is wrong."""
    import json

    from hydra_suite.detectkit.jobs.direct_calibration import (
        resolve_calibration_dataset_yaml,
    )

    source = _run_dataset(tmp_path)
    sliced = tmp_path / "sliced"
    sliced.mkdir()
    (sliced / "dataset.yaml").write_text("path: .\n")
    (sliced / "manifest.json").write_text(
        json.dumps({"type": "sliced_obb", "source": str(source)})
    )
    assert resolve_calibration_dataset_yaml(sliced) == source / "dataset.yaml"


def test_history_calibration_reads_geometry_from_a_legacy_flat_sidecar(
    monkeypatch, tmp_path
):
    """Critical 3: a legacy FLAT sidecar must still yield a real grid."""
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd
    from hydra_suite.core.inference.direct_calibration_grid import build_candidate_grid

    dlg, model_path = _history_dialog_with_calibratable_run(tmp_path)
    _stamp_sidecar(
        model_path,
        {
            "geometry_mode": "auto_object",
            "imgsz": 960,
            "object_tile_fraction": 0.5,
            "overlap": 0.35,
            "reference_body_px": 480.0,
        },
    )
    seen = {}
    monkeypatch.setattr(
        hd, "open_direct_calibration", lambda *a, **k: seen.update(k) or []
    )
    dlg._calibrate_selected()
    dlg.close()

    geometry = seen["training_geometry"]
    assert geometry["reference_body_px"] == 480.0
    assert geometry["object_tile_fraction"] == 0.5
    grid = build_candidate_grid(geometry)
    trained = next(c for c in grid if c.label == "Training geometry")
    assert trained.object_tile_fraction == 0.5
    assert trained.overlap == 0.35


def test_history_calibration_targets_the_published_artifact(tmp_path):
    """Important 6: profiles on a project export are invisible to the model."""
    from hydra_suite.detectkit.gui.dialogs.history_dialog import (
        _calibration_model_path,
        _entry_model_path,
    )

    entry = {
        "project_model_paths": ["/exports/copy.pt"],
        "published_model_path": "/models/published.pt",
    }
    assert _entry_model_path(entry) == "/exports/copy.pt"
    assert _calibration_model_path(entry) == ("/models/published.pt", True)
    assert _calibration_model_path({"project_model_paths": ["/exports/copy.pt"]}) == (
        "/exports/copy.pt",
        False,
    )


def test_tools_menu_refuses_rather_than_claiming_obb(monkeypatch, tmp_path):
    """Important 7: a segment/detect model must not be swept as OBB."""
    from hydra_suite.detectkit.gui.main_window import _resolve_calibration_task

    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    monkeypatch.setattr(
        "hydra_suite.training.model_publish.load_model_registry", lambda: {}
    )
    assert _resolve_calibration_task(model, {}) is None
    assert _resolve_calibration_task(model, {"task": "segment"}) == "segment"

    monkeypatch.setattr(
        "hydra_suite.training.model_publish.load_model_registry",
        lambda: {
            "entries": {
                "m.pt": {"usage_role": "segment_direct", "task_family": "segment"}
            }
        },
    )
    monkeypatch.setattr(
        "hydra_suite.training.model_publish._registry_key_for_model",
        lambda path: "m.pt",
    )
    assert _resolve_calibration_task(model, {}) == "segment"


def test_tools_menu_does_not_open_calibration_for_an_unknown_task(
    monkeypatch, tmp_path
):
    from hydra_suite.detectkit.gui import main_window as mw

    model_path = tmp_path / "unknown.pt"
    model_path.write_bytes(b"weights")
    win, mw_mod = _main_window_with_project(tmp_path)
    monkeypatch.setattr(
        mw_mod.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(model_path), "")),
    )
    monkeypatch.setattr(mw, "_resolve_calibration_task", lambda *a, **k: None)
    warned = []
    monkeypatch.setattr(
        mw_mod.QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a))
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.direct_calibration_wizard."
        "open_direct_calibration",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not sweep an unknown task")
        ),
    )
    win.calibrate_registered_model()
    assert warned
    win.deleteLater()


def test_training_dialog_threads_geometry_and_dataset_yaml(monkeypatch, tmp_path):
    """Critical 2 + 3 on the Review & Register path."""
    from hydra_suite.detectkit.gui.dialogs import training_dialog as td

    dataset = _run_dataset(tmp_path)
    dlg = _make_training_dialog(tmp_path)
    published = tmp_path / "published" / "m.pt"
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_bytes(b"weights")
    _stamp_sidecar(
        published,
        {
            "training_geometry": {
                "geometry_mode": "auto_object",
                "imgsz": 1024,
                "object_tile_fraction": 0.6,
                "overlap": 0.25,
                "reference_body_px": 512.0,
            }
        },
    )
    dlg._last_training_results = [
        {
            "role": "obb_direct",
            "success": True,
            "published_model_path": str(published),
            "spec": {"derived_dataset_dir": str(dataset)},
        }
    ]
    seen = {}
    monkeypatch.setattr(
        td, "open_direct_calibration", lambda *a, **k: seen.update(k) or []
    )
    dlg.calibrate_then_register()
    dlg.close()

    assert seen["dataset_yaml"] is not None
    assert seen["training_geometry"]["reference_body_px"] == 512.0
    assert seen["training_geometry"]["object_tile_fraction"] == 0.6


def test_detect_format_labels_are_read_from_a_val_split(tmp_path):
    """Critical 2 trap: _split_frames was dead code and had never seen a
    5-field detect label."""
    from hydra_suite.detectkit.jobs.direct_calibration import collect_evidence

    root = tmp_path / "detect_ds"
    images = root / "images" / "val"
    labels = root / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    cv2.imwrite(str(images / "a.png"), np.zeros((200, 300, 3), np.uint8))
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    yaml = root / "dataset.yaml"
    yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: ant\n"
    )
    evidence = collect_evidence(dataset_yaml=yaml, sources=[])
    assert evidence.split == "val"
    assert evidence.instances == 1


# ---------------------------------------------------------------------------
# Important 8 / 9 -- registry agreement and measurement provenance.
# ---------------------------------------------------------------------------


def test_measurement_records_real_provenance(tmp_path):
    dialog = _overlay_dialog(
        tmp_path,
        [
            _overlay_point(
                confidence=0.35, merge_threshold=0.5, candidate_index=0, label="T"
            )
        ],
        [],
    )
    dialog._runtime_tier = "gpu_fast"
    dialog._evidence_split = "val"
    dialog._label_set_fingerprint = "sha256:abc"
    try:
        measurement = dialog.measurement_for_row(0)
    finally:
        dialog.close()
    assert measurement["runtime"] == "gpu_fast"
    assert measurement["split"] == "val"
    assert measurement["label_set_fingerprint"] == "sha256:abc"
    assert measurement["merge_backend"] == "cv2"
    assert measurement["runtime"] != "measured"


def test_wizard_threads_measurement_provenance_into_the_results_dialog(
    monkeypatch, tmp_path
):
    """The dialog cannot stamp provenance it was never given."""
    import inspect

    from hydra_suite.detectkit.gui.dialogs import direct_calibration_wizard as wiz

    source = inspect.getsource(wiz.open_direct_calibration)
    assert "runtime_tier=request.runtime_tier" in source
    assert "evidence_split=request.evidence.split" in source
    assert "label_set_fingerprint=request.evidence.fingerprint" in source


def test_accept_refreshes_the_registry_profile_summary(monkeypatch, tmp_path):
    """Important 8: sidecar and registry summary can never be allowed to drift."""
    import hydra_suite.training.model_publish as mp
    from hydra_suite.core.inference.slice_meta import read_slice_meta

    model = tmp_path / "m.pt"
    registry = {"schema_version": 2, "entries": {"m.pt": {"slice_profiles": {}}}}
    monkeypatch.setattr(mp, "_registry_key_for_model", lambda path: "m.pt")
    monkeypatch.setattr(mp, "load_model_registry", lambda: registry)
    saved = []
    monkeypatch.setattr(mp, "save_model_registry", lambda reg: saved.append(reg))

    dialog = _overlay_dialog(
        tmp_path,
        [
            _overlay_point(
                confidence=0.35, merge_threshold=0.5, candidate_index=0, label="T"
            )
        ],
        [],
    )
    dialog.table_rows.selectRow(0)
    dialog.save_profile("Balanced", primary=True)
    dialog.accept()

    assert saved, "the registry must be rewritten alongside the sidecar"
    summary = registry["entries"]["m.pt"]["slice_profiles"]
    assert summary["names"] == ["Balanced"]
    assert summary["count"] == 1
    meta = read_slice_meta(model)
    assert summary["primary_profile_id"] == meta["primary_profile_id"]


def test_accept_leaves_an_unregistered_model_alone(monkeypatch, tmp_path):
    import hydra_suite.training.model_publish as mp

    monkeypatch.setattr(mp, "_registry_key_for_model", lambda path: "absent.pt")
    monkeypatch.setattr(
        mp, "load_model_registry", lambda: {"schema_version": 2, "entries": {}}
    )
    monkeypatch.setattr(
        mp,
        "save_model_registry",
        lambda reg: (_ for _ in ()).throw(
            AssertionError("must not write the registry for an unregistered model")
        ),
    )
    dialog = _overlay_dialog(
        tmp_path,
        [
            _overlay_point(
                confidence=0.35, merge_threshold=0.5, candidate_index=0, label="T"
            )
        ],
        [],
    )
    dialog.table_rows.selectRow(0)
    dialog.save_profile("Balanced")
    dialog.accept()


def test_overlay_cap_matches_productions_effective_cap(tmp_path):
    """A requested cap above MAX_DOWNSTREAM_CROPS_PER_FRAME is clamped by
    production, so the overlay must clamp it too."""
    from hydra_suite.core.inference.stages.filtering import (
        MAX_DOWNSTREAM_CROPS_PER_FRAME,
    )
    from hydra_suite.detectkit.jobs.direct_calibration import CalibrationPreview

    count = MAX_DOWNSTREAM_CROPS_PER_FRAME + 5
    point = _overlay_point(
        confidence=0.1,
        merge_threshold=0.5,
        candidate_index=0,
        label="T",
        cap=100000,
    )
    preview = CalibrationPreview(
        candidate_label="T",
        frames=[
            (tmp_path / "frame.png", [], [_square(10.0 * i) for i in range(count)])
        ],
        candidate_index=0,
        merge_threshold=0.5,
        pred_confidences=[[0.9] * count],
        pred_sizes=[[float(i) for i in range(count)]],
    )
    dialog = _overlay_dialog(tmp_path, [point], [preview])
    try:
        assert (
            len(dialog._row_predictions(preview, point, 0))
            == MAX_DOWNSTREAM_CROPS_PER_FRAME
        )
    finally:
        dialog.close()


def test_training_dialog_falls_back_to_the_prepared_dataset_dir(monkeypatch, tmp_path):
    """Belt-and-braces: production entries are run records carrying `spec`,
    but an entry without one must still resolve the role's prepared dataset."""
    from hydra_suite.detectkit.gui.dialogs import training_dialog as td

    dataset = _run_dataset(tmp_path)
    dlg = _make_training_dialog(tmp_path)
    published = tmp_path / "published" / "m.pt"
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_bytes(b"weights")
    dlg.role_dataset_dirs = {"obb_direct": str(dataset)}
    dlg._last_training_results = [
        {
            "role": "obb_direct",
            "success": True,
            "published_model_path": str(published),
        }
    ]
    seen = {}
    monkeypatch.setattr(
        td, "open_direct_calibration", lambda *a, **k: seen.update(k) or []
    )
    dlg.calibrate_then_register()
    dlg.close()
    assert seen["dataset_yaml"] == dataset / "dataset.yaml"
