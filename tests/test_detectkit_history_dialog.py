"""Tests for DetectKit HistoryDialog."""

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


_FAKE_RUNS = [
    {
        "run_id": "run_001",
        "role": "obb_direct",
        "status": "completed",
        "started_at": "2026-04-01T10:00:00",
        "spec": {"base_model": "yolo26s-obb.pt", "hyperparams": {"epochs": 50}},
        "artifact_paths": ["/some/source_model.pt"],
        "project_model_path": "/some/model.pt",
        "published_model_path": "/some/model.pt",
    },
    {
        "run_id": "run_002",
        "role": "seq_detect",
        "status": "failed",
        "started_at": "2026-04-02T10:00:00",
        "spec": {"base_model": "yolo26s.pt", "hyperparams": {"epochs": 30}},
        "artifact_paths": [],
        "published_model_path": "",
    },
]


def test_history_dialog_is_base_dialog(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog
    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    assert isinstance(dlg, BaseDialog)


def test_history_dialog_has_close_accept_buttons(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    close_btn = dlg._buttons.button(QDialogButtonBox.StandardButton.Close)
    assert close_btn is not None


def test_history_dialog_populates_table(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    assert dlg.table.rowCount() == 2


def test_history_dialog_uses_classkit_style_table(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    assert dlg.table.alternatingRowColors() is True
    assert "alternate-background-color: #2d2d30" in dlg.table.styleSheet()


def test_history_dialog_empty_when_no_runs(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: [])
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    assert dlg.table.rowCount() == 0


def test_history_dialog_load_for_inference_sets_active_model(
    qapp, tmp_path, monkeypatch
):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    proj = _make_proj(tmp_path)
    dlg = HistoryDialog(proj)
    dlg.table.selectRow(0)
    dlg._load_for_inference()
    assert proj.active_model_path == "/some/model.pt"


def test_history_dialog_has_detail_label(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "detail_label")
    assert "run_001" in dlg.detail_label.text()


def test_history_dialog_has_export_button(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "_btn_export")


def test_view_stored_calibration_button_needs_saved_evidence(
    qapp, tmp_path, monkeypatch
):
    """The "View stored calibration" action must only be enabled once
    evidence has actually been saved for the project -- and must open the
    results dialog in STORED mode (never triggering a fresh sweep)."""
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"weights")
    runs = [
        {
            "run_id": "run_001",
            "role": "obb_direct",
            "status": "completed",
            "started_at": "2026-04-01T10:00:00",
            "spec": {"base_model": "yolo26s-obb.pt", "hyperparams": {"epochs": 50}},
            "artifact_paths": [str(model_path)],
            "project_model_path": "",
            "published_model_path": str(model_path),
        }
    ]
    monkeypatch.setattr(hd, "_load_runs", lambda proj: runs)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    proj = _make_proj(tmp_path)
    dlg = HistoryDialog(proj)
    dlg.table.selectRow(0)

    assert dlg._btn_view_calibration.isEnabled() is False

    # Save real evidence via the actual job save function -- not a hand
    # rolled dict -- so this exercises the real load path.
    from hydra_suite.core.inference.direct_calibration import (
        CalibrationScore,
        DirectCalibrationPoint,
    )
    from hydra_suite.detectkit.jobs.direct_calibration import (
        DirectCalibrationOutcome,
        DirectCalibrationRequest,
        EvidenceSet,
        save_direct_calibration,
    )

    point = DirectCalibrationPoint(
        label="Training geometry",
        enabled=True,
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
        score=CalibrationScore(
            frames=20,
            matched=200,
            missed=10,
            extra=10,
            duplicate=1,
            precision=0.95,
            recall=0.95,
            f1=0.95,
            mean_iou=0.81,
            mean_quality=0.8,
        ),
    )
    evidence_dir = dlg._evidence_dir()
    request = DirectCalibrationRequest(
        model_path=model_path,
        task="obb",
        evidence=EvidenceSet(
            frames=[],
            split="val",
            instances=0,
            size_range=((0, 0), (0, 0)),
            sampled_from=0,
            fingerprint="fp",
        ),
        candidates=[],
        confidences=(0.35,),
        merge_settings=(),
        runtime_tier="cpu",
        max_targets=64,
        evidence_dir=evidence_dir,
    )
    save_direct_calibration(
        evidence_dir, DirectCalibrationOutcome(points=[point]), request
    )

    dlg._on_selection_changed()
    assert dlg._btn_view_calibration.isEnabled() is True

    opened = {}

    class _FakeDialog:
        def __init__(self, *_a, **kwargs):
            opened.update(kwargs)

        def exec(self):
            return 0

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.direct_calibration_results."
        "DirectCalibrationResultsDialog",
        _FakeDialog,
    )
    dlg._view_stored_calibration()
    assert opened.get("stored") is True
    assert opened["outcome"].points[0].label == "Training geometry"
