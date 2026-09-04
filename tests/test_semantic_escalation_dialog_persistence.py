"""SAM3 dialog layout, persistence, and calibration-window regressions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def available_checkpoint(monkeypatch):
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as mod

    class _Available:
        usable = True
        checkpoint_missing = False
        reason = ""

    monkeypatch.setattr(mod, "probe_checkpoint", lambda *_a, **_k: _Available())


def _source(name: str = "source") -> OBBSource:
    return OBBSource(name=name, level="polygon", path="/tmp/nonexistent")


def _point_dict() -> dict:
    return {
        "tile_fraction": 0.08,
        "tile_px": 1000,
        "tiles_per_frame": 9,
        "seconds_per_frame": 2.5,
        "confidence": 0.4,
        "missed_per_frame": 0.2,
        "extra_per_frame": 1.0,
        "recall": 0.95,
        "n_matched": 42,
        "area_min_px2": 130.0,
        "area_max_px2": 1500.0,
        "mean_quality": 0.66,
        "median_iou": 0.55,
        "median_area_ratio": 0.71,
    }


def test_semantic_dialog_restores_and_persists_project_settings(
    qapp, available_checkpoint
):
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    project = DetectKitProject(project_dir=Path("/tmp/project"))
    project.semantic_escalation_settings = {
        "prompt": "worker ant",
        "confidence": 0.42,
        "tile_fraction": 0.08,
        "reference_body_px": 76.0,
        "source_names": ["source"],
        "exhaustive": True,
    }
    saves: list[dict] = []
    dialog = SemanticEscalationDialog(
        [_source()],
        50.0,
        project=project,
        persist_callback=lambda: saves.append(
            dict(project.semantic_escalation_settings)
        ),
    )

    assert dialog.prompt() == "worker ant"
    assert dialog.parameters()["confidence"] == pytest.approx(0.42)
    assert dialog.parameters()["tile_fraction"] == pytest.approx(0.08)
    assert dialog.parameters()["reference_body_px"] == pytest.approx(76.0)
    assert [source.name for source in dialog.selected_sources()] == ["source"]

    dialog._confidence.setValue(0.55)
    dialog.accept()

    assert project.semantic_escalation_settings["confidence"] == pytest.approx(0.55)
    assert saves


def test_semantic_dialog_restores_saved_calibration(qapp, available_checkpoint):
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    project = DetectKitProject(project_dir=Path("/tmp/project"))
    project.semantic_calibration = {
        "created_at": "2026-08-30T12:00:00+00:00",
        "recommended_index": 0,
        "reason": "saved recommendation",
        "points": [_point_dict()],
    }
    dialog = SemanticEscalationDialog([_source()], 50.0, project=project)

    assert len(dialog.calibration_points) == 1
    assert dialog._btn_view_calibration.isEnabled()
    assert dialog._btn_calibrate.text().startswith("Recalibrate")
    assert "Saved calibration" in dialog._status.text()


def test_completed_calibration_is_written_to_project(qapp, available_checkpoint):
    from hydra_suite.core.inference.semantic.calibration import CalibrationPoint
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    project = DetectKitProject(project_dir=Path("/tmp/project"))
    saves: list[bool] = []
    dialog = SemanticEscalationDialog(
        [_source()],
        50.0,
        project=project,
        persist_callback=lambda: saves.append(True),
    )
    point = CalibrationPoint(**_point_dict())

    dialog._store_calibration([point], point, "measured recommendation")

    assert project.semantic_calibration["recommended_index"] == 0
    assert project.semantic_calibration["points"] == [_point_dict()]
    assert project.semantic_calibration["reason"] == "measured recommendation"
    assert saves


def test_completed_calibration_persists_visual_preview_artifact(
    qapp, available_checkpoint, tmp_path
):
    import cv2
    import numpy as np

    from hydra_suite.core.inference.semantic.calibration import (
        CalibrationGroundTruth,
        CalibrationPoint,
        CalibrationPreviewFrame,
    )
    from hydra_suite.core.inference.semantic.tiling import TileCandidate
    from hydra_suite.detectkit.gui.calibration_preview_store import (
        load_calibration_previews,
    )
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    image = tmp_path / "labelled.png"
    cv2.imwrite(str(image), np.zeros((40, 50, 3), dtype=np.uint8))
    polygon = np.asarray([[2, 2], [12, 2], [12, 12], [2, 12]], dtype=np.float32)
    preview = CalibrationPreviewFrame(
        image_path=image,
        ground_truth=(CalibrationGroundTruth(0, polygon),),
        candidates_by_fraction={0.08: (TileCandidate(polygon.copy(), 0.9, 0),)},
    )
    project = DetectKitProject(project_dir=tmp_path)
    dialog = SemanticEscalationDialog([_source()], 50.0, project=project)
    point = CalibrationPoint(**_point_dict())

    dialog._store_calibration([point], point, "", preview_frames=[preview])

    artifact = project.semantic_calibration["preview_artifact"]
    assert (tmp_path / artifact / "manifest.json").is_file()
    assert len(load_calibration_previews(tmp_path, artifact)) == 1


def test_semantic_dialog_uses_compact_multicolumn_settings(qapp, available_checkpoint):
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    dialog = SemanticEscalationDialog([_source()], 82.2)

    assert dialog._settings_grid.columnCount() == 4
    assert dialog.minimumWidth() >= 720
    assert dialog.width() >= 800
    assert dialog._tile_label.text().splitlines() == ["1644 px", "82 px / 0.05"]


def test_recalibration_warns_before_replacing_saved_frontier(
    qapp, available_checkpoint, monkeypatch
):
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as mod

    project = DetectKitProject(project_dir=Path("/tmp/project"))
    project.semantic_calibration = {
        "created_at": "2026-08-30T12:00:00+00:00",
        "recommended_index": 0,
        "points": [_point_dict()],
    }
    dialog = mod.SemanticEscalationDialog([_source()], 50.0, project=project)
    dialog._exhaustive.setChecked(True)
    prompts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        mod.QMessageBox,
        "question",
        staticmethod(
            lambda _parent, title, text, *_a, **_k: (
                prompts.append((title, text)),
                mod.QMessageBox.StandardButton.No,
            )[1]
        ),
    )
    monkeypatch.setattr(
        dialog,
        "confirm_checkpoint",
        lambda: (_ for _ in ()).throw(
            AssertionError("declining overwrite must stop before calibration")
        ),
    )

    dialog._run_calibration()

    assert prompts
    assert "replace" in (prompts[0][0] + prompts[0][1]).lower()


def test_calibration_progress_is_window_modal_and_raised(
    qapp, available_checkpoint, monkeypatch
):
    from PySide6 import QtWidgets

    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as mod
    from hydra_suite.detectkit.jobs import semantic_escalation as jobs

    events: list[object] = []

    class _Signal:
        def connect(self, callback):
            self.callback = callback

    class _Progress:
        def __init__(self, *_a, **_k):
            self.canceled = _Signal()

        def setWindowTitle(self, value):
            events.append(("title", value))

        def setMinimumDuration(self, _value):
            pass

        def setWindowModality(self, value):
            events.append(("modality", value))

        def setModal(self, value):
            events.append(("modal", value))

        def setAttribute(self, *_a):
            pass

        def setMinimumWidth(self, _value):
            pass

        def setValue(self, _value):
            pass

        def setLabelText(self, _value):
            pass

        def close(self):
            events.append("close")

        def show(self):
            events.append("show")

        def raise_(self):
            events.append("raise")

        def activateWindow(self):
            events.append("activate")

    class _Worker:
        cancelled = False

        def __init__(self, *_a, **_k):
            self.progress = _Signal()
            self.status = _Signal()
            self.result_ready = _Signal()
            self.finished = _Signal()

        def cancel(self):
            pass

        def start(self):
            events.append("start")

    monkeypatch.setattr(QtWidgets, "QProgressDialog", _Progress)
    monkeypatch.setattr(jobs, "CalibrationWorker", _Worker)
    dialog = mod.SemanticEscalationDialog([_source()], 50.0)
    dialog._exhaustive.setChecked(True)
    monkeypatch.setattr(dialog, "confirm_checkpoint", lambda: True)

    dialog._run_calibration()

    assert ("modality", Qt.WindowModality.WindowModal) in events
    assert ("modal", True) in events
    assert events.index("show") < events.index("raise") < events.index("start")
    assert events.index("activate") < events.index("start")


def test_choosing_a_frontier_point_carries_its_area_band_into_the_request(
    qapp, available_checkpoint
):
    """The gate calibration scored under must be the gate the run emits under.

    ``parameters()`` is splatted straight into SemanticEscalationRequest, so
    the band has to appear there or the 30-hour run silently reverts to the
    ungated behaviour that produced the mistargeted masks.
    """
    from hydra_suite.core.inference.semantic.calibration import CalibrationPoint
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    dialog = SemanticEscalationDialog([_source()], 50.0)
    assert dialog.parameters()["area_min_px2"] == 0.0  # ungated until calibrated

    point = CalibrationPoint(**_point_dict())
    dialog.apply_calibration_choice(point)
    params = dialog.parameters()
    assert params["area_min_px2"] == pytest.approx(130.0)
    assert params["area_max_px2"] == pytest.approx(1500.0)


def test_the_area_band_survives_a_settings_round_trip(qapp, available_checkpoint):
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    project = DetectKitProject(project_dir=Path("/tmp/project"))
    project.semantic_escalation_settings = {
        "area_min_px2": 130.0,
        "area_max_px2": 1500.0,
        "prompt": "ant",
        "source_names": ["source"],
    }
    dialog = SemanticEscalationDialog([_source()], 50.0, project=project)
    assert dialog.parameters()["area_min_px2"] == pytest.approx(130.0)
    dialog.accept()
    assert project.semantic_escalation_settings["area_max_px2"] == pytest.approx(1500.0)


def test_a_fresh_calibration_replaces_the_band_even_if_no_point_is_chosen(
    qapp, available_checkpoint
):
    """A stale band is worse than none: it gates NEW data by OLD label sizes.

    Recalibrating on a changed label set and then closing the results dialog
    without picking a point used to leave the previous run's band in place,
    so the escalation ran against a size gate fitted to different animals.
    The band is a property of the LABELS, not of the chosen operating point.
    """
    from hydra_suite.core.inference.semantic.calibration import CalibrationPoint
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    project = DetectKitProject(project_dir=Path("/tmp/project"))
    project.semantic_escalation_settings = {
        "area_min_px2": 130.0,
        "area_max_px2": 1500.0,
    }
    dialog = SemanticEscalationDialog([_source()], 50.0, project=project)
    assert dialog.parameters()["area_min_px2"] == pytest.approx(130.0)

    fresh = CalibrationPoint(
        **{**_point_dict(), "area_min_px2": 900.0, "area_max_px2": 9000.0}
    )
    dialog._store_calibration([fresh], None, "")
    assert dialog.parameters()["area_min_px2"] == pytest.approx(900.0)
    assert dialog.parameters()["area_max_px2"] == pytest.approx(9000.0)


def test_dialog_offers_the_project_classes_separately_from_the_prompt(
    qapp, available_checkpoint
):
    """The prompt is what the MODEL is asked to find; the class is what the
    findings ARE. Conflating them staged the prompt as a class the project
    had never heard of, which the overlay and the dataset builder both drop."""
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    project = DetectKitProject(project_dir=Path("/tmp/project"))
    project.class_names = ["ant", "beetle"]
    dialog = SemanticEscalationDialog([_source()], 50.0, project=project)

    assert dialog._class_name.isEnabled()
    assert [
        dialog._class_name.itemText(i) for i in range(dialog._class_name.count())
    ] == ["ant", "beetle"]
    # Defaults to the first project class, NOT to the prompt.
    assert dialog.class_name() == "ant"
    assert dialog.parameters()["class_name"] == "ant"


def test_dialog_restores_a_saved_class_assignment(qapp, available_checkpoint):
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    project = DetectKitProject(project_dir=Path("/tmp/project"))
    project.class_names = ["ant", "beetle"]
    project.semantic_escalation_settings = {"class_name": "beetle"}
    dialog = SemanticEscalationDialog([_source()], 50.0, project=project)

    assert dialog.class_name() == "beetle"


def test_a_project_with_no_classes_disables_the_selector(qapp, available_checkpoint):
    """Rather than silently inventing one: class_name() is then "" and the
    job falls back to the prompt, the pre-split behaviour."""
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    project = DetectKitProject(project_dir=Path("/tmp/project"))
    project.class_names = []
    dialog = SemanticEscalationDialog([_source()], 50.0, project=project)

    assert not dialog._class_name.isEnabled()
    assert dialog.class_name() == ""
