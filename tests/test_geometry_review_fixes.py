"""Regressions for the adversarial review of geometry-escalation SAHI."""

import os
import types

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.core.inference.sam2.calibration import (  # noqa: E402
    GeometryCalibrationPoint,
)
from hydra_suite.core.inference.sam2.checkpoints import (  # noqa: E402
    DEFAULT_VARIANT,
    available_variants,
)
from hydra_suite.core.inference.semantic.tiling import TilingSettings  # noqa: E402
from tests.test_calibration_frames import OBB, POLY, make_source  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _point(frac=0.3):
    return GeometryCalibrationPoint(frac, 263, 14.7, 0.74, 0.756, 0.636, 0.0, 0.0, 697)


class _FakeWorker(QObject):
    """Mirrors Sam2CalibrationWorker's cancel contract; finishes on start()."""

    progress = Signal(int)
    status = Signal(str)
    result_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, *_a, **_k):
        super().__init__()
        self._cancel = False
        self.sampled_frames = ["x.png"]

    def cancel(self):
        self._cancel = True

    @property
    def cancelled(self):
        return self._cancel

    def start(self):
        self.progress.emit(100)
        self.result_ready.emit([_point(0.3), _point(None)])
        self.finished.emit()


def _project(tmp_path):
    return types.SimpleNamespace(
        project_dir=str(tmp_path),
        geometry_calibration={},
        geometry_escalation_settings={},
    )


def test_completed_sam2_calibration_is_persisted(tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.escalate_sam2_dialog import (
        EscalateSam2Dialog,
    )
    from hydra_suite.detectkit.jobs import sam2_escalation

    monkeypatch.setattr(sam2_escalation, "Sam2CalibrationWorker", _FakeWorker)
    poly = make_source(tmp_path, "poly", {"a": POLY})
    project = _project(tmp_path)
    saves = []
    dlg = EscalateSam2Dialog(
        [poly],
        project=project,
        reference_body_px=79.0,
        persist_callback=lambda: saves.append(1),
    )
    dlg._run_calibration()
    record = project.geometry_calibration[DEFAULT_VARIANT]
    assert len(record["points"]) == 2 and saves


def test_completed_sam3_calibration_is_persisted(tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as mod
    from hydra_suite.detectkit.gui.models import DetectKitProject
    from hydra_suite.detectkit.jobs import semantic_escalation as jobs

    class _Sam3Worker(_FakeWorker):
        preview_frames = []

        def start(self):
            from hydra_suite.core.inference.semantic.calibration import CalibrationPoint

            self.progress.emit(100)
            self.result_ready.emit(
                [CalibrationPoint(None, None, 1, 1.0, 0.4, 0.1, 0.1, 0.95, 40)]
            )
            self.finished.emit()

    class _Available:
        usable, checkpoint_missing, reason = True, False, ""

    monkeypatch.setattr(mod, "probe_checkpoint", lambda *_a, **_k: _Available())
    monkeypatch.setattr(jobs, "CalibrationWorker", _Sam3Worker)
    monkeypatch.setattr(
        mod.SemanticEscalationDialog, "_show_calibration_results", lambda *a, **k: None
    )
    poly = make_source(tmp_path, "poly", {"a": POLY})
    project = DetectKitProject(project_dir=tmp_path)
    dlg = mod.SemanticEscalationDialog([poly], 20.0, project=project)
    dlg._exhaustive.setChecked(True)
    dlg._run_calibration()
    assert project.semantic_calibration.get("points")


def test_switching_variant_never_carries_another_variants_fraction(tmp_path):
    from dataclasses import asdict

    from hydra_suite.detectkit.gui.dialogs.escalate_sam2_dialog import (
        EscalateSam2Dialog,
    )

    other = next(v for v in available_variants() if v != DEFAULT_VARIANT)
    box = make_source(tmp_path, "box", {"a": OBB}, level="obb")
    project = _project(tmp_path)
    project.geometry_calibration = {
        DEFAULT_VARIANT: {"chosen_index": 0, "points": [asdict(_point(0.3))]}
    }
    dlg = EscalateSam2Dialog([box], project=project, reference_body_px=79.0)
    assert dlg.tile_fraction() == pytest.approx(0.3)
    dlg._variant.setCurrentText(other)
    assert dlg.tile_fraction() is None and dlg._results.rowCount() == 0


def test_an_accepted_full_frame_choice_survives_a_variant_round_trip(tmp_path):
    from dataclasses import asdict

    from hydra_suite.detectkit.gui.dialogs.escalate_sam2_dialog import (
        EscalateSam2Dialog,
    )

    other = next(v for v in available_variants() if v != DEFAULT_VARIANT)
    box = make_source(tmp_path, "box", {"a": OBB}, level="obb")
    project = _project(tmp_path)
    project.geometry_calibration = {
        DEFAULT_VARIANT: {"chosen_index": 0, "points": [asdict(_point(0.3))]}
    }
    project.geometry_escalation_settings = {
        "variant": DEFAULT_VARIANT,
        "tile_fraction": 0.0,
    }
    dlg = EscalateSam2Dialog([box], project=project, reference_body_px=79.0)
    assert dlg.tile_fraction() is None
    dlg._variant.setCurrentText(other)
    dlg._variant.setCurrentText(DEFAULT_VARIANT)
    assert dlg.tile_fraction() is None


def test_plan_above_the_tile_ceiling_falls_back_to_full_frame():
    plan = TilingSettings(reference_body_px=20.0, tile_fraction=0.4).plan_for(
        (4512, 4512)
    )
    assert plan.tiles == [(0, 0, 4512, 4512)]


def test_untileable_frames_are_counted_not_fatal(tmp_path):
    from hydra_suite.detectkit.gui.models import OBBSource
    from hydra_suite.detectkit.jobs.sam2_escalation import (
        EscalationRequest,
        run_escalation,
    )

    root = tmp_path / "src"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "a.png"), np.zeros((100, 100, 3), np.uint8))
    (root / "labels" / "a.txt").write_text(OBB)
    src = OBBSource(path=str(root), name="src", level="obb")
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])

    class Ex:
        def set_image(self, img):
            self.shape = img.shape[:2]

        def segment(self, box, pos, neg):
            return np.ones(self.shape, bool), 0.9

    result = run_escalation(EscalationRequest(project, ["src"], "v", tile_px=500), Ex())
    assert result.tiled_frames == 0 and result.untiled_frames == 1
    assert result.staged == ["src"]


def test_tiling_note_reports_what_actually_ran():
    from hydra_suite.detectkit.gui.escalation_actions import _tiling_note

    assert _tiling_note(None, 0, 0, 0) == ""
    assert "could not apply to any frame" in _tiling_note(263, 0, 5, 0)
    note = _tiling_note(263, 4, 1, 2)
    assert "on 4 frame(s)" in note and "1 frame(s) could not be tiled" in note
    assert "2 animal(s) crossed a tile seam" in note


def test_quick_median_reads_no_pixels_and_matches_the_decoding_one(
    tmp_path, monkeypatch
):
    from hydra_suite.detectkit.jobs import calibration_frames as cf

    src = make_source(tmp_path, "s", {"a": POLY, "b": OBB}, level="obb")
    decoded = cf.measure_median_body_px([src])[0]

    def _boom(*_a, **_k):
        raise AssertionError("decoded an image")

    monkeypatch.setattr(cf.cv2, "imread", _boom)
    assert cf.quick_median_body_px([src]) == pytest.approx(decoded)
