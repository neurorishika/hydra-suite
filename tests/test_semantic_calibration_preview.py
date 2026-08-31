"""Visual-evidence regressions for the SAM3 calibration frontier."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.core.inference.semantic.calibration import (  # noqa: E402
    CalibrationGroundTruth,
    CalibrationPoint,
    CalibrationPreviewFrame,
)
from hydra_suite.core.inference.semantic.tiling import TileCandidate  # noqa: E402
from hydra_suite.detectkit.gui.calibration_preview_store import (  # noqa: E402
    load_calibration_previews,
    save_calibration_previews,
)
from hydra_suite.detectkit.gui.dialogs.calibration_results_dialog import (  # noqa: E402
    CalibrationResultsDialog,
)


def _polys(canvas, key):
    return [i for b in canvas.layer_items(key).values() for i in b.obb_items]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _square(x: float, y: float, side: float = 12.0) -> np.ndarray:
    return np.asarray(
        [[x, y], [x + side, y], [x + side, y + side], [x, y + side]],
        dtype=np.float32,
    )


def _point(confidence: float) -> CalibrationPoint:
    return CalibrationPoint(
        tile_fraction=0.1,
        tile_px=120,
        tiles_per_frame=4,
        seconds_per_frame=1.0,
        confidence=confidence,
        missed_per_frame=0.0,
        extra_per_frame=0.0,
        recall=1.0,
        n_matched=30,
    )


def _preview(path: Path, offset: float = 0.0) -> CalibrationPreviewFrame:
    return CalibrationPreviewFrame(
        image_path=path,
        ground_truth=(CalibrationGroundTruth(4, _square(10 + offset, 10)),),
        candidates_by_fraction={
            0.1: (
                TileCandidate(_square(10 + offset, 10), 0.8, 0),
                TileCandidate(_square(45 + offset, 45), 0.3, 1),
            )
        },
    )


def test_preview_artifact_round_trips_candidates_and_ground_truth(tmp_path):
    image = tmp_path / "frame.png"
    cv2.imwrite(str(image), np.zeros((80, 100, 3), dtype=np.uint8))
    relative = save_calibration_previews(tmp_path, [_preview(image)])

    restored = load_calibration_previews(tmp_path, relative)

    assert len(restored) == 1
    assert restored[0].image_path == image
    assert restored[0].ground_truth[0].class_id == 4
    assert np.array_equal(restored[0].ground_truth[0].polygon_px, _square(10, 10))
    assert [item.confidence for item in restored[0].candidates_by_fraction[0.1]] == [
        0.8,
        0.3,
    ]


def test_results_dialog_rethresholds_selected_row_and_navigates_frames(qapp, tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    cv2.imwrite(str(first), np.zeros((100, 120, 3), dtype=np.uint8))
    cv2.imwrite(str(second), np.zeros((100, 120, 3), dtype=np.uint8))
    strict = _point(0.5)
    permissive = _point(0.2)
    dialog = CalibrationResultsDialog(
        [permissive, strict],
        strict,
        "",
        project_frames=10,
        preview_frames=[_preview(first), _preview(second, 5)],
    )

    assert dialog._table.currentRow() == 0
    assert len(_polys(dialog._canvas, "pred")) == 1
    assert "first.png" in dialog._frame_label.text()

    dialog._canvas._set_zoom(2.0)
    zoom_before_row_change = dialog._canvas._current_zoom()
    dialog._table.selectRow(1)
    qapp.processEvents()
    assert len(_polys(dialog._canvas, "pred")) == 2
    assert dialog._canvas._current_zoom() == pytest.approx(zoom_before_row_change)

    dialog._next_frame.click()
    assert "Frame 2 of 2" in dialog._frame_label.text()
    assert "second.png" in dialog._frame_label.text()
    assert len(_polys(dialog._canvas, "gt")) == 1


def test_legacy_calibration_explains_that_visual_evidence_is_unavailable(qapp):
    point = _point(0.5)
    dialog = CalibrationResultsDialog([point], point, "", project_frames=10)

    assert "Re-run calibration" in dialog._preview_status.text()
    assert not dialog._previous_frame.isEnabled()
    assert not dialog._next_frame.isEnabled()
