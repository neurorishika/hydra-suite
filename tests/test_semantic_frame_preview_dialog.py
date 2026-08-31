"""Visual complete-frame check regressions for SAM3 escalation."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from hydra_suite.core.inference.semantic.base import SemanticInstance  # noqa: E402
from hydra_suite.data.al.escalation import LabelRecord  # noqa: E402
from hydra_suite.detectkit.gui.dialogs.semantic_frame_preview_dialog import (  # noqa: E402
    SemanticFramePreviewDialog,
)
from hydra_suite.detectkit.jobs.semantic_escalation import (  # noqa: E402
    FramePreviewResult,
)
from hydra_suite.utils.geometry_levels import GeometryLevel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _polygon(x: float, y: float) -> np.ndarray:
    return np.asarray(
        [[x, y], [x + 20, y], [x + 20, y + 20], [x, y + 20]],
        dtype=np.float32,
    )


def test_complete_frame_dialog_shows_overlays_and_measured_estimates(qapp, tmp_path):
    image_path = tmp_path / "random-frame.png"
    cv2.imwrite(str(image_path), np.zeros((120, 160, 3), dtype=np.uint8))
    result = FramePreviewResult(
        image_path=image_path,
        source_name="source-a",
        predictions=[SemanticInstance(_polygon(10, 10), 0.82)],
        ground_truth=[
            LabelRecord(
                class_id=0,
                confidence=1.0,
                points=_polygon(12, 12),
                level=GeometryLevel.POLYGON,
            )
        ],
        seconds=2.0,
        tile_px=80,
        tiles_per_frame=9,
    )

    dialog = SemanticFramePreviewDialog(result, selected_frames=60, project_frames=120)
    text = " ".join(label.text() for label in dialog.findChildren(QLabel))

    assert "random-frame.png" in text
    assert "source-a" in text
    assert "60 images" in text and "2 min" in text
    assert "120 images" in text and "4 min" in text
    assert len(dialog._canvas._pred_obb_items) == 1
    assert len(dialog._canvas._gt_obb_items) == 1
    assert dialog._canvas._pix_item is not None


def test_ground_truth_toggle_is_disabled_for_unlabelled_random_frame(qapp, tmp_path):
    image_path = tmp_path / "unlabelled.png"
    cv2.imwrite(str(image_path), np.zeros((60, 80, 3), dtype=np.uint8))
    result = FramePreviewResult(
        image_path=image_path,
        source_name="source-a",
        predictions=[],
        ground_truth=[],
        seconds=1.0,
        tile_px=None,
        tiles_per_frame=1,
    )

    dialog = SemanticFramePreviewDialog(result, selected_frames=10, project_frames=10)

    assert not dialog._show_gt.isEnabled()
    assert dialog._show_predictions.isEnabled()
