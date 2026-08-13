"""Tests for OBBCanvas dual-layer overlay (GT + predictions)."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_DET = [{"class_id": 0, "polygon_px": [(0, 0), (10, 0), (10, 10), (0, 10)]}]
_DET2 = [{"class_id": 1, "polygon_px": [(5, 5), (15, 5), (15, 15), (5, 15)]}]


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_canvas_gt_items_populated(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET)
    assert len(canvas._gt_obb_items) == 1
    assert len(canvas._gt_label_items) == 1


def test_canvas_pred_items_populated(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_pred_detections(_DET2)
    assert len(canvas._pred_obb_items) == 1
    assert len(canvas._pred_label_items) == 1


def test_canvas_gt_and_pred_independent(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET)
    canvas.set_pred_detections(_DET2)
    assert len(canvas._gt_obb_items) == 1
    assert len(canvas._pred_obb_items) == 1


def test_canvas_clear_gt_does_not_clear_pred(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET)
    canvas.set_pred_detections(_DET2)
    canvas.clear_gt_detections()
    assert len(canvas._gt_obb_items) == 0
    assert len(canvas._pred_obb_items) == 1


def test_canvas_clear_pred_does_not_clear_gt(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET)
    canvas.set_pred_detections(_DET2)
    canvas.clear_pred_detections()
    assert len(canvas._pred_obb_items) == 0
    assert len(canvas._gt_obb_items) == 1


def test_canvas_set_overlay_visibility_hides_gt(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET)
    canvas.set_overlay_visibility(show_gt=False, show_pred=True)
    for item in canvas._gt_obb_items:
        assert not item.isVisible()


def test_canvas_set_overlay_visibility_hides_pred(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_pred_detections(_DET)
    canvas.set_overlay_visibility(show_gt=True, show_pred=False)
    for item in canvas._pred_obb_items:
        assert not item.isVisible()


def test_canvas_set_overlay_visibility_shows_both(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET)
    canvas.set_pred_detections(_DET2)
    canvas.set_overlay_visibility(show_gt=True, show_pred=True)
    for item in canvas._gt_obb_items:
        assert item.isVisible()
    for item in canvas._pred_obb_items:
        assert item.isVisible()


def test_canvas_set_class_filter(qapp):
    """Only class IDs in visible_class_ids should be shown."""
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET)  # class_id=0
    canvas.set_gt_detections(_DET2, append=True)  # class_id=1
    canvas.set_class_filter({0})
    # class 0 should be visible, class 1 hidden
    gt_visible = [i for i in canvas._gt_obb_items if i.isVisible()]
    gt_hidden = [i for i in canvas._gt_obb_items if not i.isVisible()]
    assert len(gt_visible) == 1
    assert len(gt_hidden) == 1


def test_canvas_set_detections_backward_compat(qapp):
    """set_detections() must still work as a GT alias."""
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_detections(_DET)
    assert len(canvas._gt_obb_items) == 1


def test_canvas_clear_detections_backward_compat(qapp):
    """clear_detections() must clear GT layer."""
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_detections(_DET)
    canvas.clear_detections()
    assert len(canvas._gt_obb_items) == 0


def test_canvas_old_label_items_still_accessible(qapp):
    """_label_items must remain a view of GT label items for backward compat."""
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_detections(_DET, class_names=["ant"])
    assert len(canvas._label_items) == 1
    assert canvas._label_items[0].toPlainText() == "ant (0)"
