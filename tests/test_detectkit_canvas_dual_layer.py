"""Tests for OBBCanvas dual-layer overlay (GT + predictions)."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


def _polys(canvas, key):
    return [i for b in canvas.layer_items(key).values() for i in b.obb_items]


def _labels(canvas, key):
    return [
        i
        for b in canvas.layer_items(key).values()
        for i in b.label_items
        if i is not None
    ]


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
    assert len(_polys(canvas, "gt")) == 1
    assert len(_labels(canvas, "gt")) == 1


def test_canvas_pred_items_populated(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_pred_detections(_DET2)
    assert len(_polys(canvas, "pred")) == 1
    assert len(_labels(canvas, "pred")) == 1


def test_canvas_gt_and_pred_independent(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET)
    canvas.set_pred_detections(_DET2)
    assert len(_polys(canvas, "gt")) == 1
    assert len(_polys(canvas, "pred")) == 1


def test_canvas_clear_gt_does_not_clear_pred(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET)
    canvas.set_pred_detections(_DET2)
    canvas.clear_gt_detections()
    assert len(_polys(canvas, "gt")) == 0
    assert len(_polys(canvas, "pred")) == 1


def test_canvas_clear_pred_does_not_clear_gt(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET)
    canvas.set_pred_detections(_DET2)
    canvas.clear_pred_detections()
    assert len(_polys(canvas, "pred")) == 0
    assert len(_polys(canvas, "gt")) == 1


def test_canvas_set_overlay_visibility_hides_gt(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET)
    canvas.set_overlay_visibility(show_gt=False, show_pred=True)
    for item in _polys(canvas, "gt"):
        assert not item.isVisible()


def test_canvas_set_overlay_visibility_hides_pred(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_pred_detections(_DET)
    canvas.set_overlay_visibility(show_gt=True, show_pred=False)
    for item in _polys(canvas, "pred"):
        assert not item.isVisible()


def test_canvas_set_overlay_visibility_shows_both(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET)
    canvas.set_pred_detections(_DET2)
    canvas.set_overlay_visibility(show_gt=True, show_pred=True)
    for item in _polys(canvas, "gt"):
        assert item.isVisible()
    for item in _polys(canvas, "pred"):
        assert item.isVisible()


def test_canvas_set_class_filter(qapp):
    """Only class IDs in visible_class_ids should be shown."""
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_gt_detections(_DET + _DET2)  # class_id 0 and 1
    canvas.set_class_filter({0})
    # class 0 should be visible, class 1 hidden
    gt_visible = [i for i in _polys(canvas, "gt") if i.isVisible()]
    gt_hidden = [i for i in _polys(canvas, "gt") if not i.isVisible()]
    assert len(gt_visible) == 1
    assert len(gt_hidden) == 1


def test_canvas_set_detections_backward_compat(qapp):
    """set_detections() must still work as a GT alias."""
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_detections(_DET)
    assert len(_polys(canvas, "gt")) == 1


def test_canvas_clear_detections_backward_compat(qapp):
    """clear_detections() must clear GT layer."""
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_detections(_DET)
    canvas.clear_detections()
    assert len(_polys(canvas, "gt")) == 0
