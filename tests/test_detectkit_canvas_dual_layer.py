"""Tests for OBBCanvas dual-layer overlay (GT + predictions)."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.detectkit.gui.overlays import (  # noqa: E402
    ColourPolicy,
    LabelMode,
    LayerStyle,
    OverlayLayer,
)
from hydra_suite.utils.geometry_levels import GeometryLevel  # noqa: E402


def _layer(key, detections, *, level=GeometryLevel.AABB, names=None, **kw):
    """A single-level layer: no derivation, explicit solid style."""
    base = dict(
        key=key,
        detections=detections,
        native_level=level,
        class_names=names,
        colour_policy=ColourPolicy.PER_CLASS,
        derive_levels=False,
        style=LayerStyle(Qt.PenStyle.SolidLine, Qt.BrushStyle.NoBrush, 0),
    )
    base.update(kw)
    return OverlayLayer(**base)


def _pred(detections, **kw):
    """The single-level prediction layer: dashed, confidence labels, on top."""
    return _layer(
        "pred",
        detections,
        style=LayerStyle(Qt.PenStyle.DashLine, Qt.BrushStyle.NoBrush, 0),
        label_mode=LabelMode.NAME_AND_CONFIDENCE,
        z=20,
        **kw,
    )


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
    canvas.set_layer(_layer("gt", _DET))
    assert len(_polys(canvas, "gt")) == 1
    assert len(_labels(canvas, "gt")) == 1


def test_canvas_pred_items_populated(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_layer(_pred(_DET2))
    assert len(_polys(canvas, "pred")) == 1
    assert len(_labels(canvas, "pred")) == 1


def test_canvas_gt_and_pred_independent(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_layer(_layer("gt", _DET))
    canvas.set_layer(_pred(_DET2))
    assert len(_polys(canvas, "gt")) == 1
    assert len(_polys(canvas, "pred")) == 1


def test_canvas_clear_gt_does_not_clear_pred(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_layer(_layer("gt", _DET))
    canvas.set_layer(_pred(_DET2))
    canvas.remove_layer("gt")
    assert len(_polys(canvas, "gt")) == 0
    assert len(_polys(canvas, "pred")) == 1


def test_canvas_clear_pred_does_not_clear_gt(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_layer(_layer("gt", _DET))
    canvas.set_layer(_pred(_DET2))
    canvas.remove_layer("pred")
    assert len(_polys(canvas, "pred")) == 0
    assert len(_polys(canvas, "gt")) == 1


def test_canvas_set_overlay_visibility_hides_gt(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_layer(_layer("gt", _DET))
    canvas.set_layer_visible("gt", False)
    canvas.set_layer_visible("pred", True)
    for item in _polys(canvas, "gt"):
        assert not item.isVisible()


def test_canvas_set_overlay_visibility_hides_pred(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_layer(_pred(_DET))
    canvas.set_layer_visible("gt", True)
    canvas.set_layer_visible("pred", False)
    for item in _polys(canvas, "pred"):
        assert not item.isVisible()


def test_canvas_set_overlay_visibility_shows_both(qapp):
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_layer(_layer("gt", _DET))
    canvas.set_layer(_pred(_DET2))
    canvas.set_layer_visible("gt", True)
    canvas.set_layer_visible("pred", True)
    for item in _polys(canvas, "gt"):
        assert item.isVisible()
    for item in _polys(canvas, "pred"):
        assert item.isVisible()


def test_canvas_set_class_filter(qapp):
    """Only class IDs in visible_class_ids should be shown."""
    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    canvas = OBBCanvas()
    canvas.set_layer(_layer("gt", _DET + _DET2))  # class_id 0 and 1
    canvas.set_class_filter({0})
    # class 0 should be visible, class 1 hidden
    gt_visible = [i for i in _polys(canvas, "gt") if i.isVisible()]
    gt_hidden = [i for i in _polys(canvas, "gt") if not i.isVisible()]
    assert len(gt_visible) == 1
    assert len(gt_hidden) == 1
