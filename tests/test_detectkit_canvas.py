"""Tests for OBB label parsing (canvas drawing tested manually)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.detectkit.gui.canvas import OBBCanvas  # noqa: E402
from hydra_suite.detectkit.gui.utils import parse_obb_label  # noqa: E402


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_parse_obb_label(tmp_path: Path):
    lbl = tmp_path / "test.txt"
    lbl.write_text("0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n", encoding="utf-8")
    dets = parse_obb_label(lbl, img_w=100, img_h=100)
    assert len(dets) == 1
    assert dets[0]["class_id"] == 0
    assert len(dets[0]["polygon_px"]) == 4
    assert abs(dets[0]["polygon_px"][0][0] - 10.0) < 0.1
    assert abs(dets[0]["polygon_px"][0][1] - 20.0) < 0.1


def test_parse_obb_label_empty(tmp_path: Path):
    lbl = tmp_path / "empty.txt"
    lbl.write_text("", encoding="utf-8")
    dets = parse_obb_label(lbl, img_w=100, img_h=100)
    assert dets == []


def test_parse_obb_label_invalid_line(tmp_path: Path):
    lbl = tmp_path / "bad.txt"
    lbl.write_text("0 0.1 0.2\n0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n", encoding="utf-8")
    dets = parse_obb_label(lbl, img_w=100, img_h=100)
    assert len(dets) == 1


def test_parse_obb_label_accepts_5_field_aabb_line(tmp_path: Path):
    """Regression: a plain YOLO-detect AABB line (class_id cx cy w h, 5
    fields) must not be silently dropped -- it's the format
    data/al/labels.py writes for GeometryLevel.AABB, and DetectKit's GT
    canvas rendering must show it as a quad, not skip it entirely."""
    lbl = tmp_path / "aabb.txt"
    lbl.write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    dets = parse_obb_label(lbl, img_w=100, img_h=100)
    assert len(dets) == 1
    assert dets[0]["class_id"] == 0
    poly = dets[0]["polygon_px"]
    assert len(poly) == 4
    xs = sorted(p[0] for p in poly)
    ys = sorted(p[1] for p in poly)
    assert abs(xs[0] - 30.0) < 0.1 and abs(xs[-1] - 70.0) < 0.1
    assert abs(ys[0] - 40.0) < 0.1 and abs(ys[-1] - 60.0) < 0.1


def test_parse_obb_label_accepts_variable_length_polygon_line(tmp_path: Path):
    """Regression: a genuine polygon contour (class_id + n point pairs,
    n > 4) must not be silently dropped -- it's the format
    data/al/labels.py writes for GeometryLevel.POLYGON, and it's now also
    what an accepted SAM2 escalation promotes into a source's canonical
    labels (Part A). A rigid ==9-fields check drops every such line."""
    lbl = tmp_path / "polygon.txt"
    lbl.write_text(
        "0 0.1 0.1 0.3 0.05 0.5 0.1 0.5 0.4 0.3 0.5 0.1 0.4\n", encoding="utf-8"
    )
    dets = parse_obb_label(lbl, img_w=100, img_h=100)
    assert len(dets) == 1
    assert dets[0]["class_id"] == 0
    assert len(dets[0]["polygon_px"]) == 6
    assert abs(dets[0]["polygon_px"][0][0] - 10.0) < 0.1
    assert abs(dets[0]["polygon_px"][0][1] - 10.0) < 0.1


def test_parse_obb_label_still_skips_degenerate_and_odd_length_lines(tmp_path: Path):
    """A line with fewer than 3 usable points, or an odd coordinate count
    that isn't the 4-value AABB shape, is not a valid shape and must still
    be skipped (not crash, not render as a spurious point)."""
    lbl = tmp_path / "degenerate.txt"
    lbl.write_text(
        "0 0.1 0.2 0.3\n"  # 3 fields: 2 coords, not AABB (needs 4), too few for polygon
        "0 0.1 0.2 0.3 0.4 0.5\n"  # 5 coords: odd, not AABB (4), not a valid polygon
        "0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n",  # valid quad
        encoding="utf-8",
    )
    dets = parse_obb_label(lbl, img_w=100, img_h=100)
    assert len(dets) == 1


def test_canvas_uses_class_name_lookup_for_labels(qapp):
    canvas = OBBCanvas()
    canvas.set_detections(
        [
            {
                "class_id": 1,
                "polygon_px": [(10.0, 10.0), (40.0, 10.0), (40.0, 30.0), (10.0, 30.0)],
            }
        ],
        class_names=["ant", "bee"],
    )

    assert len(canvas._label_items) == 1
    assert canvas._label_items[0].toPlainText() == "bee (1)"


class _StubWheelEvent:
    def __init__(self, delta: int = 120) -> None:
        self._delta = delta
        self.accepted = False

    def modifiers(self):
        return Qt.ControlModifier

    def angleDelta(self):
        from PySide6.QtCore import QPoint

        return QPoint(0, self._delta)

    def accept(self) -> None:
        self.accepted = True


def test_canvas_auto_fits_loaded_image_and_clamps_zoom_controls(qapp):
    canvas = OBBCanvas()
    canvas.resize(640, 480)
    canvas.show()
    qapp.processEvents()

    img = np.zeros((1800, 2400, 3), dtype=np.uint8)
    assert canvas.set_image_array(img) is True
    qapp.processEvents()

    assert canvas._zoom < 1.0

    canvas._set_zoom(10.0)
    assert canvas._zoom == canvas._max_zoom

    canvas.fit_in_view()
    assert canvas._fit_mode is True


def test_canvas_ctrl_wheel_zoom_changes_zoom(qapp):
    canvas = OBBCanvas()
    canvas.resize(640, 480)
    canvas.show()
    qapp.processEvents()

    img = np.zeros((800, 1200, 3), dtype=np.uint8)
    assert canvas.set_image_array(img) is True
    qapp.processEvents()

    before = canvas._zoom
    event = _StubWheelEvent(120)
    canvas.wheelEvent(event)

    assert event.accepted is True
    assert canvas._zoom > before


def test_set_gt_detections_multi_level_polygon_native_draws_three_layers(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    polygon = [(10.0, 10.0), (50.0, 5.0), (90.0, 40.0), (60.0, 90.0), (20.0, 60.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": polygon}],
        native_level=GeometryLevel.POLYGON,
        reviewed=True,
    )
    assert set(canvas._gt_level_items.keys()) == {
        GeometryLevel.POLYGON,
        GeometryLevel.OBB,
        GeometryLevel.AABB,
    }
    for level in (GeometryLevel.POLYGON, GeometryLevel.OBB, GeometryLevel.AABB):
        assert len(canvas._gt_level_items[level]) == 1


def test_set_gt_detections_multi_level_obb_native_draws_two_layers(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    quad = [(10.0, 10.0), (90.0, 20.0), (80.0, 90.0), (0.0, 80.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": quad}],
        native_level=GeometryLevel.OBB,
        reviewed=True,
    )
    assert set(canvas._gt_level_items.keys()) == {GeometryLevel.OBB, GeometryLevel.AABB}


def test_set_gt_detections_multi_level_aabb_native_draws_one_layer(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    quad = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": quad}],
        native_level=GeometryLevel.AABB,
        reviewed=True,
    )
    assert set(canvas._gt_level_items.keys()) == {GeometryLevel.AABB}


def test_set_gt_detections_multi_level_unreviewed_uses_hatched_brush_on_native_only(
    qapp,
):
    from PySide6.QtCore import Qt

    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    quad = [(10.0, 10.0), (90.0, 20.0), (80.0, 90.0), (0.0, 80.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": quad}],
        native_level=GeometryLevel.OBB,
        reviewed=False,
    )
    native_item = canvas._gt_level_items[GeometryLevel.OBB][0]
    derived_item = canvas._gt_level_items[GeometryLevel.AABB][0]
    assert native_item.brush().style() == Qt.BrushStyle.BDiagPattern
    assert derived_item.brush().style() != Qt.BrushStyle.BDiagPattern


def test_set_gt_detections_multi_level_unreviewed_native_pen_differs_from_aabb(
    qapp,
):
    """Regression: the unreviewed-native override must keep the native
    level's own pen style, not hardcode SolidLine -- otherwise an
    unreviewed OBB-native quad (DashLine) collides with derived AABB's
    own SolidLine pen and the two boxes become visually indistinguishable."""
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    quad = [(10.0, 10.0), (90.0, 20.0), (80.0, 90.0), (0.0, 80.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": quad}],
        native_level=GeometryLevel.OBB,
        reviewed=False,
    )
    native_item = canvas._gt_level_items[GeometryLevel.OBB][0]
    aabb_item = canvas._gt_level_items[GeometryLevel.AABB][0]
    assert native_item.pen().style() != aabb_item.pen().style()
    assert native_item.pen().style() == Qt.PenStyle.DashLine
    # Fill override is preserved -- only the pen style itself was fixed.
    assert native_item.brush().style() == Qt.BrushStyle.BDiagPattern


def test_set_gt_detections_append_after_multi_level_is_visibility_controlled(
    qapp,
):
    """Regression: set_gt_detections(..., append=True) called after a
    multi-level draw must route the new item into the native level's
    bucket, or _apply_visibility (which iterates _gt_level_items once it's
    populated) never visits it and show/hide toggles silently no-op on it."""
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    quad = [(10.0, 10.0), (90.0, 20.0), (80.0, 90.0), (0.0, 80.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": quad}],
        native_level=GeometryLevel.OBB,
        reviewed=True,
    )
    appended_quad = [(15.0, 15.0), (95.0, 25.0), (85.0, 95.0), (5.0, 85.0)]
    canvas.set_gt_detections(
        [{"class_id": 0, "polygon_px": appended_quad}], append=True
    )

    # The appended item must have landed in the native level's bucket, not
    # only the flat list.
    assert len(canvas._gt_level_items[GeometryLevel.OBB]) == 2
    appended_item = canvas._gt_level_items[GeometryLevel.OBB][-1]
    assert appended_item.isVisible() is True

    canvas.set_overlay_visibility(show_gt=False, show_pred=True)
    assert appended_item.isVisible() is False

    canvas.set_overlay_visibility(show_gt=True, show_pred=True)
    assert appended_item.isVisible() is True


def test_set_derived_levels_visible_hides_non_native_layers_only(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    polygon = [(10.0, 10.0), (50.0, 5.0), (90.0, 40.0), (60.0, 90.0), (20.0, 60.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": polygon}],
        native_level=GeometryLevel.POLYGON,
        reviewed=True,
    )

    canvas.set_derived_levels_visible(False)
    assert canvas._gt_level_items[GeometryLevel.POLYGON][0].isVisible() is True
    assert canvas._gt_level_items[GeometryLevel.OBB][0].isVisible() is False
    assert canvas._gt_level_items[GeometryLevel.AABB][0].isVisible() is False

    canvas.set_derived_levels_visible(True)
    assert canvas._gt_level_items[GeometryLevel.OBB][0].isVisible() is True
    assert canvas._gt_level_items[GeometryLevel.AABB][0].isVisible() is True


def test_clear_gt_detections_clears_per_level_state(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    quad = [(10.0, 10.0), (90.0, 20.0), (80.0, 90.0), (0.0, 80.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": quad}], native_level=GeometryLevel.OBB
    )
    canvas.clear_gt_detections()
    assert canvas._gt_level_items == {}
    assert canvas._gt_obb_items == []


def test_set_gt_detections_single_layer_still_works_unchanged(qapp):
    """Backward-compat: the old single-layer API is untouched for any
    caller that doesn't pass native_level/reviewed -- including its
    visibility wiring, which _apply_visibility must still drive via the
    flat _gt_obb_items list when _gt_level_items was never populated."""
    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    canvas.set_gt_detections(
        [
            {
                "class_id": 0,
                "polygon_px": [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)],
            }
        ]
    )
    assert len(canvas._gt_obb_items) == 1
    assert canvas._gt_level_items == {}  # single-layer path never touches this

    canvas.set_overlay_visibility(show_gt=False, show_pred=True)
    assert canvas._gt_obb_items[0].isVisible() is False

    canvas.set_overlay_visibility(show_gt=True, show_pred=True)
    assert canvas._gt_obb_items[0].isVisible() is True


def test_clear_all_then_apply_visibility_does_not_raise(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    quad = [(10.0, 10.0), (90.0, 20.0), (80.0, 90.0), (0.0, 80.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": quad}], native_level=GeometryLevel.OBB
    )
    canvas.clear_all()
    canvas.set_overlay_visibility(show_gt=True, show_pred=True)  # must not raise


# --- staged-escalation overlay ------------------------------------------
# Before this layer existed, staged SAM3/SAM2 masks were accepted or
# rejected sight-unseen: the review dialog is a text list and nothing ever
# parsed staged_path/labels for display.


def _poly():
    return [(10.0, 10.0), (50.0, 5.0), (90.0, 40.0), (60.0, 90.0), (20.0, 60.0)]


def test_escalation_layer_draws_native_plus_derived_levels(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    canvas.set_escalation_detections(
        [{"class_id": 0, "polygon_px": _poly(), "confidence": 0.8}],
        native_level=GeometryLevel.POLYGON,
    )
    assert set(canvas._esc_level_items.keys()) == {
        GeometryLevel.POLYGON,
        GeometryLevel.OBB,
        GeometryLevel.AABB,
    }
    for level in canvas._esc_level_items:
        assert len(canvas._esc_level_items[level]) == 1


def test_escalation_layer_uses_its_own_colour_not_the_class_palette(qapp):
    """A staged mask must never be mistaken for ground truth.

    Class 0's palette entry is green and GT draws it green; the escalation
    layer must draw the same class in its own hue.
    """
    from hydra_suite.detectkit.gui.canvas import _PALETTE, ESCALATION_COLOUR
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    det = [{"class_id": 0, "polygon_px": _poly()}]
    canvas.set_gt_detections_multi_level(det, native_level=GeometryLevel.POLYGON)
    canvas.set_escalation_detections(det, native_level=GeometryLevel.POLYGON)

    gt_pen = canvas._gt_level_items[GeometryLevel.POLYGON][0].pen().color()
    esc_pen = canvas._esc_level_items[GeometryLevel.POLYGON][0].pen().color()
    assert gt_pen == _PALETTE[0]
    assert esc_pen == ESCALATION_COLOUR
    assert esc_pen != gt_pen


def test_escalation_layer_leaves_the_gt_layer_alone(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    det = [{"class_id": 0, "polygon_px": _poly()}]
    canvas.set_gt_detections_multi_level(det, native_level=GeometryLevel.POLYGON)
    before = len(canvas._gt_obb_items)
    canvas.set_escalation_detections(det, native_level=GeometryLevel.POLYGON)
    assert len(canvas._gt_obb_items) == before
    canvas.clear_escalation_detections()
    assert len(canvas._gt_obb_items) == before
    assert canvas._esc_obb_items == []


def test_escalation_layer_visibility_toggles_independently(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    det = [{"class_id": 0, "polygon_px": _poly()}]
    canvas.set_gt_detections_multi_level(det, native_level=GeometryLevel.POLYGON)
    canvas.set_escalation_detections(det, native_level=GeometryLevel.POLYGON)

    canvas.set_escalation_visible(False)
    assert not canvas._esc_level_items[GeometryLevel.POLYGON][0].isVisible()
    assert canvas._gt_level_items[GeometryLevel.POLYGON][0].isVisible()
    canvas.set_escalation_visible(True)
    assert canvas._esc_level_items[GeometryLevel.POLYGON][0].isVisible()


def test_escalation_derived_levels_follow_the_shared_derived_toggle(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    canvas.set_escalation_detections(
        [{"class_id": 0, "polygon_px": _poly()}], native_level=GeometryLevel.POLYGON
    )
    canvas.set_derived_levels_visible(False)
    assert canvas._esc_level_items[GeometryLevel.POLYGON][0].isVisible()
    assert not canvas._esc_level_items[GeometryLevel.AABB][0].isVisible()


def test_escalation_layer_ignores_the_class_filter(qapp):
    """Staged class ids are the PROMPT's, not the project's.

    classes.txt in a staging dir holds the noun phrase, so its ids do not
    index the project's class list; filtering the layer by project class id
    would hide masks for a class the user never deselected.
    """
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    canvas.set_escalation_detections(
        [{"class_id": 0, "polygon_px": _poly()}], native_level=GeometryLevel.POLYGON
    )
    canvas.set_class_filter({7})
    assert canvas._esc_level_items[GeometryLevel.POLYGON][0].isVisible()


def test_escalation_labels_carry_the_confidence(qapp):
    """Which masks a re-threshold would drop has to be readable off the frame."""
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    canvas.set_escalation_detections(
        [{"class_id": 0, "polygon_px": _poly(), "confidence": 0.42}],
        class_names=["ant"],
        native_level=GeometryLevel.POLYGON,
    )
    text = canvas._esc_level_label_items[GeometryLevel.POLYGON][0].toPlainText()
    assert "0.42" in text and "ant" in text


def test_clear_all_drops_the_escalation_layer(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    canvas.set_escalation_detections(
        [{"class_id": 0, "polygon_px": _poly()}], native_level=GeometryLevel.POLYGON
    )
    canvas.clear_all()
    assert canvas._esc_obb_items == []
    assert canvas._esc_level_items == {}
    canvas._apply_visibility()  # must not raise


def test_a_detection_without_confidence_is_labelled_by_name_alone(qapp):
    """Staged labels carry no confidence -- data/al/labels.py writes only the
    class id and coordinates. Falling back to "name (class_id)" printed
    "worker ant (0)" over every staged mask, and a (0) next to a mask reads
    as confidence 0.00, which is the opposite of the truth."""
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    canvas.set_escalation_detections(
        [{"class_id": 0, "polygon_px": _poly()}],
        class_names=["worker ant"],
        native_level=GeometryLevel.POLYGON,
    )
    text = canvas._esc_level_label_items[GeometryLevel.POLYGON][0].toPlainText()
    assert text == "worker ant"
