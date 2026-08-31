"""Golden characterization of every DetectKit overlay layer.

This file is the gate for the overlay-registry refactor: it records what
the canvas draws BEFORE the refactor and asserts the refactor reproduces
it exactly. The golden lives in a committed JSON file, not in a
before-vs-after comparison inside one process -- a post-refactor oracle
built from the refactored code proves only that the code equals itself.

Regenerate ONLY for a deliberate, reviewed rendering change:
    python -m pytest tests/test_detectkit_overlay_golden.py --update-golden
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QGraphicsPolygonItem,
    QGraphicsTextItem,
)

from hydra_suite.detectkit.gui.canvas import OBBCanvas  # noqa: E402
from hydra_suite.utils.geometry_levels import GeometryLevel  # noqa: E402

GOLDEN = Path(__file__).parent / "goldens" / "detectkit_overlay_characterization.json"


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def describe_scene(canvas: OBBCanvas) -> list[dict]:
    """Serialise every overlay item, INCLUDING its stacking position.

    QGraphicsScene.items() returns items front-to-back, so the index in
    that list IS the stacking order. Capturing it is what makes this
    oracle able to catch a z-order change; a set-only comparison would
    wave one through, and the front-to-back order of the escalation and
    prediction layers is exactly the thing a registry with explicit z
    values is most likely to invert.
    """
    ordered = canvas._scene.items()
    out: list[dict] = []
    for stack_index, item in enumerate(ordered):
        if isinstance(item, QGraphicsPolygonItem):
            poly = item.polygon()
            pen = item.pen()
            brush = item.brush()
            out.append(
                {
                    "type": "polygon",
                    "stack_index": stack_index,
                    "points": [
                        [round(poly.at(i).x(), 4), round(poly.at(i).y(), 4)]
                        for i in range(poly.count())
                    ],
                    "pen_colour": pen.color().name(),
                    "pen_style": int(pen.style().value),
                    "pen_width": pen.width(),
                    "pen_cosmetic": pen.isCosmetic(),
                    "brush_style": int(brush.style().value),
                    "brush_alpha": brush.color().alpha(),
                    "visible": item.isVisible(),
                }
            )
        elif isinstance(item, QGraphicsTextItem):
            out.append(
                {
                    "type": "label",
                    "stack_index": stack_index,
                    "text": item.toPlainText(),
                    "colour": item.defaultTextColor().name(),
                    "pos": [round(item.pos().x(), 4), round(item.pos().y(), 4)],
                    "font_px": item.font().pixelSize(),
                    "visible": item.isVisible(),
                }
            )
    return out


_GT = [
    {"class_id": 0, "polygon_px": [(10, 10), (60, 12), (58, 40), (8, 38)]},
    {"class_id": 1, "polygon_px": [(100, 100), (150, 105), (145, 140), (95, 135)]},
]
_PRED = [
    {
        "class_id": 0,
        "polygon_px": [(12, 11), (62, 13), (60, 41), (10, 39)],
        "confidence": 0.87,
    },
    {"class_id": 3, "polygon_px": [(200, 200), (240, 200), (240, 230), (200, 230)]},
]
_ESC = [
    {"class_id": 0, "polygon_px": [(20, 20), (70, 22), (68, 50), (18, 48)]},
    {"class_id": 7, "polygon_px": [(300, 300), (360, 310), (350, 350), (295, 340)]},
]
_NAMES = ["ant", "worker", "queen", "larva"]


def _build_main_window_scene(canvas: OBBCanvas) -> None:
    """show_image's exact call order (main_window.py:2107-2151):
    GT first, then the staged escalation, then predictions. Predictions
    therefore sit ON TOP of staged masks -- this scene is what pins that."""
    canvas.set_gt_detections_multi_level(
        _GT, class_names=_NAMES, native_level=GeometryLevel.POLYGON, reviewed=True
    )
    canvas.set_escalation_detections(
        _ESC, class_names=["prompt_a", "prompt_b"], native_level=GeometryLevel.OBB
    )
    canvas.set_pred_detections(_PRED, class_names=_NAMES)


def _build_unreviewed_scene(canvas: OBBCanvas) -> None:
    """show_image when _resolve_source_render_state says reviewed=False:
    the native level gets the BDiagPattern hatch, keeping its own pen."""
    canvas.set_gt_detections_multi_level(
        _GT, class_names=_NAMES, native_level=GeometryLevel.OBB, reviewed=False
    )


def _build_dialog_scene(canvas: OBBCanvas) -> None:
    """semantic_frame_preview_dialog.py:131-132 and
    calibration_results_dialog.py:315-316: single-level GT and predictions
    with explicit fills and dict class_names, no level derivation."""
    names = {0: "Ground truth", 2: "Prediction"}
    canvas.set_gt_detections(_GT, names, fill_alpha=65)
    canvas.set_pred_detections(_PRED, names, fill_alpha=55)


def _build_aabb_native_scene(canvas: OBBCanvas) -> None:
    """An AABB-native source: only one level exists, nothing is derived."""
    canvas.set_gt_detections_multi_level(
        _GT, class_names=_NAMES, native_level=GeometryLevel.AABB, reviewed=True
    )


SCENES = {
    "main_window": _build_main_window_scene,
    "unreviewed": _build_unreviewed_scene,
    "dialog": _build_dialog_scene,
    "aabb_native": _build_aabb_native_scene,
}


def _render(name: str) -> list[dict]:
    canvas = OBBCanvas()
    SCENES[name](canvas)
    return describe_scene(canvas)


def test_overlay_rendering_matches_the_committed_golden(qapp, request):
    rendered = {name: _render(name) for name in sorted(SCENES)}

    if request.config.getoption("--update-golden", default=False):
        # A golden of empty scenes would pass forever while proving
        # nothing -- the empty-CSV equivalence trap. Refuse to write one.
        assert all(rendered.values()), "refusing to write an empty golden"
        assert len(rendered["main_window"]) > len(rendered["aabb_native"]), (
            "three layers over multiple derived levels must produce more "
            "items than one layer at one level"
        )
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(rendered, indent=2) + "\n")
        pytest.skip("golden updated")

    expected = json.loads(GOLDEN.read_text())
    assert rendered == expected


def test_predictions_stack_above_staged_escalations(qapp):
    """An explicit, standalone pin on the ordering the registry's z values
    must reproduce. show_image draws GT -> escalation -> predictions, so
    predictions are frontmost; QGraphicsScene.items() is front-to-back, so
    the prediction items hold the LOWEST stack indices."""
    canvas = OBBCanvas()
    _build_main_window_scene(canvas)
    described = describe_scene(canvas)
    magenta = [d for d in described if d.get("pen_colour") == "#ff3cc7"]
    others = [
        d
        for d in described
        if d["type"] == "polygon" and d.get("pen_colour") != "#ff3cc7"
    ]
    assert magenta, "the escalation layer must be drawn"
    # Predictions were drawn last => frontmost => lowest stack_index.
    assert min(d["stack_index"] for d in others) < min(
        d["stack_index"] for d in magenta
    )


from hydra_suite.detectkit.gui.overlays import (  # noqa: E402
    ColourPolicy,
    LabelMode,
    OverlayLayer,
)


def _gt_layer(**kw):
    base = dict(
        key="gt",
        detections=_GT,
        native_level=GeometryLevel.POLYGON,
        class_names=_NAMES,
        colour_policy=ColourPolicy.PER_CLASS,
        label_mode=LabelMode.NAME_AND_CLASS_ID,
    )
    base.update(kw)
    return OverlayLayer(**base)


def _pred_layer(**kw):
    from PySide6.QtCore import Qt

    from hydra_suite.detectkit.gui.overlays import LayerStyle

    base = dict(
        key="pred",
        detections=_PRED,
        native_level=GeometryLevel.AABB,
        class_names=_NAMES,
        colour_policy=ColourPolicy.PER_CLASS,
        derive_levels=False,
        style=LayerStyle(Qt.PenStyle.DashLine, Qt.BrushStyle.NoBrush, 0),
        label_mode=LabelMode.NAME_AND_CONFIDENCE,
        z=20,
    )
    base.update(kw)
    return OverlayLayer(**base)


def test_set_layer_replaces_by_key_instead_of_accumulating(qapp):
    """The escalation layer's stale-mask bug was a caller forgetting to
    clear before refreshing. set_layer makes that unrepresentable."""
    canvas = OBBCanvas()
    canvas.set_layer(_gt_layer())
    once = len(canvas._scene.items())
    canvas.set_layer(_gt_layer())
    assert len(canvas._scene.items()) == once


def test_remove_layer_removes_only_that_key(qapp):
    canvas = OBBCanvas()
    canvas.set_layer(_gt_layer())
    gt_only = len(canvas._scene.items())
    canvas.set_layer(_pred_layer())
    assert len(canvas._scene.items()) > gt_only
    canvas.remove_layer("pred")
    assert len(canvas._scene.items()) == gt_only


def test_remove_layer_is_a_no_op_for_an_unknown_key(qapp):
    OBBCanvas().remove_layer("nope")  # must not raise


def test_set_layer_visible_is_remembered_before_the_layer_is_drawn(qapp):
    """_on_overlay_changed can fire before the first show_image. A
    visibility call for a not-yet-drawn key must not be silently lost."""
    canvas = OBBCanvas()
    canvas.set_layer_visible("gt", False)
    canvas.set_layer(_gt_layer())
    assert all(
        not i["visible"] for i in describe_scene(canvas) if i["type"] == "polygon"
    )


def test_layer_items_exposes_the_per_level_buckets(qapp):
    canvas = OBBCanvas()
    canvas.set_layer(_gt_layer(native_level=GeometryLevel.POLYGON))
    buckets = canvas.layer_items("gt")
    assert set(buckets) == {
        GeometryLevel.POLYGON,
        GeometryLevel.OBB,
        GeometryLevel.AABB,
    }
    assert canvas.layer_items("absent") == {}
