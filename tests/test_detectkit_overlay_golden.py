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
from hydra_suite.detectkit.gui.colors import ESCALATION_COLOUR  # noqa: E402
from hydra_suite.detectkit.gui.dialogs._overlay_helpers import (  # noqa: E402
    dialog_gt_layer,
    dialog_pred_layer,
)
from hydra_suite.detectkit.gui.overlays import (  # noqa: E402
    ColourPolicy,
    Emphasis,
    LabelMode,
    OverlayLayer,
)
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
                    # ORACLE GAP: brush.color().name() (the fill's hue) is not
                    # recorded here, only its style and alpha. A fill painted
                    # in the wrong hue with the right alpha would pass this
                    # golden and every other test. There is no live bug today
                    # -- canvas.py derives the fill colour from the pen colour
                    # -- but closing this gap requires adding the field and
                    # regenerating the golden, which must NOT be done outside
                    # an intentional rendering change (a regenerated golden
                    # recorded from post-refactor code would be tautological
                    # for anything the refactor itself might have broken).
                    # Add "brush_colour": brush.color().name() here the next
                    # time the golden is legitimately regenerated.
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


def _escalation_layer(**kw):
    base = dict(
        key="staged",
        detections=_ESC,
        native_level=GeometryLevel.OBB,
        class_names=["prompt_a", "prompt_b"],
        colour_policy=ColourPolicy.FIXED,
        fixed_colour=ESCALATION_COLOUR,
        class_filtered=False,
        label_mode=LabelMode.NAME_AND_CONFIDENCE,
        z=10,
    )
    base.update(kw)
    return OverlayLayer(**base)


def _build_main_window_scene(canvas: OBBCanvas) -> None:
    """show_image's exact call order (main_window.py:2107-2151):
    GT first, then the staged escalation, then predictions. Predictions
    therefore sit ON TOP of staged masks -- this scene is what pins that."""
    canvas.set_layer(_gt_layer())
    canvas.set_layer(_escalation_layer())
    canvas.set_layer(_pred_layer())


def _build_unreviewed_scene(canvas: OBBCanvas) -> None:
    """show_image when _resolve_source_render_state says reviewed=False:
    the native level gets the BDiagPattern hatch, keeping its own pen."""
    canvas.set_layer(
        _gt_layer(native_level=GeometryLevel.OBB, emphasis=Emphasis.UNREVIEWED)
    )


def _build_dialog_scene(canvas: OBBCanvas) -> None:
    """semantic_frame_preview_dialog.py:131-132 and
    calibration_results_dialog.py:315-316: single-level GT and predictions
    with explicit fills and dict class_names, no level derivation."""
    names = {0: "Ground truth", 2: "Prediction"}
    canvas.set_layer(dialog_gt_layer(_GT, names))
    canvas.set_layer(dialog_pred_layer(_PRED, names))


def _build_aabb_native_scene(canvas: OBBCanvas) -> None:
    """An AABB-native source: only one level exists, nothing is derived."""
    canvas.set_layer(_gt_layer(native_level=GeometryLevel.AABB))


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


import itertools  # noqa: E402


def _visible_counts(canvas) -> dict:
    return {
        key: sum(
            1
            for b in canvas.layer_items(key).values()
            for o in b.obb_items
            if o.isVisible()
        )
        for key in ("gt", "pred", "staged")
    }


def test_hiding_gt_leaves_the_other_two_layers_fully_visible(qapp):
    canvas = OBBCanvas()
    _build_main_window_scene(canvas)
    before = _visible_counts(canvas)
    canvas.set_layer_visible("gt", False)
    after = _visible_counts(canvas)
    assert after["gt"] == 0
    assert after["pred"] == before["pred"]
    assert after["staged"] == before["staged"]


def test_hiding_derived_levels_keeps_exactly_the_native_shapes(qapp):
    """GT is POLYGON-native and escalation OBB-native in this scene, so
    each keeps its own native bucket and loses the rest. Predictions never
    derive, so they are untouched."""
    canvas = OBBCanvas()
    _build_main_window_scene(canvas)
    canvas.set_derived_levels_visible(False)
    assert _visible_counts(canvas) == {
        "gt": len(_GT),
        "staged": len(_ESC),
        "pred": len(_PRED),
    }


def test_the_class_filter_never_hides_the_escalation_layer(qapp):
    """Staged class ids index the STAGING dir's classes.txt, not the
    project's class list, so the project filter cannot address them."""
    canvas = OBBCanvas()
    _build_main_window_scene(canvas)
    canvas.set_class_filter({999})
    counts = _visible_counts(canvas)
    assert counts["gt"] == 0
    assert counts["staged"] == len(_ESC) * 2  # OBB native + derived AABB


def test_an_empty_class_filter_means_show_all(qapp):
    canvas = OBBCanvas()
    _build_main_window_scene(canvas)
    canvas.set_class_filter({0})
    filtered = _visible_counts(canvas)["gt"]
    canvas.set_class_filter(set())
    assert _visible_counts(canvas)["gt"] > filtered


@pytest.mark.parametrize(
    "show_gt,show_pred,show_esc,show_derived,class_filter",
    list(
        itertools.product(
            [True, False],
            [True, False],
            [True, False],
            [True, False],
            [set(), {0}],
        )
    ),
)
def test_visibility_matrix_is_self_consistent(
    qapp, show_gt, show_pred, show_esc, show_derived, class_filter
):
    """Broad sweep for crashes and cross-layer leakage: no combination may
    make a hidden layer visible or a shown-and-unfiltered layer invisible."""
    canvas = OBBCanvas()
    _build_main_window_scene(canvas)
    canvas.set_layer_visible("gt", show_gt)
    canvas.set_layer_visible("pred", show_pred)
    canvas.set_layer_visible("staged", show_esc)
    canvas.set_derived_levels_visible(show_derived)
    canvas.set_class_filter(class_filter)

    counts = _visible_counts(canvas)
    if not show_gt:
        assert counts["gt"] == 0
    if not show_pred:
        assert counts["pred"] == 0
    if not show_esc:
        assert counts["staged"] == 0
    if show_esc:
        # class_filtered=False, so the filter can never reduce it
        expected = len(_ESC) * (2 if show_derived else 1)
        assert counts["staged"] == expected


def test_dialog_scene_is_buildable_from_layers_alone(qapp):
    """The two calibration dialogs draw single-level filled GT and dashed
    predictions. This is the shape the registry must express without the
    transitional adapters -- and it must match the committed golden."""
    names = {0: "Ground truth", 2: "Prediction"}
    canvas = OBBCanvas()
    canvas.set_layer(dialog_gt_layer(_GT, names))
    canvas.set_layer(dialog_pred_layer(_PRED, names))
    expected = json.loads(GOLDEN.read_text())["dialog"]
    assert describe_scene(canvas) == expected
