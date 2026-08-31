# DetectKit Overlay Layer Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `OBBCanvas`'s three hand-maintained overlay layers (parallel item lists, per-layer draw/clear/visibility methods, per-layer branches in `_apply_visibility`) with one keyed registry of `OverlayLayer` value objects, and move "what ground truth / predictions / staged escalations *are*" out of the canvas into three small providers — with pixel-identical rendering.

**Architecture:** A new Qt-light `gui/overlays/layer.py` defines `OverlayLayer` (plus `ColourPolicy`, `LabelMode`, `LayerStyle`, `Emphasis`, `InstanceRef`). `OBBCanvas` becomes a pure renderer holding `self._layers: dict[str, OverlayLayer]` and `self._items: dict[(key, GeometryLevel), _LevelItems]`, with a four-method public surface. A new `gui/overlays/providers.py` holds `GroundTruthProvider`, `PredictionProvider`, `StagedEscalationProvider`, each answering "given this frame, what layer should be drawn?". `MainWindow.show_image` builds a `FrameContext` and asks each provider.

**Tech Stack:** Python 3.11+, PySide6 (`QGraphicsView`/`QGraphicsScene`), pytest with the `qapp` fixture, conda env `hydra-mps`.

**Spec:** `docs/superpowers/specs/2026-08-31-detectkit-overlay-layer-registry-design.md`

**Scheduling note:** the spec defers itself until a fourth layer or the first per-instance interaction arrives. Neither has. It is scheduled here by explicit user request, ahead of the frame-granular review work it was originally sequenced behind. Expect the *frame-granular review* spec (`2026-08-31-detectkit-frame-granular-review-design.md`) to need amending afterwards, not the other way round.

## Global Constraints

- **Rendering must be pixel-identical.** This is a structural refactor. Every pen colour/style/width, brush style/alpha, polygon point, label string, and visibility flag on every drawn item stays exactly as it is today. Task 1 builds the oracle that proves it; every later task re-runs it.
- **The oracle is a committed golden file, not an in-process before/after comparison.** A before/after check written after the refactor is tautological — the trap recorded in `project_shared_engine_param_builder`. Task 1 commits the golden *before* any production code changes.
- **The canvas stays read-only.** No shape editing, no mouse interaction on items.
- **The class filter stays a per-layer boolean.** It can only ever address project class ids, so it is `class_filtered: bool`, never a pluggable predicate.
- **The per-class palette and the fixed-hue policy stay separate.** A class colour and a "this is a proposal" colour are different kinds of statement.
- **No equivalence gate applies.** This is GUI-only and touches nothing on the tracking pipeline path. Do **not** run `tools/equivalence/run_matrix.sh`.
- **Environment:** `source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps` before running tests or `black`/`isort`. `black` is broken in the base env.
- **Formatting:** run `black` and `isort` **only on the paths you touched**, never `make format` (it reformats unrelated files).
- Work happens in the worktree `.worktrees/overlay-registry` on branch `feat/detectkit-overlay-registry`. All paths below are relative to it.

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `src/hydra_suite/detectkit/gui/overlays/__init__.py` | Re-exports the public names from `layer.py` and `providers.py`. |
| `src/hydra_suite/detectkit/gui/overlays/layer.py` | The `OverlayLayer` value object and its enums. No canvas, no project, no I/O. |
| `src/hydra_suite/detectkit/gui/overlays/providers.py` | `FrameContext` + the three providers. Knows the domain; knows nothing about `QGraphicsScene`. |
| `tests/test_detectkit_overlay_golden.py` | The characterization oracle + the visibility-toggle matrix. |
| `tests/detectkit_overlay_golden.json` | The committed golden data. |
| `tests/test_detectkit_overlay_layer.py` | Unit tests for the value object's invariants. |
| `tests/test_detectkit_overlay_providers.py` | Behavioural provider tests (replacing today's `inspect.getsource` assertions). |

**Modified:**

| File | Change |
|---|---|
| `src/hydra_suite/detectkit/gui/canvas.py` | Registry rewrite; old method families deleted in Task 8. |
| `src/hydra_suite/detectkit/gui/main_window.py` | Four call sites rewired to providers. |
| `src/hydra_suite/detectkit/gui/dialogs/semantic_frame_preview_dialog.py` | Two-layer draw rewired. |
| `src/hydra_suite/detectkit/gui/dialogs/calibration_results_dialog.py` | Same. |
| `tests/test_detectkit_canvas.py`, `tests/test_detectkit_canvas_dual_layer.py`, `tests/test_detectkit_staged_escalation_overlay.py`, `tests/test_detectkit_show_image_multi_level.py`, `tests/test_detectkit_tools_panel.py` | Rewritten against the new API. |

**Sequencing rule that makes every task independently green:** Task 3 introduces the registry *underneath* the existing public methods, reimplementing them as thin adapters. Tasks 5–7 migrate callers one at a time. Task 8 deletes the adapters. The spec's "retired, not wrapped" is satisfied at the end of the branch; the adapters never survive it. Without this, Task 3 would break the app and six call sites in one unreviewable commit.

---

### Task 1: The golden characterization oracle

The riskiest change in this refactor is a silent visual regression on the path that renders every frame. Build the detector first, commit it, and never touch it again.

**Files:**
- Create: `tests/test_detectkit_overlay_golden.py`
- Create: `tests/detectkit_overlay_golden.json` (generated by a step below, then committed)

**Interfaces:**
- Consumes: nothing (this task runs against `main`'s canvas API).
- Produces: `describe_scene(canvas) -> list[dict]` — imported by Tasks 3–8 to re-assert the golden. Each dict has keys `level_hint`, `type` (`"polygon"` | `"label"`), `visible`, `z`, and for polygons `points`, `pen_colour`, `pen_style`, `pen_width`, `brush_style`, `brush_alpha`; for labels `text`, `colour`, `pos`.

- [ ] **Step 1: Write the scene-description helper and the three fixture scenes**

Create `tests/test_detectkit_overlay_golden.py`:

```python
"""Golden characterization of every DetectKit overlay layer.

This file is the gate for the overlay-registry refactor: it records what
the canvas draws BEFORE the refactor and asserts the refactor reproduces
it byte-for-byte. The golden lives in a committed JSON file, not in a
before-vs-after comparison inside one process -- a post-refactor oracle
built from the refactored code proves only that the code equals itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QGraphicsPolygonItem, QGraphicsTextItem

from hydra_suite.detectkit.gui.canvas import OBBCanvas
from hydra_suite.training.geometry_levels import GeometryLevel

GOLDEN = Path(__file__).parent / "detectkit_overlay_golden.json"


def describe_scene(canvas: OBBCanvas) -> list[dict]:
    """Serialise every overlay item in the scene, order-independently."""
    out: list[dict] = []
    for item in canvas._scene.items():
        if isinstance(item, QGraphicsPolygonItem):
            poly = item.polygon()
            pen = item.pen()
            brush = item.brush()
            out.append(
                {
                    "type": "polygon",
                    "points": [
                        [round(poly.at(i).x(), 4), round(poly.at(i).y(), 4)]
                        for i in range(poly.count())
                    ],
                    "pen_colour": pen.color().name(),
                    "pen_style": int(pen.style().value),
                    "pen_width": pen.width(),
                    "brush_style": int(brush.style().value),
                    "brush_alpha": brush.color().alpha(),
                    "visible": item.isVisible(),
                    "z": item.zValue(),
                }
            )
        elif isinstance(item, QGraphicsTextItem):
            out.append(
                {
                    "type": "label",
                    "text": item.toPlainText(),
                    "colour": item.defaultTextColor().name(),
                    "pos": [round(item.pos().x(), 4), round(item.pos().y(), 4)],
                    "visible": item.isVisible(),
                    "z": item.zValue(),
                }
            )
    # QGraphicsScene.items() order is not part of the contract we are
    # freezing; only the SET of drawn items and their properties is.
    return sorted(out, key=lambda d: json.dumps(d, sort_keys=True))


_GT = [
    {"class_id": 0, "polygon_px": [(10, 10), (60, 12), (58, 40), (8, 38)]},
    {"class_id": 1, "polygon_px": [(100, 100), (150, 105), (145, 140), (95, 135)]},
]
_PRED = [
    {"class_id": 0, "polygon_px": [(12, 11), (62, 13), (60, 41), (10, 39)],
     "confidence": 0.87},
    {"class_id": 3, "polygon_px": [(200, 200), (240, 200), (240, 230), (200, 230)]},
]
_ESC = [
    {"class_id": 0, "polygon_px": [(20, 20), (70, 22), (68, 50), (18, 48)]},
    {"class_id": 7, "polygon_px": [(300, 300), (360, 310), (350, 350), (295, 340)]},
]
_NAMES = ["ant", "worker", "queen", "larva"]
```

- [ ] **Step 2: Write the scene builders — one per rendering situation that exists today**

Append to the same file. Each builder reproduces exactly one production call pattern; the comment names the caller so a future reader can tell whether a situation is still live.

```python
def _build_main_window_scene(canvas: OBBCanvas) -> None:
    """main_window.show_image + _refresh_prediction_overlay +
    _refresh_escalation_overlay: multi-level GT (reviewed), single-level
    dashed predictions, multi-level staged escalation."""
    canvas.set_gt_detections_multi_level(
        _GT, class_names=_NAMES, native_level=GeometryLevel.POLYGON, reviewed=True
    )
    canvas.set_pred_detections(_PRED, class_names=_NAMES)
    canvas.set_escalation_detections(
        _ESC, class_names=["prompt_a", "prompt_b"], native_level=GeometryLevel.OBB
    )


def _build_unreviewed_scene(canvas: OBBCanvas) -> None:
    """show_image when _resolve_source_render_state says reviewed=False:
    the native level gets the BDiagPattern hatch, keeping its own pen."""
    canvas.set_gt_detections_multi_level(
        _GT, class_names=_NAMES, native_level=GeometryLevel.OBB, reviewed=False
    )


def _build_dialog_scene(canvas: OBBCanvas) -> None:
    """semantic_frame_preview_dialog.py:131-132 and
    calibration_results_dialog.py:315-316: single-level GT and predictions
    with explicit fills, dict class_names, no level derivation."""
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
```

- [ ] **Step 3: Write the golden generator and the assertion test**

Append:

```python
def _render(name: str) -> list[dict]:
    canvas = OBBCanvas()
    SCENES[name](canvas)
    return describe_scene(canvas)


def regenerate_golden() -> None:
    """Run via: python -c 'import tests.test_detectkit_overlay_golden as t;
    t.regenerate_golden()'  -- ONLY on pre-refactor code."""
    GOLDEN.write_text(
        json.dumps({name: _render(name) for name in SCENES}, indent=2) + "\n"
    )


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_overlay_rendering_matches_the_committed_golden(qapp, scene):
    expected = json.loads(GOLDEN.read_text())[scene]
    assert _render(scene) == expected
```

- [ ] **Step 4: Generate the golden against unmodified canvas code**

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps
cd .worktrees/overlay-registry
git status --porcelain src/  # MUST be empty: the golden is only valid
                             # if it was recorded from untouched code
QT_QPA_PLATFORM=offscreen python -c "
import tests.test_detectkit_overlay_golden as t
from PySide6.QtWidgets import QApplication; QApplication([])
t.regenerate_golden()"
```

- [ ] **Step 5: Sanity-check the golden is not empty or degenerate**

```bash
python -c "
import json; g=json.load(open('tests/detectkit_overlay_golden.json'))
for k,v in g.items(): print(k, len(v))
assert all(len(v) > 0 for v in g.values())
assert len(g['main_window']) > len(g['aabb_native'])
"
```
Expected: four scenes, all non-empty. `main_window` (three layers, multiple derived levels) must have strictly more items than `aabb_native` (one layer, one level). A golden of all-zeros would pass every later assertion while proving nothing — this is the same failure mode as an empty-CSV equivalence pass.

- [ ] **Step 6: Run the test to verify it passes on unmodified code**

Run: `python -m pytest tests/test_detectkit_overlay_golden.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
black tests/test_detectkit_overlay_golden.py
isort tests/test_detectkit_overlay_golden.py
git add tests/test_detectkit_overlay_golden.py tests/detectkit_overlay_golden.json
git commit -m "test(detectkit): characterize overlay rendering before the registry refactor"
```

---

### Task 2: The `OverlayLayer` value object

**Files:**
- Create: `src/hydra_suite/detectkit/gui/overlays/__init__.py`
- Create: `src/hydra_suite/detectkit/gui/overlays/layer.py`
- Test: `tests/test_detectkit_overlay_layer.py`

**Interfaces:**
- Consumes: `hydra_suite.training.geometry_levels.GeometryLevel`.
- Produces: `ColourPolicy`, `LabelMode`, `Emphasis`, `LayerStyle`, `OverlayLayer`, `InstanceRef` — all importable from `hydra_suite.detectkit.gui.overlays`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_detectkit_overlay_layer.py`:

```python
"""Invariants of the OverlayLayer value object.

The point of the value object is that a layer's semantics are STATED, not
implied by which of three set_* methods a caller happened to pick. These
tests pin the combinations that are contradictory, so a provider cannot
construct one.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from hydra_suite.detectkit.gui.overlays import (
    ColourPolicy,
    InstanceRef,
    LabelMode,
    LayerStyle,
    OverlayLayer,
)
from hydra_suite.training.geometry_levels import GeometryLevel

_DET = [{"class_id": 0, "polygon_px": [(0, 0), (10, 0), (10, 10), (0, 10)]}]


def _layer(**kw):
    base = dict(
        key="gt",
        detections=_DET,
        native_level=GeometryLevel.POLYGON,
        class_names=["ant"],
        colour_policy=ColourPolicy.PER_CLASS,
    )
    base.update(kw)
    return OverlayLayer(**base)


def test_fixed_colour_policy_requires_a_colour():
    with pytest.raises(ValueError, match="fixed_colour"):
        _layer(colour_policy=ColourPolicy.FIXED)


def test_per_class_policy_rejects_a_fixed_colour():
    """Supplying both would leave which one wins to the renderer."""
    with pytest.raises(ValueError, match="fixed_colour"):
        _layer(colour_policy=ColourPolicy.PER_CLASS, fixed_colour=QColor("red"))


def test_an_explicit_style_forbids_level_derivation():
    """A single LayerStyle cannot describe three derived levels, each of
    which has its own pen and brush."""
    style = LayerStyle(Qt.PenStyle.SolidLine, Qt.BrushStyle.SolidPattern, 65)
    with pytest.raises(ValueError, match="derive_levels"):
        _layer(style=style, derive_levels=True)


def test_defaults_describe_the_ground_truth_layer():
    layer = _layer()
    assert layer.derive_levels is True
    assert layer.style is None
    assert layer.class_filtered is True
    assert layer.label_mode is LabelMode.NAME_AND_CLASS_ID
    assert layer.emphasis is None
    assert layer.z == 0


def test_the_layer_is_frozen():
    layer = _layer()
    with pytest.raises(Exception):
        layer.key = "other"


def test_instance_ref_is_hashable_and_carries_the_provenance_triple():
    ref = InstanceRef(layer_key="escalation", frame_key="a/b.png", index=3)
    assert hash(ref)
    assert (ref.layer_key, ref.frame_key, ref.index) == ("escalation", "a/b.png", 3)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_detectkit_overlay_layer.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'hydra_suite.detectkit.gui.overlays'`.

- [ ] **Step 3: Write the implementation**

Create `src/hydra_suite/detectkit/gui/overlays/layer.py`:

```python
"""A DetectKit canvas overlay layer, as a value object.

Before this existed, a layer's semantics -- whether it is class-filtered,
which colour policy it uses, whether its labels carry confidence, what it
stacks above -- were encoded in WHICH of three set_* methods you called,
and adding a layer meant editing five places in canvas.py. Two of the four
findings an adversarial review raised against the third layer were
layer-bookkeeping defects of exactly that kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

if TYPE_CHECKING:
    from hydra_suite.training.geometry_levels import GeometryLevel


class ColourPolicy(Enum):
    PER_CLASS = auto()  # index the palette by class_id
    FIXED = auto()      # one hue for the whole layer


class LabelMode(Enum):
    NAME_AND_CLASS_ID = auto()     # "ant (0)"
    NAME_AND_CONFIDENCE = auto()   # "ant (0.42)", or bare "ant" when absent


class Emphasis(Enum):
    UNREVIEWED = auto()  # hatch the native level; keep its own pen style


@dataclass(frozen=True)
class LayerStyle:
    pen_style: "Qt.PenStyle"
    brush_style: "Qt.BrushStyle"
    fill_alpha: int  # 0-255; ignored when brush_style is NoBrush


@dataclass(frozen=True)
class InstanceRef:
    """Identifies one drawn shape back to its source line.

    Nothing consumes this yet. It is stamped now because retrofitting
    identity later means revisiting every provider and every draw path,
    whereas adding it while those paths are being rewritten is nearly free.
    It is the precondition for per-instance accept/reject.
    """

    layer_key: str
    frame_key: str | None
    index: int


@dataclass(frozen=True)
class OverlayLayer:
    key: str
    detections: list[dict]
    native_level: "GeometryLevel"
    class_names: list[str] | dict[int, str] | None
    colour_policy: ColourPolicy
    fixed_colour: Optional[QColor] = None
    z: int = 0
    class_filtered: bool = True
    label_mode: LabelMode = LabelMode.NAME_AND_CLASS_ID
    emphasis: Optional[Emphasis] = None
    derive_levels: bool = True
    style: Optional[LayerStyle] = None
    frame_key: str | None = None

    def __post_init__(self) -> None:
        if self.colour_policy is ColourPolicy.FIXED and self.fixed_colour is None:
            raise ValueError("ColourPolicy.FIXED requires fixed_colour")
        if self.colour_policy is ColourPolicy.PER_CLASS and self.fixed_colour is not None:
            raise ValueError(
                "fixed_colour is meaningless under ColourPolicy.PER_CLASS"
            )
        if self.style is not None and self.derive_levels:
            raise ValueError(
                "an explicit style applies to the native level only; "
                "set derive_levels=False"
            )
```

Create `src/hydra_suite/detectkit/gui/overlays/__init__.py`:

```python
"""Overlay layer value objects and per-source providers for DetectKit."""

from .layer import (
    ColourPolicy,
    Emphasis,
    InstanceRef,
    LabelMode,
    LayerStyle,
    OverlayLayer,
)

__all__ = [
    "ColourPolicy",
    "Emphasis",
    "InstanceRef",
    "LabelMode",
    "LayerStyle",
    "OverlayLayer",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_detectkit_overlay_layer.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
black src/hydra_suite/detectkit/gui/overlays tests/test_detectkit_overlay_layer.py
isort src/hydra_suite/detectkit/gui/overlays tests/test_detectkit_overlay_layer.py
git add src/hydra_suite/detectkit/gui/overlays tests/test_detectkit_overlay_layer.py
git commit -m "feat(detectkit): add the OverlayLayer value object"
```

---

### Task 3: The canvas registry, with the old methods as adapters

The canvas gains its four-method registry surface. Every existing public method is reimplemented on top of it, so the app and all six call sites keep working and the golden keeps passing. Task 8 deletes the adapters.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/canvas.py`
- Test: `tests/test_detectkit_overlay_golden.py` (unchanged — it is the gate)

**Interfaces:**
- Consumes: `OverlayLayer`, `ColourPolicy`, `LabelMode`, `LayerStyle`, `Emphasis`, `InstanceRef` from Task 2.
- Produces:
  - `OBBCanvas.set_layer(layer: OverlayLayer) -> None` — add or replace by `layer.key`; idempotent.
  - `OBBCanvas.remove_layer(key: str) -> None` — no-op if absent.
  - `OBBCanvas.set_layer_visible(key: str, visible: bool) -> None` — remembered even for a key not yet drawn.
  - `OBBCanvas.set_class_filter(visible_class_ids: set[int]) -> None` — unchanged signature.
  - `OBBCanvas.set_derived_levels_visible(visible: bool) -> None` — unchanged; the flag stays global.
  - Layer keys: `"gt"` (z=0), `"pred"` (z=10), `"escalation"` (z=20).

- [ ] **Step 1: Write the failing test for the registry surface**

Append to `tests/test_detectkit_overlay_golden.py`:

```python
from hydra_suite.detectkit.gui.overlays import (
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
    canvas.set_layer(
        OverlayLayer(
            key="pred",
            detections=_PRED,
            native_level=GeometryLevel.POLYGON,
            class_names=_NAMES,
            colour_policy=ColourPolicy.PER_CLASS,
            derive_levels=False,
            style=None,
            label_mode=LabelMode.NAME_AND_CONFIDENCE,
            z=10,
        )
    )
    assert len(canvas._scene.items()) > gt_only
    canvas.remove_layer("pred")
    assert len(canvas._scene.items()) == gt_only


def test_remove_layer_is_a_no_op_for_an_unknown_key(qapp):
    canvas = OBBCanvas()
    canvas.remove_layer("nope")  # must not raise


def test_set_layer_visible_is_remembered_before_the_layer_is_drawn(qapp):
    """_on_overlay_changed fires before show_image on startup. A visibility
    call for a not-yet-drawn key must not be silently lost."""
    canvas = OBBCanvas()
    canvas.set_layer_visible("gt", False)
    canvas.set_layer(_gt_layer())
    assert all(
        not i["visible"] for i in describe_scene(canvas) if i["type"] == "polygon"
    )


def test_every_drawn_item_carries_an_instance_ref(qapp):
    canvas = OBBCanvas()
    canvas.set_layer(_gt_layer(frame_key="frames/0001.png"))
    refs = [
        i.data(0)
        for i in canvas._scene.items()
        if i.data(0) is not None
    ]
    assert refs
    assert {r.layer_key for r in refs} == {"gt"}
    assert {r.frame_key for r in refs} == {"frames/0001.png"}
    assert set(r.index for r in refs) <= {0, 1}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_detectkit_overlay_golden.py -v`
Expected: the 4 golden tests PASS; the 5 new tests FAIL with `AttributeError: 'OBBCanvas' object has no attribute 'set_layer'`.

- [ ] **Step 3: Replace the canvas's layer state and draw path**

In `src/hydra_suite/detectkit/gui/canvas.py`, replace the block in `__init__` that defines `_gt_*`, `_esc_*`, `_pred_*`, `_show_gt`, `_show_pred`, `_show_escalation`, `_obb_items`, `_label_items` (currently lines 106–140) with:

```python
        self._layers: dict[str, OverlayLayer] = {}
        # (layer_key, level) -> _LevelItems. One flat registry; there is no
        # per-layer branch anywhere below it.
        self._items: dict[tuple, _LevelItems] = {}
        self._layer_visible: dict[str, bool] = {}
        self._show_derived_levels: bool = True
        self._visible_class_ids: set[int] = set()
```

Add near `_LevelStyle`:

```python
@dataclass
class _LevelItems:
    """The scene items one layer drew at one geometry level."""

    obb_items: list
    label_items: list
    class_ids: list[int]
```

Add the resolved-style helper beside `_level_styles()`:

```python
def _styles_for(layer: "OverlayLayer", level, native_level) -> _LevelStyle:
    """The pen/brush a layer uses at one level.

    An explicit ``layer.style`` wins outright (single-level layers such as
    the dialogs' filled GT and the dashed prediction layer). Otherwise the
    per-level defaults apply, with Emphasis.UNREVIEWED substituting a
    hatched fill on the NATIVE level only -- and keeping that level's own
    pen style, because hardcoding SolidLine there once made an unreviewed
    OBB-native quad indistinguishable from its derived AABB outline.
    """
    if layer.style is not None:
        return _LevelStyle(
            layer.style.pen_style, layer.style.brush_style, layer.style.fill_alpha
        )
    style = _level_styles()[level]
    if layer.emphasis is Emphasis.UNREVIEWED and level == native_level:
        return _LevelStyle(style.pen_style, Qt.BrushStyle.BDiagPattern, 140)
    return style
```

- [ ] **Step 4: Write `set_layer` / `remove_layer` / `set_layer_visible` and the single-loop `_apply_visibility`**

Add to `OBBCanvas`:

```python
    def set_layer(self, layer: "OverlayLayer") -> None:
        """Add or replace the layer with ``layer.key``.

        Idempotent by key: this removes that key's items before redrawing,
        which is what makes "clear before refresh" structural instead of a
        rule every caller has to remember.
        """
        self.remove_layer(layer.key)
        self._layers[layer.key] = layer

        if layer.derive_levels:
            level_iter = self._levels_with_shapes(
                layer.detections, layer.native_level
            )
        else:
            level_iter = [(layer.native_level, layer.detections)]

        for level, level_detections in level_iter:
            style = _styles_for(layer, level, layer.native_level)
            items = _LevelItems([], [], [])
            self._draw_detections(
                level_detections,
                items,
                layer,
                style,
                show_labels=(level == layer.native_level),
            )
            self._items[(layer.key, level)] = items
        self._apply_visibility()

    def remove_layer(self, key: str) -> None:
        """Remove one layer's items. No-op if the key was never drawn."""
        for (layer_key, _level), items in list(self._items.items()):
            if layer_key != key:
                continue
            for item in items.obb_items:
                if item is not None:
                    self._scene.removeItem(item)
            for item in items.label_items:
                if item is not None:
                    self._scene.removeItem(item)
        self._items = {k: v for k, v in self._items.items() if k[0] != key}
        self._layers.pop(key, None)

    def set_layer_visible(self, key: str, visible: bool) -> None:
        """Toggle a layer. Remembered even for a key not yet drawn --
        _on_overlay_changed can fire before the first show_image."""
        self._layer_visible[key] = bool(visible)
        self._apply_visibility()

    def _apply_visibility(self) -> None:
        """One loop over the registry. No per-layer branch."""
        for (layer_key, level), items in self._items.items():
            layer = self._layers.get(layer_key)
            if layer is None:
                continue
            layer_visible = self._layer_visible.get(layer_key, True) and (
                level == layer.native_level or self._show_derived_levels
            )
            for obb, lbl, cid in zip(
                items.obb_items, items.label_items, items.class_ids
            ):
                visible = layer_visible and (
                    not layer.class_filtered
                    or not self._visible_class_ids
                    or cid in self._visible_class_ids
                )
                obb.setVisible(visible)
                if lbl is not None:
                    lbl.setVisible(visible)
```

- [ ] **Step 5: Rewrite `_draw_detections` against the layer, stamping `InstanceRef` and `z`**

Replace the whole `_draw_detections` body with:

```python
    def _draw_detections(
        self,
        detections: list[dict],
        items: "_LevelItems",
        layer: "OverlayLayer",
        style: "_LevelStyle",
        *,
        show_labels: bool = True,
    ) -> None:
        font = QFont()
        font.setPixelSize(DEFAULT_OBB_FONT_SIZE)
        lookup = self._build_class_lookup(layer.class_names)

        for index, det in enumerate(detections):
            class_id: int = det.get("class_id", 0)
            polygon_px = det.get("polygon_px", [])
            if len(polygon_px) < 3:
                continue
            confidence = det.get("confidence", None)

            # A FIXED-policy layer paints one hue because its class ids do
            # not address the project's classes: staged escalation ids index
            # the STAGING dir's classes.txt (the prompt), so indexing the
            # palette with them would assert a class identity the staged
            # labels do not carry.
            colour = (
                layer.fixed_colour
                if layer.colour_policy is ColourPolicy.FIXED
                else _PALETTE[class_id % len(_PALETTE)]
            )
            qpoly = QPolygonF()
            for x, y in polygon_px:
                qpoly.append(QPointF(x, y))
            qpoly.append(QPointF(*polygon_px[0]))

            pen = QPen(colour, DEFAULT_OBB_LINE_WIDTH)
            pen.setCosmetic(True)
            pen.setStyle(style.pen_style)

            if style.brush_style != Qt.BrushStyle.NoBrush:
                fill_colour = QColor(colour)
                fill_colour.setAlpha(style.fill_alpha)
                brush = QBrush(fill_colour, style.brush_style)
            else:
                brush = QBrush(Qt.BrushStyle.NoBrush)

            ref = InstanceRef(layer.key, layer.frame_key, index)
            poly_item = self._scene.addPolygon(qpoly, pen, brush)
            poly_item.setZValue(layer.z)
            poly_item.setData(0, ref)
            items.obb_items.append(poly_item)
            items.class_ids.append(class_id)

            if not show_labels:
                items.label_items.append(None)
                continue

            label_name = lookup.get(class_id, f"class_{class_id}")
            if layer.label_mode is LabelMode.NAME_AND_CONFIDENCE:
                # A layer that asked for confidence and has none must NOT
                # fall back to the class id: "(0)" beside a mask reads as a
                # confidence of 0.00. Staged escalation labels carry no
                # confidence -- data/al/labels.py writes class id + coords
                # only -- so this is the live path for that layer.
                label_text = (
                    f"{label_name} ({confidence:.2f})"
                    if confidence is not None
                    else label_name
                )
            else:
                label_text = f"{label_name} ({class_id})"
            txt_item = QGraphicsTextItem(label_text)
            txt_item.setFont(font)
            txt_item.setDefaultTextColor(colour)
            txt_item.setPos(QPointF(*polygon_px[0]))
            txt_item.setZValue(layer.z)
            txt_item.setData(0, ref)
            self._scene.addItem(txt_item)
            items.label_items.append(txt_item)
```

Add the imports at the top of `canvas.py`:

```python
from .overlays import (
    ColourPolicy,
    Emphasis,
    InstanceRef,
    LabelMode,
    LayerStyle,
    OverlayLayer,
)
```

- [ ] **Step 6: Reimplement every old public method as an adapter**

Replace `set_gt_detections`, `set_gt_detections_multi_level`, `set_pred_detections`, `set_escalation_detections`, `set_escalation_visible`, `set_overlay_visibility`, `clear_gt_detections`, `clear_pred_detections`, `clear_escalation_detections`, `set_detections`, `clear_detections` with:

```python
    # ------------------------------------------------------------------
    # TRANSITIONAL ADAPTERS -- deleted in the final task of this refactor.
    # They exist only so the six call sites can migrate one at a time
    # instead of in one unreviewable commit.
    # ------------------------------------------------------------------

    def set_gt_detections(
        self,
        detections: list[dict],
        class_names=None,
        *,
        fill_alpha: int = 0,
    ) -> None:
        self.set_layer(
            OverlayLayer(
                key="gt",
                detections=detections,
                native_level=self._aabb_level(),
                class_names=class_names,
                colour_policy=ColourPolicy.PER_CLASS,
                derive_levels=False,
                style=LayerStyle(
                    Qt.PenStyle.SolidLine,
                    (
                        Qt.BrushStyle.SolidPattern
                        if fill_alpha > 0
                        else Qt.BrushStyle.NoBrush
                    ),
                    fill_alpha,
                ),
                label_mode=LabelMode.NAME_AND_CLASS_ID,
                z=0,
            )
        )

    def set_gt_detections_multi_level(
        self, detections: list[dict], class_names=None, *, native_level, reviewed=True
    ) -> None:
        self.set_layer(
            OverlayLayer(
                key="gt",
                detections=detections,
                native_level=native_level,
                class_names=class_names,
                colour_policy=ColourPolicy.PER_CLASS,
                label_mode=LabelMode.NAME_AND_CLASS_ID,
                emphasis=None if reviewed else Emphasis.UNREVIEWED,
                z=0,
            )
        )

    def set_pred_detections(
        self, detections: list[dict], class_names=None, *, fill_alpha: int = 0
    ) -> None:
        self.set_layer(
            OverlayLayer(
                key="pred",
                detections=detections,
                native_level=self._aabb_level(),
                class_names=class_names,
                colour_policy=ColourPolicy.PER_CLASS,
                derive_levels=False,
                style=LayerStyle(
                    Qt.PenStyle.DashLine,
                    (
                        Qt.BrushStyle.SolidPattern
                        if fill_alpha > 0
                        else Qt.BrushStyle.NoBrush
                    ),
                    fill_alpha,
                ),
                label_mode=LabelMode.NAME_AND_CONFIDENCE,
                z=10,
            )
        )

    def set_escalation_detections(
        self, detections: list[dict], class_names=None, *, native_level
    ) -> None:
        self.set_layer(
            OverlayLayer(
                key="escalation",
                detections=detections,
                native_level=native_level,
                class_names=class_names,
                colour_policy=ColourPolicy.FIXED,
                fixed_colour=ESCALATION_COLOUR,
                class_filtered=False,
                label_mode=LabelMode.NAME_AND_CONFIDENCE,
                z=20,
            )
        )

    def set_escalation_visible(self, visible: bool) -> None:
        self.set_layer_visible("escalation", visible)

    def set_overlay_visibility(self, show_gt: bool, show_pred: bool) -> None:
        self.set_layer_visible("gt", show_gt)
        self.set_layer_visible("pred", show_pred)

    def clear_gt_detections(self) -> None:
        self.remove_layer("gt")

    def clear_pred_detections(self) -> None:
        self.remove_layer("pred")

    def clear_escalation_detections(self) -> None:
        self.remove_layer("escalation")

    def set_detections(self, detections: list[dict], class_names=None) -> None:
        self.set_gt_detections(detections, class_names)

    def clear_detections(self) -> None:
        self.remove_layer("gt")

    @staticmethod
    def _aabb_level():
        """The native level a single-level layer declares.

        Single-level layers never derive, so the value only has to satisfy
        `level == native_level` (so the layer is drawn and labelled, and
        stays visible when derived levels are hidden). AABB is the floor of
        the ordering, so it can never be mistaken for a derived level.
        """
        from hydra_suite.training.geometry_levels import GeometryLevel

        return GeometryLevel.AABB
```

Also update `clear_all` to reset the registry:

```python
    def clear_all(self) -> None:
        """Remove everything from the scene."""
        self._scene.clear()
        self._pix_item = None
        self._layers.clear()
        self._items.clear()
        self._zoom = 1.0
        self._fit_mode = True
        self.setCursor(Qt.CursorShape.ArrowCursor)
```

Note `_layer_visible` is deliberately **not** cleared: it mirrors the user's checkbox state, which survives a scene reset.

- [ ] **Step 7: Delete `set_gt_detections(append=True)` and update its two tests**

`append=True` has no production caller in `src/`. Verify:

```bash
grep -rn --include='*.py' "append=" src/hydra_suite/detectkit
```
Expected: no hit involving `set_gt_detections`.

Delete the `append` parameter (already absent from the adapter above), then in `tests/test_detectkit_canvas_dual_layer.py` replace lines 114-116:

```python
    canvas.set_gt_detections(_DET + _DET2)  # class ids 0 and 1 in one call
    canvas.set_class_filter({0})
```

and delete `test_set_gt_detections_append_after_multi_level_is_visibility_controlled` from `tests/test_detectkit_canvas.py` — the bug it guarded (appended items falling outside `_apply_visibility`'s iteration) cannot exist once every layer lives in the one registry.

- [ ] **Step 8: Run the golden and the whole DetectKit canvas suite**

```bash
python -m pytest tests/test_detectkit_overlay_golden.py \
  tests/test_detectkit_canvas.py tests/test_detectkit_canvas_dual_layer.py \
  tests/test_detectkit_show_image_multi_level.py \
  tests/test_detectkit_staged_escalation_overlay.py \
  tests/test_detectkit_tools_panel.py -v
```
Expected: the 4 golden scenes PASS unchanged — this is the whole point of the task. The 5 new registry tests PASS.

`tests/test_detectkit_staged_escalation_overlay.py:106,119` assert on `inspect.getsource(OBBCanvas.set_escalation_detections)` containing `_levels_with_shapes` and a `clear_` call. Those methods are now one-line adapters, so the assertions are false by construction. **Delete those two tests now** (`test_escalation_overlay_derives_the_levels_beneath_its_native_one` and the clear-ordering test); their behaviour is covered by the golden `main_window` scene and by `test_set_layer_replaces_by_key_instead_of_accumulating`.

- [ ] **Step 9: If the golden fails, do not edit the golden**

A golden mismatch means the refactor changed rendering. Diff it:

```bash
QT_QPA_PLATFORM=offscreen python -c "
import json, tests.test_detectkit_overlay_golden as t
from PySide6.QtWidgets import QApplication; QApplication([])
exp = json.load(open('tests/detectkit_overlay_golden.json'))
for name in t.SCENES:
    got = t._render(name)
    if got != exp[name]:
        print('=== ', name)
        for a, b in zip(exp[name], got):
            if a != b: print(' exp', a); print(' got', b)
        print(' counts', len(exp[name]), len(got))
"
```

The three most likely causes, in order: the `_aabb_level()` choice changing which single-level items count as native (label suppression); `Emphasis.UNREVIEWED` applied at the wrong level; `z` values added where items previously had `zValue()==0`. **`z` is the one legitimate golden change** — the pre-refactor scene has no explicit z. If the ONLY diffs are `"z"` keys, regenerate the golden *and say so in the commit message*; if any other key differs, fix the code.

- [ ] **Step 10: Commit**

```bash
black src/hydra_suite/detectkit/gui/canvas.py tests/
isort src/hydra_suite/detectkit/gui/canvas.py tests/
git add -A
git commit -m "refactor(detectkit): make OBBCanvas a keyed layer registry"
```

---

### Task 4: The visibility-toggle matrix

Today's tests cover the toggles one at a time. The registry rewrite is exactly where an untested combination regresses.

**Files:**
- Modify: `tests/test_detectkit_overlay_golden.py`

**Interfaces:**
- Consumes: `describe_scene`, `SCENES`, `_gt_layer` from Tasks 1 and 3.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_detectkit_overlay_golden.py`:

```python
import itertools


def _full_scene(canvas: OBBCanvas) -> None:
    _build_main_window_scene(canvas)


@pytest.mark.parametrize(
    "show_gt,show_pred,show_esc,show_derived,class_filter",
    list(
        itertools.product(
            [True, False], [True, False], [True, False], [True, False],
            [set(), {0}],
        )
    ),
)
def test_visibility_matrix(
    qapp, show_gt, show_pred, show_esc, show_derived, class_filter
):
    canvas = OBBCanvas()
    _full_scene(canvas)
    canvas.set_layer_visible("gt", show_gt)
    canvas.set_layer_visible("pred", show_pred)
    canvas.set_layer_visible("escalation", show_esc)
    canvas.set_derived_levels_visible(show_derived)
    canvas.set_class_filter(class_filter)

    want = {"gt": show_gt, "pred": show_pred, "escalation": show_esc}
    for (key, level), items in canvas._items.items():
        layer = canvas._layers[key]
        for obb, cid in zip(items.obb_items, items.class_ids):
            expected = (
                want[key]
                and (level == layer.native_level or show_derived)
                and (
                    not layer.class_filtered
                    or not class_filter
                    or cid in class_filter
                )
            )
            assert obb.isVisible() == expected, (key, level, cid)


def test_the_class_filter_never_hides_the_escalation_layer(qapp):
    """Staged class ids index the STAGING dir's classes.txt, not the
    project's class list, so the project filter cannot address them."""
    canvas = OBBCanvas()
    _full_scene(canvas)
    canvas.set_class_filter({999})
    esc = [
        obb
        for (key, _lvl), items in canvas._items.items()
        if key == "escalation"
        for obb in items.obb_items
    ]
    assert esc and all(o.isVisible() for o in esc)
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_detectkit_overlay_golden.py -v -k "matrix or class_filter"`
Expected: 33 passed (32 matrix combinations + 1). If any fail, the fault is in `_apply_visibility` — fix the code, not the assertion.

- [ ] **Step 3: Commit**

```bash
git add tests/test_detectkit_overlay_golden.py
git commit -m "test(detectkit): cover the full overlay visibility toggle matrix"
```

---

### Task 5: The providers

**Files:**
- Create: `src/hydra_suite/detectkit/gui/overlays/providers.py`
- Modify: `src/hydra_suite/detectkit/gui/overlays/__init__.py`
- Test: `tests/test_detectkit_overlay_providers.py`

**Interfaces:**
- Consumes: Task 2's value objects; `find_label_for_image`, `find_staged_label_for_image`, `staged_class_names`, `source_class_id_map`, `parse_obb_label` from `detectkit.gui.utils`; `_resolve_source_render_state` and `_resolve_pending_level` from `main_window` (moved into `providers.py` by this task).
- Produces:
  - `FrameContext(project, source_path: str, image_path: str, size: tuple[int, int], predictions: list[dict], frame_key: str)`
  - `OverlayProvider` Protocol with `key: str` and `build(ctx) -> OverlayLayer | None`
  - `GroundTruthProvider()`, `PredictionProvider()`, `StagedEscalationProvider()`

- [ ] **Step 1: Write the failing test**

Create `tests/test_detectkit_overlay_providers.py`:

```python
"""Behavioural tests for the overlay providers.

These replace the inspect.getsource assertions that used to stand in for
them: a provider is now a plain object that can be called without a
MainWindow, so the tests can check what it BUILDS rather than what its
caller's source text contains.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hydra_suite.detectkit.gui.overlays import (
    ColourPolicy,
    Emphasis,
    FrameContext,
    GroundTruthProvider,
    LabelMode,
    PredictionProvider,
    StagedEscalationProvider,
)
from hydra_suite.training.geometry_levels import GeometryLevel


def _write_label(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def source_tree(tmp_path):
    src = tmp_path / "src_a"
    (src / "images").mkdir(parents=True)
    img = src / "images" / "f0001.png"
    img.write_bytes(b"")
    _write_label(
        src / "labels" / "f0001.txt",
        ["0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"],
    )
    return SimpleNamespace(root=src, image=img)


def _ctx(project, source_tree, **kw):
    base = dict(
        project=project,
        source_path=str(source_tree.root),
        image_path=str(source_tree.image),
        size=(100, 100),
        predictions=[],
        frame_key="images/f0001.png",
    )
    base.update(kw)
    return FrameContext(**base)


def _project(source_tree, **source_kw):
    source = SimpleNamespace(
        path=source_tree.root,
        name="src_a",
        level="polygon",
        reviewed=True,
        pending_escalation=None,
    )
    for k, v in source_kw.items():
        setattr(source, k, v)
    return SimpleNamespace(class_names=["ant", "worker"], sources=[source])


def test_ground_truth_provider_builds_a_per_class_multi_level_layer(source_tree):
    layer = GroundTruthProvider().build(_ctx(_project(source_tree), source_tree))
    assert layer.key == "gt"
    assert layer.colour_policy is ColourPolicy.PER_CLASS
    assert layer.label_mode is LabelMode.NAME_AND_CLASS_ID
    assert layer.derive_levels is True
    assert layer.class_filtered is True
    assert layer.native_level is GeometryLevel.POLYGON
    assert layer.emphasis is None
    assert len(layer.detections) == 1


def test_ground_truth_provider_flags_an_unreviewed_source(source_tree):
    project = _project(source_tree, reviewed=False)
    layer = GroundTruthProvider().build(_ctx(project, source_tree))
    assert layer.emphasis is Emphasis.UNREVIEWED


def test_ground_truth_provider_returns_none_when_the_frame_has_no_label(
    source_tree, tmp_path
):
    (source_tree.root / "labels" / "f0001.txt").unlink()
    assert GroundTruthProvider().build(_ctx(_project(source_tree), source_tree)) is None


def test_prediction_provider_labels_with_confidence(source_tree):
    preds = [{"class_id": 0, "polygon_px": [(1, 1), (5, 1), (5, 5)], "confidence": 0.5}]
    layer = PredictionProvider().build(
        _ctx(_project(source_tree), source_tree, predictions=preds)
    )
    assert layer.key == "pred"
    assert layer.label_mode is LabelMode.NAME_AND_CONFIDENCE
    assert layer.derive_levels is False
    assert layer.style is not None


def test_prediction_provider_returns_none_with_no_predictions(source_tree):
    assert PredictionProvider().build(_ctx(_project(source_tree), source_tree)) is None


def test_staged_provider_is_fixed_colour_and_unfiltered(source_tree, tmp_path):
    staged = tmp_path / "staged"
    _write_label(
        staged / "labels" / "images" / "f0001.txt",
        ["0 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4"],
    )
    (staged / "classes.txt").write_text("prompt_a\n")
    project = _project(
        source_tree,
        pending_escalation=SimpleNamespace(
            staged_path=str(staged), target_level="obb"
        ),
    )
    layer = StagedEscalationProvider().build(_ctx(project, source_tree))
    assert layer.key == "escalation"
    assert layer.colour_policy is ColourPolicy.FIXED
    assert layer.fixed_colour is not None
    assert layer.class_filtered is False
    assert layer.native_level is GeometryLevel.OBB
    assert layer.label_mode is LabelMode.NAME_AND_CONFIDENCE


def test_staged_provider_honours_the_escalations_own_target_level(
    source_tree, tmp_path
):
    """A SAM2 run can stage OBB. Hardcoding POLYGON here once gave a staged
    OBB polygon styling plus a duplicate derived OBB outline."""
    staged = tmp_path / "staged"
    _write_label(
        staged / "labels" / "images" / "f0001.txt",
        ["0 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4"],
    )
    project = _project(
        source_tree,
        pending_escalation=SimpleNamespace(
            staged_path=str(staged), target_level="aabb"
        ),
    )
    layer = StagedEscalationProvider().build(_ctx(project, source_tree))
    assert layer.native_level is GeometryLevel.AABB


def test_staged_provider_returns_none_without_a_pending_escalation(source_tree):
    assert (
        StagedEscalationProvider().build(_ctx(_project(source_tree), source_tree))
        is None
    )


def test_every_provider_stamps_the_frame_key(source_tree):
    ctx = _ctx(
        _project(source_tree),
        source_tree,
        predictions=[{"class_id": 0, "polygon_px": [(1, 1), (5, 1), (5, 5)]}],
    )
    for provider in (GroundTruthProvider(), PredictionProvider()):
        assert provider.build(ctx).frame_key == "images/f0001.png"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_detectkit_overlay_providers.py -v`
Expected: collection error, `ImportError: cannot import name 'FrameContext'`.

- [ ] **Step 3: Write the providers**

Create `src/hydra_suite/detectkit/gui/overlays/providers.py`:

```python
"""One provider per overlay data source.

The canvas used to conflate rendering shapes with knowing what ground
truth, predictions and staged escalations ARE -- and the three genuinely
differ (class-id space, whether confidence exists, lifecycle, geometry
level), so those differences leaked into show_image. Each quirk now lives
in exactly one small class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from PySide6.QtCore import Qt

from hydra_suite.training.geometry_levels import GeometryLevel

from ..constants import ESCALATION_COLOUR
from ..utils import (
    find_label_for_image,
    find_staged_label_for_image,
    parse_obb_label,
    source_class_id_map,
    staged_class_names,
)
from .layer import (
    ColourPolicy,
    Emphasis,
    LabelMode,
    LayerStyle,
    OverlayLayer,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameContext:
    """Everything every provider needs about the frame on screen."""

    project: Any
    source_path: str
    image_path: str
    # (h, w) taken from the LOADED PIXMAP, never decoded again. Re-decoding
    # the file per provider cost ~100 ms per keypress on 4512^2 frames.
    size: tuple[int, int]
    predictions: list[dict] = field(default_factory=list)
    frame_key: str = ""

    def source(self):
        if self.project is None:
            return None
        return next(
            (
                s
                for s in self.project.sources
                if str(s.path) == str(self.source_path)
            ),
            None,
        )


class OverlayProvider(Protocol):
    key: str

    def build(self, ctx: FrameContext) -> Optional[OverlayLayer]: ...


def _source_render_state(project, source_path) -> tuple[GeometryLevel, bool]:
    """The (native level, reviewed) of a source. Moved verbatim from
    main_window._resolve_source_render_state."""
    level = GeometryLevel.OBB
    reviewed = True
    if project is None:
        return level, reviewed
    source = next(
        (s for s in project.sources if str(s.path) == str(source_path)), None
    )
    if source is None:
        return level, reviewed
    try:
        level = GeometryLevel.from_str(str(getattr(source, "level", "obb")))
    except Exception:
        level = GeometryLevel.OBB
    reviewed = bool(getattr(source, "reviewed", True))
    return level, reviewed


def _pending_level(pending) -> GeometryLevel:
    """The escalation's OWN target level. Moved verbatim from
    main_window._resolve_pending_level: a SAM2 run can stage OBB, and
    hardcoding POLYGON gave it polygon styling plus a duplicate derived
    OBB outline."""
    try:
        return GeometryLevel.from_str(str(getattr(pending, "target_level", "polygon")))
    except Exception:
        return GeometryLevel.POLYGON


class GroundTruthProvider:
    key = "gt"

    def build(self, ctx: FrameContext) -> Optional[OverlayLayer]:
        label_path = find_label_for_image(Path(ctx.image_path), ctx.source_path)
        if label_path is None:
            return None
        h, w = ctx.size
        class_names = (
            ctx.project.class_names if ctx.project is not None else ["object"]
        )
        class_id_map = None
        if ctx.project is not None:
            try:
                class_id_map = source_class_id_map(ctx.source_path, class_names)
            except Exception:
                class_id_map = {}
                logger.warning(
                    "Skipping incompatible source labels for preview: %s",
                    ctx.source_path,
                    exc_info=True,
                )
        dets = parse_obb_label(label_path, w, h, class_id_map=class_id_map)
        native_level, reviewed = _source_render_state(ctx.project, ctx.source_path)
        return OverlayLayer(
            key=self.key,
            detections=dets,
            native_level=native_level,
            class_names=class_names,
            colour_policy=ColourPolicy.PER_CLASS,
            label_mode=LabelMode.NAME_AND_CLASS_ID,
            emphasis=None if reviewed else Emphasis.UNREVIEWED,
            z=0,
            frame_key=ctx.frame_key,
        )


class PredictionProvider:
    key = "pred"

    def build(self, ctx: FrameContext) -> Optional[OverlayLayer]:
        if not ctx.predictions:
            return None
        class_names = (
            ctx.project.class_names if ctx.project is not None else ["object"]
        )
        return OverlayLayer(
            key=self.key,
            detections=list(ctx.predictions),
            native_level=GeometryLevel.AABB,
            class_names=class_names,
            colour_policy=ColourPolicy.PER_CLASS,
            derive_levels=False,
            style=LayerStyle(Qt.PenStyle.DashLine, Qt.BrushStyle.NoBrush, 0),
            label_mode=LabelMode.NAME_AND_CONFIDENCE,
            z=10,
            frame_key=ctx.frame_key,
        )


class StagedEscalationProvider:
    key = "escalation"

    def build(self, ctx: FrameContext) -> Optional[OverlayLayer]:
        source = ctx.source()
        pending = getattr(source, "pending_escalation", None) if source else None
        if pending is None or not str(getattr(pending, "staged_path", "")).strip():
            return None
        label_path = find_staged_label_for_image(
            Path(ctx.image_path), ctx.source_path, pending.staged_path
        )
        if label_path is None:
            return None
        h, w = ctx.size
        # No class_id_map: staged ids index the STAGING dir's classes.txt,
        # not the project's class list, so remapping them would mislabel.
        dets = parse_obb_label(label_path, w, h)
        if not dets:
            return None
        return OverlayLayer(
            key=self.key,
            detections=dets,
            native_level=_pending_level(pending),
            class_names=staged_class_names(pending.staged_path),
            colour_policy=ColourPolicy.FIXED,
            fixed_colour=ESCALATION_COLOUR,
            class_filtered=False,
            label_mode=LabelMode.NAME_AND_CONFIDENCE,
            z=20,
            frame_key=ctx.frame_key,
        )


PROVIDERS: tuple = (
    GroundTruthProvider(),
    PredictionProvider(),
    StagedEscalationProvider(),
)
```

- [ ] **Step 4: Move `ESCALATION_COLOUR` to `constants.py`**

`providers.py` needs the colour and `canvas.py` needs it too; importing it from `canvas` into `providers` would make the domain layer depend on the renderer. Cut the `ESCALATION_COLOUR` definition (and its comment) out of `canvas.py` and paste it into `src/hydra_suite/detectkit/gui/constants.py`, adding `from PySide6.QtGui import QColor` there. In `canvas.py` import it: `from .constants import CANVAS_BG_COLOR, DEFAULT_OBB_FONT_SIZE, DEFAULT_OBB_LINE_WIDTH, ESCALATION_COLOUR`. Keep a re-export in `canvas.py` (`ESCALATION_COLOUR` is imported by name in `tests/test_detectkit_canvas.py`) — the import above already provides it as a module attribute.

- [ ] **Step 5: Extend `overlays/__init__.py`**

```python
from .providers import (
    PROVIDERS,
    FrameContext,
    GroundTruthProvider,
    OverlayProvider,
    PredictionProvider,
    StagedEscalationProvider,
)
```
and add those six names to `__all__`.

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_detectkit_overlay_providers.py -v`
Expected: 9 passed.

- [ ] **Step 7: Commit**

```bash
black src/hydra_suite/detectkit/gui tests/test_detectkit_overlay_providers.py
isort src/hydra_suite/detectkit/gui tests/test_detectkit_overlay_providers.py
git add -A
git commit -m "feat(detectkit): add per-source overlay providers"
```

---

### Task 6: Rewire `MainWindow`

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/main_window.py` (`show_image`, `_refresh_prediction_overlay`, `_refresh_escalation_overlay`, `_on_overlay_changed`; delete `_resolve_pending_level` and `_resolve_source_render_state`)
- Modify: `tests/test_detectkit_show_image_multi_level.py`, `tests/test_detectkit_staged_escalation_overlay.py`, `tests/test_detectkit_tools_panel.py`

**Interfaces:**
- Consumes: `FrameContext`, `PROVIDERS`, and the canvas registry surface.
- Produces: `MainWindow._frame_context() -> FrameContext | None` and `MainWindow._refresh_overlays(keys: tuple[str, ...] | None = None) -> None`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_detectkit_show_image_multi_level.py`, replace the source-text assertions at lines 17-26 with:

```python
def test_show_image_drives_the_overlay_providers(qapp):
    """show_image must not know how any single layer is built; it builds
    the frame context and asks each provider."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow.show_image)
    assert "_refresh_overlays" in source
    assert "set_gt_detections" not in source
    assert "clear_escalation_detections" not in source


def test_refresh_overlays_removes_a_layer_whose_provider_returns_none(qapp):
    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow._refresh_overlays)
    assert "remove_layer" in source
    assert "set_layer" in source
```

In `tests/test_detectkit_staged_escalation_overlay.py`, delete the remaining `inspect.getsource` assertions at lines 94-95 and 119 and replace them with:

```python
def test_the_staged_layer_is_refreshed_by_the_same_path_as_every_other(qapp):
    """The escalation layer's refresh used to fire only incidentally, and
    its clear used to sit below an early return. Both are structural now:
    one _refresh_overlays call, one idempotent set_layer per key."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow._refresh_overlays)
    assert "PROVIDERS" in source
```

In `tests/test_detectkit_tools_panel.py`, change line 85 `"set_derived_levels_visible"` (unchanged, still exists) and line 219 `"set_escalation_visible"` to `'set_layer_visible("escalation"'`.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_detectkit_show_image_multi_level.py tests/test_detectkit_staged_escalation_overlay.py tests/test_detectkit_tools_panel.py -v`
Expected: FAIL — `AttributeError: type object 'MainWindow' has no attribute '_refresh_overlays'`.

- [ ] **Step 3: Add `_frame_context` and `_refresh_overlays`**

In `main_window.py`, add:

```python
    def _frame_context(self) -> "FrameContext | None":
        """Everything the providers need about the frame on screen.

        The size comes from the pixmap load_image just built. Decoding the
        file again per provider cost ~100 ms per keypress on 4512^2 frames.
        """
        if not self._current_source_path or not self._current_image_path:
            return None
        size = self._canvas.image_size()
        if size is None:
            return None
        settings = self._tools_panel.get_overlay_settings()
        predictions: list[dict] = []
        signature = self._dataset_signature(settings)
        if (
            signature is not None
            and signature == self._dataset_prediction_signature
            and self._current_image_path in self._dataset_predictions
        ):
            predictions = _filter_detections_by_confidence(
                self._dataset_predictions.get(self._current_image_path, []),
                settings.confidence_threshold,
            )
        try:
            frame_key = str(
                Path(self._current_image_path).relative_to(self._current_source_path)
            )
        except ValueError:
            frame_key = str(self._current_image_path)
        return FrameContext(
            project=self._project,
            source_path=self._current_source_path,
            image_path=self._current_image_path,
            size=size,
            predictions=predictions,
            frame_key=frame_key,
        )

    def _refresh_overlays(self, keys: "tuple[str, ...] | None" = None) -> None:
        """Ask each provider for its layer and set or remove it.

        A provider returning None means "this layer does not apply to this
        frame", which removes it. There is no path where a stale layer can
        survive a frame change -- the bug that left the previous frame's
        staged masks floating over the new pixmap.
        """
        ctx = self._frame_context()
        for provider in PROVIDERS:
            if keys is not None and provider.key not in keys:
                continue
            layer = provider.build(ctx) if ctx is not None else None
            if layer is None:
                self._canvas.remove_layer(provider.key)
            else:
                self._canvas.set_layer(layer)
```

Add the import: `from .overlays import PROVIDERS, FrameContext`.

- [ ] **Step 4: Rewrite `show_image`**

Replace the body between `self._last_prediction_request = None` and the trailing `self._refresh_escalation_overlay()` with:

```python
        self._last_prediction_request = None
        # Every layer is cleared BEFORE the load can bail: otherwise
        # navigating to an unreadable frame left the previous frame's
        # overlays floating over the previous frame's pixmap.
        for provider in PROVIDERS:
            self._canvas.remove_layer(provider.key)
        if not self._canvas.load_image(image_path):
            return
        self._refresh_overlays()
```

Delete the now-unused module-level `_resolve_pending_level` and `_resolve_source_render_state` from `main_window.py` (they live in `providers.py` now).

- [ ] **Step 5: Rewrite the other three call sites**

```python
    def _refresh_prediction_overlay(self, *, force: bool = False) -> None:
        self._refresh_overlays(keys=("pred",))
        if self._project is not None:
            settings = self._tools_panel.get_overlay_settings()
            self._tools_panel.update_inference_stats(
                self._visible_inference_stats(settings.confidence_threshold),
                class_names=self._project.class_names,
            )

    def _refresh_escalation_overlay(self) -> None:
        self._refresh_overlays(keys=("escalation",))
```

`_refresh_escalation_overlay` is kept as a one-line named method because `escalation_actions.on_review_escalations` calls it directly after the review dialog — that direct call is what makes the post-review refresh deliberate rather than incidental.

In `_on_overlay_changed`, replace the four canvas calls at lines 1691-1694 with:

```python
        self._canvas.set_layer_visible("gt", settings.show_gt)
        self._canvas.set_layer_visible("pred", settings.show_pred)
        self._canvas.set_layer_visible("escalation", settings.show_escalation)
        self._canvas.set_class_filter(settings.visible_class_ids)
        self._canvas.set_derived_levels_visible(settings.show_derived_levels)
```

and replace each of its three `self._canvas.clear_pred_detections()` calls with `self._canvas.remove_layer("pred")`.

- [ ] **Step 6: Run the DetectKit suite**

```bash
python -m pytest tests/test_detectkit_show_image_multi_level.py \
  tests/test_detectkit_staged_escalation_overlay.py \
  tests/test_detectkit_tools_panel.py \
  tests/test_detectkit_overlay_golden.py \
  tests/test_detectkit_canvas.py tests/test_detectkit_canvas_dual_layer.py \
  tests/test_detectkit_prediction_preview.py \
  tests/test_detectkit_review_escalations_dialog.py -v
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
black src/hydra_suite/detectkit/gui/main_window.py tests/
isort src/hydra_suite/detectkit/gui/main_window.py tests/
git add -A
git commit -m "refactor(detectkit): drive MainWindow overlays through providers"
```

---

### Task 7: Rewire the two dialogs

These are the call sites the spec's migration section missed. They draw a single level with an explicit fill and dict class names — the case `derive_levels=False` + `style=` exists for.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/semantic_frame_preview_dialog.py:131-137`
- Modify: `src/hydra_suite/detectkit/gui/dialogs/calibration_results_dialog.py:243-245,315-316`

**Interfaces:**
- Consumes: the canvas registry surface and `OverlayLayer`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_detectkit_overlay_golden.py`:

```python
def test_dialog_scene_is_buildable_from_layers_alone(qapp):
    """The two calibration dialogs draw single-level filled GT and dashed
    predictions. This is the shape the registry has to express without the
    transitional adapters."""
    from PySide6.QtCore import Qt

    from hydra_suite.detectkit.gui.overlays import LayerStyle

    names = {0: "Ground truth", 2: "Prediction"}
    canvas = OBBCanvas()
    canvas.set_layer(
        OverlayLayer(
            key="gt",
            detections=_GT,
            native_level=GeometryLevel.AABB,
            class_names=names,
            colour_policy=ColourPolicy.PER_CLASS,
            derive_levels=False,
            style=LayerStyle(
                Qt.PenStyle.SolidLine, Qt.BrushStyle.SolidPattern, 65
            ),
            label_mode=LabelMode.NAME_AND_CLASS_ID,
            z=0,
        )
    )
    canvas.set_layer(
        OverlayLayer(
            key="pred",
            detections=_PRED,
            native_level=GeometryLevel.AABB,
            class_names=names,
            colour_policy=ColourPolicy.PER_CLASS,
            derive_levels=False,
            style=LayerStyle(
                Qt.PenStyle.DashLine, Qt.BrushStyle.SolidPattern, 55
            ),
            label_mode=LabelMode.NAME_AND_CONFIDENCE,
            z=10,
        )
    )
    expected = json.loads(GOLDEN.read_text())["dialog"]
    got = describe_scene(canvas)
    assert [{k: v for k, v in d.items() if k != "z"} for d in got] == [
        {k: v for k, v in d.items() if k != "z"} for d in expected
    ]
```

- [ ] **Step 2: Run to verify it fails or passes**

Run: `python -m pytest tests/test_detectkit_overlay_golden.py -k dialog_scene -v`
Expected: PASS. If it fails, the adapter in Task 3 and this explicit construction disagree — reconcile them before touching the dialogs, because the dialogs are about to be rewritten to the explicit form.

- [ ] **Step 3: Rewrite `semantic_frame_preview_dialog.py`**

Replace lines 131-132 with a module-level pair of builders and two `set_layer` calls, and lines 135-138 with `set_layer_visible`:

```python
        self._canvas.set_layer(_dialog_gt_layer(ground_truth, names))
        self._canvas.set_layer(_dialog_pred_layer(predictions, names))
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        self._canvas.set_layer_visible("gt", self._show_gt.isChecked())
        self._canvas.set_layer_visible("pred", self._show_predictions.isChecked())
```

Add to `src/hydra_suite/detectkit/gui/dialogs/_overlay_helpers.py` (new, shared by both dialogs — they draw the identical two layers):

```python
"""The two-layer GT/prediction overlay both calibration dialogs draw."""

from __future__ import annotations

from PySide6.QtCore import Qt

from hydra_suite.detectkit.gui.overlays import (
    ColourPolicy,
    LabelMode,
    LayerStyle,
    OverlayLayer,
)
from hydra_suite.training.geometry_levels import GeometryLevel


def dialog_gt_layer(detections, class_names) -> OverlayLayer:
    return OverlayLayer(
        key="gt",
        detections=detections,
        native_level=GeometryLevel.AABB,
        class_names=class_names,
        colour_policy=ColourPolicy.PER_CLASS,
        derive_levels=False,
        style=LayerStyle(Qt.PenStyle.SolidLine, Qt.BrushStyle.SolidPattern, 65),
        label_mode=LabelMode.NAME_AND_CLASS_ID,
        z=0,
    )


def dialog_pred_layer(detections, class_names) -> OverlayLayer:
    return OverlayLayer(
        key="pred",
        detections=detections,
        native_level=GeometryLevel.AABB,
        class_names=class_names,
        colour_policy=ColourPolicy.PER_CLASS,
        derive_levels=False,
        style=LayerStyle(Qt.PenStyle.DashLine, Qt.BrushStyle.SolidPattern, 55),
        label_mode=LabelMode.NAME_AND_CONFIDENCE,
        z=10,
    )
```

Import them in both dialogs as `from ._overlay_helpers import dialog_gt_layer, dialog_pred_layer` and use those names (drop the leading underscores used in the sketch above).

- [ ] **Step 4: Rewrite `calibration_results_dialog.py` the same way**

Lines 315-316 become the two `set_layer` calls; lines 243-245 become the two `set_layer_visible` calls.

- [ ] **Step 5: Run the dialog tests**

```bash
python -m pytest tests/test_detectkit_evaluation_dialog.py \
  tests/test_detectkit_sliced_preview.py \
  tests/test_detectkit_preview_target.py \
  tests/test_detectkit_overlay_golden.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
black src/hydra_suite/detectkit/gui/dialogs tests/
isort src/hydra_suite/detectkit/gui/dialogs tests/
git add -A
git commit -m "refactor(detectkit): draw calibration dialog overlays as layers"
```

---

### Task 8: Delete the transitional adapters

Every caller is migrated. The parallel-list vocabulary now has zero users; the spec's "retired, not wrapped" comes due.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/canvas.py`
- Modify: `tests/test_detectkit_canvas.py`, `tests/test_detectkit_canvas_dual_layer.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the final `OBBCanvas` public overlay surface — `set_layer`, `remove_layer`, `set_layer_visible`, `set_class_filter`, `set_derived_levels_visible`, `clear_all`, `image_size`, `load_image`, `set_image_array`, `fit_in_view`.

- [ ] **Step 1: Confirm every adapter is unreferenced**

```bash
grep -rn --include='*.py' -E "set_gt_detections|set_pred_detections|set_escalation_detections|clear_gt_detections|clear_pred_detections|clear_escalation_detections|set_overlay_visibility|set_escalation_visible" src | grep -v "gui/canvas.py:"
```
Expected: **no output.** Any hit is a call site Tasks 6–7 missed — go fix it there, do not delete the method out from under it.

- [ ] **Step 2: Delete the adapter block**

Remove the whole `# TRANSITIONAL ADAPTERS` section from `canvas.py`, including `_aabb_level` (its only callers were the adapters).

- [ ] **Step 3: Rewrite the two canvas test files against the registry API**

In `tests/test_detectkit_canvas_dual_layer.py` and `tests/test_detectkit_canvas.py`, replace every `canvas.set_gt_detections(...)` / `set_pred_detections(...)` / `set_escalation_detections(...)` / `set_detections(...)` with the equivalent `canvas.set_layer(OverlayLayer(...))`, every `clear_*_detections()` with `canvas.remove_layer("<key>")`, and every `set_overlay_visibility(show_gt=A, show_pred=B)` with a pair of `set_layer_visible` calls. Add a shared local helper at the top of each file so the construction is not repeated per test:

```python
def _layer(key, detections, *, level=GeometryLevel.AABB, names=None, **kw):
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
```

Delete `test_canvas_backward_compat_set_detections_alias` and `test_canvas_backward_compat_clear_detections_alias` (dual_layer.py:127-140) outright: the aliases they cover are gone and had no production caller.

- [ ] **Step 4: Run the full DetectKit test set**

```bash
for f in tests/test_detectkit_*.py tests/test_al_detectkit_equivalence.py; do
  echo "== $f"; python -m pytest "$f" -q 2>&1 | tail -3
done
```
Expected: every file green except the known pre-existing failure `tests/test_detectkit_dataset_panel.py::test_export_level_refresh_cannot_skip_identity_config_loading`, which fails on `main` too. Confirm that by running it on `main` before blaming this branch. Run per-file, not `pytest tests/` — the whole suite never finishes (`project_main_suite_blockers`).

- [ ] **Step 5: Launch the app and click through the three layers**

```bash
detectkit
```
Open a project with a source that has labels, toggle each of Show ground truth / Show predictions / Show staged escalation / Show derived levels, apply a class filter, run inference, and navigate between frames including to a frame with no label. Confirm nothing is stale, nothing is missing, and the magenta staged masks appear only where a staged escalation exists. The golden covers item properties; it cannot catch a wiring error that leaves a layer permanently absent from a live window.

- [ ] **Step 6: Commit**

```bash
black src/hydra_suite/detectkit tests/
isort src/hydra_suite/detectkit tests/
make lint-moderate
git add -A
git commit -m "refactor(detectkit): retire the per-layer canvas method families"
```

---

### Task 9: Documentation and spec closeout

**Files:**
- Modify: `docs/superpowers/specs/2026-08-31-detectkit-overlay-layer-registry-design.md`
- Modify: `docs/superpowers/specs/2026-08-31-detectkit-frame-granular-review-design.md`

- [ ] **Step 1: Check the docs build**

```bash
make docs-build
```
Expected: this currently fails on `hydra_suite.core.detectors` (a retired module) on `main` too. Confirm no *new* failure mentions `detectkit.gui.overlays` or `canvas`.

- [ ] **Step 2: Amend the frame-granular review spec**

That spec was written against a three-layer canvas with `set_escalation_detections`. Update its references to the new API and note that folding inference predictions into staged reviews now means adding a provider, not a canvas layer. Do not change any of its decisions.

- [ ] **Step 3: Update the registry spec's status header**

Change `**Status:**` to `Shipped — merged to main (<sha>)` only after the merge; until then leave the plan reference.

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs(detectkit): close out the overlay registry spec"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 `OverlayLayer` value object | 2 |
| §2 `OBBCanvas` four-method surface + one registry | 3 |
| §3 Providers, one per data source | 5 |
| §4 Stable instance identity (`InstanceRef`) | 3 (stamped in `_draw_detections`) |
| Migration — six call sites, adapters deleted | 6, 7, 8 |
| Testing 1 — golden characterization, committed | 1 |
| Testing 2 — visibility-toggle matrix | 4 |
| Testing 3 — provider unit tests replacing `getsource` | 5, 6 |
| Testing 4 — existing suites pass | 3, 6, 7, 8 |

**Known deviations from the spec, deliberate:**

1. **Adapters exist transiently** (Tasks 3–7) where the spec says "retired, not wrapped". They are deleted in Task 8, within the same branch. Without them Task 3 is a single unreviewable commit that breaks six call sites at once.
2. **`LabelMode.NAME` is not defined.** The spec lists three modes, but the staged-escalation layer today passes `show_confidence=True` and relies on the degrade-to-bare-name branch. Defining `NAME` and assigning it to that layer would render identically *today* but would silently discard confidence the moment the escalation job starts writing it (the values already exist in `staged_root/candidates.json`).
3. **`LayerStyle` / `derive_levels` / `frame_key` were added to the value object.** The spec's version could not express the two dialog call sites or the single-level prediction layer. Spec amended in the same branch.
