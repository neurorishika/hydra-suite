# DetectKit Overlay Layer Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `OBBCanvas`'s three hand-maintained overlay layers (parallel item lists, per-layer draw/clear/visibility methods, per-layer branches in `_apply_visibility`) with one keyed registry of `OverlayLayer` value objects, and move "what ground truth / predictions / staged escalations *are*" out of the canvas into three small providers — with pixel-identical rendering.

**Architecture:** A new `gui/overlays/layer.py` defines `OverlayLayer` (plus `ColourPolicy`, `LabelMode`, `LayerStyle`, `Emphasis`). `OBBCanvas` becomes a pure renderer holding `self._layers: dict[str, OverlayLayer]` and `self._items: dict[(key, GeometryLevel), _LevelItems]`, with a five-method public overlay surface. A new `gui/overlays/providers.py` holds `GroundTruthProvider`, `PredictionProvider`, `StagedEscalationProvider`, each answering "given this frame, what layer should be drawn?". `MainWindow.show_image` builds a `FrameContext` and asks each provider.

**Tech Stack:** Python 3.11+, PySide6 (`QGraphicsView`/`QGraphicsScene`), pytest, conda env `hydra-mps`.

**Spec:** `docs/superpowers/specs/2026-08-31-detectkit-overlay-layer-registry-design.md`

**Scheduling note:** the spec defers itself until a fourth overlay layer or the first per-instance interaction arrives. The user has stated that **neither is planned**, and has sequenced this work *after* the frame-granular review programme (`2026-08-31-detectkit-frame-granular-review-design.md`). What remains as justification is the maintainability half of the spec's Problem section — adding a layer means editing five places, and two of the four findings against the third layer were layer-bookkeeping defects. Because the per-instance trigger is dead, **spec §4 (`InstanceRef` stable instance identity) is dropped from this plan.** Its sole stated justification was "the precondition for per-instance accept/reject". Stamping provenance onto every drawn item on the hot draw path with no consumer, now or planned, is cost without benefit.

## Global Constraints

- **Rendering must be pixel-identical, *including stacking order*.** Every pen (colour, style, width, cosmetic flag), brush (style, alpha), polygon point, label string, label font size, item visibility, and the front-to-back order of items in the scene stays exactly as it is today. Task 1 builds the oracle that proves it; every later task re-runs it.
- **Today's stacking is insertion order, and it must be preserved.** `show_image` draws GT, then the staged escalation, then predictions (`main_window.py:2107-2151`), so **dashed predictions render above magenta staged masks** where they overlap. The `z` values assigned in Task 3 (`gt=0`, `escalation=10`, `pred=20`) are chosen to reproduce exactly that. Inverting the stack so proposals sit on top may well be an improvement, but it is a *deliberate behaviour change* and is out of scope here — raise it separately rather than folding it into a structural refactor.
- **The oracle is a committed golden file, regenerated only through pytest.** A before/after check written after the refactor is tautological — the trap recorded in `project_shared_engine_param_builder`. Task 1 commits the golden *before* any production code changes.
- **Never run a bare `python -c` against the worktree.** In `.worktrees/overlay-registry`, `import hydra_suite` resolves to the **main repo's** editable install, not this branch's `src/`. Any script that regenerates or diffs the golden that way records or compares the wrong tree — the `feedback_equivalence_pythonpath_gotcha` trap, and it fails silently. `pytest` is safe (`tests/conftest.py:9` puts the worktree `src/` at `sys.path[0]`), so **all golden work goes through pytest**, using the repo's existing `--update-golden` option (`tests/conftest.py:18-24`; reference implementation in `tests/test_al_selection_golden.py`).
- **There is no global `qapp` fixture.** `tests/conftest.py` does not define one and `pytest-qt` is not installed. Every Qt test file defines its own, preceded by the offscreen-platform preamble — copy this block verbatim into each new Qt test file (pattern: `tests/test_detectkit_canvas.py:1-29`):

  ```python
  import os
  import sys

  import pytest

  os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

  pytest.importorskip("PySide6")

  from PySide6.QtWidgets import QApplication  # noqa: E402


  @pytest.fixture()
  def qapp():
      app = QApplication.instance()
      if app is None:
          app = QApplication(sys.argv)
      return app
  ```

- **The canvas stays read-only.** No shape editing, no mouse interaction on items.
- **The class filter stays a per-layer boolean.** It can only ever address project class ids, so it is `class_filtered: bool`, never a pluggable predicate.
- **`GeometryLevel` is imported from `hydra_suite.utils.geometry_levels`**, its canonical bottom-layer home (`utils/geometry_levels.py:11-13`); `training.geometry_levels` merely re-exports it.
- **No equivalence gate applies.** GUI-only; touches nothing on the tracking pipeline path. Do **not** run `tools/equivalence/run_matrix.sh`.
- **Environment:** `source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps` before running tests or `black`/`isort`. `black` is broken in the base env.
- **Formatting:** run `black` and `isort` **only on the paths you touched**, never `make format` (it reformats unrelated files).
- Work happens in the worktree `.worktrees/overlay-registry` on branch `feat/detectkit-overlay-registry`. All paths below are relative to it.

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `src/hydra_suite/detectkit/gui/colors.py` | `ESCALATION_COLOUR`. A new module, not `constants.py`: the headless escalation jobs (`jobs/sam2_escalation.py:21`, `jobs/semantic_escalation.py:46`) import `gui.constants` for `IMG_EXTS`, and that module is deliberately Qt-free. |
| `src/hydra_suite/detectkit/gui/overlays/__init__.py` | Re-exports the public names from `layer.py` and `providers.py`. |
| `src/hydra_suite/detectkit/gui/overlays/layer.py` | The `OverlayLayer` value object and its enums. No canvas, no project, no I/O. |
| `src/hydra_suite/detectkit/gui/overlays/providers.py` | `FrameContext` + the three providers. Knows the domain; knows nothing about `QGraphicsScene`. |
| `src/hydra_suite/detectkit/gui/dialogs/_overlay_helpers.py` | The two-layer GT/prediction overlay both calibration dialogs draw. |
| `tests/test_detectkit_overlay_golden.py` | The characterization oracle + the visibility-toggle matrix. |
| `tests/goldens/detectkit_overlay_characterization.json` | The committed golden data (alongside the existing `tests/goldens/`). |
| `tests/test_detectkit_overlay_layer.py` | Unit tests for the value object's invariants. |
| `tests/test_detectkit_overlay_providers.py` | Behavioural provider tests. |

**Modified:**

| File | Change |
|---|---|
| `src/hydra_suite/detectkit/gui/canvas.py` | Registry rewrite; old method families deleted in Task 8. |
| `src/hydra_suite/detectkit/gui/main_window.py` | Four call sites rewired to providers; two helpers moved out. |
| `src/hydra_suite/detectkit/gui/dialogs/semantic_frame_preview_dialog.py` | Two-layer draw rewired. |
| `src/hydra_suite/detectkit/gui/dialogs/calibration_results_dialog.py` | Same. |
| `tests/test_detectkit_canvas.py`, `tests/test_detectkit_canvas_dual_layer.py` | Assert on canvas internals that Task 3 deletes — **rewritten in Task 3**. |
| `tests/test_semantic_calibration_preview.py`, `tests/test_semantic_frame_preview_dialog.py` | Assert on `dialog._canvas._pred_obb_items`/`_gt_obb_items` — **rewritten in Task 3** (they break as soon as the internals change, not when the dialogs are rewired). |
| `tests/test_detectkit_staged_escalation_overlay.py`, `tests/test_detectkit_show_image_multi_level.py`, `tests/test_detectkit_tools_panel.py` | Assert on method names / helpers that Task 6 deletes — rewritten in Task 6. |

**Sequencing rule that makes every task green:** Task 3 introduces the registry *underneath* the existing public methods, reimplementing them as thin adapters, **and rewrites in the same commit every test that reads the internals it deletes**. Tasks 5–7 migrate callers one at a time. Task 8 deletes the adapters. The spec's "retired, not wrapped" is satisfied at the end of the branch; the adapters never survive it.

**Known pre-existing failure**, unrelated to this work — confirm it fails on `main` before blaming this branch: `tests/test_detectkit_dataset_panel.py::test_export_level_refresh_cannot_skip_identity_config_loading`. Also, `make docs-build` already fails on the retired `hydra_suite.core.detectors` module.

---

### Task 1: The golden characterization oracle

The riskiest change in this refactor is a silent visual regression on the path that renders every frame. Build the detector first, commit it, and never weaken it.

**Files:**
- Create: `tests/test_detectkit_overlay_golden.py`
- Create: `tests/goldens/detectkit_overlay_characterization.json`

**Interfaces:**
- Consumes: nothing (this task runs against the current canvas API).
- Produces: `describe_scene(canvas) -> list[dict]` — imported by Tasks 3–7. Each dict carries `type` (`"polygon"` | `"label"`), `stack_index`, `visible`, and for polygons `points`, `pen_colour`, `pen_style`, `pen_width`, `pen_cosmetic`, `brush_style`, `brush_alpha`; for labels `text`, `colour`, `pos`, `font_px`.

- [ ] **Step 1: Write the file preamble, the scene description helper, and the fixtures**

Create `tests/test_detectkit_overlay_golden.py`:

```python
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

Note `_PRED`'s second detection deliberately has **no** `confidence` key: it exercises the degrade-to-bare-name branch that exists because `name (0)` beside a shape reads as confidence 0.00.

- [ ] **Step 2: Write the scene builders — one per rendering situation that exists today**

Append. Each builder reproduces exactly one production call pattern, in the same call order, because that order is what determines stacking.

```python
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
```

- [ ] **Step 3: Write the golden test, using the repo's `--update-golden` option**

Append:

```python
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
```

(`#ff3cc7` is `ESCALATION_COLOUR`, `QColor(255, 60, 199)`; verify with `QColor(255, 60, 199).name()` if the assertion misses.)

- [ ] **Step 4: Generate the golden against unmodified canvas code, through pytest**

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/overlay-registry
git status --porcelain src/   # MUST be empty: the golden is only valid if
                              # it was recorded from untouched source
python -m pytest tests/test_detectkit_overlay_golden.py --update-golden -v
```
Expected: the golden test SKIPs with "golden updated"; `test_predictions_stack_above_staged_escalations` PASSES.

Use **pytest**, never `python -c`: a bare interpreter in this worktree imports the main repo's `src/`, so the golden would record the wrong tree's canvas.

- [ ] **Step 5: Sanity-check the generated golden**

```bash
python -m pytest tests/test_detectkit_overlay_golden.py -v
python - <<'PY'
import json
g = json.load(open("tests/goldens/detectkit_overlay_characterization.json"))
for k, v in g.items():
    print(k, len(v), "items")
assert all(len(v) > 0 for v in g.values())
mw = g["main_window"]
assert any(d.get("pen_colour") == "#ff3cc7" for d in mw), "no escalation items"
assert any(d["type"] == "label" and d["text"] == "ant (0.87)" for d in mw), \
    "confidence label missing"
assert any(d["type"] == "label" and d["text"] == "larva" for d in mw), \
    "the degrade-to-bare-name branch is not exercised"
assert any(d.get("brush_style") == 11 for d in g["unreviewed"]), \
    "BDiagPattern hatch missing from the unreviewed scene"
print("OK")
PY
```
Expected: all four scenes non-empty and `OK`. This one `python -` block reads only the generated JSON, imports no `hydra_suite`, and is therefore safe.

- [ ] **Step 6: Commit**

```bash
black tests/test_detectkit_overlay_golden.py
isort tests/test_detectkit_overlay_golden.py
git add tests/test_detectkit_overlay_golden.py \
        tests/goldens/detectkit_overlay_characterization.json
git commit -m "test(detectkit): characterize overlay rendering before the registry refactor"
```

---

### Task 2: The `OverlayLayer` value object

**Files:**
- Create: `src/hydra_suite/detectkit/gui/overlays/__init__.py`
- Create: `src/hydra_suite/detectkit/gui/overlays/layer.py`
- Test: `tests/test_detectkit_overlay_layer.py`

**Interfaces:**
- Consumes: `hydra_suite.utils.geometry_levels.GeometryLevel`.
- Produces: `ColourPolicy`, `LabelMode`, `Emphasis`, `LayerStyle`, `OverlayLayer` — importable from `hydra_suite.detectkit.gui.overlays`.

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

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402

from hydra_suite.detectkit.gui.overlays import (  # noqa: E402
    ColourPolicy,
    LabelMode,
    LayerStyle,
    OverlayLayer,
)
from hydra_suite.utils.geometry_levels import GeometryLevel  # noqa: E402

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

from PySide6.QtGui import QColor

if TYPE_CHECKING:
    from PySide6.QtCore import Qt

    from hydra_suite.utils.geometry_levels import GeometryLevel


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

    def __post_init__(self) -> None:
        if self.colour_policy is ColourPolicy.FIXED and self.fixed_colour is None:
            raise ValueError("ColourPolicy.FIXED requires fixed_colour")
        if (
            self.colour_policy is ColourPolicy.PER_CLASS
            and self.fixed_colour is not None
        ):
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
    LabelMode,
    LayerStyle,
    OverlayLayer,
)

__all__ = [
    "ColourPolicy",
    "Emphasis",
    "LabelMode",
    "LayerStyle",
    "OverlayLayer",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_detectkit_overlay_layer.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
black src/hydra_suite/detectkit/gui/overlays tests/test_detectkit_overlay_layer.py
isort src/hydra_suite/detectkit/gui/overlays tests/test_detectkit_overlay_layer.py
git add src/hydra_suite/detectkit/gui/overlays tests/test_detectkit_overlay_layer.py
git commit -m "feat(detectkit): add the OverlayLayer value object"
```

---

### Task 3: The canvas registry, with the old methods as adapters

The canvas gains its registry surface. Every existing public method is reimplemented on top of it, so the app and all six call sites keep working and the golden keeps passing. This task **also rewrites every test that reads the canvas internals it deletes** — those internals disappear here, so their tests cannot wait for Task 8.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/canvas.py`
- Modify: `tests/test_detectkit_canvas.py`, `tests/test_detectkit_canvas_dual_layer.py`, `tests/test_semantic_calibration_preview.py`, `tests/test_semantic_frame_preview_dialog.py`
- Test: `tests/test_detectkit_overlay_golden.py` (unchanged — it is the gate)

**Interfaces:**
- Consumes: `OverlayLayer`, `ColourPolicy`, `LabelMode`, `LayerStyle`, `Emphasis` from Task 2.
- Produces:
  - `OBBCanvas.set_layer(layer: OverlayLayer) -> None` — add or replace by `layer.key`; idempotent.
  - `OBBCanvas.remove_layer(key: str) -> None` — no-op if absent.
  - `OBBCanvas.set_layer_visible(key: str, visible: bool) -> None` — remembered even for a key not yet drawn.
  - `OBBCanvas.set_class_filter(visible_class_ids: set[int]) -> None` — unchanged signature.
  - `OBBCanvas.set_derived_levels_visible(visible: bool) -> None` — unchanged; the flag stays global.
  - `OBBCanvas.layer_items(key) -> dict[GeometryLevel, _LevelItems]` — the read accessor tests use in place of the deleted `_gt_level_items` etc.
  - Layer keys and z values: `"gt"` z=0, `"escalation"` z=10, `"pred"` z=20.

- [ ] **Step 1: Write the failing tests for the registry surface**

Append to `tests/test_detectkit_overlay_golden.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_detectkit_overlay_golden.py -v`
Expected: the golden test and the stacking test PASS; the 5 new tests FAIL with `AttributeError: 'OBBCanvas' object has no attribute 'set_layer'`.

- [ ] **Step 3: Replace the canvas's layer state**

In `src/hydra_suite/detectkit/gui/canvas.py`, replace the block in `__init__` defining `_gt_*`, `_esc_*`, `_pred_*`, `_show_gt`, `_show_pred`, `_show_escalation`, `_obb_items`, `_label_items` (lines 106-140) with:

```python
        self._layers: dict[str, OverlayLayer] = {}
        # (layer_key, level) -> _LevelItems. One flat registry; there is no
        # per-layer branch anywhere below it.
        self._items: dict[tuple, _LevelItems] = {}
        self._layer_visible: dict[str, bool] = {}
        self._show_derived_levels: bool = True
        self._visible_class_ids: set[int] = set()
```

Add beside `_LevelStyle`:

```python
@dataclass
class _LevelItems:
    """The scene items one layer drew at one geometry level."""

    obb_items: list
    label_items: list
    class_ids: list[int]
```

And the resolved-style helper beside `_level_styles()`:

```python
def _styles_for(layer: "OverlayLayer", level, native_level) -> _LevelStyle:
    """The pen/brush a layer uses at one level.

    An explicit ``layer.style`` wins outright (single-level layers: the
    dialogs' filled GT and the dashed prediction layer). Otherwise the
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

Add the imports at the top of `canvas.py`:

```python
from .colors import ESCALATION_COLOUR
from .overlays import (
    ColourPolicy,
    Emphasis,
    LabelMode,
    LayerStyle,
    OverlayLayer,
)
```

and create `src/hydra_suite/detectkit/gui/colors.py`, moving the `ESCALATION_COLOUR` definition and its comment out of `canvas.py:52-56`:

```python
"""Canvas colours that are not per-class palette entries."""

from __future__ import annotations

from PySide6.QtGui import QColor

# The staged-escalation layer's single hue, deliberately OUTSIDE the class
# palette: a staged SAM3/SAM2 mask is a proposal, not a labelled class, so
# the distinction it must carry is "not ground truth" -- never a class
# identity.
ESCALATION_COLOUR = QColor(255, 60, 199)  # magenta
```

This goes in its own module rather than `constants.py` because the headless escalation jobs (`jobs/sam2_escalation.py:21`, `jobs/semantic_escalation.py:46`) import `gui.constants` for `IMG_EXTS`, and that module is deliberately Qt-free. The `from .colors import ESCALATION_COLOUR` line keeps `canvas.ESCALATION_COLOUR` working for `tests/test_detectkit_canvas.py`, which imports it by that path.

- [ ] **Step 4: Write `set_layer` / `remove_layer` / `set_layer_visible` / `layer_items` and the single-loop `_apply_visibility`**

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
            level_iter = self._levels_with_shapes(layer.detections, layer.native_level)
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

    def layer_items(self, key: str) -> dict:
        """The per-level item buckets one layer drew. Read-only accessor
        for tests; production code never needs it."""
        return {
            level: items
            for (layer_key, level), items in self._items.items()
            if layer_key == key
        }

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

- [ ] **Step 5: Rewrite `_draw_detections` against the layer**

Replace the whole method:

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

        for det in detections:
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

            poly_item = self._scene.addPolygon(qpoly, pen, brush)
            poly_item.setZValue(layer.z)
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
            self._scene.addItem(txt_item)
            items.label_items.append(txt_item)
```

**The z values are load-bearing and must reproduce today's insertion-order stacking.** `show_image` draws GT, then escalation, then predictions, so predictions sit frontmost. The adapters in Step 6 therefore assign `gt=0`, `escalation=10`, `pred=20`. Do not "improve" this ordering here.

- [ ] **Step 6: Reimplement every old public method as an adapter**

Replace `set_gt_detections`, `set_gt_detections_multi_level`, `set_pred_detections`, `set_escalation_detections`, `set_escalation_visible`, `set_overlay_visibility`, `clear_gt_detections`, `clear_pred_detections`, `clear_escalation_detections`, `set_detections`, `clear_detections` with:

```python
    # ------------------------------------------------------------------
    # TRANSITIONAL ADAPTERS -- deleted in Task 8 of this refactor. They
    # exist only so the six call sites can migrate one at a time instead
    # of in one unreviewable commit.
    # ------------------------------------------------------------------

    def set_gt_detections(self, detections, class_names=None, *, fill_alpha=0) -> None:
        self.set_layer(
            OverlayLayer(
                key="gt",
                detections=detections,
                native_level=self._single_level(),
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
        self, detections, class_names=None, *, native_level, reviewed=True
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
        self, detections, class_names=None, *, fill_alpha=0
    ) -> None:
        self.set_layer(
            OverlayLayer(
                key="pred",
                detections=detections,
                native_level=self._single_level(),
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
                z=20,
            )
        )

    def set_escalation_detections(
        self, detections, class_names=None, *, native_level
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
                z=10,
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

    def set_detections(self, detections, class_names=None) -> None:
        self.set_gt_detections(detections, class_names)

    def clear_detections(self) -> None:
        self.remove_layer("gt")

    @staticmethod
    def _single_level():
        """The native level a non-deriving layer declares.

        Single-level layers never derive, so the value only has to satisfy
        `level == native_level` -- which is what makes the layer labelled
        and keeps it visible when derived levels are hidden, matching the
        old flat-list branch of _apply_visibility. AABB is the floor of the
        ordering, so it can never be mistaken for a derived level.
        """
        from hydra_suite.utils.geometry_levels import GeometryLevel

        return GeometryLevel.AABB
```

Update `clear_all`:

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

`_layer_visible` is deliberately **not** cleared: it mirrors the user's checkbox state, which survives a scene reset.

- [ ] **Step 7: Delete `set_gt_detections(append=True)`**

`append=True` has no production caller. Verify:

```bash
grep -rn --include='*.py' "append=True" src/hydra_suite/detectkit
```
Expected: no hit. The adapter above already omits the parameter.

- [ ] **Step 8: Rewrite every test that reads the deleted internals**

These break the moment Step 3 lands. Four files:

`tests/test_detectkit_canvas.py` — replace `canvas._gt_level_items` → `canvas.layer_items("gt")`, `canvas._esc_level_items` → `canvas.layer_items("escalation")`, `canvas._gt_level_label_items[L]` → `canvas.layer_items("gt")[L].label_items`, `canvas._gt_obb_items` → `[i for b in canvas.layer_items("gt").values() for i in b.obb_items]`. Delete `test_set_gt_detections_append_after_multi_level_is_visibility_controlled` (line 272): the bug it guarded — appended items falling outside `_apply_visibility`'s iteration — cannot exist once every layer lives in the one registry, and its API is gone.

`tests/test_detectkit_canvas_dual_layer.py` — same substitutions with `"pred"`. At lines 114-116 replace the two `set_gt_detections(..., append=True)` calls with one `canvas.set_gt_detections(_DET + _DET2)`. Keep `test_canvas_old_label_items_still_accessible` only if it can be expressed through `layer_items`; otherwise delete it — the `_label_items` alias it names is gone.

`tests/test_semantic_calibration_preview.py:104-117` and `tests/test_semantic_frame_preview_dialog.py:66-67` — replace `dialog._canvas._pred_obb_items` / `_gt_obb_items` with a helper at the top of each file:

```python
def _polys(canvas, key):
    return [i for b in canvas.layer_items(key).values() for i in b.obb_items]
```

- [ ] **Step 9: Delete the two `inspect.getsource` tests that are now false by construction**

In `tests/test_detectkit_staged_escalation_overlay.py`, delete `test_escalation_overlay_derives_the_levels_beneath_its_native_one` (line 106, asserts `_levels_with_shapes` appears in `set_escalation_detections`'s source) and the clear-ordering test at line 119. Both methods are now one-line adapters. Their behaviour is covered by the golden `main_window` scene and `test_set_layer_replaces_by_key_instead_of_accumulating`.

- [ ] **Step 10: Run the golden and every affected suite**

```bash
python -m pytest tests/test_detectkit_overlay_golden.py \
  tests/test_detectkit_canvas.py tests/test_detectkit_canvas_dual_layer.py \
  tests/test_semantic_calibration_preview.py \
  tests/test_semantic_frame_preview_dialog.py \
  tests/test_detectkit_show_image_multi_level.py \
  tests/test_detectkit_staged_escalation_overlay.py \
  tests/test_detectkit_tools_panel.py -v
```
Expected: all pass. The golden reproducing exactly — including `stack_index` — is the whole point of the task.

- [ ] **Step 11: If the golden fails, do not regenerate it**

A mismatch means the refactor changed rendering. Diff it **through pytest** so the correct source tree is imported:

```bash
python -m pytest tests/test_detectkit_overlay_golden.py::test_overlay_rendering_matches_the_committed_golden -vv 2>&1 | head -80
```

Compare by matching items on `points`/`text`, not by zipping the two lists — a single stacking change shifts every subsequent `stack_index` and floods a positional diff. Most likely causes, in order: the `z` assignment not matching insertion order (check `gt=0 < escalation=10 < pred=20`); `Emphasis.UNREVIEWED` applied at the wrong level; the `_single_level()` choice changing which items count as native and therefore get labels; a dropped `setCosmetic(True)`.

**Regenerating the golden is not an available fix in this task.** Every change here is structural; there is no legitimate rendering diff to absorb.

- [ ] **Step 12: Commit**

```bash
black src/hydra_suite/detectkit/gui tests/
isort src/hydra_suite/detectkit/gui tests/
git add -A
git commit -m "refactor(detectkit): make OBBCanvas a keyed layer registry"
```

---

### Task 4: The visibility-toggle matrix

Today's tests cover the toggles one at a time. The registry rewrite is exactly where an untested combination regresses.

**Files:**
- Modify: `tests/test_detectkit_overlay_golden.py`

- [ ] **Step 1: Write the hand-computed cases first**

A matrix test that recomputes expected visibility with the same boolean expression as `_apply_visibility` proves only that the code equals itself. Write the anchors by hand, then the sweep. Append:

```python
import itertools  # noqa: E402


def _visible_counts(canvas) -> dict:
    return {
        key: sum(
            1
            for b in canvas.layer_items(key).values()
            for o in b.obb_items
            if o.isVisible()
        )
        for key in ("gt", "pred", "escalation")
    }


def test_hiding_gt_leaves_the_other_two_layers_fully_visible(qapp):
    canvas = OBBCanvas()
    _build_main_window_scene(canvas)
    before = _visible_counts(canvas)
    canvas.set_layer_visible("gt", False)
    after = _visible_counts(canvas)
    assert after["gt"] == 0
    assert after["pred"] == before["pred"]
    assert after["escalation"] == before["escalation"]


def test_hiding_derived_levels_keeps_exactly_the_native_shapes(qapp):
    """GT is POLYGON-native and escalation OBB-native in this scene, so
    each keeps its own native bucket and loses the rest. Predictions never
    derive, so they are untouched."""
    canvas = OBBCanvas()
    _build_main_window_scene(canvas)
    canvas.set_derived_levels_visible(False)
    assert _visible_counts(canvas) == {
        "gt": len(_GT),
        "escalation": len(_ESC),
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
    assert counts["escalation"] == len(_ESC) * 2  # OBB native + derived AABB


def test_an_empty_class_filter_means_show_all(qapp):
    canvas = OBBCanvas()
    _build_main_window_scene(canvas)
    canvas.set_class_filter({0})
    filtered = _visible_counts(canvas)["gt"]
    canvas.set_class_filter(set())
    assert _visible_counts(canvas)["gt"] > filtered
```

- [ ] **Step 2: Add the exhaustive sweep as a crash/consistency guard**

```python
@pytest.mark.parametrize(
    "show_gt,show_pred,show_esc,show_derived,class_filter",
    list(
        itertools.product(
            [True, False], [True, False], [True, False], [True, False],
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
    canvas.set_layer_visible("escalation", show_esc)
    canvas.set_derived_levels_visible(show_derived)
    canvas.set_class_filter(class_filter)

    counts = _visible_counts(canvas)
    if not show_gt:
        assert counts["gt"] == 0
    if not show_pred:
        assert counts["pred"] == 0
    if not show_esc:
        assert counts["escalation"] == 0
    if show_esc:
        # class_filtered=False, so the filter can never reduce it
        expected = len(_ESC) * (2 if show_derived else 1)
        assert counts["escalation"] == expected
```

- [ ] **Step 3: Run**

Run: `python -m pytest tests/test_detectkit_overlay_golden.py -v -k "visib or class_filter or derived"`
Expected: 4 hand-written + 32 sweep cases pass. If any fail, the fault is in `_apply_visibility` — fix the code, not the assertion.

- [ ] **Step 4: Commit**

```bash
black tests/test_detectkit_overlay_golden.py
isort tests/test_detectkit_overlay_golden.py
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
- Consumes: Task 2's value objects; `find_label_for_image`, `find_staged_label_for_image`, `staged_class_names`, `source_class_id_map`, `parse_obb_label` from `detectkit.gui.utils`.
- Produces:
  - `FrameContext(project, source_path: str, image_path: str, size: tuple[int, int], predictions: list[dict])`
  - `OverlayProvider` Protocol with `key: str` and `build(ctx) -> OverlayLayer | None`
  - `GroundTruthProvider()`, `PredictionProvider()`, `StagedEscalationProvider()`, `PROVIDERS`
  - `resolve_source_render_state(project, source_path)`, `resolve_pending_level(pending)` — moved **verbatim** from `main_window.py:98-145`, renamed without the leading underscore because they are now imported across modules.

- [ ] **Step 1: Write the providers**

Create `src/hydra_suite/detectkit/gui/overlays/providers.py`. The two resolver helpers are moved **byte-for-byte** apart from the rename: same `s.path == source_path` comparison, same `except ValueError` (not `except Exception` — `tests/test_detectkit_show_image_multi_level.py:35-36` asserts on that exact text), same `logger.warning` calls.

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

from hydra_suite.utils.geometry_levels import GeometryLevel

from ..colors import ESCALATION_COLOUR
from ..utils import (
    find_label_for_image,
    find_staged_label_for_image,
    parse_obb_label,
    source_class_id_map,
    staged_class_names,
)
from .layer import ColourPolicy, Emphasis, LabelMode, LayerStyle, OverlayLayer

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

    def source(self):
        if self.project is None:
            return None
        return next(
            (s for s in self.project.sources if s.path == self.source_path), None
        )


class OverlayProvider(Protocol):
    key: str

    def build(self, ctx: FrameContext) -> Optional[OverlayLayer]: ...


def resolve_pending_level(pending):
    """Geometry level a staged escalation's labels are in.

    ``PendingEscalation.target_level`` is load-bearing: SAM2 converts
    existing boxes IN PLACE and can stage OBB, while SAM3 stages polygons.
    Drawing an OBB quad as polygon-native gave it the polygon style AND a
    derived OBB of the same quad -- a duplicate outline in the wrong style.

    Like ``OBBSource.level`` this is an unvalidated string from project
    JSON, so it degrades rather than raising.
    """
    raw = str(getattr(pending, "target_level", "") or "")
    try:
        return GeometryLevel.from_str(raw)
    except ValueError:
        logger.warning(
            "Unknown target_level %r on a staged escalation; rendering as polygon",
            raw,
        )
        return GeometryLevel.POLYGON


def resolve_source_render_state(project, source_path):
    """Return (native_level, reviewed) for the OBBSource at *source_path*.

    Falls back to (GeometryLevel.OBB, True) if project is None, no source
    matches, or the matched source's level string doesn't parse --
    OBBSource.level is an unvalidated string loaded from project JSON, so a
    hand-edited or future-version file must degrade gracefully here rather
    than crashing show_image on every image selection.
    """
    if project is None:
        return GeometryLevel.OBB, True

    src_obj = next((s for s in project.sources if s.path == source_path), None)
    if src_obj is None:
        return GeometryLevel.OBB, True

    try:
        native_level = GeometryLevel.from_str(src_obj.level)
    except ValueError:
        logger.warning(
            "Unknown geometry level %r for source %r; rendering as OBB",
            src_obj.level,
            source_path,
        )
        native_level = GeometryLevel.OBB

    return native_level, src_obj.reviewed


class GroundTruthProvider:
    key = "gt"

    def build(self, ctx: FrameContext) -> Optional[OverlayLayer]:
        label_path = find_label_for_image(Path(ctx.image_path), ctx.source_path)
        if label_path is None:
            return None
        h, w = ctx.size
        class_names = ctx.project.class_names if ctx.project is not None else ["object"]
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
        native_level, reviewed = resolve_source_render_state(
            ctx.project, ctx.source_path
        )
        return OverlayLayer(
            key=self.key,
            detections=dets,
            native_level=native_level,
            class_names=class_names,
            colour_policy=ColourPolicy.PER_CLASS,
            label_mode=LabelMode.NAME_AND_CLASS_ID,
            emphasis=None if reviewed else Emphasis.UNREVIEWED,
            z=0,
        )


class PredictionProvider:
    key = "pred"

    def build(self, ctx: FrameContext) -> Optional[OverlayLayer]:
        if not ctx.predictions:
            return None
        class_names = ctx.project.class_names if ctx.project is not None else ["object"]
        return OverlayLayer(
            key=self.key,
            detections=list(ctx.predictions),
            native_level=GeometryLevel.AABB,
            class_names=class_names,
            colour_policy=ColourPolicy.PER_CLASS,
            derive_levels=False,
            style=LayerStyle(Qt.PenStyle.DashLine, Qt.BrushStyle.NoBrush, 0),
            label_mode=LabelMode.NAME_AND_CONFIDENCE,
            z=20,
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
            native_level=resolve_pending_level(pending),
            class_names=staged_class_names(pending.staged_path),
            colour_policy=ColourPolicy.FIXED,
            fixed_colour=ESCALATION_COLOUR,
            class_filtered=False,
            label_mode=LabelMode.NAME_AND_CONFIDENCE,
            z=10,
        )


# Draw order == today's show_image order (GT, escalation, predictions), and
# the z values above encode the same stacking.
PROVIDERS: tuple = (
    GroundTruthProvider(),
    StagedEscalationProvider(),
    PredictionProvider(),
)
```

Extend `overlays/__init__.py`:

```python
from .providers import (
    PROVIDERS,
    FrameContext,
    GroundTruthProvider,
    OverlayProvider,
    PredictionProvider,
    StagedEscalationProvider,
    resolve_pending_level,
    resolve_source_render_state,
)
```
and add those eight names to `__all__`.

- [ ] **Step 2: Write the tests**

Create `tests/test_detectkit_overlay_providers.py`:

```python
"""Behavioural tests for the overlay providers.

These replace the inspect.getsource assertions that used to stand in for
them: a provider is a plain object that can be called without a
MainWindow, so the tests check what it BUILDS rather than what its
caller's source text contains.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from hydra_suite.detectkit.gui.overlays import (  # noqa: E402
    ColourPolicy,
    Emphasis,
    FrameContext,
    GroundTruthProvider,
    LabelMode,
    PredictionProvider,
    StagedEscalationProvider,
    resolve_pending_level,
    resolve_source_render_state,
)
from hydra_suite.utils.geometry_levels import GeometryLevel  # noqa: E402


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def source_tree(tmp_path):
    src = tmp_path / "src_a"
    (src / "images").mkdir(parents=True)
    img = src / "images" / "f0001.png"
    img.write_bytes(b"")
    _write(src / "labels" / "f0001.txt", ["0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"])
    # REQUIRED: source_class_id_map -> read_classes_txt raises RuntimeError
    # without it, the provider's except-branch zeroes the class map, and
    # parse_obb_label then drops every line. A missing classes.txt makes
    # these tests fail in a way that looks like a provider bug.
    (src / "classes.txt").write_text("ant\nworker\n")
    return SimpleNamespace(root=src, image=img)


def _project(source_tree, **kw):
    source = SimpleNamespace(
        path=str(source_tree.root),
        name="src_a",
        level="polygon",
        reviewed=True,
        pending_escalation=None,
    )
    for k, v in kw.items():
        setattr(source, k, v)
    return SimpleNamespace(class_names=["ant", "worker"], sources=[source])


def _ctx(project, source_tree, **kw):
    base = dict(
        project=project,
        source_path=str(source_tree.root),
        image_path=str(source_tree.image),
        size=(100, 100),
        predictions=[],
    )
    base.update(kw)
    return FrameContext(**base)


def test_ground_truth_provider_builds_a_per_class_multi_level_layer(source_tree):
    layer = GroundTruthProvider().build(_ctx(_project(source_tree), source_tree))
    assert layer.key == "gt"
    assert layer.colour_policy is ColourPolicy.PER_CLASS
    assert layer.label_mode is LabelMode.NAME_AND_CLASS_ID
    assert layer.derive_levels is True
    assert layer.class_filtered is True
    assert layer.native_level is GeometryLevel.POLYGON
    assert layer.emphasis is None
    assert layer.z == 0
    assert len(layer.detections) == 1


def test_ground_truth_provider_flags_an_unreviewed_source(source_tree):
    layer = GroundTruthProvider().build(
        _ctx(_project(source_tree, reviewed=False), source_tree)
    )
    assert layer.emphasis is Emphasis.UNREVIEWED


def test_ground_truth_provider_returns_none_when_the_frame_has_no_label(source_tree):
    (source_tree.root / "labels" / "f0001.txt").unlink()
    assert GroundTruthProvider().build(_ctx(_project(source_tree), source_tree)) is None


def test_prediction_provider_labels_with_confidence_and_never_derives(source_tree):
    preds = [{"class_id": 0, "polygon_px": [(1, 1), (5, 1), (5, 5)], "confidence": 0.5}]
    layer = PredictionProvider().build(
        _ctx(_project(source_tree), source_tree, predictions=preds)
    )
    assert layer.key == "pred"
    assert layer.label_mode is LabelMode.NAME_AND_CONFIDENCE
    assert layer.derive_levels is False
    assert layer.style is not None
    assert layer.z == 20


def test_prediction_provider_returns_none_with_no_predictions(source_tree):
    assert PredictionProvider().build(_ctx(_project(source_tree), source_tree)) is None


def _staged(tmp_path, target_level):
    staged = tmp_path / "staged"
    _write(
        staged / "labels" / "images" / "f0001.txt",
        ["0 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4"],
    )
    (staged / "classes.txt").write_text("prompt_a\n")
    return SimpleNamespace(staged_path=str(staged), target_level=target_level)


def test_staged_provider_is_fixed_colour_and_unfiltered(source_tree, tmp_path):
    project = _project(source_tree, pending_escalation=_staged(tmp_path, "obb"))
    layer = StagedEscalationProvider().build(_ctx(project, source_tree))
    assert layer.key == "escalation"
    assert layer.colour_policy is ColourPolicy.FIXED
    assert layer.fixed_colour is not None
    assert layer.class_filtered is False
    assert layer.native_level is GeometryLevel.OBB
    assert layer.label_mode is LabelMode.NAME_AND_CONFIDENCE
    assert layer.z == 10


def test_staged_provider_honours_the_escalations_own_target_level(
    source_tree, tmp_path
):
    """A SAM2 run can stage OBB. Hardcoding POLYGON here once gave a staged
    OBB polygon styling plus a duplicate derived OBB outline."""
    project = _project(source_tree, pending_escalation=_staged(tmp_path, "aabb"))
    layer = StagedEscalationProvider().build(_ctx(project, source_tree))
    assert layer.native_level is GeometryLevel.AABB


def test_staged_provider_returns_none_without_a_pending_escalation(source_tree):
    assert (
        StagedEscalationProvider().build(_ctx(_project(source_tree), source_tree))
        is None
    )


def test_the_escalation_layer_stacks_below_predictions(source_tree, tmp_path):
    """show_image draws GT, then the staged escalation, then predictions,
    so dashed predictions sit ON TOP of magenta staged masks. The z values
    must reproduce that -- this refactor does not change stacking."""
    project = _project(source_tree, pending_escalation=_staged(tmp_path, "obb"))
    ctx = _ctx(
        project,
        source_tree,
        predictions=[{"class_id": 0, "polygon_px": [(1, 1), (5, 1), (5, 5)]}],
    )
    gt = GroundTruthProvider().build(ctx)
    esc = StagedEscalationProvider().build(ctx)
    pred = PredictionProvider().build(ctx)
    assert gt.z < esc.z < pred.z
```

- [ ] **Step 3: Port the resolver tests that Task 6 would otherwise orphan**

`tests/test_detectkit_show_image_multi_level.py:43-84` has three behavioural tests importing `_resolve_source_render_state` from `main_window`, and `tests/test_detectkit_staged_escalation_overlay.py:178-198` one importing `_resolve_pending_level`. Task 6 deletes both helpers. Move all four into `tests/test_detectkit_overlay_providers.py` now, changing only the import and the name (`_resolve_source_render_state` → `resolve_source_render_state`, `_resolve_pending_level` → `resolve_pending_level`). Their bodies — including the `not_a_real_level` and `not_a_level` degradation cases — stay as they are.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_detectkit_overlay_providers.py -v`
Expected: 13 passed (9 provider tests + 4 ported resolver tests).

- [ ] **Step 5: Commit**

```bash
black src/hydra_suite/detectkit/gui tests/test_detectkit_overlay_providers.py
isort src/hydra_suite/detectkit/gui tests/test_detectkit_overlay_providers.py
git add -A
git commit -m "feat(detectkit): add per-source overlay providers"
```

---

### Task 6: Rewire `MainWindow`

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/main_window.py`
- Modify: `tests/test_detectkit_show_image_multi_level.py`, `tests/test_detectkit_staged_escalation_overlay.py`, `tests/test_detectkit_tools_panel.py`

**Interfaces:**
- Consumes: `FrameContext`, `PROVIDERS`, the canvas registry surface.
- Produces: `MainWindow._frame_context() -> FrameContext | None`, `MainWindow._refresh_overlays(keys=None) -> None`.

- [ ] **Step 1: Add `_frame_context` and `_refresh_overlays`**

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
            self._project is not None
            and signature is not None
            and signature == self._dataset_prediction_signature
            and self._current_image_path in self._dataset_predictions
        ):
            predictions = _filter_detections_by_confidence(
                self._dataset_predictions.get(self._current_image_path, []),
                settings.confidence_threshold,
            )
        return FrameContext(
            project=self._project,
            source_path=self._current_source_path,
            image_path=self._current_image_path,
            size=size,
            predictions=predictions,
        )

    def _refresh_overlays(self, keys: "tuple[str, ...] | None" = None) -> None:
        """Ask each provider for its layer and set or remove it.

        A provider returning None means "this layer does not apply to this
        frame", which removes it. There is no path where a stale layer can
        survive a frame change -- the bug that left the previous frame's
        staged masks floating over the new pixmap.

        PROVIDERS is iterated in draw order, which the z values also encode.
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

Add `from .overlays import PROVIDERS, FrameContext` to the imports. `Path`, `_dataset_signature` and `_filter_detections_by_confidence` are already available in this module.

- [ ] **Step 2: Rewrite `show_image` — the exact final body**

Replace `show_image` (`main_window.py:2099-2161`) in full. Note that `_last_prediction_request` and the status message, `fit_in_view()`, and the prediction restore all survive:

```python
    def show_image(self, source_path: str, image_path: str) -> None:
        """Load an image and overlay GT labels, predictions and staged masks."""
        new_source = str(source_path or "")
        if new_source != self._current_source_path:
            self._dataset_predictions = {}
            self._dataset_prediction_signature = None
        self._current_source_path = new_source
        self._current_image_path = str(image_path or "")
        self._last_prediction_request = None
        # Every layer is removed BEFORE the load can bail: otherwise
        # navigating to an unreadable frame left the previous frame's
        # overlays floating over the previous frame's pixmap.
        for provider in PROVIDERS:
            self._canvas.remove_layer(provider.key)
        if not self._canvas.load_image(image_path):
            return

        self._refresh_overlays()

        if (
            self._last_prediction_request is None
            and self._project is not None
            and str(self._project.active_model_path or "").strip()
        ):
            self.statusBar().showMessage(
                "Image loaded. Click Run Inference to refresh overlay predictions.",
                3000,
            )
        self._canvas.fit_in_view()
```

`_refresh_overlays()` covers all three layers, so the separate `_refresh_escalation_overlay()` and `_refresh_prediction_overlay(force=True)` calls the old body made are gone — but `_refresh_overlays` must still maintain `_last_prediction_request`, which Step 3 handles.

Delete the module-level `_resolve_pending_level` and `_resolve_source_render_state` (`main_window.py:98-145`) — they live in `providers.py` now.

- [ ] **Step 3: Preserve the `_last_prediction_request` bookkeeping**

`show_image`'s status message reads `_last_prediction_request` to decide whether predictions were restored. Nothing else writes it once `_refresh_prediction_overlay`'s body is replaced, so it would be permanently `None` and the message would show even when cached predictions were just drawn. Set it inside `_refresh_overlays`, right after the pred layer is handled:

```python
            if provider.key == "pred":
                self._last_prediction_request = (
                    None if layer is None else self._dataset_signature(
                        self._tools_panel.get_overlay_settings()
                    )
                )
```

- [ ] **Step 4: Rewrite the other three call sites**

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

`_refresh_escalation_overlay` stays as a named one-liner because `escalation_actions.on_review_escalations` calls it directly after the review dialog — that direct call is what makes the post-review refresh deliberate rather than incidental.

In `_on_overlay_changed`, replace lines 1691-1694 with:

```python
        self._canvas.set_layer_visible("gt", settings.show_gt)
        self._canvas.set_layer_visible("pred", settings.show_pred)
        self._canvas.set_layer_visible("escalation", settings.show_escalation)
        self._canvas.set_class_filter(settings.visible_class_ids)
        self._canvas.set_derived_levels_visible(settings.show_derived_levels)
```

and replace each of its three `self._canvas.clear_pred_detections()` calls with `self._canvas.remove_layer("pred")`.

- [ ] **Step 5: Update the tests that name the deleted symbols**

`tests/test_detectkit_show_image_multi_level.py` — the four resolver tests moved to Task 5 Step 3; delete them here. Replace `test_show_image_calls_multi_level_api_with_source_level_and_reviewed` with:

```python
def test_show_image_drives_the_overlay_providers():
    """show_image must not know how any single layer is built; it builds
    the frame context and asks each provider."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow.show_image)
    assert "_refresh_overlays" in source
    assert "set_gt_detections" not in source
    assert "clear_escalation_detections" not in source

    refresh = inspect.getsource(MainWindow._refresh_overlays)
    assert "PROVIDERS" in refresh
    assert "remove_layer" in refresh
    assert "set_layer" in refresh
```

`tests/test_detectkit_staged_escalation_overlay.py` — delete `test_the_overlay_renders_at_the_escalations_own_target_level` (line 162; it asserts `_resolve_pending_level(pending)` appears in `_refresh_escalation_overlay`, now a one-liner). Its behaviour is covered by `test_staged_provider_honours_the_escalations_own_target_level` in Task 5. Replace the source-text assertions at lines 94-95 with:

```python
def test_the_staged_layer_refreshes_through_the_same_path_as_every_other():
    """The escalation layer's refresh used to fire only incidentally, and
    its clear used to sit below an early return. Both are structural now:
    one _refresh_overlays call, one idempotent set_layer per key."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    assert "_refresh_overlays" in inspect.getsource(
        MainWindow._refresh_escalation_overlay
    )
```

`tests/test_detectkit_tools_panel.py:219` — change `"set_escalation_visible"` to `'set_layer_visible("escalation"'`. Line 85 (`set_derived_levels_visible`) is unchanged; that method survives.

- [ ] **Step 6: Run**

```bash
python -m pytest tests/test_detectkit_show_image_multi_level.py \
  tests/test_detectkit_staged_escalation_overlay.py \
  tests/test_detectkit_tools_panel.py \
  tests/test_detectkit_overlay_golden.py \
  tests/test_detectkit_overlay_providers.py \
  tests/test_detectkit_canvas.py tests/test_detectkit_canvas_dual_layer.py \
  tests/test_detectkit_prediction_preview.py \
  tests/test_detectkit_review_escalations_dialog.py \
  tests/test_detectkit_inference_cancel.py -v
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

The call sites the spec's original migration section missed. They draw a single level with an explicit fill and dict class names — the case `derive_levels=False` + `style=` exists for.

**Files:**
- Create: `src/hydra_suite/detectkit/gui/dialogs/_overlay_helpers.py`
- Modify: `src/hydra_suite/detectkit/gui/dialogs/semantic_frame_preview_dialog.py:131-137`
- Modify: `src/hydra_suite/detectkit/gui/dialogs/calibration_results_dialog.py:243-245,315-316`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_detectkit_overlay_golden.py`:

```python
def test_dialog_scene_is_buildable_from_layers_alone(qapp):
    """The two calibration dialogs draw single-level filled GT and dashed
    predictions. This is the shape the registry must express without the
    transitional adapters -- and it must match the committed golden."""
    from hydra_suite.detectkit.gui.dialogs._overlay_helpers import (
        dialog_gt_layer,
        dialog_pred_layer,
    )

    names = {0: "Ground truth", 2: "Prediction"}
    canvas = OBBCanvas()
    canvas.set_layer(dialog_gt_layer(_GT, names))
    canvas.set_layer(dialog_pred_layer(_PRED, names))
    expected = json.loads(GOLDEN.read_text())["dialog"]
    assert describe_scene(canvas) == expected
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_detectkit_overlay_golden.py -k dialog_scene -v`
Expected: `ModuleNotFoundError: ..._overlay_helpers`.

- [ ] **Step 3: Write the shared helpers**

Create `src/hydra_suite/detectkit/gui/dialogs/_overlay_helpers.py`:

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
from hydra_suite.utils.geometry_levels import GeometryLevel


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
        z=20,
    )
```

- [ ] **Step 4: Rewire `semantic_frame_preview_dialog.py`**

Replace lines 131-132 and the `_refresh_visibility` body:

```python
        self._canvas.set_layer(dialog_gt_layer(ground_truth, names))
        self._canvas.set_layer(dialog_pred_layer(predictions, names))
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        self._canvas.set_layer_visible("gt", self._show_gt.isChecked())
        self._canvas.set_layer_visible("pred", self._show_predictions.isChecked())
```

with `from ._overlay_helpers import dialog_gt_layer, dialog_pred_layer` at the top.

- [ ] **Step 5: Rewire `calibration_results_dialog.py` identically**

Lines 315-316 become the two `set_layer` calls; lines 243-245 become the two `set_layer_visible` calls.

- [ ] **Step 6: Run the dialog tests**

```bash
python -m pytest tests/test_semantic_calibration_preview.py \
  tests/test_semantic_frame_preview_dialog.py \
  tests/test_detectkit_evaluation_dialog.py \
  tests/test_detectkit_sliced_preview.py \
  tests/test_detectkit_preview_target.py \
  tests/test_detectkit_overlay_golden.py -v
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
black src/hydra_suite/detectkit/gui/dialogs tests/
isort src/hydra_suite/detectkit/gui/dialogs tests/
git add -A
git commit -m "refactor(detectkit): draw calibration dialog overlays as layers"
```

---

### Task 8: Delete the transitional adapters

Every caller is migrated; the parallel-list vocabulary has zero users.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/canvas.py`
- Modify: `tests/test_detectkit_canvas.py`, `tests/test_detectkit_canvas_dual_layer.py`

- [ ] **Step 1: Confirm every adapter is unreferenced in `src/`**

```bash
grep -rn --include='*.py' -E \
  "set_gt_detections|set_pred_detections|set_escalation_detections|clear_gt_detections|clear_pred_detections|clear_escalation_detections|set_overlay_visibility|set_escalation_visible|\.set_detections|\.clear_detections" \
  src/hydra_suite/detectkit
```
Expected: hits only inside `gui/canvas.py` (the adapter definitions themselves). Any other hit is a call site Tasks 6–7 missed — go fix it there, do not delete the method out from under it.

`src/hydra_suite/trackerkit/gui/panels/detection_panel.py:2050` calls `set_detections` on `reference_scale_preview`, a **different widget**. Out of scope; the grep above is scoped to `detectkit` so it will not appear.

- [ ] **Step 2: Delete the adapter block**

Remove the whole `# TRANSITIONAL ADAPTERS` section from `canvas.py`, including `_single_level` (its only callers were the adapters).

- [ ] **Step 3: Rewrite the two canvas test files against the registry API**

In `tests/test_detectkit_canvas.py` and `tests/test_detectkit_canvas_dual_layer.py`, replace every remaining adapter call with `canvas.set_layer(...)` / `canvas.remove_layer("<key>")` / paired `set_layer_visible`. Add a shared local helper at the top of each file so the construction is not repeated per test:

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

Delete `test_canvas_backward_compat_set_detections_alias` and `test_canvas_backward_compat_clear_detections_alias` (`test_detectkit_canvas_dual_layer.py:127-140`): the aliases they cover are gone and had no production caller.

- [ ] **Step 4: Run the full DetectKit test set, per file**

```bash
for f in tests/test_detectkit_*.py tests/test_semantic_*.py \
         tests/test_al_detectkit_equivalence.py; do
  echo "== $f"; python -m pytest "$f" -q 2>&1 | tail -3
done
```
Expected: every file green except `tests/test_detectkit_dataset_panel.py::test_export_level_refresh_cannot_skip_identity_config_loading`, which fails on `main` too — verify that before blaming this branch. Run per-file, not `pytest tests/`: the whole suite never finishes (`project_main_suite_blockers`).

- [ ] **Step 5: Launch the app and click through**

```bash
detectkit
```

The golden covers item properties; it cannot catch a wiring error that leaves a layer permanently absent from a live window. Open a project with a labelled source and check:
1. Each of Show ground truth / Show predictions / Show staged escalation / Show derived levels toggles the right layer.
2. A class filter hides GT classes but never the magenta staged masks.
3. Run Inference draws dashed predictions, and they render **on top of** magenta staged masks where they overlap.
4. Navigating between frames — including to a frame with no label, and to an unreadable file — leaves nothing stale.
5. An unreviewed source shows the hatched native fill.

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
Expected: this already fails on the retired `hydra_suite.core.detectors` module on `main`. Confirm no *new* failure mentions `detectkit.gui.overlays`, `gui.colors`, or `canvas`.

- [ ] **Step 2: Record the dropped §4 in the spec**

Add a note to §4 of the registry spec: stable instance identity was **not implemented**, because the user confirmed no per-instance review interaction is planned and it had no other consumer. Keep the section as the record of what it would take if that changes.

- [ ] **Step 3: Amend the frame-granular review spec**

It was written against a three-layer canvas with `set_escalation_detections`. Update its API references and note that folding inference predictions into staged reviews now means adding a provider, not a canvas layer. Do not change any of its decisions.

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
| §2 `OBBCanvas` registry surface + one `_apply_visibility` loop | 3 |
| §3 Providers, one per data source | 5 |
| §4 Stable instance identity | **dropped** — see the scheduling note and Task 9 Step 2 |
| Migration — six call sites, adapters deleted | 6, 7, 8 |
| Testing 1 — golden characterization, committed | 1 |
| Testing 2 — visibility-toggle matrix | 4 |
| Testing 3 — provider unit tests replacing `getsource` | 5, 6 |
| Testing 4 — existing suites pass | 3, 6, 7, 8 |

**Deliberate deviations from the spec:**

1. **Adapters exist transiently** (Tasks 3–7) where the spec says "retired, not wrapped". They are deleted in Task 8, within the same branch. Without them Task 3 is a single unreviewable commit that breaks six call sites at once.
2. **`LabelMode.NAME` is not defined.** The spec lists three modes, but the staged-escalation layer passes `show_confidence=True` today and relies on the degrade-to-bare-name branch. Assigning it `NAME` renders identically now but would silently discard confidence the moment the escalation job starts writing it (the values already exist in `staged_root/candidates.json`).
3. **`LayerStyle` and `derive_levels` were added to the value object.** The spec's version could not express the two dialog call sites or the single-level prediction layer.
4. **`InstanceRef` / spec §4 dropped**, per the scheduling note.
5. **`ESCALATION_COLOUR` moves to a new `gui/colors.py`, not `constants.py`** — the headless escalation jobs import `gui.constants`, which must stay Qt-free.

**Traps this plan is written to avoid** (each was a real defect in its first draft, caught by adversarial review):

- A bare `python -c` in this worktree imports the **main repo's** `src/`. All golden work goes through pytest.
- There is **no global `qapp` fixture**; every Qt test file defines its own.
- The oracle must capture **stacking order**, `pen.isCosmetic()`, and the label font size, or a real regression passes.
- Task 3 deletes canvas internals that **four** test files read — they are rewritten in the same task, not deferred.
- `source_class_id_map` raises without `classes.txt`; a provider fixture missing it makes the provider look broken.
- `_last_prediction_request` has a reader in `show_image`; dropping its writes silently mis-triggers a status message.
