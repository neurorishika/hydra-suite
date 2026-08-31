# DetectKit Overlay Layer Registry — Design

**Status:** implementation plan written — `docs/superpowers/plans/2026-08-31-detectkit-overlay-layer-registry.md`
**Date:** 2026-08-31
**Scope:** `src/hydra_suite/detectkit/gui/canvas.py`, `main_window.py`, new `gui/overlays/`

## Problem

`OBBCanvas` grew a third overlay layer (staged escalations, merged
2026-08-31) beside ground truth and model predictions. Each layer is a
hand-maintained triplet of parallel lists plus its own draw method, its own
clear method, its own visibility flag, and its own branch in
`_apply_visibility`:

```
_gt_obb_items   / _gt_label_items   / _gt_class_ids   + _gt_level_*  + _show_gt
_pred_obb_items / _pred_label_items / _pred_class_ids               + _show_pred
_esc_obb_items  / _esc_label_items  / _esc_class_ids  + _esc_level_* + _show_escalation
```

Three consequences, all observed rather than predicted:

1. **Adding a layer means editing five places.** The escalation layer
   touched `__init__`, `_draw_detections`, `_apply_visibility`,
   `clear_all`, and added three public methods — none of it interesting,
   all of it copy-adapted.
2. **The branches are where the bugs were.** Of the four findings an
   adversarial review raised against the escalation layer, two were
   layer-bookkeeping defects: a clear that sat below an early `return`, and
   a refresh that fired only incidentally.
3. **The layer's semantics are implied by its method name, not stated.**
   Whether a layer is class-filtered, which colour policy it uses, whether
   its labels carry confidence, and what it stacks above are all encoded in
   which of the three `set_*` methods you call.

The canvas also conflates two responsibilities: *rendering shapes* and
*knowing what ground truth, predictions and escalations are*. The three
sources genuinely differ, and the differences currently leak into
`show_image`:

| | Ground truth | Model predictions | Staged escalation |
|---|---|---|---|
| Source of data | source `labels/` | in-memory inference cache | `staged_path/labels/` |
| Class-id space | project's | project's | the staging dir's `classes.txt` |
| Confidence available | no | yes | no (dropped on write) |
| Lifecycle | permanent | transient, per model run | pending accept/reject |
| Geometry level | source's `level` | project's | escalation's `target_level` |

## Non-goals

- Changing what any layer looks like on screen. This is a structural
  refactor; the rendered output must be identical.
- Editing shapes on the canvas. The canvas stays read-only.
- Unifying the class filter across layers. It can only ever address project
  class ids, so it stays a per-layer boolean, not a pluggable predicate.
- Merging the per-class palette into the fixed-hue policy. A class colour
  and a "this is a proposal" colour are different kinds of statement.

## Known deviations from pixel-identical (accepted post-review)

The "rendered output must be identical" non-goal above holds everywhere
except the three cases below. An adversarial review of the finished branch
found each; all three were accepted as intentional, and all three are
visual-only, cheap to reverse, and unrelated to the goal of collapsing the
three hand-maintained layers into one registry.

1. **Escalation-layer stacking is now consistent, where it used to be
   call-order-dependent.** The old canvas never called `setZValue`
   anywhere, so stacking was pure scene-insertion order. On a plain
   `show_image` that meant staged escalation masks sat *below* prediction
   overlays (drawn later); but `escalation_actions.py`'s post-dialog
   `_refresh_escalation_overlay()` re-inserted the escalation layer last,
   putting it *above* predictions on that one path only. The old behaviour
   was therefore self-contradictory depending on which code path last
   touched the scene. The registry assigns explicit z-order (`gt=0`,
   `escalation=10`, `pred=20`), so staged masks now render *below*
   predictions on every path, always. The declared invariant going forward
   is **predictions above staged masks**. If this proves wrong, the fix is
   swapping two z constants in the registry — no data or layout risk.
2. **A confidence-filtered-to-empty prediction cache now surfaces a status
   message.** When an image has cached predictions but *all* fall below the
   confidence threshold, the prediction provider returns `None`, so
   `_last_prediction_request` becomes `None` and `show_image` now displays
   "Image loaded. Click Run Inference to refresh overlay predictions."
   where it previously stayed silent. Canvas pixels are identical either
   way — nothing was drawn before and nothing is drawn now — only the
   status text changed. **Known wart:** the message blames stale inference
   when the confidence slider is the actual cause; a future pass should
   give this case its own wording.
3. **`update_inference_stats` now fires whenever a project exists,
   including when the prediction overlay was just removed** (previously it
   updated only on the successful-draw path). This is arguably a
   correction rather than a regression: `_visible_inference_stats` is
   dataset-wide and frame-independent, and the old code could leave stale
   pre-filter counts on screen while the overlay showed nothing. It was
   undeclared during the branch, so it is recorded here.

## Design

### 1. `OverlayLayer` — a layer as a value object

New `gui/overlays/layer.py`, Qt-free apart from the colour type:

```python
class ColourPolicy(Enum):
    PER_CLASS = auto()   # index the palette by class_id
    FIXED = auto()       # one hue for the whole layer

class LabelMode(Enum):
    NAME = auto()                 # "worker ant"
    NAME_AND_CLASS_ID = auto()    # "ant (0)"
    NAME_AND_CONFIDENCE = auto()  # "ant (0.42)"

@dataclass(frozen=True)
class LayerStyle:
    pen_style: Qt.PenStyle
    brush_style: Qt.BrushStyle
    fill_alpha: int               # 0-255; ignored when brush is NoBrush

@dataclass(frozen=True)
class OverlayLayer:
    key: str                      # "gt" | "pred" | "escalation" | ...
    detections: list[dict]
    native_level: GeometryLevel
    class_names: list[str] | dict[int, str] | None
    colour_policy: ColourPolicy
    fixed_colour: QColor | None = None   # required iff policy is FIXED
    z: int = 0                    # stacking; higher draws above
    class_filtered: bool = True
    label_mode: LabelMode = LabelMode.NAME_AND_CLASS_ID
    emphasis: Emphasis | None = None     # e.g. the unreviewed hatch
    derive_levels: bool = True    # False => draw only at native_level
    style: LayerStyle | None = None      # None => per-level default styles
    frame_key: str | None = None         # for InstanceRef (§4; DROPPED)
```

`LabelMode` replaces today's `show_confidence` boolean, which could not
express "this layer wants confidence but has none" — the gap that rendered
`worker ant (0)` over every staged mask and read as confidence 0.00.

`derive_levels` and `style` exist because **not every layer is
multi-level**. Two of the six call sites (§Migration) draw a single level
with an explicit fill: `set_gt_detections(…, fill_alpha=65)` and
`set_pred_detections(…, fill_alpha=55)`. The prediction layer today draws
only its native level, dashed — running `_levels_with_shapes` over it would
add derived outlines that are not on screen now, which the "identical
rendering" non-goal forbids. `style=None` means "use `_level_styles()`,
one style per level"; a non-`None` `style` applies to the single native
level and requires `derive_levels=False`.

`Emphasis.UNREVIEWED` substitutes `BDiagPattern`/alpha 140 on the
**native level only**, keeping that level's own pen style. That rule lives
in the renderer, not in any provider — hardcoding `SolidLine` there once
made an unreviewed OBB-native quad indistinguishable from its derived AABB.

The `LabelMode` assignment per layer is fixed and must not drift:

| Layer | `LabelMode` | Renders today |
|---|---|---|
| Ground truth | `NAME_AND_CLASS_ID` | `ant (0)` |
| Model predictions | `NAME_AND_CONFIDENCE` | `ant (0.42)` |
| Staged escalation | `NAME_AND_CONFIDENCE` | `ant` — staged labels carry no confidence, and this mode degrades to the bare name rather than to the class id |

`LabelMode.NAME` is therefore not needed and is not defined.

### 2. `OBBCanvas` — a renderer that knows nothing about the domain

Public surface collapses to four methods:

```python
def set_layer(self, layer: OverlayLayer) -> None      # add or replace by key
def remove_layer(self, key: str) -> None
def set_layer_visible(self, key: str, visible: bool) -> None
def set_class_filter(self, visible_class_ids: set[int]) -> None
```

Internal state collapses to one registry:

```python
self._layers: dict[str, OverlayLayer]                       # by key
self._items: dict[tuple[str, GeometryLevel], _LevelItems]   # scene items
self._layer_visible: dict[str, bool]
self._show_derived_levels: bool                             # stays global
```

`_apply_visibility` becomes one loop over `self._items` — no per-layer
branch, no zipped parallel lists. `_levels_with_shapes` (already extracted)
stays as the shared derivation. Drawing order is `sorted(by z, then key)`,
so stacking is declared rather than emergent from call order.

`set_layer` is idempotent by key: it removes that key's items and redraws.
This is what makes "clear before refresh" structural instead of a rule each
caller has to remember — the escalation layer's stale-mask bug becomes
unrepresentable.

### 3. Providers — one per data source, outside the canvas

New `gui/overlays/providers.py`. Each provider answers one question: *given
the current source and frame, what layer (if any) should be drawn?*

```python
class OverlayProvider(Protocol):
    key: str
    def build(self, ctx: FrameContext) -> OverlayLayer | None: ...
```

`FrameContext` carries what all three need — project, source path, image
path, and the frame's `(h, w)` **taken from the loaded pixmap, not decoded
again**. Three implementations:

- `GroundTruthProvider` — `find_label_for_image`, project class-id map,
  `_resolve_source_render_state` for level + the unreviewed hatch.
- `PredictionProvider` — reads the existing `_dataset_predictions` cache;
  `LabelMode.NAME_AND_CONFIDENCE`.
- `StagedEscalationProvider` — `find_staged_label_for_image`,
  `staged_class_names`, `_resolve_pending_level`, `class_filtered=False`,
  `ColourPolicy.FIXED`.

`MainWindow.show_image` becomes: build the context, ask each provider,
`set_layer` or `remove_layer` per result. Every quirk in the table above
lives in exactly one small class.

### 4. Stable instance identity

**Deliberately not implemented.** The user confirmed no per-instance review
interaction (click-to-accept) is planned, which removes this section's only
justification (see the Override note below). This is a final decision, not
a deferral pending a future trigger — the section is kept only as a record
of what retrofitting instance identity would cost if that assumption ever
changes.

Each drawn item would carry `item.setData(0, InstanceRef(layer_key,
frame_rel, index))`. Nothing consumes it in this refactor, and nothing in
the shipped registry (`OverlayLayer`, the three providers, `canvas.py`)
stamps or reads one.

It is specified now because retrofitting identity later means revisiting
every provider and every draw path, whereas adding it during a rewrite of
those paths is nearly free. It is the precondition for per-instance
accept/reject — clicking one magenta polygon and mapping it back to a line
in a staged label file — which is the natural next request for a review
overlay and is impossible without it.

## Migration

The three current `set_*`/`clear_*` method families are retired, not
wrapped — leaving permanent compatibility shims would preserve the
parallel-list mental model this refactor exists to delete.

They have **six** call sites, not four, and two of them are outside
`main_window.py`:

| Call site | Uses |
|---|---|
| `main_window.show_image` | `clear_gt/pred/escalation`, `set_gt_detections_multi_level` |
| `main_window._refresh_prediction_overlay` | `clear_pred_detections`, `set_pred_detections` |
| `main_window._refresh_escalation_overlay` | `clear_escalation_detections`, `set_escalation_detections` |
| `main_window._on_overlay_changed` | `set_overlay_visibility`, `set_class_filter`, `set_derived_levels_visible`, `set_escalation_visible`, `clear_pred_detections` |
| `dialogs/semantic_frame_preview_dialog.py:131-137` | `set_gt_detections(fill_alpha=65)`, `set_pred_detections(fill_alpha=55)`, `set_overlay_visibility` |
| `dialogs/calibration_results_dialog.py:243,315-316` | same three |

Two further public methods have **no production caller at all** and are
deleted rather than ported: `set_gt_detections(append=True)` (only
`tests/test_detectkit_canvas_dual_layer.py:115` and
`tests/test_detectkit_canvas.py:290` reach it) and the back-compat aliases
`set_detections`/`clear_detections`. Retiring `append` also deletes the
flat-list-vs-per-level-bucket special case in `set_gt_detections`, which
exists purely to keep appended items inside `_apply_visibility`'s iteration.

`trackerkit/.../detection_panel.py:2050` calls `set_detections` on
`reference_scale_preview`, a **different widget**. It is out of scope.

`OverlaySettings` keeps its per-layer booleans (`show_gt`, `show_pred`,
`show_escalation`); `_on_overlay_changed` maps them onto
`set_layer_visible(key, …)` calls.

## Testing

The risk in this change is a silent visual regression in the path that
renders every frame, so the gate is characterization, not new assertions:

1. **Golden-item characterization.** Before refactoring, add a test that
   builds each of the three layers on a fixture frame and records, per
   drawn item: level, pen colour/style/width, brush style/alpha, polygon
   points, label text, and visibility. Commit that as the oracle. The
   refactor must reproduce it exactly. (Committed golden, not a
   before-vs-after comparison in one process — the same trap noted in
   `project_shared_engine_param_builder`, where a post-collapse oracle was
   tautological.)
2. **Visibility-toggle matrix.** Every combination of the three layer
   toggles x derived-levels x a class filter, asserted against the golden.
   Today's tests cover these one at a time; the registry rewrite is exactly
   where an untested combination regresses.
3. **Provider unit tests**, replacing `inspect.getsource` assertions with
   real behavioural ones — a provider is a plain object callable without a
   `MainWindow`. Every source-inspecting test that this refactor breaks **by
   construction**, because it asserts on a method name that ceases to exist:
   `test_detectkit_staged_escalation_overlay.py:94-95,106,119`,
   `test_detectkit_tools_panel.py:85,219`,
   `test_detectkit_show_image_multi_level.py:25-26`.
4. **Existing suites must pass unchanged:** `test_detectkit_canvas.py`,
   `test_detectkit_staged_escalation_overlay.py`,
   `test_detectkit_show_image_multi_level.py`, `test_detectkit_tools_panel.py`.

No equivalence gate applies: this is GUI-only and touches nothing on the
tracking pipeline path.

## Cost and trigger

Roughly 300 lines rewritten in `canvas.py`, three small provider classes,
and a rewrite of the four call sites — plus the golden-characterization
test, which is the larger half of the work and the reason to do this
deliberately rather than opportunistically.

Deliberately **not** scheduled on merge of this spec. Three layers is the
point where the abstraction is guessable but not yet forced. Build it when
either arrives:

- a **fourth layer** (a second model's output, an AL escalation preview, a
  diff between two labellers), or
- the **first per-instance review interaction** (click-to-accept), which
  needs §4 and would otherwise have to retrofit it.

**Override (2026-08-31):** the user has confirmed that **neither trigger is
planned** — no fourth layer, no per-instance interaction. The work is
scheduled anyway, on the maintainability argument in the Problem section
alone, and sequenced **after** the frame-granular review programme.

Two consequences:

- **§4 is dropped from implementation.** Stable instance identity was
  justified solely as the precondition for per-instance accept/reject.
  With that dead, stamping an `InstanceRef` onto every drawn item on the
  hot draw path buys nothing. The section stays as the record of what
  retrofitting it would cost.
- The review programme may fold model predictions into staged reviews,
  which would take the canvas from three layers to two. Re-confirm the
  registry is still worth its cost once that lands.

Implementation plan:
`docs/superpowers/plans/2026-08-31-detectkit-overlay-layer-registry.md`.
