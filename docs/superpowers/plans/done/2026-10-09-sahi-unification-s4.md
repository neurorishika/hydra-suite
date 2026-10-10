# SAHI Unification — Slice S4 (shared widget, five hosts, visual verification) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One shared SAHI settings widget (`hydra_suite/widgets/slice_settings.py`) used by all five hosts with TrackerKit's vocabulary, source badges on derived values, constrained fields disabled (never hidden), validator-backed ranges (F5), derived overlap (F7) — and a rendered visual gallery of every host for the user to confirm.

**Architecture:** A Qt-only widget in the shared `widgets/` layer whose state is the S1 contract (`TilingSpec`) plus a role-specific `extras` dict; it imports only `utils.tiling_spec`/`tiling_resolve`/`slice_geometry` and Qt. Each kit keeps a thin adapter (DetectKit: `SliceTrainingSettings`/SAM3 tiling dict/escalation params ↔ spec+extras; TrackerKit: config/advanced keys ↔ spec+extras). Hosts are migrated one at a time; host-facing contracts (`to_settings()`, `to_sam3_tiling()`, `parameters()`, `tiling_parameters()`, config JSON keys, `SLICE_*` engine params) do not change.

**Tech Stack:** PySide6 (offscreen for tests and rendering), pytest (no pytest-qt: QApplication fixture convention from `tests/test_detectkit_review_bar.py`).

**Spec:** `docs/superpowers/specs/2026-10-09-sahi-unification-design.md` §5 (UI), §6 F5/F7. Prior plans: `...-s1-s2.md`, `...-s3.md` (deviations 1–21 apply).

## Revision 1 — binding decisions from the plan-stage adversarial review (2026-10-09)

These override anything below that conflicts.

**Split.** S4a = Tasks 15, 16, 17, 19 (DetectKit hosts only) on `feat/sahi-unify-s4a`, gate = regression + adversarial review + DetectKit gallery (no tracking path touched, no equivalence run). S4b = Task 18 + the TrackerKit gallery on `feat/sahi-unify-s4b`, branched after S4a merges, gate = full MPS matrix + CUDA (diptera, explicit idle GPU UUID) + adversarial review.

22a. **(S4a review refinement)** The derived value `max(fraction)+margin` is shown as the WHOLE-ANIMAL MINIMUM: below it → warning + "Raise to X"; at/above it → muted "≥ whole-animal minimum (X)", no button. A deliberate higher overlap (SAM3 escalation 0.5, SAM3 training 0.25) is never nudged down.
22. **F7 is a suggestion, never an automatic default** (B1, M8; **confirmed by the user 2026-10-09**: "Suggest, don't apply"). Every host already persists an overlap (TrackerKit 0.2 via engine default, `SliceTrainingSettings.overlap` 0.2, `Sam3LoraParams.tile_overlap` 0.25, SAM3 escalation 0.5), so "derive when unsaved" would either be inert or change outputs. The widget shows `Suggested: <max(fraction)+margin>` next to overlap with a `Use suggested` button (a user edit); it never writes overlap on its own. `set_overlap_source` is removed from the API.
23. **One ownership model** (B2): the widget sets its own controls ONLY inside `set_spec(...)` (signals blocked, then derived labels refreshed). Derived values (tile size outside custom, suggested overlap, resolved tile label) are shown in LABELS, never written into host-persisted spins. User edits emit `field_changed(str field)`; hosts keep their own per-spin `valueChanged` wiring (the spins ARE the widget's spins). TrackerKit's panel never calls `set_spec` (S4b keeps its existing `setValue` + `_applying_slice_profile` guard logic verbatim).
24. **Full frame in escalation roles** (B3): `escalate_sam3`/`escalate_sam2` fraction spin range `0.0..FRACTION_MAX` with special text "full frame (no tiling)"; `spec()` maps 0 → `enabled=False, object_tile_fractions=()`; `set_spec(enabled=False)` shows 0. Hosts keep mapping 0 ↔ None as today.
25. **Role capabilities instead of one layout** (M4, M5, M6, M7, M11): `SliceSettingsWidget(role=..., capabilities=...)`; capabilities = `body_override: bool` (DetectKit inference dialog + escalation dialogs True; TrackerKit False → body is a display-only badge), `advanced_merge: bool` (DetectKit inference True; TrackerKit False — merge settings stay profile-owned, S3 deviation 14), `tile_label_formatter: Callable | None` (escalation hosts pass their existing formatters so pinned label texts stay: SAM3 `"{tile} px\n{body} px / {frac:.2f}"`, SAM2 `"{tile} px ({body} px / {frac:g})"`, and their "tiling is off" strings). A body of 0 shows the spin ENABLED with "unknown (tiling off)" (the I6 last link), regardless of override. `train_sam3` keeps a scalar `spin_slice_object_fraction` (4 decimals) carried in `extras["object_tile_fraction"]`, never derived from the set.
26. **Per-role extras** (answer to attack 4): `train_yolo` `{negative_tile_fraction, full_frame_mix, balance_multiscale_loss, balance_multiscale_loss_power, min_area_ratio}`; `train_sam3` `{object_tile_fraction, keep_empty_tiles, full_frame_mix, min_area_ratio}`; `infer_yolo` (DetectKit) `{merge_threshold}`; `infer_yolo` (TrackerKit) `{tile_batch_size, memory_budget_mib}`; `escalate_sam3` `{seam_margin_px, merge_iou}`; `escalate_sam2` `{}` (overlap is the constant 0.5, displayed disabled).
27. **Overlap spin max per role from the role's validator** (F5, m5): `train_sam3` 0.99 (contract `[0, 1)` — no silent clamp of a saved 0.95), every other role `OVERLAP_MAX` 0.9.
28. **Legacy names survive in DetectKit** (M2, M3): `SliceSettingsGroup` (subclass) keeps read-only property aliases for every old attribute used in tests (`chk_enabled, cmb_mode, txt_targets, spin_w, spin_h, spin_overlap, spin_object_fraction, spin_min_area, spin_neg, spin_merge, chk_full, chk_balance_loss, spin_balance_power, chk_keep_empty, auto_reference_note, _rows, preview`) and an outer `QHBoxLayout` (controls | preview). Controls absent for a role are CONSTRUCTED and hidden. Existing DetectKit tests then need no edits except where an assertion flips from hidden→disabled for a now-constrained field; enumerate each such flip in the commit message.
29. **Geometry combo readers** (M3): fix every `combo_slice_geometry.currentText()/setCurrentText()/findText()` in src AND tests: `detection_panel.py:~2476,~2867`, `config.py:454,503,1771`, `tests/test_main_window_config_persistence.py:1954,1971,2011`, `tests/test_trackerkit_slice_meta_prefill.py:88,153,159` (S4b).
30. **TrackerKit layout bug fix** (M1): S4b gives the widget explicit full-width rows in `f_yolo`, removing the `(7,1)` overlap that today paints the profile combo UNDER the Classes field (pre-existing bug, unreachable profile picker); add a test that no two visible `f_yolo` items share a grid cell.
31. **Calibrate stays host-owned** (m4): TrackerKit `setup_panel` button and the dialogs' `_btn_calibrate` stay where they are; spec §5.3's "reachable from the widget" is satisfied by placement next to the widget, not inside it.
32. **Tests fixed** (M9, M10): tooltip test uses `findChildren(cls)` per class and skips children whose parent is a spin box/combo (internal line edits); combo-tooltip test checks `itemData(i, Qt.ItemDataRole.ToolTipRole)`; drop `isVisible() is not None`; `changed`-signal test becomes `field_changed` emitted on `setValue` (user path) and NOT inside `set_spec`; Task 17 compares `parameters()` keys to a pasted literal of today's 11 keys. New tests: escalate full-frame round trip incl. `apply_calibration_choice(tile_fraction=None)`; SAM3 scalar preserved after editing the set; SAM3 escalation fresh dialog overlap 0.5; F6 display of a saved `merge_policy="nmm"` (shows "greedy NMM", raw value preserved in `extras`/advanced for the cache hash); TrackerKit fresh session + fraction 0.4 + no saved overlap → `SLICE_OVERLAP` 0.2 and no `slice_overlap` key written (S4b); every TrackerKit `advanced_config` slice key equals its spin after config load, profile apply, geometry change (S4b).
33. **Gallery** (m6): grab the detection panel's scroll CONTENT (or `panel.resize(panel.sizeHint())` first); `docs/superpowers/assets/sahi-unification/` is created by the tool; the gallery is shown to the user via the `SendUserFile` tool if available, else the PNG paths are listed for them.

## Global Constraints

- TrackerKit tracking output byte-identical; config JSON keys/values unchanged (`slice_geometry_mode` persists the ENUM string, never the human label); `get_parameters_dict`/`build_engine_params` outputs unchanged for every existing config.
- `widgets/` imports nothing from app layers (add an AST test enforcing it).
- Widget attribute names follow TrackerKit (`chk_slice_enabled`, `combo_slice_geometry`, `combo_slice_profile`, `lbl_slice_profile_status`, `spin_slice_overlap`, `spin_slice_tile_w`, `spin_slice_tile_h`, `spin_slice_object_fraction`, `spin_slice_tile_batch`, `spin_slice_memory_budget`). New names use the same `*_slice_*` pattern.
- Geometry combo items: human label text ("Fit to animal size" / "Use model input size" / "Custom tile size"), enum as item data and in the tooltip; every reader uses `currentData()`/`findData()`.
- Constrained fields are disabled with a tooltip naming their source; never hidden for being constrained (rows hidden only when the ROLE has no such field).
- Ranges come from `utils.tiling_spec` constants (`FRACTION_MIN/MAX`, `OVERLAP_MAX`, slice size 0..8192).
- Derived overlap (F7) applies only when no overlap is saved; saved configs keep their value.
- Worktree from local HEAD; `make format` (revert `core/post/merge.py` churn); no Co-Authored-By; never `git stash`; PYTHONPATH pinned; `QT_QPA_PLATFORM=offscreen`.

## Review Focus

1. Saving and reopening a TrackerKit config round-trips `slice_geometry_mode` as the enum (not "Fit to animal size") and the golden `get_parameters_dict` SLICE_* keys are unchanged. → Task 18.
2. The SAM3 dialog's `parameters()` and SAM2's `tiling_parameters()` emit exactly the request fields they did before (constructors raise on unknown keys). → Task 17.
3. A derived value the user cannot edit (body px from dataset/stamp, tile size outside custom, overlap without override) is visibly disabled with its source badge — never an editable spin that silently gets overwritten. → Task 15.
4. A saved overlap (including TrackerKit's persisted 0.2 and SAM3's 0.25) is never replaced by the derived value. → Task 15, 18.
5. Every host renders without clipped/overlapping controls at its real size (visual gallery inspected). → Task 19.

---

Setup (after S3 is merged):

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
git worktree add .worktrees/sahi-s4 -b feat/sahi-unify-s4 HEAD
cd .worktrees/sahi-s4
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps
export PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE QT_QPA_PLATFORM=offscreen
```

### Task 15: `widgets/slice_settings.py` — the shared widget

**Files:**
- Create: `src/hydra_suite/widgets/slice_settings.py` (widget + `_TileLayoutPreview` moved from `detectkit/gui/panels/slice_settings_widget.py`)
- Create: `tests/test_widgets_slice_settings.py`, `tests/test_widgets_no_app_imports.py`

**Interfaces (Produces):**
- `ROLES = ("infer_yolo", "train_yolo", "train_sam3", "escalate_sam3", "escalate_sam2")`
- `class SliceSettingsWidget(QGroupBox)`:
  - `__init__(self, parent=None, *, role: str, title: str | None = None)`; `ValueError` on unknown role.
  - `changed = Signal()` — emitted on any user edit (not on programmatic `set_*`).
  - `set_spec(spec: TilingSpec, *, extras: dict | None = None) -> None` / `spec() -> TilingSpec` / `extras() -> dict`.
  - `set_model_input_size(imgsz: int)`, `set_preview_frame_size(wh | None)`, `set_preview_frame_options(list[(w, h, count)])`.
  - `set_reference_body(value: float, source: str)` — shows the value + badge; `spin_slice_body` disabled unless `chk_slice_body_override` is checked.
  - `set_overlap_source(saved: float | None)` — when `saved is None`, overlap is derived via `resolve_overlap(fractions=...)` and the spin is disabled unless `chk_slice_overlap_override`.
  - `source_badge(field: str) -> str` (for tests): `"user" | "override" | "profile" | "stamped" | "dataset" | "derived" | "default"`.
- Rows by role (hidden only when the role has no such field):

| Row | infer_yolo | train_yolo | train_sam3 | escalate_sam3 | escalate_sam2 |
|---|---|---|---|---|---|
| `chk_slice_enabled` | ✓ | ✓ | – | – | – |
| `combo_slice_profile` + `lbl_slice_profile_status` | ✓ (host fills) | – | – | – | – |
| `combo_slice_geometry` | ✓ | ✓ | ✓ | – | – |
| `spin_slice_object_fraction` (single scale) | ✓ | – | – | ✓ (0 = full frame) | ✓ (0 = full frame) |
| `txt_slice_scales` (scale set) | – | ✓ | ✓ | – | – |
| body (`spin_slice_body`, override, badge) | ✓ | ✓ (note only, measured at build) | ✓ (note only) | ✓ | ✓ |
| tile size W×H (`spin_slice_tile_w/h`, editable only in custom; derived label otherwise) | ✓ | ✓ | ✓ | derived label | derived label |
| overlap (`spin_slice_overlap`, override) | ✓ | ✓ | ✓ | ✓ | – (constant 0.5, shown disabled) |
| preview | ✓ | ✓ | ✓ | – | – |
| Advanced: merge policy/metric/threshold | ✓ | threshold | – | merge IoU + seam margin | – |
| Advanced: min area + fragment policy | – | ✓ (drop/mask) | ✓ (crowd) | – | – |
| Advanced: full frame (`perform_standard_pred` / `full_frame_mix`) | ✓ | ✓ | ✓ | – | – |
| Advanced: tiles per call / memory | ✓ | – | – | – | – |
| Advanced: negative fraction, loss balance | – | ✓ | keep empty tiles | – | – |

- [ ] **Step 1: Failing tests** (excerpt — write all of these):

```python
# tests/test_widgets_slice_settings.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtWidgets import QApplication

from hydra_suite.utils.tiling_spec import FRACTION_MAX, FRACTION_MIN, OVERLAP_MAX, TilingSpec
from hydra_suite.widgets.slice_settings import ROLES, SliceSettingsWidget

_app = QApplication.instance() or QApplication([])


@pytest.mark.parametrize("role", ROLES)
def test_every_role_round_trips_its_spec(role):
    w = SliceSettingsWidget(role=role)
    spec = TilingSpec(enabled=True, geometry_mode="auto_object", object_tile_fractions=(0.1,),
                      reference_body_px=40.0, overlap=0.3)
    w.set_spec(spec)
    got = w.spec()
    assert got.object_tile_fractions == (0.1,)
    if role != "escalate_sam2":
        assert got.overlap == 0.3


def test_geometry_combo_shows_labels_but_stores_enums():
    w = SliceSettingsWidget(role="infer_yolo")
    assert [w.combo_slice_geometry.itemData(i) for i in range(3)] == ["auto_model", "auto_object", "custom"]
    assert "auto_object" in w.combo_slice_geometry.itemData(1)
    assert w.combo_slice_geometry.itemText(1) == "Fit to animal size"


def test_ranges_come_from_the_contract():
    """F5: SAM3 overlap used to accept 1.0."""
    for role in ("train_sam3", "infer_yolo", "escalate_sam3"):
        w = SliceSettingsWidget(role=role)
        assert w.spin_slice_overlap.maximum() == OVERLAP_MAX
    w = SliceSettingsWidget(role="infer_yolo")
    assert (w.spin_slice_object_fraction.minimum(), w.spin_slice_object_fraction.maximum()) == (FRACTION_MIN, FRACTION_MAX)


def test_tile_size_editable_only_in_custom():
    w = SliceSettingsWidget(role="infer_yolo")
    w.set_spec(TilingSpec(enabled=True, geometry_mode="auto_object", object_tile_fractions=(0.1,), reference_body_px=50.0))
    assert not w.spin_slice_tile_w.isEnabled() and w.spin_slice_tile_w.isVisible() is not None
    assert w.source_badge("tile_size") == "derived"
    assert "500" in w.lbl_slice_tile_size.text()
    w.combo_slice_geometry.setCurrentIndex(w.combo_slice_geometry.findData("custom"))
    assert w.spin_slice_tile_w.isEnabled()


def test_body_is_read_only_until_override():
    w = SliceSettingsWidget(role="escalate_sam3")
    w.set_reference_body(82.2, "dataset")
    assert not w.spin_slice_body.isEnabled()
    assert w.source_badge("reference_body_px") == "dataset"
    w.chk_slice_body_override.setChecked(True)
    assert w.spin_slice_body.isEnabled()
    assert w.source_badge("reference_body_px") == "override"


def test_derived_overlap_only_when_unsaved():
    """F7 + Review Focus 4."""
    w = SliceSettingsWidget(role="train_yolo")
    w.set_spec(TilingSpec(enabled=True, geometry_mode="auto_object", object_tile_fractions=(0.05, 0.15)))
    w.set_overlap_source(None)
    assert w.spin_slice_overlap.value() == pytest.approx(0.2)
    assert w.source_badge("overlap") == "derived" and not w.spin_slice_overlap.isEnabled()
    w.set_spec(TilingSpec(enabled=True, overlap=0.25, object_tile_fractions=(0.05, 0.15)))
    w.set_overlap_source(0.25)
    assert w.spin_slice_overlap.value() == 0.25 and w.source_badge("overlap") == "user"


def test_changed_signal_only_on_user_edits():
    w = SliceSettingsWidget(role="infer_yolo")
    hits = []
    w.changed.connect(lambda: hits.append(1))
    w.set_spec(TilingSpec(enabled=True, overlap=0.3))
    assert hits == []
    w.spin_slice_overlap.setValue(0.4)  # simulates the user (signals not blocked)
    assert hits


@pytest.mark.parametrize("role", ROLES)
def test_every_enabled_control_has_a_tooltip(role):
    from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit

    w = SliceSettingsWidget(role=role)
    for child in w.findChildren((QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit)):
        assert child.toolTip().strip(), child.objectName() or type(child).__name__
```

```python
# tests/test_widgets_no_app_imports.py
import ast
from pathlib import Path

APP_LAYERS = ("trackerkit", "posekit", "classkit", "refinekit", "detectkit", "filterkit", "launcher")


def test_widgets_import_no_app_layer():
    root = Path(__file__).resolve().parents[1] / "src" / "hydra_suite" / "widgets"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert not any(f"hydra_suite.{app}" in name or name.startswith(f"{app}") for app in APP_LAYERS), (path, name)
```

Adapt `test_tile_size_editable_only_in_custom`'s visibility assertion to the widget's real layout (rows are shown for the role; the point is *disabled*, not hidden). Use the realized tile from `tile_size_for_mode` (body 50 / 0.1 = 500 px).

- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3:** Implement. Move `_TileLayoutPreview` verbatim (plus `model_input_size` from `set_model_input_size`, no literal 640). Build rows from the table; constrained → `setEnabled(False)` + tooltip "Derived from <source>; check Override to edit" (or "Editable in Custom tile size"). Badges: a small `QLabel` per derived row with the source word, styled muted. Advanced fields in a collapsible `QToolButton`-toggled frame, collapsed by default. All programmatic setters block signals; user edits emit `changed`.
- [ ] **Step 4:** Run both new files → PASS.
- [ ] **Step 5:** Commit `feat(widgets): shared SAHI settings widget with roles, badges, contract ranges`.

### Task 16: DetectKit training + inference hosts adopt the widget

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/slice_settings_widget.py` → keep the module and the `SliceSettingsGroup` name as a thin subclass of `SliceSettingsWidget` (`role="train_yolo"` for `backend="yolo"`, `role="train_sam3"` for `backend="sam3"`) that keeps `load_from`/`to_settings`/`load_sam3_tiling`/`to_sam3_tiling` via a new adapter module
- Create: `src/hydra_suite/detectkit/gui/panels/slice_settings_adapter.py` — `settings_to_spec(SliceTrainingSettings) -> (TilingSpec, extras)`, `spec_to_settings(spec, extras, base) -> SliceTrainingSettings`, `sam3_tiling_to_spec(dict) -> (spec, extras)`, `spec_to_sam3_tiling(spec, extras) -> dict` (exactly the 9 keys `to_sam3_tiling` returns today)
- Modify: `src/hydra_suite/detectkit/gui/dialogs/inference_settings.py` — its SAHI group becomes `SliceSettingsWidget(role="infer_yolo")`; `settings()`/`load_from()` contracts unchanged
- Update tests that reference old attribute names: `tests/test_detectkit_slice_ui.py`, `tests/test_sam3_slice_settings_shared.py`, `tests/test_detectkit_inference_settings.py`, `tests/test_detectkit_training_dialog.py` (only name changes `cmb_mode`→`combo_slice_geometry` via `findData`, `txt_targets`→`txt_slice_scales`, `spin_w/h`→`spin_slice_tile_w/h`, `spin_overlap`→`spin_slice_overlap`, `spin_object_fraction`→`spin_slice_object_fraction`, `_rows` label lookups → widget accessors; and the `QHBoxLayout` assertion if the outer layout changes). Behavioral assertions stay.

- [ ] **Step 1:** Write `tests/test_detectkit_slice_adapter.py`: round-trip every `SliceTrainingSettings` field and every `to_sam3_tiling` key through the adapter (property-style over a few hand cases incl. legacy pixel-only settings and multi-scale sets); `spec_to_sam3_tiling` emits exactly `set(Sam3LoraParams fields) ∩ tiling keys` as today; F5: `spin_slice_overlap` max 0.9 in the SAM3 panel and `to_sam3_tiling()["tile_overlap"] < 1`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement adapter + host swaps.
- [ ] **Step 4:** Run the new test + the four updated files + `tests/test_sam3_gui_cli_training_parity.py tests/test_sam3_dialog_wiring.py tests/test_detectkit_fraction_only_settings.py tests/test_detectkit_preview_tiling.py` → PASS.
- [ ] **Step 5:** Commit `refactor(detectkit): training and inference hosts use the shared SAHI widget`.

### Task 17: Escalation dialogs adopt the widget's tiling rows

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/semantic_escalation_dialog.py` — body / tile fraction / tile label / overlap / merge IoU / seam margin rows come from `SliceSettingsWidget(role="escalate_sam3")`; keep `_reference_body`, `_tile_fraction`, `_overlap`, `_merge_iou`, `_seam_margin`, `_tile_label` as attributes ALIASING the widget's controls so `parameters()`, `apply_calibration_choice`, `prefill_from_sidecar`, persistence and existing tests keep working
- Modify: `src/hydra_suite/detectkit/gui/dialogs/escalate_sam2_dialog.py` — same with `role="escalate_sam2"`; `tiling_parameters()` unchanged
- Update only grid-position assertions in `tests/test_semantic_escalation_dialog_persistence.py:201-230` if the layout moves; text/value assertions stay

- [ ] **Step 1:** Write `tests/test_escalation_dialogs_use_shared_widget.py`: each dialog contains exactly one `SliceSettingsWidget` with the right role; `parameters()` key set equals `SemanticEscalationRequest` tiling+run fields used today (compare to the set captured from main before the change — paste it as a literal); `tiling_parameters()` keys == `{"reference_body_px", "tile_fraction", "overlap"}`; the body row shows a source badge (`dataset` when seeded from the project, `stamped` after `prefill_from_sidecar`).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the new test + `tests/test_semantic_escalation_dialog_persistence.py tests/test_detectkit_sam2_escalation_wiring.py tests/test_sam3_model_selection.py tests/test_geometry_review_fixes.py tests/test_escalate_sam2_dialog.py tests/test_semantic_default_tiling.py` → PASS.
- [ ] **Step 5:** Commit `refactor(detectkit): escalation dialogs use the shared SAHI widget rows`.

### Task 18: TrackerKit detection panel adopts the widget

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` (~:757-880 construction; visibility handlers; `_sync_advanced`; `_apply_slice_meta_values`) — build `SliceSettingsWidget(role="infer_yolo")` inside `yolo_group`; bind the panel's existing attribute names to the widget's controls (`self.chk_slice_enabled = w.chk_slice_enabled`, etc.) so the profile state machine and callers are unchanged
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:455,501,1771` — `combo_slice_geometry.currentText()`/`setCurrentText()` → `currentData()`/`setCurrentIndex(findData(...))`; grep `combo_slice_geometry` across `src/` and fix every reader
- Update: `tests/test_detection_panel_slice_widgets.py:21-27` (items now asserted via `itemData`), visibility tests (`isHidden` → `isEnabled` where the field is now constrained-but-visible), `tests/test_trackerkit_profile_session.py:214` and any `currentText()` geometry assertion → `currentData()`

- [ ] **Step 1:** Write `tests/test_trackerkit_sahi_widget_persistence.py`: build a main window via `tests/test_main_window_config_persistence._make_main_window`; set each geometry mode; save the config dict; assert `slice_geometry_mode` is the enum; reload a config with each enum and assert the combo's `currentData()`; assert `get_parameters_dict()` SLICE_* keys equal those from main for the same config (capture main's values as literals first, on the unmodified tree).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Execution knobs (tiles per call, memory, admission label) live in the widget's Advanced section for `infer_yolo`; the panel keeps `_update_slice_batch_admission_label` wired to them.
- [ ] **Step 4:** Run the new test + `tests/test_detection_panel_slice_widgets.py tests/test_trackerkit_slice_meta_prefill.py tests/test_trackerkit_profile_session.py tests/test_gui_cli_profile_parity.py tests/test_main_window_config_persistence.py tests/test_engine_params_slice_profile.py tests/test_get_parameters_dict_characterization.py tests/test_trackerkit_preview_worker.py` → PASS (the characterization golden fails on main for DATASET_* keys only — confirm no SLICE_* key differs).
- [ ] **Step 5:** Commit `refactor(trackerkit): detection panel uses the shared SAHI widget`.

### Task 19: Visual gallery

**Files:**
- Create: `tools/sahi_widget_gallery.py` — renders, offscreen, PNGs into an output dir: (a) the bare widget for every role with representative state (derived badges visible, Advanced collapsed and expanded); (b) the REAL hosts at their natural size: TrackerKit main window on the "Find Animals" tab, YOLO direct mode, SAHI on, with a model sidecar carrying two profiles (recipe: `tests/test_main_window_config_persistence._make_main_window`, `_show_workspace()`, `tabs.setCurrentWidget(_detection_panel)`, `combo_detection_method.setCurrentIndex(1)`, `combo_yolo_obb_mode.setCurrentIndex(0)`, `chk_slice_enabled.setChecked(True)`, `apply_slice_meta_for_model(path)`), DetectKit `TrainingDialog` (overview page), `Sam3TrainingPanel`, `InferenceSettingsDialog`, `SemanticEscalationDialog`, `EscalateSam2Dialog` (constructors per `tests/` fixtures; hermetic `HYDRA_DATA_DIR`/`HYDRA_CONFIG_DIR` in a temp dir).
- Test: `tests/test_sahi_widget_gallery.py` — runs the tool into `tmp_path` and asserts one non-empty PNG per expected name.

- [ ] **Step 1:** Write the test (expects files `role_<role>.png`, `role_<role>_advanced.png`, `host_trackerkit.png`, `host_detectkit_training.png`, `host_sam3_training.png`, `host_inference_settings.png`, `host_sam3_escalation.png`, `host_sam2_escalation.png`).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement with `widget.grab().save(path)`; `QApplication.processEvents()` before grabbing; set a fixed window size per host. **Step 4:** PASS.
- [ ] **Step 5:** Generate the gallery into `docs/superpowers/assets/sahi-unification/` and LOOK at every PNG (Read tool). Fix clipped/overlapping layout, missing badges, wrong enablement; regenerate. Commit tool + test + PNGs: `docs(sahi): visual gallery of the shared widget in every host`.

### Task 20: S4 gate

- [ ] **Step 1:** Regression: every test file named in Tasks 15–19 + `tests/test_tiling_*.py tests/test_slice_*.py tests/test_detectkit_*.py tests/test_sam3_*.py tests/test_trackerkit_*.py tests/test_core_import_is_light.py`; `make lint`; compare failure sets to main.
- [ ] **Step 2:** Equivalence MPS full matrix (`WT=$PWD` main checkout for harness+fixtures, `WT_SRC` = worktree src) — TrackerKit persistence changed, so all clips. CUDA on diptera per S3 deviation 20 (or mehek if back online).
- [ ] **Step 3:** Adversarial review (Fable): execute the gallery and the hosts; try to break round-trips (save/reload in both kits, profile switching to Custom, derived→override→back, role-hidden fields leaking values), widget layering, and GUI/CLI parity (`tests/test_gui_cli_profile_parity.py` semantics). Plus a normal reviewer. Fix wave; re-review if the fix exceeds the reported lines.
- [ ] **Step 4:** Present the gallery PNGs to the user for visual confirmation (SendUserFile), then merge `--no-ff`.
