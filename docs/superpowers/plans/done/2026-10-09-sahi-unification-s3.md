# SAHI Unification — Slice S3 (callers + behavioral fixes) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move DetectKit's SAHI callers onto the S1/S2 contract and fix F1 (640 anchor), F2 (preview ignores the model sidecar), F3 (headless SAM3 never tiles), F4/F8 (scattered defaults), with TrackerKit tracking output byte-identical.

**Architecture:** One defaults table (`utils/tiling_spec.BACKEND_DEFAULTS`) feeds every dataclass default; DetectKit's inference/training settings become fraction-only; the DetectKit preview resolves its tiling from the model's `.slice_meta.json` through the SAME ladder TrackerKit uses (`slice_meta_to_panel_values`); SAM3 escalation gets a `default_semantic_tiling` shared by the dialog and the CLI (mirrors SAM2's `default_geometry_tiling`).

**Tech Stack:** Python 3, PySide6 (offscreen for tests), numpy, pytest. Env `hydra-mps`.

**Spec:** `docs/superpowers/specs/2026-10-09-sahi-unification-design.md` (§6 F1–F8). Prior slices: `docs/superpowers/plans/2026-10-09-sahi-unification-s1-s2.md` (deviations 1–13 still apply).

## Global Constraints

- TrackerKit tracking output byte-identical: `SLICE_*` keys/defaults, `SliceConfig` fields, `_slice_config_hash`, `tests/data/get_parameters_dict_golden/*.json` unchanged. TrackerKit is the canonical framework; S3 changes only its literal defaults into named constants with the SAME values.
- `REFERENCE_BODY_SIZE` is never written.
- `merge_policy="nmm"` keeps flowing raw through TrackerKit/engine params and the cache hash (F6 resolved: no UI exposes it; `canonicalize` reads it as `greedy_nmm`; nothing else changes). Do NOT edit the 10 test files that assert `"nmm"`.
- Layering: `utils` ← `core`/`training` ← app layers. `detectkit` may import `core.inference.tiling_meta` / `slice_meta`.
- Worktree from local HEAD; `make format` (revert unrelated `core/post/merge.py` churn); no Co-Authored-By trailer; never `git stash`; PYTHONPATH pinned to the worktree `src`.

## Decisions recorded for S3 (deviations 14–18)

14. **F6 needs no code**: no UI exposes `merge_policy`; `nmm` stays a raw pass-through value for cache-hash stability.
15. **SAM3 escalation seed stays 0.05** (plan review M6): it is on the calibration grid `TILE_FRACTION_GRID`, the dialog label/persistence tests and staged-cache fingerprints depend on it, and the table's `sam3` entry (0.055) is the TRAINING default. `SEMANTIC_TILE_FRACTION_SEED` gets a comment naming that split; no value changes.
16. **DetectKit preview owns `enabled`, body px and imgsz; the model owns geometry**: when the user has NOT opened the inference-settings override, the preview takes geometry mode, scale, overlap, tile size and merge settings from the model's sidecar (primary profile, else training geometry) via `slice_meta_to_panel_values`, TrackerKit's fresh-load ladder. Body px follows spec §3.3 (dataset first): the project's label-measured `slice_settings.reference_body_px` if > 0, else the stamped `trained_body_px`. imgsz is the project's `imgsz_obb_direct` (the user's knob; TrackerKit serves at its own imgsz too). With the override dialog open, its values win.
17. **Deferred to after S4**: removing geometry from `.sam3_meta.json` (dual-write stays) and moving TrackerKit's own ladder onto `read_tiling_meta` (TrackerKit is canonical and its ladder already reads v3 additively; moving it buys nothing and risks byte-identity).
18. **F4 keeps the GUI field default `[]`** (plan review M1): a non-empty fractions default would silently override pixel `target_sizes` passed to constructors. Instead the table is the single source by deriving the pixel default `target_sizes = [f * LEGACY_TARGET_SIZE_IMGSZ for f in table]` (same `[32, 64, 96, 128]`) and `SliceTrainingConfig.target_size_fractions` from the table; GUI and headless `target_fractions()` already agree and are pinned by a test.
19. **Scope honesty** (plan review m4): after S3, `TilingSpec`/`canonicalize` and `read_tiling_meta` still have no production callers (the preview uses `slice_meta_to_panel_values`, TrackerKit's own ladder); YOLO frame-level `AreaBand`, SAM2 job and DetectKit calibration callers are untouched. The shared widget (S4) is their first production consumer.
20. **CUDA gate host**: mehek is offline (last seen 4 days ago); the CUDA gate runs on diptera with `CUDA_MAJOR=12`, an explicitly chosen idle GPU UUID from `nvidia-smi` (never `--gpus auto`; another session currently uses GPU 2), cleanup scoped to `pgrep -u rishika`.
21. **Training widget stops deriving pixels**: `to_settings()` emits `target_size_fractions` only; `target_sizes` keeps its dataclass default and is ignored whenever fractions are present (`target_fractions_from` rule).

## Review Focus

1. A DetectKit project whose model has a calibrated primary profile previews with that profile's fraction/overlap/merge — the same numbers TrackerKit would use for that model with a fresh config. → Task 12.
2. A project saved before S3 (pixel `target_sizes`, empty fractions) opens with the same effective scales it had. → Task 11.
3. `detectkit escalate sam3` with no flags tiles exactly as the dialog would open (saved → calibrated → stamped → full frame), never silently full-frame when a body size and fraction are known. → Task 13.
4. Changing `slice_meta.json` (e.g. recalibrating) invalidates the DetectKit preview cache. → Task 12.
5. TrackerKit `get_parameters_dict` goldens unchanged and equivalence byte-identical on MPS and CUDA. → Task 14.

---

Setup:

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
git worktree add .worktrees/sahi-s3 -b feat/sahi-unify-s3 HEAD
cd .worktrees/sahi-s3
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps
export PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE QT_QPA_PLATFORM=offscreen
```

### Task 10: One defaults table (F4, F8)

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py:175,852,873` (`SliceConfig.object_tile_fraction`, `_slice_config_from_params` fallbacks) — reference `BACKEND_DEFAULTS["yolo_infer"]` / `DEFAULT_OVERLAP` (same values)
- Modify: `src/hydra_suite/trackerkit/engine_params.py:1167,1170` and `src/hydra_suite/trackerkit/gui/panels/detection_panel.py:817,837-840` — same constants (same values)
- Modify: `src/hydra_suite/detectkit/gui/models.py:169-190` — `SliceTrainingSettings.target_sizes` default = `[f * LEGACY_TARGET_SIZE_IMGSZ for f in BACKEND_DEFAULTS["yolo_train"].object_tile_fractions]` (same values); `target_size_fractions` default stays `[]` (deviation 18)
- Modify: `src/hydra_suite/detectkit/config/training.py:204` — `target_size_fractions` default from the table
- Modify: `src/hydra_suite/training/contracts.py:273` — `Sam3LoraParams.object_tile_fraction` from `BACKEND_DEFAULTS["sam3"]`
- Modify: `src/hydra_suite/core/inference/semantic/tiling.py:34` — comment only (deviation 15): the escalation seed (0.05, on `TILE_FRACTION_GRID`) is deliberately distinct from the SAM3 training default in `BACKEND_DEFAULTS["sam3"]`
- Test: `tests/test_slice_config_sahi_path_guard.py` — SAHI-ENABLED byte-identity guard (plan review M7; no equivalence fixture enables slicing)
- Test: `tests/test_tiling_defaults_single_source.py`

- [x] **Step 1: Failing tests**

```python
# tests/test_tiling_defaults_single_source.py
from hydra_suite.core.inference.config import SliceConfig, _slice_config_from_params
from hydra_suite.core.inference.semantic.tiling import SEMANTIC_TILE_FRACTION_SEED
from hydra_suite.detectkit.config.training import SliceTrainingConfig
from hydra_suite.detectkit.gui.models import SliceTrainingSettings
from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.utils.tiling_spec import BACKEND_DEFAULTS, DEFAULT_OVERLAP


def test_inference_defaults_come_from_the_table():
    assert SliceConfig().object_tile_fraction == BACKEND_DEFAULTS["yolo_infer"].object_tile_fractions[0] == 0.15
    cfg = _slice_config_from_params({}, "SLICE_", reference_body_px=0.0)
    assert cfg.object_tile_fraction == 0.15
    assert cfg.overlap_width_ratio == DEFAULT_OVERLAP == 0.2


def test_training_scale_set_defaults_come_from_one_table():
    """F4: GUI and headless resolve the same set, both from the table."""
    expected = list(BACKEND_DEFAULTS["yolo_train"].object_tile_fractions)
    assert SliceTrainingSettings().target_fractions() == expected
    assert list(SliceTrainingConfig().target_fractions()) == expected
    assert list(SliceTrainingConfig().target_size_fractions) == expected
    assert SliceTrainingSettings().target_sizes == [32.0, 64.0, 96.0, 128.0]  # derived, same values


def test_constructor_pixels_still_win_over_the_default():
    """Plan review M1: a non-empty fractions default would have ignored these."""
    assert SliceTrainingSettings(target_sizes=[200.0]).target_fractions() == [200.0 / 640.0]


def test_sam3_defaults_come_from_the_table():
    assert Sam3LoraParams.__dataclass_fields__["object_tile_fraction"].default == 0.055
    assert SEMANTIC_TILE_FRACTION_SEED == 0.05  # escalation seed, deviation 15


def test_legacy_project_without_fractions_keeps_its_scales():
    """Review Focus 2: an old project (pixel target_sizes, no fractions key)."""
    old = SliceTrainingSettings.from_dict({"target_sizes": [64.0, 128.0]})
    assert old.target_fractions() == [0.1, 0.2]
```

```python
# tests/test_slice_config_sahi_path_guard.py
"""SAHI-ENABLED byte-identity guard: no equivalence fixture enables slicing."""
from dataclasses import asdict

from hydra_suite.core.inference.cache.keys import _slice_config_hash
from hydra_suite.core.inference.config import _slice_config_from_params

# Captured from main BEFORE Task 10 (Step 1 records the literal values below by
# running this module's _snapshot() on the unmodified tree and pasting them).
PARAMS = {"SLICE_ENABLED": True, "SLICE_GEOMETRY_MODE": "auto_object", "REFERENCE_BODY_SIZE": 40.0,
          "RESIZE_FACTOR": 1.0}


def _snapshot():
    cfg = _slice_config_from_params(PARAMS, "SLICE_", reference_body_px=40.0)
    return asdict(cfg), _slice_config_hash(cfg)


def test_sahi_enabled_slice_config_is_unchanged():
    cfg, digest = _snapshot()
    assert cfg == EXPECTED_CFG
    assert digest == EXPECTED_HASH
```

Step 1 for this file: on the UNMODIFIED worktree run `python -c "from tests.test_slice_config_sahi_path_guard import _snapshot; print(repr(_snapshot()))"` (define `EXPECTED_CFG = EXPECTED_HASH = None` first so it imports), paste the printed dict and hash as `EXPECTED_CFG`/`EXPECTED_HASH`, and confirm the test passes BEFORE any Task 10 change. Read `cache/keys.py` for `_slice_config_hash`'s real signature (it may take more arguments, e.g. a ROI flag) and adapt `_snapshot` accordingly.

```python
```

Before writing `test_legacy_project_without_fractions_keeps_its_scales`, read `SliceTrainingSettings.from_dict`: if a missing `target_size_fractions` key would now pick up the new non-empty default and override the legacy pixels, make `from_dict` use `[]` when the key is ABSENT (legacy) and the default only for a fresh instance. The test must pass for the reason stated.

- [x] **Step 2:** `python -m pytest tests/test_tiling_defaults_single_source.py -q` → FAIL (F4 + seed asserts).
- [x] **Step 3:** Replace each literal with the table/constant import. Values for TrackerKit/core must be bit-identical (0.15, 0.2). `_slice_config_from_params` keeps its clamp ranges.
- [x] **Step 4:** Run the new tests plus `tests/test_slice_config_sahi_path_guard.py tests/test_detectkit_slice_settings.py tests/test_resolve_scales.py tests/test_inference_config.py tests/test_engine_params_slice_profile.py tests/test_detection_panel_slice_widgets.py tests/test_semantic_tiling.py tests/test_sam3_slice_settings_shared.py tests/test_detectkit_training_cli.py tests/test_detectkit_inference_settings.py tests/test_semantic_escalation_dialog_persistence.py tests/test_get_parameters_dict_characterization.py` → all PASS with no edits to existing tests.
- [x] **Step 5:** Commit `refactor(tiling): one defaults table feeds every SAHI dataclass (F4, F8)`.

### Task 11: Fraction-only DetectKit settings (F1)

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/inference_settings.py` — replace `spin_target_size` (px ÷ 640) with `spin_object_fraction` (QDoubleSpinBox, range `FRACTION_MIN..FRACTION_MAX`, 3 decimals, step 0.005, label "Object scale (fraction of tile)"), plus a read-only `lbl_scale_px` showing `fraction × imgsz` px at the project's direct imgsz
- Modify: `src/hydra_suite/detectkit/gui/panels/slice_settings_widget.py` — `to_settings()` (~:623-645) stops emitting the 640 pixel list; help text and schematic use the real model input size (`self._model_input_size`), not a literal 640
- Test: `tests/test_detectkit_fraction_only_settings.py`; update `tests/test_detectkit_inference_settings.py:77` and `tests/test_sam3_slice_settings_shared.py:137` (they lock the 640 anchor — the bug)

- [x] **Step 1: Failing tests**

```python
# tests/test_detectkit_fraction_only_settings.py
import pytest

pytest.importorskip("PySide6")

from hydra_suite.detectkit.gui.models import InferenceRunSettings, SliceTrainingSettings


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    # Repo convention (no pytest-qt): see tests/test_detectkit_review_bar.py.
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture
def dialog():
    from hydra_suite.detectkit.gui.dialogs.inference_settings import InferenceSettingsDialog

    defaults = InferenceRunSettings(device="cpu", confidence_threshold=0.25, slice_settings=SliceTrainingSettings())
    return InferenceSettingsDialog(defaults, defaults, model_input_size=1024)  # see Step 3 for the new kwarg


def test_dialog_round_trips_a_fraction_without_640(dialog):
    dialog.chk_sliced.setChecked(True)
    dialog.spin_object_fraction.setValue(0.12)
    sliced = dialog.settings().slice_settings
    assert sliced.target_size_fractions == [0.12]
    assert sliced.object_tile_fraction == 0.12
    assert sliced.target_fractions() == [0.12]


def test_dialog_shows_pixels_at_the_real_input_size(dialog):
    dialog.spin_object_fraction.setValue(0.1)
    assert "102.4" in dialog.lbl_scale_px.text()  # 0.1 x 1024


def test_dialog_loads_a_legacy_pixel_project_as_its_fraction(dialog):
    legacy = SliceTrainingSettings.from_dict({"enabled": True, "target_sizes": [96.0]})
    dialog.load_from(InferenceRunSettings(device="cpu", confidence_threshold=0.25, slice_settings=legacy))
    assert dialog.spin_object_fraction.value() == pytest.approx(0.15)


def test_training_widget_emits_no_640_pixel_list():
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import SliceSettingsGroup

    group = SliceSettingsGroup()
    group.set_model_input_size(1024)
    group.txt_targets.setText("0.1, 0.2")
    settings = group.to_settings()
    assert settings.target_size_fractions == [0.1, 0.2]
    assert settings.target_sizes == SliceTrainingSettings().target_sizes  # untouched default, ignored
    assert settings.target_sizes_for(1024) == [102.4, 204.8]
```

Read the dialog's current constructor first; if it cannot learn the model input size, add a keyword `model_input_size: int = 640` and pass `self._project.imgsz_obb_direct` from `main_window._open_inference_settings_dialog` (~:1610). Read `SliceSettingsGroup`'s constructor/`txt_targets` name and adapt the test to the real widget API without changing intent.

- [x] **Step 2:** Run → FAIL.
- [x] **Step 3:** Implement. `load_from` sets the spin from `median(sliced.target_fractions())` (fallback `BACKEND_DEFAULTS["yolo_infer"].object_tile_fractions[0]`); `settings()` writes `target_size_fractions=[f]`, `object_tile_fraction=f`, leaves `target_sizes` at its default. `_refresh_enabled_state` enables the fraction spin in `auto_object` like before. `to_settings()` drops the `target_sizes=[fraction * 640 ...]` line.
- [x] **Step 4:** Update the two locking tests: `test_detectkit_inference_settings.py:77` asserts `target_size_fractions == [<fraction>]` instead of `target_sizes == [200.0]`; `test_sam3_slice_settings_shared.py:137` asserts the fractions and that no 640-derived list is emitted. Run the new file plus `tests/test_detectkit_inference_settings.py tests/test_sam3_slice_settings_shared.py tests/test_detectkit_slice_ui.py tests/test_detectkit_slice_settings.py tests/test_detectkit_training_cli.py tests/test_detectkit_dataset_preparation_sidecar.py` → PASS.
- [x] **Step 5:** Commit `fix(detectkit): fraction-only SAHI settings; no 640 anchor (F1)`.

### Task 12: Preview resolves tiling from the model sidecar (F2)

**Files:**
- Create: `src/hydra_suite/detectkit/jobs/preview_tiling.py`
- Modify: `src/hydra_suite/detectkit/gui/main_window.py` (worker construction ~:1813-1833 AND `_dataset_signature` ~:1640-1649) — pass the resolved preview tiling, and build the in-session signature from it (plan review M3: the GUI reuse check at ~:1676/:2061 must see recalibration)
- Modify: `src/hydra_suite/detectkit/jobs/dataset_inference.py:41-58` — settings dict carries `slice_merge_policy`, `slice_merge_metric`, `slice_imgsz` (so the cache key covers them)
- Modify: `src/hydra_suite/detectkit/sidecars/operations.py:181-206` and `src/hydra_suite/detectkit/gui/prediction_preview.py:74-97` — use the resolved merge policy/metric and imgsz
- Test: `tests/test_detectkit_preview_tiling.py`

**Interfaces:**
- Produces: `resolve_preview_tiling(model_path: str | Path | None, project_settings: SliceTrainingSettings, *, project_imgsz: int, override: SliceTrainingSettings | None) -> PreviewTiling`, where `@dataclass(frozen=True) class PreviewTiling: slice_settings: SliceTrainingSettings; imgsz: int; merge_policy: str; merge_metric: str; source: str` (`source` ∈ `"override"`, `"profile:<name>"`, `"training"`, `"project"`).

Rule (deviation 16): `override` given → it, `source="override"`. Else `meta = read_slice_meta(model_path)`; if a meta exists → `values = slice_meta_to_panel_values(meta, None)` (primary profile, else training geometry — TrackerKit's fresh-load ladder) mapped onto a copy of `project_settings` with: `enabled` = project's; `geometry_mode`, `overlap`, `slice_width/height` from values; `object_tile_fraction` and `target_size_fractions=[values["object_tile_fraction"]]`; `reference_body_px` = the project's (label-measured) if > 0 else `values["trained_body_px"]` (spec §3.3: dataset before stamp); `merge_threshold` = values' if not None else project's; `merge_policy/metric` = values' if not None else `"greedy_nmm"`/`"ios"`; `imgsz` = `project_imgsz`; `source = f"profile:{values['profile_name']}"` or `"training"`. No meta → project settings, `"project"`. Never raises (corrupt sidecar → project). Pass a profile's `merge_policy` through raw (`nmm` stays `nmm`).

- [x] **Step 1: Failing tests**

```python
# tests/test_detectkit_preview_tiling.py
from hydra_suite.core.inference.slice_meta import upsert_slice_profile, write_slice_meta
from hydra_suite.detectkit.gui.models import SliceTrainingSettings
from hydra_suite.detectkit.jobs.preview_tiling import resolve_preview_tiling

GEOM = {"geometry_mode": "auto_object", "imgsz": 1024, "object_tile_fraction": 0.1,
        "target_sizes": [51.2, 102.4, 153.6, 204.8], "overlap": 0.25, "reference_body_px": 44.0}


def _model(tmp_path, meta=None):
    model = tmp_path / "det.pt"
    model.write_bytes(b"x")
    if meta is not None:
        write_slice_meta(model, meta)
    return model


def test_no_sidecar_uses_project(tmp_path):
    project = SliceTrainingSettings(enabled=True, overlap=0.3)
    got = resolve_preview_tiling(_model(tmp_path), project, project_imgsz=640, override=None)
    assert got.source == "project" and got.slice_settings == project and got.imgsz == 640


def test_training_geometry_like_trackerkit(tmp_path):
    project = SliceTrainingSettings(enabled=True, reference_body_px=10.0)
    got = resolve_preview_tiling(_model(tmp_path, {"training_geometry": GEOM}), project, project_imgsz=640, override=None)
    assert got.source == "training"
    assert got.imgsz == 640  # the project's knob
    assert got.slice_settings.enabled is True
    assert got.slice_settings.overlap == 0.25
    assert got.slice_settings.object_tile_fraction == 0.125  # median(target_sizes)/imgsz = 128/1024, TrackerKit's value
    assert got.slice_settings.reference_body_px == 10.0  # dataset (project) before stamp


def test_stamped_body_when_project_has_none(tmp_path):
    got = resolve_preview_tiling(_model(tmp_path, {"training_geometry": GEOM}), SliceTrainingSettings(enabled=True), project_imgsz=640, override=None)
    assert got.slice_settings.reference_body_px == 44.0
    assert (got.merge_policy, got.merge_metric) == ("greedy_nmm", "ios")


def test_primary_profile_wins(tmp_path):
    """Review Focus 1."""
    meta = upsert_slice_profile({"training_geometry": GEOM}, name="High recall",
                                settings={"object_tile_fraction": 0.08, "overlap": 0.35,
                                          "merge_policy": "nms", "merge_metric": "iou",
                                          "merge_threshold": 0.6}, primary=True)
    got = resolve_preview_tiling(_model(tmp_path, meta), SliceTrainingSettings(enabled=True), project_imgsz=640, override=None)
    assert got.source == "profile:High recall"
    assert got.slice_settings.object_tile_fraction == 0.08
    assert got.slice_settings.target_size_fractions == [0.08]
    assert got.slice_settings.overlap == 0.35
    assert got.slice_settings.merge_threshold == 0.6
    assert (got.merge_policy, got.merge_metric) == ("nms", "iou")


def test_project_toggle_owns_enabled(tmp_path):
    got = resolve_preview_tiling(_model(tmp_path, {"training_geometry": GEOM}), SliceTrainingSettings(enabled=False), project_imgsz=640, override=None)
    assert got.slice_settings.enabled is False


def test_override_wins(tmp_path):
    override = SliceTrainingSettings(enabled=True, overlap=0.4)
    got = resolve_preview_tiling(_model(tmp_path, {"training_geometry": GEOM}), SliceTrainingSettings(), project_imgsz=640, override=override)
    assert got.source == "override" and got.slice_settings == override


def test_corrupt_sidecar_falls_back_to_project(tmp_path):
    model = _model(tmp_path)
    model.with_name(model.name + ".slice_meta.json").write_text("{nope")
    got = resolve_preview_tiling(model, SliceTrainingSettings(enabled=True), project_imgsz=640, override=None)
    assert got.source == "project"


def test_cache_key_changes_when_sidecar_changes(tmp_path):
    """Review Focus 4: recalibrating must invalidate cached predictions."""
    from hydra_suite.detectkit.jobs.dataset_inference import preview_settings_dict  # see Step 3

    model = _model(tmp_path, {"training_geometry": GEOM})
    a = preview_settings_dict(resolve_preview_tiling(model, SliceTrainingSettings(enabled=True), project_imgsz=640, override=None))
    write_slice_meta(model, upsert_slice_profile({"training_geometry": GEOM}, name="P", settings={"overlap": 0.4}, primary=True))
    b = preview_settings_dict(resolve_preview_tiling(model, SliceTrainingSettings(enabled=True), project_imgsz=640, override=None))
    assert a != b
```

Add one GUI-level test (same QApplication fixture convention as Task 11) that constructs the DetectKit main window's signature helper — or, if `_dataset_signature` cannot be reached without the full window, factor it into a pure function `dataset_signature(source, model_path, tiling: PreviewTiling, ...)` and test that: the signature changes when the sidecar's primary profile changes.

- [x] **Step 2:** Run → FAIL (module missing).
- [x] **Step 3:** Implement `preview_tiling.py`. In `dataset_inference.py`, factor the settings-dict construction into `preview_settings_dict(tiling: PreviewTiling, **other) -> dict` (or an equivalent helper the worker uses) so `slice_settings`, `slice_merge_policy`, `slice_merge_metric`, `slice_imgsz` all enter `prediction_cache_key`; keep `imgsz_obb_direct` for back-compat but have the sliced branch read `slice_imgsz` when present. `operations.run_dataset_inference` passes `merge_policy`/`merge_metric` to `predict_sliced_obb_result` → `_preview_slice_merge_config(threshold, policy, metric)` (defaults `greedy_nmm`/`ios` keep the old byte behavior). `main_window` computes `resolve_preview_tiling(model_path, self._project.slice_settings, project_imgsz=self._project.imgsz_obb_direct, override=<dialog override slice_settings or None>)` where the worker is built, and logs `source` in the status bar ("SAHI preview: training geometry" / "profile 'X'" / "project settings" / "override").
- [x] **Step 4:** Run the new file plus `tests/test_detectkit_sliced_preview.py tests/test_detectkit_preview_target.py tests/test_detectkit_inference_cancel.py tests/test_detectkit_inference_stager.py tests/test_detectkit_inference_settings.py` → PASS.
- [x] **Step 5:** Commit `fix(detectkit): preview tiles like TrackerKit from the model sidecar (F2)`.

### Task 13: Headless SAM3 tiles like the dialog (F3)

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/semantic_escalation.py` — add `default_semantic_tiling(project, variant) -> dict`
- Modify: `src/hydra_suite/detectkit/escalate_cli.py` — `sam3` gets `--tile-fraction`, `--reference-body-px`; `run_sam3` passes the resolved tiling in `params`
- Modify: `src/hydra_suite/detectkit/gui/dialogs/semantic_escalation_dialog.py` — opening state for tile fraction/body px comes from `default_semantic_tiling` when the saved settings are for a different variant or absent
- Test: `tests/test_semantic_default_tiling.py`

**Interfaces:**
- Produces: `default_semantic_tiling(project, variant: str) -> {"reference_body_px": float, "tile_fraction": float | None, "overlap": float, "merge_iou": float, "area_min_px2": float, "area_max_px2": float, "origin": str}`. Precedence (the SAM2 shape): (1) `project.semantic_escalation_settings` when its `variant == variant` (all fields as saved; `tile_fraction` 0 → None); (2) the serving calibration (`resolve_serving_calibration(sidecar_for(variant), project.semantic_calibration)`), only when `record.get("variant") == variant`: `tile_fraction = record["points"][record["recommended_index"]]["tile_fraction"]`, body = `record["parameters"]["reference_body_px"]` (SAM3 records have no `chosen_index` and no top-level body — plan review M4; guard every lookup); (3) the model stamp: `stamped_object_tile_fraction(sidecar_for(variant))` with body = the DIALOG's body chain (below), else the stamped `reference_body_px`; (4) the dialog's own opening state (plan review B1): `tile_fraction = SEMANTIC_TILE_FRACTION_SEED`, body = the dialog's body chain — `escalation_actions.resolve_reference_body_px(project)` (project `slice_settings.reference_body_px` → label median), whatever function the dialog actually calls at construction; read it and reuse it, do not re-implement. Full frame (`tile_fraction=None`) only when the resolved body is 0, exactly as `resolve_tile_px` treats it. `overlap` default `semantic.tiling.DEFAULT_OVERLAP`, `merge_iou` default `DEFAULT_MERGE_IOU`. `origin` ∈ `saved|calibration|stamped|default|full_frame`. Never raises.

- [x] **Step 1: Failing tests**

```python
# tests/test_semantic_default_tiling.py
import argparse
from types import SimpleNamespace

import pytest

from hydra_suite.detectkit.jobs import semantic_escalation as se


def _project(**kw):
    base = dict(semantic_escalation_settings={}, semantic_calibration={}, project_dir="/tmp/sahi_s3_project",
                slice_settings=SimpleNamespace(reference_body_px=0.0), sources=[])
    base.update(kw)
    return SimpleNamespace(**base)


def test_saved_settings_for_same_variant_win(monkeypatch):
    monkeypatch.setattr(se, "sidecar_for", lambda key: {"object_tile_fraction": 0.2})
    p = _project(semantic_escalation_settings={"variant": "sam3", "tile_fraction": 0.07, "reference_body_px": 40.0, "overlap": 0.3})
    got = se.default_semantic_tiling(p, "sam3")
    assert (got["tile_fraction"], got["reference_body_px"], got["overlap"], got["origin"]) == (0.07, 40.0, 0.3, "saved")


def test_saved_full_frame_is_respected(monkeypatch):
    p = _project(semantic_escalation_settings={"variant": "sam3", "tile_fraction": 0.0, "reference_body_px": 40.0})
    assert se.default_semantic_tiling(p, "sam3")["tile_fraction"] is None


def test_stamped_fraction_and_body_when_nothing_saved(monkeypatch):
    """Review Focus 3: a finetuned model tiles at its trained scale headlessly."""
    monkeypatch.setattr(se, "sidecar_for", lambda key: {"object_tile_fraction": 0.055, "reference_body_px": 53.4})
    got = se.default_semantic_tiling(_project(), "ft-model")
    assert (got["tile_fraction"], got["reference_body_px"], got["origin"]) == (0.055, 53.4, "stamped")


def test_stock_variant_with_known_body_tiles_at_the_seed(monkeypatch):
    """Plan review B1: the dialog's opening state, not full frame."""
    monkeypatch.setattr(se, "sidecar_for", lambda key: None)
    p = _project(slice_settings=SimpleNamespace(reference_body_px=82.2))
    got = se.default_semantic_tiling(p, "sam3")
    assert (got["tile_fraction"], got["reference_body_px"], got["origin"]) == (0.05, 82.2, "default")


def test_stock_variant_with_no_body_is_full_frame(monkeypatch):
    monkeypatch.setattr(se, "sidecar_for", lambda key: None)
    got = se.default_semantic_tiling(_project(), "sam3")
    assert got["tile_fraction"] is None and got["origin"] == "full_frame"


def test_calibration_record_recommended_point(monkeypatch):
    monkeypatch.setattr(se, "sidecar_for", lambda key: None)
    record = {"variant": "sam3", "recommended_index": 1, "parameters": {"reference_body_px": 61.0},
              "points": [{"tile_fraction": 0.03}, {"tile_fraction": 0.1}]}
    got = se.default_semantic_tiling(_project(semantic_calibration=record), "sam3")
    assert (got["tile_fraction"], got["reference_body_px"], got["origin"]) == (0.1, 61.0, "calibration")
    other = dict(record, variant="ft-model")
    assert se.default_semantic_tiling(_project(semantic_calibration=other), "sam3")["origin"] != "calibration"


def test_cli_sam3_accepts_tiling_flags():
    from hydra_suite.detectkit.escalate_cli import build_parser

    ns = build_parser().parse_args(["sam3", "--project", "p", "--prompt", "ant", "--tile-fraction", "0.06", "--reference-body-px", "50"])
    assert ns.tile_fraction == 0.06 and ns.reference_body_px == 50.0


def test_cli_sam3_payload_carries_resolved_tiling(monkeypatch):
    from hydra_suite.detectkit import escalate_cli as cli

    captured = {}
    monkeypatch.setattr(cli, "_open", lambda path: _project(), raising=False)
    monkeypatch.setattr(cli, "_selected", lambda project, names: [], raising=False)
    monkeypatch.setattr(cli, "_resolve_class_name", lambda *a: "ant", raising=False)
    monkeypatch.setattr(se, "sidecar_for", lambda key: {"object_tile_fraction": 0.055, "reference_body_px": 53.4})
    import hydra_suite.core.inference.semantic.checkpoints as ck
    import hydra_suite.detectkit.sidecars.operations as ops

    monkeypatch.setattr(ck, "probe_dependencies", lambda: SimpleNamespace(usable=True, reason=""))
    monkeypatch.setattr(ops, "run_semantic_escalation_sidecar", lambda payload, progress: captured.update(payload) or {"semantic_result": {}})
    ns = cli.build_parser().parse_args(["sam3", "--project", "p", "--prompt", "ant", "--variant", "ft-model"])
    assert cli.run_sam3(ns) == 0
    params = captured["params"]
    assert params["tile_fraction"] == 0.055 and params["reference_body_px"] == 53.4
```

`sidecar_for` must be a module-level import in `semantic_escalation.py` so `monkeypatch.setattr(se, "sidecar_for", ...)` (raising=True) works. The label-median fallback in tests with empty `sources` must resolve to 0 without error. Adapt other monkeypatch targets to where names are actually imported (e.g. if `semantic_escalation` imports `sidecar_for` inside a function, patch `hydra_suite.core.inference.semantic.checkpoints.sidecar_for`). Read `test_detectkit_escalate_cli.py` for the existing patching pattern of `run_sam2`.

- [x] **Step 2:** Run → FAIL.
- [x] **Step 3:** Implement `default_semantic_tiling` (module-level import of `sidecar_for` so it is patchable), the CLI flags (help text mirrors sam2's: "0 = full frame; default: what the DetectKit dialog would open with"), and `run_sam3`: `tiling = default_semantic_tiling(project, args.variant)`; flags override (`--tile-fraction 0` → None; explicit fraction with no known body → same fallback chain as `_sam2_tiling`: project slice_settings body, then `quick_median_body_px`); `params.update(reference_body_px=..., tile_fraction=..., overlap=..., merge_iou=..., area_min_px2=..., area_max_px2=...)` — check `SemanticEscalationRequest` field names and only pass fields it has. Print `Tiling: fraction F of a B px body (origin)` or `Tiling: full frame`. Dialog: when saved settings are absent or for another variant, seed `_tile_fraction` and `_reference_body` from `default_semantic_tiling` — which by rung (4) equals today's opening state for a stock model, so `tests/test_semantic_escalation_dialog_persistence.py` must pass UNCHANGED; a finetuned model now opens at its stamped scale.
- [x] **Step 4:** Run the new file plus `tests/test_detectkit_escalate_cli.py tests/test_semantic_escalation_dialog_persistence.py tests/test_sam3_model_selection.py tests/test_semantic_calibration_sidecar.py tests/test_geometry_review_fixes.py` → PASS.
- [x] **Step 5:** Update `docs/user-guide/sam-install-and-run.md:126` CLI synopsis with the two flags. Commit `fix(detectkit): headless SAM3 escalation tiles like the dialog (F3)`.

### Task 14: S3 gate — regression, equivalence (MPS + CUDA), adversarial review, merge

- [x] **Step 1: Regression**: every test file named in Tasks 10–13, plus `tests/test_tiling_*.py tests/test_slice_*.py tests/test_gui_cli_profile_parity.py tests/test_trackerkit_cli_sahi_profile.py tests/test_trackerkit_slice_meta_prefill.py tests/test_profile_cache_keys.py tests/test_direct_calibration_grid.py tests/test_geometry_drift_guard.py tests/test_core_import_is_light.py` and `tests/test_get_parameters_dict_characterization.py`, `tests/test_slice_config_sahi_path_guard.py`; `make lint` (compare to main's known 10 findings).
- [x] **Step 2: Equivalence, MPS (this box)** — kill stale sleap/hydra first:
```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
pgrep -u "$USER" -fl "sleap|hydra" ; pkill -u "$USER" -f "sleap|hydra" || true
REPO=$PWD WT=$PWD/.worktrees/sahi-s3 MAIN_SRC=$PWD/src WT_SRC=$PWD/.worktrees/sahi-s3/src \
  OUT=/tmp/equiv_sahi_s3_mps RUNTIME=mps bash tools/equivalence/run_matrix.sh
```
Expected: every clip EQUIVALENT at its determinism floor, row counts > 1 (`wc -l`).
- [x] **Step 3: Equivalence, CUDA (diptera; mehek is offline — deviation 20)** — check `ssh rishika@diptera.rockefeller.edu nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv` and pick an IDLE GPU's UUID (not one another job uses). Sync both src trees (local `main` is ahead of origin; do not push): `rsync -a --delete src/ rishika@diptera.rockefeller.edu:/tmp/sahi_s3/main_src/` and the worktree's `src/` to `/tmp/sahi_s3/wt_src/`. On diptera in `~/hydra-suite` (harness + fixtures; if its `tools/equivalence/` is older than main's, rsync main's `tools/equivalence/` into `/tmp/sahi_s3/harness/` and use `WT=/tmp/sahi_s3/harness_root` containing it plus the fixtures symlinked): `source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda && export CUDA_VISIBLE_DEVICES=<uuid>`; clean only `pgrep -u rishika -f "sleap|hydra"` processes belonging to this gate; `REPO=$PWD WT=$PWD MAIN_SRC=/tmp/sahi_s3/main_src WT_SRC=/tmp/sahi_s3/wt_src OUT=/tmp/equiv_sahi_s3_cuda RUNTIME=cuda bash tools/equivalence/run_matrix.sh fly_obb worm_bgsub ant_obb_sleap`. No fixture enables slicing, so this proves the SAHI-disabled path; the SAHI-enabled path is guarded by `test_slice_config_sahi_path_guard.py`.
- [x] **Step 4: Adversarial review** (Fable, independent): break F1–F4/F8 claims by executing code — legacy project files from before S3, preview cache invalidation, `detectkit escalate sam3` parity with the dialog's opening state, TrackerKit byte-identity (golden + a constructed config through `build_engine_params` vs `main`), and any GUI path that still converts by 640 (`grep -rn "640" src/hydra_suite/detectkit`). Plus a normal reviewer. Fix wave; re-review if a fix exceeds the reported lines.
- [x] **Step 5: Merge** `--no-ff` into local `main`, remove the worktree.
