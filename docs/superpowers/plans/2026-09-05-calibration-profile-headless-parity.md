# Calibration Profile Headless Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DetectKit calibration profiles work identically in `trackerkit track` and in the GUI, selectable from the CLI, and durable across publish and model import.

**Architecture:** `build_engine_params` gains a profile overlay that mirrors — line for line, with the GUI lines cited — the resolution ladder already implemented in `detection_panel._apply_slice_meta_values`. The two pure translators it needs (`slice_meta_to_panel_values`, `slice_meta_values_from_settings`) already exist in `core/inference/slice_meta.py`; no third ladder is invented. A `--sahi-profile` flag rewrites the config's `slice_profile_id` before the builder runs. One shared sidecar-copy helper closes the two places a sidecar is dropped.

**Tech Stack:** Python 3.11, PySide6 (GUI tasks only), pytest, numpy.

**Spec:** `docs/superpowers/specs/2026-09-05-calibration-profile-headless-parity-design.md`
(and the audit it argues from: `docs/superpowers/specs/2026-09-05-calibration-profile-headless-parity-audit.md`)

## Global Constraints

- **Python 3.11 target.** No PEP 701 multi-line f-strings (they are a `SyntaxError` on 3.11 and this repo has shipped that bug before). No `match` on 3.10-incompatible patterns.
- **Equivalence must stay byte-identical on MPS and CUDA — but inertness is MACHINE STATE, not a property of the code.** No fixture config in `tools/equivalence/fixtures/configs/` names a profile (verified: 0 `slice_profile*` hits across all ten), so the overlay can only fire via a sidecar sitting beside a fixture model. Those models live in the shared models directory that DetectKit's calibration wizard writes sidecars into (`detectkit/gui/dialogs/direct_calibration_results.py:724`), and a config with no `slice_profile_id` still resolves to the sidecar's PRIMARY profile. **Before each gate run, on each box, verify no fixture model has a `.slice_meta.json`:** `find "$(python -c 'from hydra_suite.paths import get_models_dir; print(get_models_dir())')" -name '*.slice_meta.json'`. The local box was clean at plan time; mehek is unverified. If a sidecar is present, either point `HYDRA_DATA_DIR` at a clean models dir for the run or record the sidecar's effect as expected, and say which in the gate report.
- **`build_engine_params` must never raise on missing metadata.** It is called in tests with model paths that do not exist. Every sidecar read degrades to "no overlay".
- **Layering:** `core/` must never import an app layer. `trackerkit` and `detectkit` must not import each other. `trackerkit/engine_params.py` is Qt-free and must stay Qt-free.
- **`tests/test_profile_cache_keys.py` semantics are load-bearing and unchanged:** geometry / overlap / merge settings split the detection cache key; confidence deliberately does not.
- **No new params keys.** Provenance is a log line, not a `SLICE_PROFILE_*` param — a key nobody consumes would force a golden regeneration for nothing.
- Run `make format` before each commit; `make lint-moderate` before the final task.

## Revision 2 — rulings from the adversarial review

Revision 1 was reviewed against source and found to mirror the GUI wrongly in
three places. The review is at `/tmp/plan_adversarial_review.md`; its findings
are resolved here, and the resolution changed the design's central rule.

**The rule (replaces revision 1's "the profile overrides the config"):**

> A key the GUI writes into the saved config is **authoritative from the
> config**. A key that lives only in `advanced_config` is **supplied by the
> overlay**.

`build_config_dict` (`gui/orchestrators/config.py:1697-1730`) persists
`slice_enabled`, `slice_geometry_mode` and `yolo_confidence_threshold` — those
three already carry whatever the profile put in the widgets when the session
was saved, so re-deriving them from the sidecar can only *disagree* with what
the user saw. The nine keys the audit found dying (`overlap`,
`object_tile_fraction`, `slice_width`, `slice_height`, `trained_body_px`, and
the four `merge_*`) exist only in `advanced_config`, which the GUI never
persists — those are exactly what the overlay supplies.

This rule resolves four findings at once and makes the overlay a **no-op in the
GUI** (where `advanced_config` already holds the panel-applied values), which is
why it is safe that `get_parameters_dict()` is itself a `build_engine_params`
call (review C1, `gui/orchestrators/config.py:2153-2160`).

**Ruling R1 (review C2 — confidence).** *User decision.* On restore the config
wins; the profile never overwrites a saved threshold. `_load_config_yolo`
applies the profile via model selection (`:530-533`) and *then* sets the spin
from the config (`:614-616`), so config-wins is what the GUI already does. The
overlay therefore never touches `YOLO_CONFIDENCE_THRESHOLD`; it **warns** when
the config's threshold disagrees with the claimed profile's measured one, so a
mismatch is visible in a batch log instead of silent. Cost if wrong: a user who
expects a profile to fully pin its operating point must re-select the profile
in the GUI to pick up its confidence.

**Ruling R2 (review C3 — training rung).** *User decision.* The training rung
**does** apply: `_training_values` (`slice_meta.py:287-320`) returns real
overlap / `object_tile_fraction` / `trained_body_px`, and every sliced-training
publish produces exactly that sidecar, so revision 1 left F1 open for the
commonest case. The overlay applies those values. It does **not** propagate
`_training_values`' `enabled=True` — under the rule above `slice_enabled` comes
from the config, which also sidesteps the GUI bug the review found (a one-item
profile combo makes `_mark_slice_profile_custom` early-return at `:2617`, so the
panel can silently re-tick a `slice_enabled` the user had cleared). We do not
copy that bug.

**Ruling R3 (unclaimed merge keys).** Unclaimed (`None`) merge keys mean
`SLICE_MERGE_DEFAULTS`, always — matching `detection_panel.py:2723-2732`'s
non-restore branch. The panel's restore branch instead *keeps* whatever was in
`advanced_config` (`resetting=False`, `:2721-2732`), which on a fresh window is
the machine-global advanced file — i.e. the same leak the CLI has. Task 4's
GUI↔CLI oracle will expose any resulting disagreement; if it does, the fix is to
drop that guard in the panel, not to weaken the builder. Cost if wrong: a
restored GUI session's merge settings change to defaults where they previously
inherited a machine-global value.

**Mechanical corrections carried into the tasks:** Task 4's oracle must actually
select the model (review C4 — `build_config_dict` reads the path from
`_get_selected_yolo_model_path()`, and `TOTAL_FRAMES`/`FRAME_WIDTH`/
`FRAME_HEIGHT` are not params keys; only `FPS` is, `engine_params.py:893`);
Task 6 must connect the signal after `spin_yolo_confidence` is constructed at
`:1057`, not at `:878` (review C5); every test that asserts a *default* must
pass an explicit `advanced_config` rather than depending on the developer's
on-disk file (review I4).

## Verified API reference

Every signature below was read from source on `main` @ b8c424e7. Use these verbatim; do not infer variants.

| Symbol | Location | Signature / fact |
|---|---|---|
| `read_slice_meta` | `core/inference/slice_meta.py:29` | `(model_path: str \| Path) -> dict \| None` — returns `None` on absent/corrupt, never raises |
| `write_slice_meta` | `slice_meta.py:38` | `(model_path, meta) -> Path` — atomic |
| `sidecar_path` | `slice_meta.py:23` | `(model_path) -> Path`, appends `.slice_meta.json` to the **full** name (`foo.pt` → `foo.pt.slice_meta.json`) |
| `available_slice_profiles` | `slice_meta.py:158` | `(meta) -> list[dict]` with keys `id`, `name`, `note`, `settings`, `measurement` |
| `primary_slice_profile` | `slice_meta.py:213` | `(meta) -> dict \| None` — explicit primary only, never invented |
| `profile_by_id` | `slice_meta.py:251` | `(meta, profile_id) -> dict \| None`. **Special-cases only `"__training__"`.** A `"__custom__"` id falls through to the primary. |
| `slice_meta_to_panel_values` | `slice_meta.py:321` | `(meta, profile_id=None) -> dict`. Keys: `enabled`, `geometry_mode`, `overlap`, `object_tile_fraction`, `trained_body_px`, `slice_width`, `slice_height`, `confidence_threshold`, `merge_policy`, `merge_metric`, `merge_threshold`, `merge_backend`, `profile_id`, `profile_name`, `resolution`. `resolution ∈ {"requested","primary","training"}`. Merge keys and `confidence_threshold` are `None` when unclaimed. |
| `slice_meta_values_from_settings` | `slice_meta.py:385` | `(meta, settings) -> dict` — same keys; `resolution == "saved_settings"`, `profile_id is None`, `profile_name == "Saved settings"` |
| `profile_summary` | `slice_meta.py:191` | `(meta) -> {"count": int, "primary_profile_id": str, "names": list[str]}` |
| `merge_training_geometry` | `slice_meta.py:201` | `(existing \| None, training_geometry) -> dict` |
| `build_engine_params` | `trackerkit/engine_params.py:362` | `(config: Mapping, *, runtime: RuntimeContext, advanced_config: Mapping \| None = None) -> dict` |
| `SLICE_MERGE_DEFAULTS` | `engine_params.py:43` | dict with keys `merge_policy`, `merge_metric`, `merge_threshold`, `merge_backend` |
| `resolve_model_path` | imported at `engine_params.py:25` from `core.inference.model_paths` | `(path_str) -> str` |
| `yolo_direct_path` | `engine_params.py:626` | already-resolved direct model path, in scope before the `SLICE_*` block at :913 |
| `RuntimeContext` | `engine_params.py:98` | dataclass; required fields `fps`, `total_frames`, `frame_width`, `frame_height` |
| `load_tracker_cli_session` | `trackerkit/cli_config.py:196` | `(video_path, *, config_path=None, config_data=None, video_probe=None, advanced_config=None) -> TrackerCliSession` |
| `run_tracking_cli` | `trackerkit/cli.py:22` | `(video_paths, *, config_path=None, keystone_override=False) -> int` |
| `track_parser` | `trackerkit/app.py:87` | the `track` subparser; args validated at `app.py:114`, dispatched at `app.py:236-241` |
| `_copy_model_metadata_sidecars` | `detectkit/gui/project.py:193` | `(source: Path, destination: Path) -> None`; suffixes tuple at `project.py:35-39` |
| `publish_trained_model` sidecar gate | `training/model_publish.py:872-878` | `if slice_geometry and role in _DIRECT_DETECTOR_ROLES and dst.suffix.lower() == ".pt"` |
| registry profile stamp | `model_publish.py:930-935` | inside `if slice_geometry and role in _DIRECT_DETECTOR_ROLES` |
| `verify_profile_summary` | `model_publish.py` (called at :933) | `(dst, summary) -> None` |
| TrackerKit model import copy | `gui/orchestrators/config.py:3688` | `shutil.copy2(src, dest_path)` — weights only |
| `_mark_slice_profile_custom` | `gui/panels/detection_panel.py:2603` | guards on `self._applying_slice_profile` and `main_window._restoring_config`; sets `advanced_config["slice_profile_id"] = "__custom__"` |
| `_applying_slice_profile` | `detection_panel.py:769` | the suppression flag set around programmatic widget writes (`:2693`, cleared `:2763`) |
| `spin_yolo_confidence` | `detection_panel.py:1057` | `QDoubleSpinBox`, range 0.01–1.0, **no signal connection anywhere** |
| `_slice_profile_settings_snapshot` | `gui/orchestrators/config.py:411` | writes the 12 knobs + `base_profile_name`; always present in configs saved post-feature |
| `_make_panel_with_sidecar` | `tests/test_trackerkit_slice_meta_prefill.py:19` | builds a real offscreen `MainWindow` + stub `model.pt`; returns `(panel, window, model_path)` |

## The GUI ladder being mirrored (read this before Task 2)

From `detection_panel.py:2665-2683` and `:2812-2823`:

```python
known_ids = {p["id"] for p in available_slice_profiles(meta)}
is_custom_restore = bool(profile_id == "__custom__" and saved_settings)
use_saved_settings = is_custom_restore or bool(
    profile_id
    and profile_id not in ("__training__", "__custom__")
    and profile_id not in known_ids
    and saved_settings
)
values = (slice_meta_values_from_settings(meta, saved_settings)
          if use_saved_settings else slice_meta_to_panel_values(meta, profile_id))
```

Then (`:2695-2743`) the values are applied: `enabled`, `geometry_mode`, `overlap`,
`object_tile_fraction`, `trained_body_px`, `slice_width`, `slice_height`
unconditionally; merge keys unconditionally with `None → SLICE_MERGE_DEFAULTS[key]`;
`confidence_threshold` **only when not `None`**.

---

## Task 1: Pure ladder resolver in `core/inference/slice_meta.py`

The branching above is currently inlined in a Qt widget. Extract it as a pure
function so the builder and the panel cannot drift.

**Files:**
- Modify: `src/hydra_suite/core/inference/slice_meta.py`
- Test: `tests/test_slice_profile_resolution.py` (create)

**Interfaces:**
- Consumes: `available_slice_profiles`, `slice_meta_to_panel_values`, `slice_meta_values_from_settings` (all already in this module)
- Produces: `resolve_slice_profile_values(meta: dict, profile_id: str | None, saved_settings: dict | None) -> dict` — returns the same value dict shape as the two translators, with `resolution` set by whichever branch ran.

- [ ] **Step 1: Write the failing tests**

```python
import json
from hydra_suite.core.inference.slice_meta import resolve_slice_profile_values

TRAINING = {"geometry_mode": "auto_object", "imgsz": 640, "overlap": 0.2}
SETTINGS = {
    "enabled": True, "geometry_mode": "auto_object",
    "slice_width": 0, "slice_height": 0,
    "overlap": 0.31, "object_tile_fraction": 0.11, "trained_body_px": 560.0,
    "confidence_threshold": 0.35,
    "merge_policy": "nmm", "merge_metric": "iou",
    "merge_threshold": 0.6, "merge_backend": "cv2",
}
META = {
    "schema_version": 2,
    "training_geometry": TRAINING,
    "primary_profile_id": "balanced",
    "profiles": [
        {"id": "balanced", "name": "Balanced", "settings": SETTINGS},
        {"id": "recall", "name": "High recall",
         "settings": dict(SETTINGS, overlap=0.45)},
    ],
}
SNAPSHOT = dict(SETTINGS, overlap=0.07, base_profile_name="Balanced")


def test_live_id_wins_over_snapshot():
    v = resolve_slice_profile_values(META, "recall", SNAPSHOT)
    assert v["resolution"] == "requested"
    assert v["overlap"] == 0.45


def test_custom_with_snapshot_uses_snapshot_not_primary():
    v = resolve_slice_profile_values(META, "__custom__", SNAPSHOT)
    assert v["resolution"] == "saved_settings"
    assert v["overlap"] == 0.07


def test_custom_without_snapshot_falls_back_to_primary():
    v = resolve_slice_profile_values(META, "__custom__", None)
    assert v["resolution"] == "primary"
    assert v["profile_id"] == "balanced"


def test_missing_id_with_snapshot_uses_snapshot():
    v = resolve_slice_profile_values(META, "deleted-profile", SNAPSHOT)
    assert v["resolution"] == "saved_settings"
    assert v["overlap"] == 0.07


def test_missing_id_without_snapshot_uses_primary():
    v = resolve_slice_profile_values(META, "deleted-profile", None)
    assert v["resolution"] == "primary"


def test_training_request_ignores_snapshot():
    v = resolve_slice_profile_values(META, "__training__", SNAPSHOT)
    assert v["resolution"] == "training"
    assert v["profile_id"] is None


def test_empty_id_uses_primary():
    assert resolve_slice_profile_values(META, "", None)["resolution"] == "primary"


def test_no_primary_and_no_id_is_training():
    meta = dict(META, primary_profile_id="")
    assert resolve_slice_profile_values(meta, "", None)["resolution"] == "training"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_slice_profile_resolution.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_slice_profile_values'`

- [ ] **Step 3: Implement**

Append to `slice_meta.py` (after `slice_meta_values_from_settings`):

```python
def resolve_slice_profile_values(
    meta: dict[str, Any],
    profile_id: str | None,
    saved_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve a saved session's SAHI request into applied panel values.

    This is the ONE ladder. TrackerKit's detection panel
    (``_apply_slice_meta_values``) and the Qt-free ``build_engine_params``
    both route through it, so the GUI and the CLI cannot resolve the same
    saved config to different settings -- the divergence this exists to fix.

    ``"__custom__"`` must be checked explicitly: ``profile_by_id`` special-
    cases only ``"__training__"``, so a raw custom id would otherwise fall
    through to the primary profile and silently overwrite the very edits the
    snapshot was captured to preserve.
    """
    known_ids = {profile["id"] for profile in available_slice_profiles(meta)}
    is_custom_restore = bool(profile_id == "__custom__" and saved_settings)
    use_saved_settings = is_custom_restore or bool(
        profile_id
        and profile_id not in ("__training__", "__custom__")
        and profile_id not in known_ids
        and saved_settings
    )
    if use_saved_settings:
        return slice_meta_values_from_settings(meta, dict(saved_settings or {}))
    return slice_meta_to_panel_values(meta, profile_id)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_slice_profile_resolution.py tests/test_slice_meta_read.py tests/test_slice_profile_mutations.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/slice_meta.py tests/test_slice_profile_resolution.py
git commit -m "feat(slice-meta): extract the one profile-resolution ladder"
```

---

## Task 2: Route the detection panel through the shared resolver

Prove the extraction is faithful by making the GUI use it. If the panel's
behaviour changes, the extraction was wrong — that is the point of doing this
before the builder.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py:2658-2683`
- Test: `tests/test_trackerkit_slice_meta_prefill.py` (existing — must stay green)

**Interfaces:**
- Consumes: `resolve_slice_profile_values` from Task 1.
- Produces: no new interface; `_apply_slice_meta_values` keeps its signature.

- [ ] **Step 1: Run the existing prefill suite to capture the pre-change baseline**

Run: `python -m pytest tests/test_trackerkit_slice_meta_prefill.py tests/test_detection_panel_slice_widgets.py -v`
Expected: PASS (record the count; it must be identical after the change)

- [ ] **Step 2: Replace the inlined branching**

In `_apply_slice_meta_values`, replace the import block and the
`known_ids` / `is_custom_restore` / `use_saved_settings` / `values = ...`
lines with:

```python
        from hydra_suite.core.inference.slice_meta import (
            resolve_slice_profile_values,
        )

        if self._slice_meta is None:
            return
        self._slice_profile_requested_id = profile_id
        is_custom_restore = bool(profile_id == "__custom__" and saved_settings)
        values = resolve_slice_profile_values(
            self._slice_meta, profile_id, saved_settings
        )
        use_saved_settings = values["resolution"] == "saved_settings"
```

Leave every line below (`self._slice_profile_applied_id = ...` onward)
untouched — `use_saved_settings` and `is_custom_restore` keep their existing
meanings and are still consumed at `:2683`, `:2707` and `:2753`.

- [ ] **Step 3: Run the suite**

Run: `python -m pytest tests/test_trackerkit_slice_meta_prefill.py tests/test_detection_panel_slice_widgets.py tests/test_trackerkit_profile_session.py -v`
Expected: identical pass count to Step 1

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/panels/detection_panel.py
git commit -m "refactor(trackerkit): panel resolves profiles via the shared ladder"
```

---

## Task 3: Profile overlay in `build_engine_params` (the F1 fix)

Supplies the nine `advanced_config`-only SAHI keys from the resolved profile.
Does **not** touch `SLICE_ENABLED`, `SLICE_GEOMETRY_MODE` or
`YOLO_CONFIDENCE_THRESHOLD` — those come from the saved config (Revision 2 rule,
rulings R1/R2).

**Files:**
- Modify: `src/hydra_suite/trackerkit/engine_params.py` (helper near `_default_advanced_config_fallback` at :348; call site immediately before the `SLICE_*` block at :913)
- Test: `tests/test_engine_params_slice_profile.py` (create)

**Interfaces:**
- Consumes: `resolve_slice_profile_values` (Task 1), `read_slice_meta`, `SLICE_MERGE_DEFAULTS`, the already-resolved `yolo_direct_path` local (`engine_params.py:626`).
- Produces: `_slice_profile_overlay(cfg, model_path) -> dict[str, Any] | None` — the resolved value dict, or `None` when no overlay applies.

- [ ] **Step 1: Write the failing tests**

Every test passes an explicit `advanced_config` so a default assertion cannot
be satisfied (or broken) by the developer's on-disk advanced file.

```python
import json
from pathlib import Path

import pytest

from hydra_suite.trackerkit.engine_params import (
    SLICE_MERGE_DEFAULTS,
    RuntimeContext,
    build_engine_params,
)

RUNTIME = RuntimeContext(fps=30.0, total_frames=100, frame_width=640, frame_height=480)

# Explicit advanced baseline: every SAHI key at a value distinguishable from
# both the profile's and the module defaults, so "the overlay did nothing" and
# "the overlay wrote defaults" are different observations.
ADVANCED = {
    "slice_overlap": 0.2,
    "slice_object_tile_fraction": 0.15,
    "slice_width": 0,
    "slice_height": 0,
    "slice_trained_body_px": 0.0,
    "slice_merge_policy": "greedy_nmm",
    "slice_merge_metric": "ios",
    "slice_merge_threshold": 0.5,
    "slice_merge_backend": "cv2",
}

SETTINGS = {
    "enabled": True, "geometry_mode": "auto_object",
    "slice_width": 704, "slice_height": 512,
    "overlap": 0.31, "object_tile_fraction": 0.11, "trained_body_px": 560.0,
    "confidence_threshold": 0.42,
    "merge_policy": "nmm", "merge_metric": "iou",
    "merge_threshold": 0.6, "merge_backend": "cv2",
}


def _sidecar(tmp_path: Path, payload: dict, name: str = "model.pt") -> str:
    model = tmp_path / name
    model.write_text("stub", encoding="utf-8")
    (tmp_path / (name + ".slice_meta.json")).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return str(model)


def _profiled(tmp_path: Path) -> str:
    return _sidecar(
        tmp_path,
        {
            "schema_version": 2,
            "training_geometry": {"geometry_mode": "auto_model", "imgsz": 640},
            "primary_profile_id": "balanced",
            "profiles": [
                {"id": "balanced", "name": "Balanced", "settings": SETTINGS}
            ],
        },
    )


def _cfg(model_path: str, **extra) -> dict:
    base = {
        "yolo_obb_mode": "direct",
        "yolo_obb_direct_model_path": model_path,
        "slice_enabled": False,
        "slice_geometry_mode": "auto_model",
        "yolo_confidence_threshold": 0.25,
    }
    base.update(extra)
    return base


def _build(cfg):
    return build_engine_params(cfg, runtime=RUNTIME, advanced_config=dict(ADVANCED))


def test_named_profile_supplies_the_advanced_only_keys(tmp_path):
    params = _build(_cfg(_profiled(tmp_path), slice_profile_id="balanced"))
    assert params["SLICE_OVERLAP"] == 0.31
    assert params["SLICE_OBJECT_TILE_FRACTION"] == 0.11
    assert params["SLICE_WIDTH"] == 704
    assert params["SLICE_HEIGHT"] == 512
    assert params["SLICE_TRAINED_BODY_PX"] == 560.0
    assert params["SLICE_MERGE_POLICY"] == "nmm"
    assert params["SLICE_MERGE_METRIC"] == "iou"
    assert params["SLICE_MERGE_THRESHOLD"] == 0.6
    assert params["SLICE_MERGE_BACKEND"] == "cv2"


def test_config_owns_enabled_geometry_and_confidence(tmp_path):
    """Ruling R1/R2: the three keys build_config_dict persists are authoritative.

    The profile claims enabled=True, auto_object and 0.42; the config says
    False, auto_model and 0.25. The config must win -- it already records what
    the user saw when the session was saved.
    """
    params = _build(_cfg(_profiled(tmp_path), slice_profile_id="balanced"))
    assert params["SLICE_ENABLED"] is False
    assert params["SLICE_GEOMETRY_MODE"] == "auto_model"
    assert params["YOLO_CONFIDENCE_THRESHOLD"] == 0.25


def test_confidence_disagreement_warns(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _build(_cfg(_profiled(tmp_path), slice_profile_id="balanced"))
    assert any(
        "Balanced" in record.message and "0.42" in record.message
        for record in caplog.records
    )


def test_matching_confidence_does_not_warn(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _build(
            _cfg(
                _profiled(tmp_path),
                slice_profile_id="balanced",
                yolo_confidence_threshold=0.42,
            )
        )
    assert not [r for r in caplog.records if "confidence" in r.message.lower()]


def test_primary_applies_when_config_names_nothing(tmp_path):
    assert _build(_cfg(_profiled(tmp_path)))["SLICE_OVERLAP"] == 0.31


def test_custom_id_uses_saved_snapshot_not_primary(tmp_path):
    snapshot = dict(SETTINGS, overlap=0.07, merge_policy="nms")
    params = _build(
        _cfg(
            _profiled(tmp_path),
            slice_profile_id="__custom__",
            slice_profile_settings=snapshot,
        )
    )
    assert params["SLICE_OVERLAP"] == 0.07
    assert params["SLICE_MERGE_POLICY"] == "nms"


def test_training_geometry_is_applied(tmp_path):
    """Ruling R2: the commonest sidecar in the wild has no profiles at all."""
    model = _sidecar(
        tmp_path,
        {
            "schema_version": 2,
            "training_geometry": {
                "geometry_mode": "auto_object",
                "imgsz": 640,
                "overlap": 0.27,
                "reference_body_px": 480.0,
            },
            "primary_profile_id": "",
            "profiles": [],
        },
    )
    params = _build(_cfg(model, slice_enabled=True))
    assert params["SLICE_OVERLAP"] == 0.27
    assert params["SLICE_TRAINED_BODY_PX"] == 480.0
    # _training_values returns enabled=True; the config must still own it.
    assert params["SLICE_ENABLED"] is True
    params_off = _build(_cfg(model, slice_enabled=False))
    assert params_off["SLICE_ENABLED"] is False


def test_legacy_v1_flat_sidecar_is_applied(tmp_path):
    model = _sidecar(
        tmp_path,
        {"geometry_mode": "auto_object", "imgsz": 640, "overlap": 0.33},
    )
    assert _build(_cfg(model))["SLICE_OVERLAP"] == 0.33


def test_unclaimed_merge_keys_become_defaults(tmp_path):
    """Ruling R3: unclaimed means the DEFAULT, never 'whatever was there'."""
    settings = {k: v for k, v in SETTINGS.items() if not k.startswith("merge_")}
    model = _sidecar(
        tmp_path,
        {
            "schema_version": 2,
            "training_geometry": {"geometry_mode": "auto_model"},
            "primary_profile_id": "p",
            "profiles": [{"id": "p", "name": "P", "settings": settings}],
        },
    )
    advanced = dict(ADVANCED, slice_merge_policy="nmm", slice_merge_threshold=0.9)
    params = build_engine_params(
        _cfg(model), runtime=RUNTIME, advanced_config=advanced
    )
    assert params["SLICE_MERGE_POLICY"] == SLICE_MERGE_DEFAULTS["merge_policy"]
    assert params["SLICE_MERGE_THRESHOLD"] == SLICE_MERGE_DEFAULTS["merge_threshold"]


def test_missing_sidecar_leaves_advanced_untouched(tmp_path):
    model = tmp_path / "bare.pt"
    model.write_text("stub", encoding="utf-8")
    advanced = dict(ADVANCED, slice_overlap=0.44)
    params = build_engine_params(
        _cfg(str(model), slice_profile_id="balanced"),
        runtime=RUNTIME,
        advanced_config=advanced,
    )
    assert params["SLICE_OVERLAP"] == 0.44


def test_nonexistent_model_path_does_not_raise(tmp_path):
    assert _build(_cfg(str(tmp_path / "nope.pt")))["SLICE_OVERLAP"] == 0.2


def test_corrupt_sidecar_is_a_no_op(tmp_path):
    model = tmp_path / "c.pt"
    model.write_text("stub", encoding="utf-8")
    (tmp_path / "c.pt.slice_meta.json").write_text("{not json", encoding="utf-8")
    assert _build(_cfg(str(model)))["SLICE_OVERLAP"] == 0.2


def test_sequential_mode_gets_no_overlay(tmp_path):
    params = _build(
        _cfg(
            _profiled(tmp_path),
            yolo_obb_mode="sequential",
            slice_profile_id="balanced",
        )
    )
    assert params["SLICE_OVERLAP"] == 0.2
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_engine_params_slice_profile.py -v`
Expected: the overlay tests FAIL (`SLICE_OVERLAP == 0.2`); the three
config-owns-it assertions in `test_config_owns_enabled_geometry_and_confidence`
already PASS — that is intended, they pin behaviour that must **not** change.

- [ ] **Step 3: Implement the helper**

Add near `_default_advanced_config_fallback` (`engine_params.py:348`). Confirm
`logger` exists at module scope in this file; if it does not, add
`logger = logging.getLogger(__name__)` with the import.

```python
def _slice_profile_overlay(
    cfg: Mapping[str, Any], model_path: str
) -> dict[str, Any] | None:
    """Resolve the calibration profile a saved config asks for.

    ``advanced_config.json`` is machine-global and the GUI never writes it
    back, so before this existed the CLI ran a config's named profile with
    DEFAULT tile geometry and merge settings -- same config, different
    detections (design doc F1). The per-video config already carries
    ``slice_profile_id`` and a full effective-settings snapshot; this reads
    them through the same ladder the panel uses, so the two cannot drift.

    Scope is deliberate and asymmetric: this supplies only the keys that live
    *solely* in ``advanced_config``. ``slice_enabled``, ``slice_geometry_mode``
    and ``yolo_confidence_threshold`` are persisted by ``build_config_dict``
    and are authoritative from the config -- they already hold whatever the
    profile put in the widgets when the session was saved, so re-deriving them
    here could only disagree with what the user saw.

    Returns ``None`` -- "leave advanced_config alone" -- for sequential mode,
    a missing/corrupt sidecar, or an empty model path. Never raises: this
    builder is called with model paths that do not exist.
    """
    from hydra_suite.core.inference.slice_meta import (
        read_slice_meta,
        resolve_slice_profile_values,
    )

    mode = str(_cfg_get(cfg, "yolo_obb_mode", default="direct")).strip().lower()
    if mode != "direct" or not model_path:
        return None
    meta = read_slice_meta(model_path)
    if not meta:
        return None
    profile_id = str(_cfg_get(cfg, "slice_profile_id", default="") or "")
    saved = _cfg_get(cfg, "slice_profile_settings", default=None)
    saved_settings = dict(saved) if isinstance(saved, dict) and saved else None
    values = resolve_slice_profile_values(meta, profile_id or None, saved_settings)

    if profile_id and profile_id not in ("__training__", "__custom__"):
        if values.get("resolution") == "primary":
            logger.warning(
                "Config names SAHI profile %r, which is not in %s; "
                "falling back to the primary profile %r.",
                profile_id,
                model_path,
                values.get("profile_name"),
            )
    claimed = values.get("confidence_threshold")
    if claimed is not None:
        configured = float(_cfg_get(cfg, "yolo_confidence_threshold", default=0.25))
        if abs(float(claimed) - configured) > 1e-9:
            # The config wins (design ruling R1) -- but silently running an
            # operating point the profile never measured is the provenance lie
            # this whole feature exists to remove.
            logger.warning(
                "SAHI profile %r was measured at confidence %.3f; this config "
                "overrides it with %.3f.",
                values.get("profile_name"),
                float(claimed),
                configured,
            )
    logger.info(
        "SAHI calibration: %s (id=%s, resolution=%s)",
        values.get("profile_name", "?"),
        values.get("profile_id"),
        values.get("resolution"),
    )
    return values
```

Note the training rung is **not** filtered out (ruling R2): `_training_values`
returns real overlap / fraction / `trained_body_px`, and that is the sidecar
every sliced-training publish writes.

- [ ] **Step 4: Wire it at the call site**

Immediately before the params dict is constructed, after `yolo_direct_path` is
resolved (`engine_params.py:626`) and before the `SLICE_*` block (:913):

```python
    # Calibration-profile overlay. Supplies ONLY the advanced_config-only SAHI
    # keys; SLICE_ENABLED / SLICE_GEOMETRY_MODE / YOLO_CONFIDENCE_THRESHOLD are
    # persisted config fields and stay untouched (design rulings R1, R2).
    _profile_values = _slice_profile_overlay(cfg, yolo_direct_path)
    if _profile_values is not None:
        advanced["slice_overlap"] = float(_profile_values["overlap"])
        advanced["slice_object_tile_fraction"] = float(
            _profile_values["object_tile_fraction"]
        )
        advanced["slice_trained_body_px"] = float(_profile_values["trained_body_px"])
        advanced["slice_width"] = int(_profile_values["slice_width"])
        advanced["slice_height"] = int(_profile_values["slice_height"])
        for _key in (
            "merge_policy",
            "merge_metric",
            "merge_threshold",
            "merge_backend",
        ):
            _value = _profile_values[_key]
            advanced[f"slice_{_key}"] = (
                SLICE_MERGE_DEFAULTS[_key] if _value is None else _value
            )
```

**Do not modify** the `SLICE_ENABLED`, `SLICE_GEOMETRY_MODE` or
`YOLO_CONFIDENCE_THRESHOLD` entries in the params dict. Leave their existing
`_cfg_get` expressions and default literals exactly as they are.

- [ ] **Step 5: Run the new tests and the affected suite**

Run: `python -m pytest tests/test_engine_params_slice_profile.py tests/test_profile_cache_keys.py tests/test_engine_params_extraction.py tests/test_trackerkit_cli_config.py -v`
Expected: all PASS

- [ ] **Step 6: Prove the characterization goldens are untouched**

Run: `python -m pytest tests/test_get_parameters_dict_characterization.py tests/test_gui_cli_param_equivalence.py -v`
Expected: PASS with no golden regeneration. If a golden changes, STOP: either a
fixture model has acquired a sidecar in this machine's models dir (see the
Global Constraints' inertness clause) or the overlay is not the GUI no-op it is
designed to be.

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/engine_params.py tests/test_engine_params_slice_profile.py
git commit -m "fix(trackerkit): headless param builder honours calibration profiles"
```

---

## Task 4: GUI↔CLI parity guard for a non-default profile

The audit's root observation: this bug survived the GUI/CLI unification gate
because no golden or oracle covers a non-default SAHI case. The fix is not
durable until that hole is closed. This task also discharges review finding C1:
`get_parameters_dict()` is itself a `build_engine_params` call
(`gui/orchestrators/config.py:2153-2160`), so the overlay must provably be a
no-op there.

**Files:**
- Test: `tests/test_gui_cli_profile_parity.py` (create)

**Interfaces:**
- Consumes: `_make_panel_with_sidecar` (`tests/test_trackerkit_slice_meta_prefill.py:19`); `build_engine_params`; `MainWindow.get_parameters_dict` (`gui/main_window.py:3026`); `self._config_orch.build_config_dict()`.

- [ ] **Step 1: Read the two things this test depends on**

Read `tests/test_gui_cli_param_equivalence.py` for its excluded-key set and its
`RuntimeContext` construction, and reuse both verbatim. Read
`_get_selected_yolo_model_path` in `detection_panel.py` and find the accessor
the GUI uses to select a model — `apply_slice_meta_for_model` alone does **not**
select it, and `build_config_dict` reads the path from the selector. Without a
real selection the CLI side sees an empty model path, the overlay returns
`None`, and the test compares nothing (review C4).

- [ ] **Step 2: Write the test**

```python
"""A calibrated profile must produce identical params in the GUI and the CLI.

This is the hole that let the headless path ignore profiles for a whole
release: every golden and oracle covered DEFAULT SAHI settings, where the
GUI's in-memory advanced_config and the CLI's machine-global
advanced_config.json happen to agree.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from hydra_suite.trackerkit.engine_params import (  # noqa: E402
    RuntimeContext,
    build_engine_params,
)
from tests.test_trackerkit_slice_meta_prefill import (  # noqa: E402
    _make_panel_with_sidecar,
)

PROFILE_SETTINGS = {
    "enabled": True,
    "geometry_mode": "auto_object",
    "slice_width": 704,
    "slice_height": 512,
    "overlap": 0.31,
    "object_tile_fraction": 0.11,
    "trained_body_px": 560.0,
    "confidence_threshold": 0.42,
    "merge_policy": "nmm",
    "merge_metric": "iou",
    "merge_threshold": 0.6,
    "merge_backend": "cv2",
}

SLICE_KEYS = (
    "SLICE_ENABLED",
    "SLICE_GEOMETRY_MODE",
    "SLICE_OVERLAP",
    "SLICE_OBJECT_TILE_FRACTION",
    "SLICE_WIDTH",
    "SLICE_HEIGHT",
    "SLICE_TRAINED_BODY_PX",
    "SLICE_MERGE_POLICY",
    "SLICE_MERGE_METRIC",
    "SLICE_MERGE_THRESHOLD",
    "SLICE_MERGE_BACKEND",
    "YOLO_CONFIDENCE_THRESHOLD",
)


def _select_profiled_model(tmp_path, monkeypatch):
    panel, window, model_path = _make_panel_with_sidecar(tmp_path, monkeypatch)
    (model_path.parent / (model_path.name + ".slice_meta.json")).write_text(
        json.dumps(
            {
                "schema_version": 2,
                "training_geometry": {"geometry_mode": "auto_model", "imgsz": 640},
                "primary_profile_id": "balanced",
                "profiles": [
                    {
                        "id": "balanced",
                        "name": "Balanced",
                        "settings": PROFILE_SETTINGS,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # Select the model through the real selector -- build_config_dict reads the
    # path from it, not from apply_slice_meta_for_model's argument.
    window._set_yolo_model_selection(str(model_path))
    panel.apply_slice_meta_for_model(str(model_path))
    return panel, window, model_path


def test_gui_and_builder_agree_on_a_profiled_model(tmp_path, monkeypatch):
    panel, window, model_path = _select_profiled_model(tmp_path, monkeypatch)

    gui_params = window.get_parameters_dict()
    cfg = window._config_orch.build_config_dict()
    assert cfg["yolo_obb_direct_model_path"], "model must be selected in the config"

    cli_params = build_engine_params(
        cfg, runtime=RuntimeContext(fps=float(gui_params.get("FPS", 30.0)),
                                    total_frames=None,
                                    frame_width=None,
                                    frame_height=None)
    )
    for key in SLICE_KEYS:
        assert cli_params[key] == gui_params[key], key
    # Guard against a vacuous pass: the profile must genuinely be in effect.
    assert gui_params["SLICE_OVERLAP"] == 0.31
    assert gui_params["SLICE_MERGE_POLICY"] == "nmm"


def test_overlay_is_a_no_op_inside_the_gui(tmp_path, monkeypatch):
    """get_parameters_dict() routes through build_engine_params too (C1).

    The GUI's advanced_config already holds the panel-applied profile values,
    so the overlay must recompute exactly those -- if it disagrees, selecting a
    profile in the GUI would change params behind the user's back.
    """
    panel, window, model_path = _select_profiled_model(tmp_path, monkeypatch)
    cfg = window._config_orch.build_config_dict()
    runtime = RuntimeContext(fps=30.0, total_frames=None,
                             frame_width=None, frame_height=None)

    with_overlay = build_engine_params(
        cfg, runtime=runtime, advanced_config=dict(window.advanced_config)
    )
    without_model = build_engine_params(
        dict(cfg, yolo_obb_direct_model_path=""),
        runtime=runtime,
        advanced_config=dict(window.advanced_config),
    )
    for key in SLICE_KEYS:
        if key in ("SLICE_ENABLED", "SLICE_GEOMETRY_MODE"):
            continue  # config-owned; unaffected by the model path
        assert with_overlay[key] == without_model[key], key
```

If `window._set_yolo_model_selection` is not the right accessor, use whichever
one `tests/test_trackerkit_slice_meta_prefill.py` or the config orchestrator
uses — read, do not guess.

- [ ] **Step 3: Run**

Run: `python -m pytest tests/test_gui_cli_profile_parity.py -v`
Expected: PASS. The first test would have FAILED before Task 3 — note that in
the commit message. If `test_overlay_is_a_no_op_inside_the_gui` fails on the
merge keys, that is ruling R3's predicted collision with the panel's
`resetting=False` restore branch (`detection_panel.py:2721-2732`); fix it by
dropping that guard in the panel, not by weakening the builder.

- [ ] **Step 4: Commit**

```bash
git add tests/test_gui_cli_profile_parity.py
git commit -m "test(trackerkit): GUI/CLI parity oracle for a non-default SAHI profile"
```

---

## Task 5: `--sahi-profile` CLI flag

**Files:**
- Modify: `src/hydra_suite/trackerkit/app.py:87-110` (parser), `:236-241` (dispatch)
- Modify: `src/hydra_suite/trackerkit/cli.py:22-70`
- Modify: `src/hydra_suite/trackerkit/cli_config.py` (new pure helper)
- Test: `tests/test_trackerkit_cli_sahi_profile.py` (create)

**Interfaces:**
- Consumes: `read_slice_meta`, `available_slice_profiles`, `resolve_model_path`.
- Produces:
  - `apply_sahi_profile_override(cfg: dict, profile: str) -> dict` in `cli_config.py` — returns a new cfg with `slice_profile_id` set to the resolved id; raises `ValueError` when the name/id is not in the model's sidecar.
  - `run_tracking_cli(..., sahi_profile: str | None = None)`.

- [ ] **Step 1: Write the failing tests**

```python
import json
import pytest

from hydra_suite.trackerkit.cli_config import apply_sahi_profile_override


def _cfg_with_model(tmp_path, profiles):
    model = tmp_path / "m.pt"
    model.write_text("stub", encoding="utf-8")
    (tmp_path / "m.pt.slice_meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "training_geometry": {"geometry_mode": "auto_model"},
                "primary_profile_id": "",
                "profiles": profiles,
            }
        ),
        encoding="utf-8",
    )
    return {"yolo_obb_direct_model_path": str(model)}


PROFILES = [
    {"id": "p-abc", "name": "High recall", "settings": {"enabled": True}},
    {"id": "p-def", "name": "Balanced", "settings": {"enabled": True}},
]


def test_override_by_name(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    assert apply_sahi_profile_override(cfg, "Balanced")["slice_profile_id"] == "p-def"


def test_override_by_id(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    assert apply_sahi_profile_override(cfg, "p-abc")["slice_profile_id"] == "p-abc"


def test_override_clears_stale_snapshot(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    cfg["slice_profile_settings"] = {"overlap": 0.9}
    out = apply_sahi_profile_override(cfg, "Balanced")
    assert "slice_profile_settings" not in out


def test_unknown_profile_is_a_hard_error(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    with pytest.raises(ValueError) as excinfo:
        apply_sahi_profile_override(cfg, "Nope")
    assert "Nope" in str(excinfo.value)
    assert "High recall" in str(excinfo.value)


def test_missing_sidecar_is_a_hard_error(tmp_path):
    model = tmp_path / "bare.pt"
    model.write_text("stub", encoding="utf-8")
    with pytest.raises(ValueError):
        apply_sahi_profile_override(
            {"yolo_obb_direct_model_path": str(model)}, "Balanced"
        )


def test_sequential_mode_is_a_hard_error(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    cfg["yolo_obb_mode"] = "sequential"
    with pytest.raises(ValueError) as excinfo:
        apply_sahi_profile_override(cfg, "Balanced")
    assert "sequential" in str(excinfo.value)


def test_training_keyword_is_accepted(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    out = apply_sahi_profile_override(cfg, "__training__")
    assert out["slice_profile_id"] == "__training__"


def test_original_cfg_is_not_mutated(tmp_path):
    cfg = _cfg_with_model(tmp_path, PROFILES)
    apply_sahi_profile_override(cfg, "Balanced")
    assert "slice_profile_id" not in cfg
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_trackerkit_cli_sahi_profile.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_sahi_profile_override'`

- [ ] **Step 3: Implement the helper in `cli_config.py`**

```python
def apply_sahi_profile_override(
    cfg: Mapping[str, Any], profile: str
) -> dict[str, Any]:
    """Return ``cfg`` with an explicitly requested SAHI profile selected.

    An explicit ``--sahi-profile`` that does not resolve is a HARD ERROR,
    deliberately unlike a config's own saved id (which falls back down the
    ladder with a warning). The user named an operating point; quietly
    running a different one is the exact failure this feature exists to
    remove.
    """
    from hydra_suite.core.inference.model_paths import resolve_model_path
    from hydra_suite.core.inference.slice_meta import (
        available_slice_profiles,
        read_slice_meta,
    )

    requested = str(profile).strip()
    result = dict(cfg)
    if requested == "__training__":
        result["slice_profile_id"] = "__training__"
        result.pop("slice_profile_settings", None)
        return result

    model_path = resolve_model_path(
        str(
            result.get("yolo_obb_direct_model_path")
            or result.get("yolo_model_path")
            or ""
        )
    )
    meta = read_slice_meta(model_path) if model_path else None
    profiles = available_slice_profiles(meta) if meta else []
    match = next(
        (item for item in profiles if item["id"] == requested),
        None,
    ) or next((item for item in profiles if item["name"] == requested), None)
    if match is None:
        known = ", ".join(f"{item['name']!r} ({item['id']})" for item in profiles)
        raise ValueError(
            f"--sahi-profile {requested!r} is not a profile of "
            f"{model_path or '<no direct model in config>'}. "
            f"Available: {known or '(none)'}"
        )
    if str(result.get("yolo_obb_mode", "direct")).strip().lower() != "direct":
        raise ValueError(
            "--sahi-profile applies to direct-detector inference only; this "
            "config runs in sequential mode, where profiles are not consumed."
        )
    result["slice_profile_id"] = match["id"]
    # A snapshot captured under a DIFFERENT profile would win the ladder over
    # the id we just set (rung 3 beats a missing id, and an explicit override
    # must beat both).
    result.pop("slice_profile_settings", None)
    return result
```

- [ ] **Step 4: Thread it through `cli.py` and `app.py`**

In `cli.py`, add `sahi_profile: str | None = None` to `run_tracking_cli`'s
keyword-only signature. Where the per-item config is resolved (around the
`load_tracker_cli_session` call at `cli.py:64-70`), when `sahi_profile` is
set, load the config explicitly and override it before handing it over:

```python
            if sahi_profile:
                base_cfg = (
                    effective_config_data
                    if effective_config_data is not None
                    else load_tracker_cli_config(item.config_path)
                )
                effective_config_data = apply_sahi_profile_override(
                    base_cfg, sahi_profile
                )
```

placed immediately before the `session = load_tracker_cli_session(...)` call,
and pass `config_path=None` when `effective_config_data is not None` (the
existing conditional at `:66-68` already does this).

Two things the review checked and the plan must preserve: the override is
re-applied per video (correct — videos 2..N take the keystone baseline, which
is re-overridden each iteration), and the keystone dump written at
`cli.py:80-84` must record the OVERRIDDEN config, not the pre-override one, or
the provenance file will name a profile the run did not use. Verify which
object is dumped and adjust if needed.

In `app.py`, add to `track_parser` (after `--keystone-override`, `:108`):

```python
    track_parser.add_argument(
        "--sahi-profile",
        type=str,
        help=(
            "Name or id of a calibration profile from the direct model's "
            ".slice_meta.json sidecar. Overrides the profile saved in the "
            "config. Use __training__ for the model's training geometry. "
            "Applies to every video in the batch."
        ),
    )
```

and at the dispatch (`app.py:239`):

```python
            exit_code = run_tracking_cli(
                resolved_videos,
                config_path=args.config,
                keystone_override=bool(args.keystone_override),
                sahi_profile=getattr(args, "sahi_profile", None),
            )
```

- [ ] **Step 5: Run**

Run: `python -m pytest tests/test_trackerkit_cli_sahi_profile.py tests/test_trackerkit_cli_config.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/app.py src/hydra_suite/trackerkit/cli.py src/hydra_suite/trackerkit/cli_config.py tests/test_trackerkit_cli_sahi_profile.py
git commit -m "feat(trackerkit): --sahi-profile selects a calibration profile headlessly"
```

---

## Task 6: Confidence edits mark the profile Custom (the F2 fix)

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` (near the spin wiring at :866-878)
- Test: `tests/test_trackerkit_slice_meta_prefill.py` (extend)

**Interfaces:**
- Consumes: `_mark_slice_profile_custom` (`:2603`), `_applying_slice_profile` (`:769`).
- Produces: no new interface.

- [ ] **Step 1: Write both directions of the failing test**

A test that only checks "user edit marks Custom" passes with a fix that breaks
profile application. Both assertions are required.

```python
def test_confidence_edit_marks_profile_custom(tmp_path, monkeypatch):
    panel, mw, model_path = _make_panel_with_sidecar(tmp_path, monkeypatch)
    _write_sidecar_with_profile(model_path)   # reuse this module's helper
    panel.apply_slice_meta_for_model(str(model_path))
    assert mw.advanced_config["slice_profile_id"] != "__custom__"

    panel.spin_yolo_confidence.setValue(0.15)

    assert mw.advanced_config["slice_profile_id"] == "__custom__"
    assert panel.combo_slice_profile.currentData() == "__custom__"


def test_applying_a_profile_does_not_mark_custom(tmp_path, monkeypatch):
    panel, mw, model_path = _make_panel_with_sidecar(tmp_path, monkeypatch)
    _write_sidecar_with_profile(model_path)   # profile claims confidence 0.42
    panel.apply_slice_meta_for_model(str(model_path))

    assert panel.spin_yolo_confidence.value() == pytest.approx(0.42)
    assert mw.advanced_config["slice_profile_id"] != "__custom__"
```

Add `_write_sidecar_with_profile(model_path)` to the module if it does not
already exist, writing a v2 sidecar whose single profile claims
`confidence_threshold: 0.42` and is the primary.

- [ ] **Step 2: Run to verify the first test fails**

Run: `python -m pytest tests/test_trackerkit_slice_meta_prefill.py -k confidence -v`
Expected: `test_confidence_edit_marks_profile_custom` FAILS; the second PASSES

- [ ] **Step 3: Connect the signal**

`spin_yolo_confidence` is constructed at `detection_panel.py:1057` — connecting
it at :878 (where the SAHI spin loop lives) raises `AttributeError` during
`__init__`. Insert **after** the spin's `setToolTip` block ends at :1066, before
`spin_yolo_iou` is created:

```python
        # The profile-owned SAHI spins above route through _sync_advanced;
        # confidence is different -- it is the user's global YOLO threshold,
        # written straight into the config, not into advanced_config. It must
        # still un-claim the profile: a session that reports "Balanced" while
        # running a threshold Balanced never measured is a provenance lie.
        # _mark_slice_profile_custom's own _applying_slice_profile guard makes
        # this safe against the programmatic setValue in
        # _apply_slice_meta_values:2740.
        self.spin_yolo_confidence.valueChanged.connect(
            lambda _value: self._mark_slice_profile_custom()
        )
```

Verify by reading `_apply_slice_meta_values:2739-2743` that the confidence
`setValue` happens **inside** the `self._applying_slice_profile = True` block
(it does, `:2693`/`:2763`) — that is what makes the second test pass.

- [ ] **Step 4: Run**

Run: `python -m pytest tests/test_trackerkit_slice_meta_prefill.py tests/test_detection_panel_slice_widgets.py tests/test_trackerkit_profile_session.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/gui/panels/detection_panel.py tests/test_trackerkit_slice_meta_prefill.py
git commit -m "fix(trackerkit): confidence edits un-claim the calibration profile"
```

---

## Task 7: Sidecars survive publish and import (F3 + F4)

**Files:**
- Modify: `src/hydra_suite/training/model_publish.py:872-935`
- Create the shared helper in: `src/hydra_suite/core/inference/model_paths.py`
- Modify: `src/hydra_suite/detectkit/gui/project.py:193-207` (delegate to it)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:3688`
- Test: `tests/test_model_sidecar_durability.py` (create)

**Interfaces:**
- Produces: `copy_model_metadata_sidecars(source: Path, destination: Path) -> None` in `core/inference/model_paths.py`, plus the suffix tuple `MODEL_METADATA_SIDECAR_SUFFIXES`.
- Consumes: `read_slice_meta`, `normalized_slice_meta`, `profile_summary`, `verify_profile_summary`.

`core/inference/model_paths.py` is the right home: both kits already import
from it, so neither kit imports the other.

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

from hydra_suite.core.inference.model_paths import copy_model_metadata_sidecars

META = {
    "schema_version": 2,
    "training_geometry": {"geometry_mode": "auto_model"},
    "primary_profile_id": "p",
    "profiles": [{"id": "p", "name": "P", "settings": {"enabled": True}}],
}


def test_copies_every_sidecar_suffix(tmp_path):
    src = tmp_path / "a.pt"
    src.write_text("w", encoding="utf-8")
    for suffix in (".slice_meta.json", ".canonical_meta.json", ".runtime_meta.json"):
        Path(str(src) + suffix).write_text("{}", encoding="utf-8")
    src.with_suffix(".v2meta.json").write_text("{}", encoding="utf-8")

    dst = tmp_path / "out" / "b.pt"
    dst.parent.mkdir()
    dst.write_text("w", encoding="utf-8")
    copy_model_metadata_sidecars(src, dst)

    for suffix in (".slice_meta.json", ".canonical_meta.json", ".runtime_meta.json"):
        assert Path(str(dst) + suffix).exists(), suffix
    assert dst.with_suffix(".v2meta.json").exists()


def test_absent_sidecars_are_not_an_error(tmp_path):
    src = tmp_path / "a.pt"
    src.write_text("w", encoding="utf-8")
    dst = tmp_path / "b.pt"
    dst.write_text("w", encoding="utf-8")
    copy_model_metadata_sidecars(src, dst)   # must not raise
```

And, for publish:

```python
def test_publish_without_slice_geometry_keeps_profiles(tmp_path, monkeypatch):
    """A calibrated-then-registered model must not lose its profiles."""
    # Arrange a source .pt + v2 sidecar, publish it with slice_geometry=None
    # into a temp HYDRA_DATA_DIR, then assert:
    #   1. the destination sidecar exists and still lists profile "p"
    #   2. the registry entry carries slice_profiles == {"count": 1, ...}
```

**This stub must become a real test, not prose.** Read `publish_trained_model`'s
signature at `model_publish.py` and use the exact keyword names; construct the
source `.pt` + v2 sidecar, call publish with `slice_geometry=None`, and assert
(1) `read_slice_meta(dst)` still lists profile `"p"`, and (2) the registry entry
carries `slice_profiles == {"count": 1, "primary_profile_id": "p", "names": ["P"]}`.
Also assert the negative: publishing a source with NO sidecar still writes no
sidecar (this is `test_no_slice_geometry_writes_no_sidecar`'s existing
expectation — the review confirmed it survives, and it must keep passing). Set
`HYDRA_DATA_DIR` via `monkeypatch.setenv` as
`tests/test_trackerkit_slice_meta_prefill.py:20-27` does.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_model_sidecar_durability.py -v`
Expected: FAIL — `ImportError` for the helper, and the publish test fails on a missing destination sidecar

- [ ] **Step 3: Implement the shared helper**

In `core/inference/model_paths.py`:

```python
MODEL_METADATA_SIDECAR_SUFFIXES = (
    ".slice_meta.json",
    ".canonical_meta.json",
    ".runtime_meta.json",
)


def copy_model_metadata_sidecars(source, destination) -> None:
    """Copy inference metadata stored beside a model, preserving its naming.

    Calibration profiles, canonical geometry and runtime stamps are only
    meaningful beside their weights. Copying a ``.pt`` without them produces
    a model that silently loses its operating points -- which is what both
    TrackerKit's import and one branch of publish used to do.
    """
    import shutil
    from pathlib import Path

    src = Path(source)
    dst = Path(destination)
    pairs = [
        (
            src.with_suffix(src.suffix + suffix),
            dst.with_suffix(dst.suffix + suffix),
        )
        for suffix in MODEL_METADATA_SIDECAR_SUFFIXES
    ]
    pairs.append(
        (src.with_suffix(".v2meta.json"), dst.with_suffix(".v2meta.json"))
    )
    for src_sidecar, dst_sidecar in pairs:
        if src_sidecar.exists():
            dst_sidecar.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_sidecar), str(dst_sidecar))
```

- [ ] **Step 4: Delegate the DetectKit copy**

Replace the body of `detectkit/gui/project.py:_copy_model_metadata_sidecars`
with a call to the shared helper, keeping the private name and its callers
untouched (`project.py:282, 604`). Keep
`_MODEL_METADATA_SIDECAR_SUFFIXES` as an alias of the shared tuple if it is
referenced elsewhere in that module.

- [ ] **Step 5: Close the publish gap**

In `model_publish.py`, after the existing `if slice_geometry and role in
_DIRECT_DETECTOR_ROLES and dst.suffix.lower() == ".pt":` block, add:

```python
    elif role in _DIRECT_DETECTOR_ROLES and dst.suffix.lower() == ".pt":
        # No training geometry to stamp, but the SOURCE may still carry
        # user-approved calibration profiles. Dropping them here silently
        # un-calibrated any model registered outside a sliced training run.
        source_meta = read_slice_meta(src)
        if source_meta:
            merged_slice_meta = normalized_slice_meta(source_meta)
            slice_sidecar = dst.with_suffix(dst.suffix + ".slice_meta.json")
            slice_sidecar.write_text(
                json.dumps(merged_slice_meta, indent=2), encoding="utf-8"
            )
            slice_geom_sidecar_name = slice_sidecar.name
```

and change the registry stamp (`:930-935`) so the profile inventory is not
gated on `slice_geometry`:

```python
    if slice_geometry and role in _DIRECT_DETECTOR_ROLES:
        metadata["slice_geometry"] = dict(slice_geometry)
    if role in _DIRECT_DETECTOR_ROLES and merged_slice_meta is not None:
        metadata["slice_profiles"] = profile_summary(merged_slice_meta)
        verify_profile_summary(dst, metadata["slice_profiles"])
        if slice_geom_sidecar_name:
            metadata["slice_meta_sidecar"] = slice_geom_sidecar_name
```

Import `normalized_slice_meta` alongside the existing `read_slice_meta` /
`merge_training_geometry` imports at the top of the module.

- [ ] **Step 6: Fix TrackerKit's import**

At `gui/orchestrators/config.py:3688`, after the successful `shutil.copy2`:

```python
        try:
            copy_model_metadata_sidecars(Path(src), Path(dest_path))
        except Exception as exc:  # sidecars are best-effort, weights are not
            logger.warning("Could not copy model metadata sidecars: %s", exc)
```

with `from hydra_suite.core.inference.model_paths import
copy_model_metadata_sidecars` added to that module's imports.

- [ ] **Step 6b: Stamp the imported model's profile inventory**

Copying the sidecar is not enough: TrackerKit's import writes its own registry
`metadata` dict (`gui/orchestrators/config.py:3699+`) and never records
`slice_profiles`, so an imported calibrated model shows a profile combo but an
empty registry inventory. After the sidecar copy, add:

```python
        imported_meta = read_slice_meta(dest_path)
        if imported_meta:
            metadata["slice_profiles"] = profile_summary(imported_meta)
```

placed where the `metadata` dict is assembled, with the two imports added to
that module. Add a test asserting the registry entry's `slice_profiles["count"]`
matches the copied sidecar.

- [ ] **Step 7: Run**

Run: `python -m pytest tests/test_model_sidecar_durability.py tests/test_slice_profile_mutations.py tests/test_augmentation_profile_canonical_copies.py -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
make format
git add src/hydra_suite/core/inference/model_paths.py src/hydra_suite/detectkit/gui/project.py src/hydra_suite/training/model_publish.py src/hydra_suite/trackerkit/gui/orchestrators/config.py tests/test_model_sidecar_durability.py
git commit -m "fix(models): calibration sidecars survive publish and import"
```

---

## Task 8: Provenance log line + docs + known residual

**Files:**
- Modify: `src/hydra_suite/trackerkit/headless_tracking.py` (session start) or `cli.py` — whichever already logs the session banner; read first
- Modify: `docs/user-guide/` page covering TrackerKit SAHI (locate with `grep -rl "slice_profile\|SAHI profile" docs/`)
- Test: `tests/test_engine_params_slice_profile.py` (extend)

**Interfaces:**
- Consumes: `_slice_profile_overlay` (Task 3).
- Produces: no new public interface. **No new params keys** (Global Constraints).

The `logger.info` / `logger.warning` calls themselves landed in Task 3 (they
live inside `_slice_profile_overlay`). This task covers the fallback warning's
test, the docs, and the residual note.

- [ ] **Step 1: Test the deleted-profile fallback warning**

The warning must fire only for a genuinely missing *named* profile — not for
`"__custom__"` without a snapshot, which also resolves to `"primary"` but is a
normal state, not a lost profile.

```python
def test_missing_requested_id_warns_and_falls_back(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        params = _build(_cfg(_profiled(tmp_path), slice_profile_id="gone"))
    assert params["SLICE_OVERLAP"] == 0.31          # primary applied
    assert any("gone" in record.message for record in caplog.records)


def test_custom_without_snapshot_does_not_warn_about_a_missing_profile(
    tmp_path, caplog
):
    with caplog.at_level("WARNING"):
        _build(_cfg(_profiled(tmp_path), slice_profile_id="__custom__"))
    assert not [r for r in caplog.records if "not in" in r.message]
```

Add both to `tests/test_engine_params_slice_profile.py`, reusing that module's
`_build` / `_cfg` / `_profiled` helpers.

- [ ] **Step 3: Document**

Add to the SAHI/calibration user-guide page:
- that `trackerkit track` now honours the profile saved in the config;
- the `--sahi-profile <name|id>` flag with its hard-error semantics;
- **that a profile does not override a saved confidence threshold** (ruling R1):
  the config's `yolo_confidence_threshold` wins and a mismatch is logged. To
  adopt a profile's measured confidence, re-select the profile in the GUI.
- the **known residual**: other `advanced_config` keys (`obb_seg_*`, etc.)
  remain machine-global on the CLI and are *not* carried by the saved config.
  This fix covers the SAHI/merge/confidence knobs only.

- [ ] **Step 4: Run the docs gate and the full affected suite**

```bash
make docs-check
python -m pytest tests/test_engine_params_slice_profile.py tests/test_gui_cli_profile_parity.py tests/test_profile_cache_keys.py tests/test_trackerkit_cli_sahi_profile.py tests/test_model_sidecar_durability.py tests/test_slice_profile_resolution.py -v
make lint-moderate
```

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "feat(trackerkit): log the applied calibration profile; document CLI profiles"
```

---

## Final verification (before merge)

- [ ] **Delta test gate.** Capture the failing-test set on the merge-base first, then on the branch; the sets must be identical minus the tests this plan adds. Batch per-file — `pytest tests/` never finishes on this repo (classkit modal-dialog hang + SIGABRT).
- [ ] **MPS equivalence matrix** from the repo root (not a worktree — fixtures live in the main repo), baseline = merge-base, `RUNTIME=mps`. Expect byte-identical on all clips: the overlay is inert on fixtures that carry no `slice_profile*` key and no sidecar. Verify CSV row counts > 1 before trusting any `EQUIVALENT`.
- [ ] **CUDA equivalence** on `rutalab@mehek.taild08eb9.ts.net` with `hydra-cuda`.
- [ ] **Manual smoke, the thing the user actually asked for:** calibrate a profile in DetectKit, publish, select it in TrackerKit, save the config, then run `trackerkit track --config <video>_config.json` and confirm the log line names the same profile and the detection cache key matches the GUI run's.
- [ ] Move this plan and the two spec docs into their `done/` subfolders in the merge commit.
