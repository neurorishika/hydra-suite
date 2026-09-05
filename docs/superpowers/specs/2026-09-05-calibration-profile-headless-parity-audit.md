# DetectKit calibration profile -> TrackerKit handoff: audit

Repo: `/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker` @ main (2bec50f5), read-only. All paths below are under `src/hydra_suite/` unless noted.

## Verdict

**GUI path: fully wired. Headless/CLI path: not wired.** In the TrackerKit GUI every profile in the sidecar appears in a "SAHI profile" combo, the explicit primary is auto-applied when a model is selected, all profile-owned knobs (enabled, geometry mode, overlap, tile fraction, tile W/H, trained body px, confidence, merge policy/metric/threshold/backend) reach `InferenceConfig.obb.direct.slice` / `confidence_threshold`, manual edits flip the combo to "Custom", and the session config records both the profile id and an effective-settings snapshot. **The single biggest gap:** `build_engine_params` reads overlap / tile fraction / tile W/H / trained_body_px / all four merge keys *only* from `advanced_config` (`trackerkit/engine_params.py:919-935`), the GUI never writes `advanced_config.json` back, and the CLI never passes an `advanced_config` -- so `trackerkit track --config <video>_config.json` silently runs the *default* SAHI geometry and merge policy, not the calibrated profile the config names. A second, smaller gap: the confidence spin is not hooked to the Custom marker, so the panel can claim a profile while running a different threshold.

## Round-trip field table

Producer = DetectKit `DirectCalibrationResultsDialog.settings_for_row` (`detectkit/gui/dialogs/direct_calibration_results.py:481-507`). Sidecar = `<model>.pt.slice_meta.json` v2 `profiles[].settings` (`core/inference/slice_meta.py:76-123`). Reader = `slice_meta_to_panel_values` (`slice_meta.py:321-382`) applied by `_apply_slice_meta_values` (`trackerkit/gui/panels/detection_panel.py:2643-2765`). Builder = `build_engine_params` (`engine_params.py`). Config = `_slice_config_from_params` / `build_obb_only_config` (`core/inference/config.py:628-680, 921-954`).

| Knob | Produced | In sidecar `settings` | Read by TrackerKit GUI | Reaches InferenceConfig (GUI) | Reaches InferenceConfig (CLI) |
|---|---|---|---|---|---|
| `enabled` (sliced vs full-frame) | yes :484 | yes | `chk_slice_enabled.setChecked` :2696 | `SLICE_ENABLED` from cfg `slice_enabled` :915 -> `SliceConfig.enabled` :650 | **yes** (persisted per-video cfg, orch config.py:1697) |
| `geometry_mode` | yes :485 | yes | `combo_slice_geometry` :2697-2699 | `SLICE_GEOMETRY_MODE` from cfg :916-918 | **yes** (cfg :1698) |
| `overlap` | yes :488 | yes | `advanced["slice_overlap"]` + spin :2701,2745 | `SLICE_OVERLAP` from **advanced only** :919 | **no** -- default 0.2 |
| `object_tile_fraction` | yes :489 | yes | `advanced["slice_object_tile_fraction"]` :2702 | `SLICE_OBJECT_TILE_FRACTION` advanced only :922 | **no** -- default 0.15 |
| `slice_width` / `slice_height` | yes :486-487 | yes | `advanced["slice_width/height"]` :2705-2706 | `SLICE_WIDTH/HEIGHT` advanced only :920-921 | **no** -- default 0 |
| `trained_body_px` | yes :490-492 (from training geometry, not per-point) | yes | `advanced["slice_trained_body_px"]` :2704 | `SLICE_TRAINED_BODY_PX` advanced only :923 -> `reference_body_px` (config.py:921-935) | **no** -- 0.0 -> falls back to `REFERENCE_BODY_SIZE*RESIZE_FACTOR` |
| `confidence_threshold` | yes :493 | yes | `spin_yolo_confidence.setValue` :2740-2743 (only when profile claims one) | `YOLO_CONFIDENCE_THRESHOLD` from cfg `yolo_confidence_threshold` :1019 | **yes** (cfg :1730) |
| `merge_policy` / `merge_metric` / `merge_threshold` / `merge_backend` | yes :494-497 | yes | `advanced["slice_merge_*"]` :2723-2732, reset to `SLICE_MERGE_DEFAULTS` when unclaimed | `SLICE_MERGE_*` advanced only :924-935 | **no** -- defaults greedy_nmm/ios/0.5/cv2 |
| `runtime_tier` | measured only: `measurement.runtime` :521 | **not in `settings`** | not read | -- | -- (by design: wizard tooltip :215-217 "the tier the project actually tracks with") |
| `max_detections` (sweep cap) | measurement only :523 | not in settings | not read | -- (deliberate: MAX_TARGETS is the tracking slot count, comment :498-506) | -- |
| `perform_standard_pred` | **not swept / not produced** | no | -- | advanced only :936 (default False) | no |
| profile id / name (provenance) | -- | `profiles[].id/name`, `primary_profile_id` | `advanced["slice_profile_id"]` :2707-2711 | not in params (deliberately not part of cache key; tests/test_profile_cache_keys.py) | per-video cfg carries `slice_profile_id` + `slice_profile_settings` (orch config.py:1699-1702) but builder ignores both |

Note `enabled=false` profiles ("Full frame" candidates from `build_candidate_grid`, `direct_calibration_grid.py:72-80`) still carry a confidence + merge settings and apply correctly on the GUI path.

## Findings (ranked)

### F1 -- Critical: headless/CLI ignores calibration profiles entirely
- `trackerkit/cli.py:63-69` -> `cli_config.load_tracker_cli_session` (`cli_config.py:198-216`) never passes `advanced_config`; `build_engine_params` then calls `_default_advanced_config_fallback` (`engine_params.py:349-360, 380`) which loads the machine-global `advanced_config.json`.
- That file is never written back by the GUI: `_save_advanced_config` (`gui/orchestrators/config.py:2556`) has no caller except its own `main_window.py:3218-3220` wrapper, which nothing calls. Profile values written into the live `advanced_config` dict (`detection_panel.py:2700-2732`) exist only in memory.
- `build_engine_params` contains no reference to `slice_profile_id` or `slice_profile_settings` (grep of `engine_params.py` for "profile": only the comment at :38-40).
- **Failure scenario:** user picks profile "High recall" (overlap 0.35, fraction 0.10, merge nmm/iou/0.6) in the GUI, runs tracking (auto-saves `<video>_config.json`, `gui/orchestrators/tracking.py:331`), then batch-runs the same config on mehek via `trackerkit track`. The CLI honors `slice_enabled=True` and the geometry *mode* but tiles at overlap 0.2 / fraction 0.15 against `REFERENCE_BODY_SIZE*RESIZE_FACTOR` instead of `trained_body_px`, and merges with greedy_nmm/ios/0.5. Detection output differs from the GUI run with no warning; the config file *looks* like it names the profile.
- Violates the CLAUDE.md convention that GUI and CLI share one param builder with identical output. The golden fixture (`tests/data/get_parameters_dict_golden/fly_obb.json`) only covers the default-SAHI case so this never trips the characterization test.

### F2 -- Important: confidence edits do not mark the profile "Custom"
- `spin_yolo_confidence` has no `valueChanged`/`editingFinished` connection anywhere in trackerkit (`detection_panel.py:1057-1081`; no other references). Every other profile-owned widget routes through `_mark_slice_profile_custom` (`detection_panel.py:874-879, 2298, 2316`).
- **Scenario:** select "Balanced" (conf 0.30), nudge the YOLO threshold to 0.15, run. Status label still reads "Balanced"; saved config records `slice_profile_id=<balanced-id>` with a snapshot whose `confidence_threshold=0.15` -- provenance claims a measured operating point the run did not use.

### F3 -- Important: publish drops the sidecar when no `slice_geometry` is passed
- `training/model_publish.py:874-878`: the sidecar is read from the source and merged **only if `slice_geometry` is truthy**. Otherwise no `.slice_meta.json` is written beside `dst`, so any profiles on the source artifact are lost (only the classifier `.v2meta.json` is copied, :863-870).
- **Scenario:** a user calibrates a project-export `.pt` (spec allows "imported or registered" models; `detectkit/gui/main_window.py:855-871` accepts any `*.pt`) then registers it through a non-sliced publish -> registry model has no profiles. DetectKit's own project copy path does copy sidecars (`detectkit/gui/project.py:193-207, 282, 604`), so this is publish-specific.

### F4 -- Important: TrackerKit "Add model" import copies only the weights
- `gui/orchestrators/config.py:3688` `shutil.copy2(src, dest_path)` -- no `.slice_meta.json` / `.canonical_meta.json` copied. A model shared between labs as `.pt + sidecar` and imported via TrackerKit's browse dialog silently arrives without profiles (and without canonical geometry).

### F5 -- Minor: trigger points and clobbering (mostly correct)
- Sidecar read on every direct-model combo change, incl. programmatic (`detection_panel.py:680-686, 2151`; `main_window.py:2478-2492`), and once per config/preset restore. Not re-read while the model stays selected: a profile saved in DetectKit while TrackerKit is open is invisible until the model is re-selected. No file-watch; no "reload profiles" button.
- User edits are **not** clobbered by a restore (`_restoring_config` guard, `detection_panel.py:2613-2615, 2721-2732`) and a *user* model switch deliberately resets profile id + merge keys (`:2788-2817`). Behaviour is correct; the only silent case is F2.
- Restore ladder is 3-rung (id -> primary -> saved snapshot) with a visible status line (`slice_profile_status_text` :2827-2884). Good.

### F6 -- Minor: staleness check is label-only and expensive
- `profile_evidence_state` (`slice_meta.py:226-248`) SHA-256s the whole checkpoint every time `slice_profile_status_text` runs -- i.e. every combo change and every spin edit (`_update_slice_profile_status_label` :2886). Tens of MB hashed on each keystroke; no mtime/size cache. The mismatch is informational only (spec-conformant: "non-fatal ... explains the fallback"), it never blocks a run nor is it recorded in the saved config.

### F7 -- Minor: exported ONNX/TensorRT artifacts
- Runtime artifacts are derived from the `.pt` path (`core/inference/runtime_artifacts.py:297-322`), and TrackerKit reads the sidecar at the resolved `.pt` path (`main_window.py:2488`), so `gpu_fast` works when a `.pt` is selected. Whether TrackerKit combos can list a bare `.onnx/.engine` as the selected model is **unverified** (listing filter not located); `_load_direct_executor` accepts them (`runtime_artifacts.py:637`), and if selected `read_slice_meta("x.onnx")` would look for `x.onnx.slice_meta.json`, which nothing writes.

### F8 -- Minor: sequential mode gets nothing
- `apply_slice_meta_for_model` runs only for direct roles (`main_window.py:2466, 2481-2492`). `YOLO_SEQ_STAGE1_SLICE_*` exists only in `core/inference/config.py:831-850`; `engine_params.py` never emits it. In scope per spec ("profiles apply only to direct-detector TrackerKit inference"), noted for completeness.

### F9 -- Minor: v1 sidecars and missing sidecars
- Handled: `training_geometry()` promotes flat payloads (`slice_meta.py:50-53`), tested in `tests/test_slice_meta_read.py`; missing/corrupt -> `None` -> combo hidden and profile state cleared (`detection_panel.py:2796-2802`). `normalized_slice_meta` never invents a primary (:66-73). No issue.

### F10 -- Layering: clean on the handoff path
- `core/inference/slice_meta.py`, `direct_calibration_grid.py`, `direct_calibration_sweep.py` import only stdlib/numpy/core. `training/model_publish.py` imports core (allowed). `detection_panel.py` imports core (allowed). The results dialog imports a private `_registry_key_for_model` from training (`direct_calibration_results.py:700-704`) -- coupling, not a direction violation. Two pre-existing detectkit -> trackerkit imports exist outside this feature (`detectkit/gui/evaluation.py:151`, `detectkit/gui/dialogs/training_dialog.py:1701`), which violates "kits must not import each other".

## Recommended improvements

### Wiring gaps to close (in order)

1. **Make `build_engine_params` profile-aware (fixes F1; ~half a day + gate).** In `engine_params.py` before the `SLICE_*` block: overlay `cfg["slice_profile_settings"]` (already saved by the GUI, survives profile deletion) onto the `advanced` slice keys, mapped through the same `slice_meta_values_from_settings` clamps; optionally as rung 2, resolve `cfg["slice_profile_id"]` against `read_slice_meta(resolve_model_path(direct_model))` so a config that names a profile but lacks a snapshot still works. Must keep `_slice_config_hash` semantics (tests/test_profile_cache_keys.py) and regenerate the characterization goldens only if their default-SAHI output changes (it should not). Run MPS + CUDA equivalence: SAHI is off in every fixture so it should be byte-identical.
2. **Hook `spin_yolo_confidence.valueChanged` -> `_mark_slice_profile_custom` (fixes F2; 15 min).** Also add `yolo_confidence` to the "resetting" contract comment at `detection_panel.py:2733-2739` so the asymmetry (confidence is restored only when claimed, but edits must still un-claim) is explicit.
3. **Always carry the sidecar on publish and import (fixes F3, F4; ~1 h).** `model_publish.py:874`: when `slice_geometry` is falsy but `read_slice_meta(src)` exists, copy/normalize it to `dst` and still stamp `slice_profiles` in the registry. `gui/orchestrators/config.py:3688`: reuse `detectkit/gui/project.py:_copy_model_metadata_sidecars` logic (move it to `core/inference/model_paths.py` or `training/model_publish.py` so both kits share it without cross-importing).
4. **Cache the checkpoint fingerprint per (path, size, mtime) (F6; 30 min)** in `slice_meta.py` or in the panel.
5. **Record the applied profile in run provenance (F5/F2 follow-up).** The per-video config already carries `slice_profile_id` + snapshot; also emit `SLICE_PROFILE_ID`/`SLICE_PROFILE_NAME` into the params dict (excluded from the cache key) so the headless log / `<stem>_logs/` profile summary states which operating point ran, and so the CLI can warn when a named profile is absent from the model's sidecar.

### New capability (what the user is asking for beyond wiring)

6. **Runtime tier as a profile setting.** Today `runtime` is measurement provenance only. If profiles should carry a preferred tier, add an optional `settings.runtime_tier`, have `slice_meta_to_panel_values` surface it, and let `_apply_slice_meta_values` set `_selected_runtime_tier` -- with the same "applied vs measured" honesty rule (the wizard measured on one tier; applying another is fine but the s/frame numbers no longer apply).
7. **Profile-aware model selector.** The registry already stores `slice_profiles` (`model_publish.py:934`); show "(3 profiles)" in the direct-model combo label and a tooltip listing names, so users know a calibration exists before selecting.
8. **Live refresh.** A "Reload profiles" action (or mtime check on focus) so a profile saved in DetectKit while TrackerKit is open appears without re-selecting the model.
9. **Headless profile selection flag.** Once (1) lands, a `--sahi-profile <name|id>` override on `trackerkit track` gives batch users the "selectable" half of the request without editing JSON.
10. (Optional) Sequential stage-1 profile application if sequential SAHI is ever exposed in the GUI (F8).
