# Shared Engine Param-Builder — Total GUI/CLI Unification Design

**Date:** 2026-08-05
**Status:** Approved (design forks confirmed 2026-08-05)
**Predecessor:** `2026-08-04-gui-post-tracking-cutover-design.md` (Slice 5), which named this as the sole remaining follow-up of the Qt-free headless program. The whole program (Slices 1–5 + cancellation fixes, branch `feat/headless-qt-free`, tip `65af12ac`) is gated byte-identical on MPS + CUDA and ready to merge; **this work executes after that merge.**

## Goal

Make the TrackerKit GUI and CLI produce **byte-identical tracking output** by construction, by collapsing the two parallel "engine params" derivations into a **single shared, Qt-free `build_engine_params(config, *, runtime_context)`** that both callers use. After this, GUI/CLI param drift is structurally impossible: the only per-caller difference is a small, output-neutral runtime overlay and a handful of GUI-only *display* keys.

## Background: the two derivations (from the 2026-08-05 structural map)

The tracking engine (`core/tracking/worker.py::set_parameters`) consumes a **flat, UPPER_SNAKE, unit-converted** params dict (pixels/frames, not multipliers/seconds; one nested `ADVANCED_CONFIG`, plus array values `ROI_MASK`/`TRAJECTORY_COLORS`). Two code paths build that dict today:

| | Reads from | Emits | Location | ~keys |
|---|---|---|---|---|
| `get_parameters_dict()` | **live Qt widgets** (~90% of ~580 lines are `self._panels.<panel>.<widget>.value()/.text()/.isChecked()/.currentText()`) | UPPER_SNAKE engine params | `trackerkit/gui/orchestrators/config.py:1948` | ~259 |
| `build_tracking_parameters(cfg, *, video_probe, advanced_config)` | a **serialized config dict** (snake_case JSON, the on-disk per-video config) | UPPER_SNAKE engine params | `trackerkit/cli_config.py:315` | ~195 |

Critically:

- `get_parameters_dict()` and `build_config_dict()` (`config.py:1412`, the snake_case JSON serializer) are **independent parallel scrapes of the same widgets** — there is no config→params flow inside the GUI. `build_config_dict()` emits the persisted config; `get_parameters_dict()` re-derives engine params directly from widgets.
- The CLI's `cfg` argument **is** `build_config_dict()`'s output shape (same snake_case keys, same flat nesting). So **`build_tracking_parameters` is already the pure `config → params` function we want** — it consumes a plain dict, is Qt-free, and its comments already state it mirrors the GUI derivation line-by-line. It is the better-architected side.

### Where GUI and CLI output can actually diverge today

The ~64-key gap between the two dicts falls into four buckets:

1. **Four true params-only widget leaks** — read by `get_parameters_dict()`, **not persisted** by `build_config_dict()`, **not emitted** by the CLI. These are live divergence today:
   - `MAX_BRIDGE_GAP_FRAMES` ← `postprocess.spin_max_bridge_gap_frames.value()` (config.py:2238)
   - `FRAGMENT_SPATIAL_VETO_THRESHOLD` ← `postprocess.spin_fragment_spatial_veto_threshold.value()` (2239)
   - `COLOR_TAG_MODEL_PATH` ← `identity.line_color_tag_model.text()` (2390)
   - `COLOR_TAG_CONFIDENCE` ← `identity.spin_color_tag_conf.value()` (2391)
2. **~50 inert-but-divergent keys** — `DATASET_*`, `SLICE_*`, `FINAL_MEDIA_EXPORT_*`, `INDIVIDUAL_*`, `YOLO_OBB_SEG_*`/`YOLO_OBB_DIRECT_TASK`/`YOLO_OBB_FIXED_ANGLE_DEG`, `SUPPRESS_FOREIGN_OBB_*`, `METRIC_*`, `ENABLE_PROFILING`, `COLOR_TAG_*` (above), legacy singular `CNN_CLASSIFIER_*`, `IDENTITY_METHOD` edge, and the two pose Minors. Proven inert for the 7 gate clips by the Slice-4 runtime param-diff, but real drift hazards.
3. **~13 genuinely GUI-only display keys** — `SHOW_FG/BG/CIRCLES/ORIENTATION/YOLO_OBB/TRAJECTORIES/LABELS/STATE/KALMAN_UNCERTAINTY` (2323–2331), `zoom_factor` (2335), `VISUALIZATION_FREE_MODE`/`TRACKING_REALTIME_MODE`/`TRACKING_WORKFLOW_MODE` (2332–2334), `TRAJECTORY_COLORS` (2322, display palette). The CLI hardcodes all to `False`/neutral (cli_config.py:1000–1012). These legitimately stay GUI-only.
4. **Per-run runtime/session values** — `ROI_MASK` (GUI uses live `self._mw.roi_mask`; CLI rasterizes from `roi_shapes` via `_build_roi_mask`, cli_config.py:271), `INDIVIDUAL_PROPERTIES_CACHE_PATH` (`current_individual_properties_cache_path`), `INDIVIDUAL_DATASET_RUN_ID` (`_individual_dataset_run_id`), and output dirs (`DATASET_OUTPUT_DIR`/`FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR`/`INDIVIDUAL_DATASET_*`, derived from `current_video_path`). These legitimately differ per caller and are output-neutral for tracking CSV content (gate-proven).

## Approved design decisions

- **Direction:** *Flip.* The CLI's `build_tracking_parameters` becomes the canonical shared `build_engine_params`; `get_parameters_dict()` becomes a thin wrapper. (Not: extract fresh from the widget-scraping GUI code.)
- **Location:** a **trackerkit-level Qt-free module** (`trackerkit/engine_params.py`). Both `cli_config.py` and `gui/orchestrators/config.py` are in the trackerkit app layer, so both import it with no layering violation, and it stays Qt-free so the CLI stays Qt-free. (Not `core/`: the derivation is trackerkit-specific param semantics, and keeping it in the app layer avoids dragging trackerkit helpers down into core.)

## Target architecture

```
build_config_dict()  ──(snake_case config dict, Qt-free)──┐
                                                          ├──►  build_engine_params(config, *, runtime_context)  ──►  UPPER_SNAKE params
CLI: load_tracker_cli_config() ──(same snake_case dict)───┘         (trackerkit/engine_params.py, Qt-free)

GUI get_parameters_dict()  =  build_engine_params(build_config_dict(), runtime_context=gui_ctx)  +  {GUI-only display keys}
CLI build_tracking_parameters()  =  build_engine_params(cfg, runtime_context=cli_ctx)   (thin shim / renamed callsite)
```

### `build_engine_params` contract

```python
def build_engine_params(
    config: Mapping[str, Any],           # the snake_case config dict (build_config_dict output / on-disk config)
    *,
    runtime: RuntimeContext,             # per-caller, output-neutral overlay (below)
    advanced_config: Mapping[str, Any] | None = None,  # defaults to load_advanced_tracker_config()
) -> dict[str, Any]:                     # flat UPPER_SNAKE engine params
```

- **Pure & Qt-free:** no widget reads, no imports from `gui/`. Deterministic function of its three inputs.
- Emits the **full engine key set** derivable from config (all of buckets 1 & 2), so both callers get every tracking-relevant key identically.
- **Does not** emit the ~13 GUI-only display keys (bucket 3) — the GUI wrapper overlays those.

### `RuntimeContext` (bucket 4 — the output-neutral overlay)

A small dataclass carrying the per-run values each caller supplies:

```python
@dataclass
class RuntimeContext:
    fps: float
    total_frames: int
    frame_width: int
    frame_height: int
    roi_mask: np.ndarray | None            # rasterized via the SHARED _build_roi_mask(roi_shapes, ...)
    dataset_output_dir: str | None
    final_media_video_output_dir: str | None
    individual_dataset_output_dir: str | None
    individual_dataset_name: str | None
    individual_dataset_run_id: str | None
    individual_properties_cache_path: str | None
```

- GUI builds it from `self._mw.*` runtime attributes; CLI builds it from `video_probe` + its output-path logic (exactly as `build_tracking_parameters` does today).
- **ROI is unified:** both callers rasterize from `config["roi_shapes"]` via the shared `_build_roi_mask` (moved to `engine_params.py`). The GUI stops passing its live `self._mw.roi_mask`. This removes the last place ROI could differ between paths. (Verification must confirm the GUI's live mask equals the shared rasterization for the gate clips — expected identical since the live mask is itself rasterized from the same shapes.)

## Closing the leaks (bucket 1) — "move derived values into the shared path"

For each of the four leaked fields, add a persisted config field so the value flows GUI-widget → `build_config_dict` → `build_engine_params`, and CLI-config → `build_engine_params`:

| Engine key | New config field (snake_case) | build_config_dict source | CLI read |
|---|---|---|---|
| `MAX_BRIDGE_GAP_FRAMES` | `max_bridge_gap_frames` | `postprocess.spin_max_bridge_gap_frames.value()` | `_cfg_get(cfg, "max_bridge_gap_frames", default=<same default as widget>)` |
| `FRAGMENT_SPATIAL_VETO_THRESHOLD` | `fragment_spatial_veto_threshold` | `postprocess.spin_fragment_spatial_veto_threshold.value()` | `_cfg_get(...)` |
| `COLOR_TAG_MODEL_PATH` | `color_tag_model_path` | `identity.line_color_tag_model.text()` | `_cfg_get(...)` |
| `COLOR_TAG_CONFIDENCE` | `color_tag_confidence` | `identity.spin_color_tag_conf.value()` | `_cfg_get(...)` |

Also add the corresponding **config-load** wiring in the GUI (`_load_config_postprocessing` / `_load_config_individual_analysis` or the matching loader) so a saved config round-trips these fields back into the widgets. Defaults must exactly match today's widget defaults so existing configs (which lack these fields) reproduce today's behavior.

## The two pose Minors (fold into this work)

- `POSE_SLEAP_BATCH` in the CLI reads its own `cfg["pose_sleap_batch"]` before falling back to `pose_batch_size`; the GUI always emits the single `POSE_BATCH_SIZE` value. In `build_engine_params`, emit `POSE_SLEAP_BATCH` from the same source as `POSE_BATCH_SIZE`.
- `_coerce_pose_keypoint_tokens` (cli_config.py:157) must emit **plain strings** matching the GUI's `_selected_pose_group_keypoints` (main_window.py:1159), so `POSE_IGNORE_KEYPOINTS`/`POSE_DIRECTION_*_KEYPOINTS` are identical token lists.

## Verification (three layers)

1. **Params-equality test (primary GUI gate — fast, exhaustive, no tracking run).** For each of the 7 gate clips' saved configs: instantiate an offscreen `MainWindow`, load the config, call `get_parameters_dict()`, and assert it equals `build_engine_params(build_config_dict(), runtime=gui_ctx) + gui_overlay` **and** equals the CLI's `build_engine_params(cfg, runtime=cli_ctx)` on every tracking-relevant key (excluding the ~13 GUI-only display keys and the output-neutral runtime paths, which are asserted structurally, not by value). This is the regression oracle: it catches every diverging key without a SLEAP run. Built from the empirical key-diff harness (below).
2. **CLI byte-identity re-gate, MPS + CUDA.** The params path changed, so re-run `tools/equivalence/run_matrix.sh` baseline (pre-slice tip) vs current across all 7 clips, both platforms, discounting the documented mehek cupy vacuous pose-clip failures (MPS is authoritative for pose clips). Acceptance identical to the program's standing gate: pos p99≈0, θ≈0, identical row counts.
3. **Extend the Slice-5 GUI==CLI final-CSV test** (`test_gui_session_cutover_equivalence.py`, currently `fly_obb`) to add **one identity clip** (`ant_cnn_identity`) and **one pose clip** (`ant_pose_headtail`), proving the whole GUI path (widgets → config → shared builder → engine → post-tracking) equals the CLI end-to-end for the hardest cases.

**Parity-safety principle (Slice-4 B/B2/B3 template):** a key inert for a clip must *stay* inert. When lifting a key's derivation into `build_engine_params`, preserve the exact default and gating so the 7 gate clips see identical values (e.g. `identity_weight=0.0` gated by `alpha>0` at `hungarian.py:239`; `ENABLE_POSE_EXTRACTOR` default False at `core/inference/config.py:843`). The empirical key-diff harness is the proof.

### Empirical key-diff harness (drives the worklist and the oracle)

Before extending the builder, dump both dicts for each gate clip's saved config and diff them (the Slice-4 diagnosis method that worked: env-gated param dump at the common choke-point + diff). This produces the authoritative, non-stale list of every diverging key and its category, rather than hardcoding a list that may drift. The harness becomes the params-equality test's fixture.

## Scope / non-goals

- **In scope:** the config→params derivation unification, the 4 leak fields, shared ROI rasterization, the two pose Minors, the three verification layers.
- **Out of scope:** `build_config_dict()` itself stays a GUI method (it *must* read widgets — that's the UI's job to serialize state). We only make it *complete* (the 4 leaks). We do not unify the widget→config layer.
- **Out of scope:** the remaining Slice-5 minor follow-ups tracked in the ledger (dead `PostProcessWorker`, vestigial `run_post_tracking(forward, backward)` params) — orthogonal, handle separately.
- **No `core/` change** is expected (the builder lives in trackerkit). If any `core/` file is touched, the CLI re-gate covers it, but the design does not require it.

## Risks

- **Hidden widget-only derivation.** The map found 4 leaks; the empirical key-diff harness (run against all 7 configs before extending the builder) is the safety net for any fifth. If a diverging key resists config-ization (reads live non-persistable state), it is either a genuine GUI-only display key (bucket 3, overlay it) or a runtime value (bucket 4, `RuntimeContext`) — no key should require widget access inside `build_engine_params`.
- **Default mismatches on old configs.** Every new config field must default to today's widget default so pre-existing saved configs reproduce current behavior. The params-equality test over real saved configs catches this.
- **ROI rasterization drift.** Unifying ROI onto the shared rasterizer is the one place a live-GUI value is replaced by a re-derivation. Verification layer 1 asserts the GUI live mask == shared rasterization for the gate clips before the switch is trusted.
