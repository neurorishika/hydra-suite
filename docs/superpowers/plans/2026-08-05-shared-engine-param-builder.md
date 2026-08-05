# Shared Engine Param-Builder — Total GUI/CLI Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the GUI's `get_parameters_dict()` and the CLI's `build_tracking_parameters()` onto a single shared, Qt-free `build_engine_params(config, *, runtime)` so GUI and CLI tracking output is byte-identical by construction.

**Architecture:** The CLI's `build_tracking_parameters` (already a pure `config→params` function) is promoted to the canonical shared builder in a new Qt-free `trackerkit/engine_params.py`. The GUI's `get_parameters_dict()` is rewritten to `build_config_dict()` → `build_engine_params(cfg, runtime=gui_ctx)` → a thin GUI-only display overlay. Four params-only widget "leaks" are moved into the persisted config so the shared path carries them. Per-run values (ROI mask, cache paths, output dirs, fps/frames) travel in an output-neutral `RuntimeContext` each caller builds.

**Tech Stack:** Python 3, PySide6 (GUI only — the shared builder is Qt-free), pytest with offscreen Qt (`QT_QPA_PLATFORM=offscreen`, no pytest-qt), the `tools/equivalence` byte-identity harness.

## Global Constraints

- **Reference / oracle:** the GUI's current `get_parameters_dict()` output IS the correctness reference (it produced the params the Slice-4/5 gate validated byte-identical). `build_engine_params` MUST reproduce it key-for-key on every non-display key. This is stricter than the tracking gate and is the primary regression oracle.
- **Qt-free:** `trackerkit/engine_params.py` must import nothing from `hydra_suite.trackerkit.gui` and must not touch PySide6. The CLI (`trackerkit/cli_config.py` / `headless_tracking.py`) stays importable with PySide6 blocked — the whole point of the parent program.
- **Parity-safety (Slice-4 B/B2/B3 template):** a key inert for a clip must stay inert. Preserve exact defaults and gating when lifting a key's derivation (e.g. `identity_weight=0.0` gated by `alpha>0` at `hungarian.py:239`; `ENABLE_POSE_EXTRACTOR` default `False` at `core/inference/config.py:843`).
- **Defaults:** every NEW persisted config field must default to today's widget default so pre-existing saved configs (which lack the field) reproduce current behavior exactly.
- **No `core/` change expected.** The builder lives in trackerkit. If a `core/` file is touched, the CLI re-gate (Task 7) must cover it; prefer not to.
- **Commit as the configured git user; NO `Co-Authored-By` trailer.**
- **Test command (from the worktree):** `PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE conda run -n hydra-mps python -m pytest <path> -q --ignore=tests/test_identity_postprocess.py`. `make format` is broken → `conda run -n hydra-mps black <files> && conda run -n hydra-mps isort <files>`.
- **Long/SLEAP tests run FOREGROUND** (blocking Bash, timeout up to 600000 ms) — background `Monitor` polling has a PID-reuse false-completion artifact. Full `pytest tests/` never finishes (classkit modal hang) → batch per file, judge failures as delta vs the pre-task baseline.
- **Line numbers in this plan are anchors from 2026-08-05 source; reconcile against live source before editing** (`get_parameters_dict` and `cli_config.py` will shift as tasks land — grep the symbol, don't trust the number).
- **Gate fixture configs** live under `tools/equivalence/fixtures/` (per-clip). The 7 clips: `fly_obb`, `worm_bgsub`, `emi_obb_identity`, `ant_pose_headtail`, `ant_obb_sleap`, `ant_obb_sequential`, `ant_cnn_identity`. Fixture clips/models are gitignored — symlink main's `tools/equivalence/fixtures/clips` into the worktree for tracking runs, remove after, never `git add`.

---

### Task 1: Extract the shared Qt-free `build_engine_params` + `RuntimeContext`

Move the CLI's already-pure derivation into a new shared module and make the CLI a thin shim over it. Params output must be **byte-identical** to today's CLI (pure refactor).

**Files:**
- Create: `src/hydra_suite/trackerkit/engine_params.py`
- Modify: `src/hydra_suite/trackerkit/cli_config.py` (`build_tracking_parameters` becomes a shim; move helpers `_build_roi_mask`, `_coerce_pose_keypoint_tokens`, `_cfg_get`, `_cfg_get_time`, `_coerce_int_list`, `_autopick_greedy`, and the inner `_seconds_to_frames` logic into the new module — keep back-compat imports in `cli_config.py` for anything referenced elsewhere)
- Test: `tests/test_engine_params_extraction.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class RuntimeContext:
      fps: float
      total_frames: int
      frame_width: int
      frame_height: int
      roi_mask: "np.ndarray | None" = None
      dataset_output_dir: str | None = None
      final_media_video_output_dir: str | None = None
      individual_dataset_output_dir: str | None = None
      individual_dataset_name: str | None = None
      individual_dataset_run_id: str | None = None
      individual_properties_cache_path: str | None = None

  def build_engine_params(
      config: Mapping[str, Any],
      *,
      runtime: RuntimeContext,
      advanced_config: Mapping[str, Any] | None = None,
  ) -> dict[str, Any]: ...

  def build_roi_mask(roi_shapes, width, height) -> "np.ndarray | None": ...  # moved from cli_config._build_roi_mask
  ```
- Consumes (from earlier program work): `load_advanced_tracker_config()`, `is_pose_inference_enabled(cfg)`, `ResolvedBackend`, `legacy_detection_runtime_fields`, `_resolve_solver_flags`, `resolve_tensorrt_max_batch_size` — import these into `engine_params.py` (they are already Qt-free).

- [ ] **Step 1: Write the failing test** — assert the shim reproduces the pre-refactor CLI params byte-for-byte for one fixture config.

```python
# tests/test_engine_params_extraction.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from hydra_suite.trackerkit import cli_config
from hydra_suite.trackerkit.engine_params import build_engine_params, RuntimeContext

def test_shim_matches_direct_builder(fly_obb_cfg, fly_obb_probe):
    # fixtures load the saved fly_obb config dict + a TrackerCliVideoProbe
    via_shim = cli_config.build_tracking_parameters(fly_obb_cfg, video_probe=fly_obb_probe)
    rt = RuntimeContext(fps=fly_obb_probe.fps, total_frames=fly_obb_probe.total_frames,
                        frame_width=fly_obb_probe.width, frame_height=fly_obb_probe.height)
    via_direct = build_engine_params(fly_obb_cfg, runtime=rt)
    # every key the shim produces from the same inputs must match the direct call
    for k in via_shim:
        assert via_shim[k] == via_direct.get(k, "__MISSING__"), f"key {k} diverged"
```

- [ ] **Step 2: Run it, expect ImportError/FAIL** (`engine_params` does not exist yet).
- [ ] **Step 3: Create `engine_params.py`** — move the body of `build_tracking_parameters` verbatim into `build_engine_params(config, *, runtime, advanced_config=None)`. Mechanical transform of the input surface:
  - Replace `video_probe.fps` → `runtime.fps`, `.total_frames` → `runtime.total_frames`, `.width/.height` → `runtime.frame_width/frame_height`.
  - Replace the inline `_build_roi_mask(...)` call with `runtime.roi_mask if runtime.roi_mask is not None else build_roi_mask(config.get("roi_shapes"), runtime.frame_width, runtime.frame_height)`.
  - Replace the CLI's output-dir derivations (`DATASET_OUTPUT_DIR`, `FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR`, `INDIVIDUAL_DATASET_*`, `INDIVIDUAL_PROPERTIES_CACHE_PATH`, `INDIVIDUAL_DATASET_RUN_ID`) with the corresponding `runtime.*` fields (fall back to the CLI's current default when the field is `None`, so the shim reproduces today's values).
  - Move `_build_roi_mask`→`build_roi_mask`, `_coerce_pose_keypoint_tokens`, `_cfg_get`, `_cfg_get_time`, `_coerce_int_list`, `_autopick_greedy`, `_seconds_to_frames` into the module.
- [ ] **Step 4: Rewrite `cli_config.build_tracking_parameters` as a shim** — keep its current signature `(cfg, *, video_probe, advanced_config=None)`; build a `RuntimeContext` from `video_probe` plus the CLI's existing output-path logic (the `_default_output_paths`/video-derived dirs it computes today), then `return build_engine_params(cfg, runtime=rt, advanced_config=advanced_config)`. Re-export moved helpers from `cli_config` (`from .engine_params import build_roi_mask as _build_roi_mask`, etc.) if any test/module imports them.
- [ ] **Step 5: Run the test, expect PASS.** Then run the existing CLI param tests: `pytest tests/test_trackerkit_cli*.py tests/test_engine_params_extraction.py -q`. Expected: no new failures vs baseline.
- [ ] **Step 6: Qt-free guard** — `PYTHONPATH=$PWD/src python -c "import sys; sys.modules['PySide6']=None; import hydra_suite.trackerkit.engine_params"` must succeed (no Qt import). Add this as an assertion in the test module.
- [ ] **Step 7: Commit** — `refactor(trackerkit): extract shared Qt-free build_engine_params + RuntimeContext from CLI builder`.

---

### Task 2: Failing params-equality oracle (GUI reference vs shared builder)

Write the exhaustive regression oracle FIRST, RED. It loads each gate clip's saved config into an offscreen `MainWindow`, reads the GUI reference `get_parameters_dict()`, and asserts `build_engine_params(build_config_dict(), runtime=gui_ctx)` matches it on every non-display key. It fails now (the ~54 diverging keys); Tasks 3–6 drive it green.

**Files:**
- Create: `tests/test_gui_cli_param_equivalence.py`
- Test helper: reuse the offscreen-`MainWindow` load pattern from `tests/test_config_build_dict.py` (module-top `os.environ.setdefault("QT_QPA_PLATFORM","offscreen")`, `QApplication` fixture, `MainWindow()` with `_save/_load_advanced_config` stubbed).

**Interfaces:**
- Consumes: `RuntimeContext`, `build_engine_params` (Task 1); `MainWindow`; `orchestrator.build_config_dict()` / `get_parameters_dict()`.
- Produces: `DISPLAY_ONLY_KEYS` and `RUNTIME_OVERLAY_KEYS` constants (the bucket-3 and bucket-4 exclusion sets from the spec), reused by Task 6.

- [ ] **Step 1: Enumerate the exclusion sets** as module constants:

```python
DISPLAY_ONLY_KEYS = {
    "SHOW_FG","SHOW_BG","SHOW_CIRCLES","SHOW_ORIENTATION","SHOW_YOLO_OBB",
    "SHOW_TRAJECTORIES","SHOW_LABELS","SHOW_STATE","SHOW_KALMAN_UNCERTAINTY",
    "zoom_factor","VISUALIZATION_FREE_MODE","TRACKING_REALTIME_MODE",
    "TRACKING_WORKFLOW_MODE","TRAJECTORY_COLORS",
}
RUNTIME_OVERLAY_KEYS = {
    "ROI_MASK","INDIVIDUAL_PROPERTIES_CACHE_PATH","INDIVIDUAL_DATASET_RUN_ID",
    "DATASET_OUTPUT_DIR","FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR",
    "INDIVIDUAL_DATASET_OUTPUT_DIR","INDIVIDUAL_DATASET_NAME",
}
```

- [ ] **Step 2: Write the parametrized failing test.**

```python
import pytest
CLIPS = ["fly_obb","worm_bgsub","emi_obb_identity","ant_pose_headtail",
         "ant_obb_sleap","ant_obb_sequential","ant_cnn_identity"]

@pytest.mark.parametrize("clip", CLIPS)
def test_shared_builder_reproduces_gui_reference(main_window, clip):
    load_gate_config_into(main_window, clip)          # via the config-load path
    orch = main_window._config_orch
    reference = orch.get_parameters_dict()            # GUI reference (widget scrape)
    cfg = orch.build_config_dict()
    rt = gui_runtime_context(main_window)             # from _mw runtime attrs
    shared = build_engine_params(cfg, runtime=rt)
    compared = (set(reference) | set(shared)) - DISPLAY_ONLY_KEYS - RUNTIME_OVERLAY_KEYS
    diffs = {k: (reference.get(k,"∅"), shared.get(k,"∅"))
             for k in compared if reference.get(k,"∅") != shared.get(k,"∅")}
    assert not diffs, f"{clip}: {len(diffs)} keys diverge: {sorted(diffs)}"
```

- [ ] **Step 3: Run it FOREGROUND, capture the diff.** Expected: FAIL on most clips. Save the printed diverging-key list to `.superpowers/sdd/<workspace>/keydiff-baseline.txt` — this is the authoritative worklist for Tasks 3–5 (do NOT trust the spec's hand-listed categories over this empirical output).
- [ ] **Step 4: Commit** — `test(trackerkit): failing GUI-reference vs shared-builder param equality oracle (7 clips)`. (Test is expected-failing; note that in the commit body and mark the assertions `xfail(strict=True)` so CI is green until Task 6 flips them — remove the xfail in Task 6.)

---

### Task 3: Close the four params-only leaks (persist them into config)

Move `MAX_BRIDGE_GAP_FRAMES`, `FRAGMENT_SPATIAL_VETO_THRESHOLD`, `COLOR_TAG_MODEL_PATH`, `COLOR_TAG_CONFIDENCE` from widget-only reads into persisted config fields, so both the GUI config→params path and the CLI carry them.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py` (`build_config_dict` — add the 4 snake_case fields; the matching `_load_config_*` loader — read them back into widgets)
- Modify: `src/hydra_suite/trackerkit/engine_params.py` (emit the 4 UPPER keys from config fields)
- Test: extend `tests/test_config_build_dict.py` (round-trip) + the Task-2 oracle diff shrinks by 4 keys

**Interfaces:**
- Config fields (snake_case) and their engine keys:
  | engine key | config field | GUI widget source | default |
  |---|---|---|---|
  | `MAX_BRIDGE_GAP_FRAMES` | `max_bridge_gap_frames` | `postprocess.spin_max_bridge_gap_frames.value()` | widget's constructed default |
  | `FRAGMENT_SPATIAL_VETO_THRESHOLD` | `fragment_spatial_veto_threshold` | `postprocess.spin_fragment_spatial_veto_threshold.value()` | widget's constructed default |
  | `COLOR_TAG_MODEL_PATH` | `color_tag_model_path` | `identity.line_color_tag_model.text()` | `""` |
  | `COLOR_TAG_CONFIDENCE` | `color_tag_confidence` | `identity.spin_color_tag_conf.value()` | widget's constructed default |

- [ ] **Step 1: Find the exact widget defaults** — grep each widget symbol in `trackerkit/gui/panels/` for its `.setValue(...)`/`.setText(...)`/constructor default; use those exact values as the config-field defaults (Global Constraint: defaults must match today's widgets).
- [ ] **Step 2: Write the failing round-trip test** in `test_config_build_dict.py`: set the 4 widgets to non-default values, `build_config_dict()`, assert the 4 snake_case fields are present with those values; then load that dict back and assert the widgets read them.
- [ ] **Step 3: Run it, expect FAIL** (fields absent from `build_config_dict`).
- [ ] **Step 4: Add the 4 fields** to `build_config_dict` (near the postprocess / individual-analysis serialization blocks) and to the matching `_load_config_*` loader (so a saved config round-trips). Use `_cfg_get(cfg, "<field>", default=<exact widget default>)` on the load side.
- [ ] **Step 5: Emit the 4 engine keys** from `config` in `build_engine_params` (they were previously CLI-absent). Use the same defaults.
- [ ] **Step 6: Run the round-trip test (PASS) + the Task-2 oracle** (FOREGROUND) — the diverging-key count must drop by exactly these 4.
- [ ] **Step 7: Commit** — `feat(trackerkit): persist max_bridge_gap/fragment_veto/color_tag config fields (close param leaks)`.

---

### Task 4: Emit the full inert-key set from config (bucket 2)

Extend `build_engine_params` to emit every remaining tracking-relevant key the GUI reference emits but the CLI dropped (`DATASET_*`, `SLICE_*`, `FINAL_MEDIA_EXPORT_*`, `INDIVIDUAL_*`, `YOLO_OBB_SEG_*`, `SUPPRESS_FOREIGN_OBB_*`, `METRIC_*`, `ENABLE_PROFILING`, legacy singular `CNN_CLASSIFIER_*`, `IDENTITY_METHOD` edge, etc.), derived from config, per parity-safety.

**Files:**
- Modify: `src/hydra_suite/trackerkit/engine_params.py`
- Reference: the GUI `get_parameters_dict()` derivation for each key (config.py:1948+) — lift the derivation, swapping widget reads for the corresponding config field
- Test: the Task-2 oracle (diff → 0 on non-display keys)

- [ ] **Step 1: Work from `keydiff-baseline.txt`** (Task 2) — the empirical remaining diverging keys. For each, open the GUI reference derivation and find the config field it corresponds to (`build_config_dict` emits it, or add it if a fifth leak surfaces — same pattern as Task 3).
- [ ] **Step 2: For each key group, add a failing sub-assertion** (or just rely on the Task-2 oracle narrowing) and implement the derivation in `build_engine_params`, preserving exact defaults/gating (parity-safety). Do the `DATASET_*` block, then `SLICE_*`, then `YOLO_OBB_SEG_*`, then `INDIVIDUAL_*`/`METRIC_*`/misc — one commit-sized group at a time is fine but this task's DoD is the whole bucket.
- [ ] **Step 3: Handle the pose Minors here or in Task 5** — `POSE_SLEAP_BATCH` must come from the same source as `POSE_BATCH_SIZE` (not a separate `pose_sleap_batch` field), and `POSE_IGNORE_KEYPOINTS`/`POSE_DIRECTION_*` token lists must be plain strings (see Task 5).
- [ ] **Step 4: Run the Task-2 oracle FOREGROUND** — on every clip, `diffs` must be empty for all keys except those in `DISPLAY_ONLY_KEYS`/`RUNTIME_OVERLAY_KEYS`. Iterate until zero.
- [ ] **Step 5: Commit** — `feat(trackerkit): shared builder emits full inert-key set from config (kills GUI/CLI drift)`.

---

### Task 5: Pose keypoint-token + SLEAP-batch parity Minors

Make the two pose Minors identical across paths (may already be closed by Task 4 — this task is the dedicated gate for them).

**Files:**
- Modify: `src/hydra_suite/trackerkit/engine_params.py` (`_coerce_pose_keypoint_tokens`, `POSE_SLEAP_BATCH`)
- Test: `tests/test_engine_params_pose_tokens.py`

- [ ] **Step 1: Write the failing test** — given a config with `pose_ignore_keypoints`/`pose_direction_anterior_keypoints` and a `pose_batch_size`, assert `build_engine_params` emits `POSE_IGNORE_KEYPOINTS` as a list of **plain strings** (matching the GUI `_selected_pose_group_keypoints` output — `main_window.py:1159`) and `POSE_SLEAP_BATCH == POSE_BATCH_SIZE`.
- [ ] **Step 2: Run, expect FAIL** if `_coerce_pose_keypoint_tokens` emits int/str tuples or `POSE_SLEAP_BATCH` reads a separate field.
- [ ] **Step 3: Fix** `_coerce_pose_keypoint_tokens` to return plain strings; make `POSE_SLEAP_BATCH` read the `POSE_BATCH_SIZE` source.
- [ ] **Step 4: Run pose-token test (PASS) + a pose-clip slice of the Task-2 oracle** (`ant_pose_headtail`, `ant_obb_sleap`).
- [ ] **Step 5: Commit** — `fix(trackerkit): pose keypoint tokens + POSE_SLEAP_BATCH parity in shared builder`.

---

### Task 6: Rewrite `get_parameters_dict()` as the thin wrapper (the collapse)

Replace the ~800-line widget scrape with `build_config_dict()` → `build_engine_params(cfg, runtime=gui_ctx)` → GUI-only display overlay. Unify ROI onto the shared rasterizer. Flip the Task-2 oracle from xfail to strict-pass.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py` (`get_parameters_dict` body)
- Modify: `tests/test_gui_cli_param_equivalence.py` (remove xfail)
- Test: the full Task-2 oracle, strict, all 7 clips

**Interfaces:**
- Consumes: `build_engine_params`, `RuntimeContext`, `build_roi_mask`, `DISPLAY_ONLY_KEYS`.
- Produces: `_gui_runtime_context(self) -> RuntimeContext` (reads `self._mw.current_video_path`-derived dirs, `current_individual_properties_cache_path`, `_individual_dataset_run_id`, fps/frame dims, and rasterizes ROI from `config["roi_shapes"]` via `build_roi_mask`) and `_gui_display_overlay(self) -> dict` (the ~13 display keys from `SHOW_*` widgets, `slider_zoom`, viz/realtime/workflow-mode state, `TRAJECTORY_COLORS`).

- [ ] **Step 1: Add a characterization test FIRST** — before touching `get_parameters_dict`, snapshot its current full output for `fly_obb` + `ant_cnn_identity` to a fixture, so the rewrite is provably output-preserving on ALL keys (including display + runtime). Assert new output == snapshot.
- [ ] **Step 2: Run it, expect PASS** (characterizing current behavior).
- [ ] **Step 3: Implement `_gui_runtime_context` and `_gui_display_overlay`.** For ROI: build from `config["roi_shapes"]` via `build_roi_mask`; assert (in a sub-test) it equals the current `self._mw.roi_mask` for the gate clips before trusting the switch (spec risk item).
- [ ] **Step 4: Rewrite `get_parameters_dict`:**

```python
def get_parameters_dict(self):
    self._mw._commit_pending_setup_edits()
    config = self.build_config_dict()
    params = build_engine_params(config, runtime=self._gui_runtime_context())
    params.update(self._gui_display_overlay())
    return params
```

- [ ] **Step 5: Run the characterization test** — every key must still match the snapshot (display + runtime + all tracking keys). Fix overlay/runtime until green. This proves the collapse is behavior-preserving for the live GUI.
- [ ] **Step 6: Remove the xfail** in `test_gui_cli_param_equivalence.py`; run it strict FOREGROUND on all 7 clips — must pass (now trivially, since `get_parameters_dict` calls the shared builder).
- [ ] **Step 7: Delete the now-dead widget-scrape helpers** in `config.py` orphaned by the rewrite (grep `src/` + `tests/` for each before deleting).
- [ ] **Step 8: Commit** — `refactor(trackerkit): get_parameters_dict = build_config_dict + shared build_engine_params + display overlay`.

---

### Task 7: Full verification — re-gate + end-to-end GUI==CLI

Prove byte-identity of the changed params path on both platforms, and that the whole GUI path equals the CLI end-to-end for the hardest clips.

**Files:**
- Modify: `tests/test_gui_session_cutover_equivalence.py` (extend from `fly_obb` to add `ant_cnn_identity` + `ant_pose_headtail`)

- [ ] **Step 1: Extend the Slice-5 GUI==CLI final-CSV test** to `ant_cnn_identity` (identity) and `ant_pose_headtail` (pose), running the SessionWorker synchronously (monkeypatch `.start`→`.execute`, per the Slice-5 test). Requires conda `hydra-mps` + `sleap` env + symlinked fixture clips. Run FOREGROUND.
- [ ] **Step 2: CLI byte-identity re-gate, MPS.** Baseline worktree at the pre-slice tip (record it: `git rev-parse HEAD` before Task 1) vs current, all 7 clips: `REPO=$PWD WT=$PWD MAIN_SRC=<baseline>/src WT_SRC=$PWD/src OUT=/tmp/equiv_paramunify RUNTIME=mps bash tools/equivalence/run_matrix.sh`. conda MUST be active; verify `wc -l` on CSVs > 1. Acceptance: pos p99≈0, θ≈0, identical row counts, all EQUIVALENT (perf flags are noise).
- [ ] **Step 3: CLI byte-identity re-gate, CUDA (mehek).** Per the program's gate how-to; discount the documented cupy-vacuous pose-clip failures (MPS authoritative for pose). Acceptance identical.
- [ ] **Step 4: Confirm no `core/` change** — `git diff <baseline>..HEAD --stat -- src/hydra_suite/core/` empty (or, if non-empty, the re-gate above covers it — state which).
- [ ] **Step 5: Update memory** `project_headless_qt_free_program.md` — mark the shared param-builder DONE, note the oracle test as the durable anti-drift guard, and record both-platform re-gate results.
- [ ] **Step 6: Commit** — `test(trackerkit): GUI==CLI end-to-end equivalence for identity + pose clips`.

---

## Self-review notes (author)

- **Spec coverage:** builder extraction (T1), 4 leaks (T3), full inert set (T2 oracle + T4), pose Minors (T5), GUI collapse + ROI unification (T6), three verification layers (T2 oracle + T6 characterization = layer 1; T7 step 2/3 = layer 2; T7 step 1 = layer 3). All spec sections mapped.
- **Oracle direction:** the plan pins the GUI reference `get_parameters_dict()` as the correctness oracle (Global Constraints) — stricter than the tracking gate, so it forces total unification, not just gate-clip inertness. This is the key insight that makes "identical output" provable without a tracking run per key.
- **Empirical worklist:** Task 2 generates `keydiff-baseline.txt` so Task 4 works from live source, not this plan's possibly-stale category list.
- **Risk of a fifth leak:** handled — Task 4 Step 1 adds any newly-surfaced widget-only key via the Task-3 pattern.
