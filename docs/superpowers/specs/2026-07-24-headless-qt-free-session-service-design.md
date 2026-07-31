# Qt-Free Headless Session Service — Design

**Date:** 2026-07-24
**Status:** Approved for planning
**Goal:** Make the headless TrackerKit CLI path fully Qt-free by extracting the post-tracking pipeline out of the GUI orchestrators into a Qt-free `TrackingSessionCore` in `core/`, consumed by both the GUI and the CLI, so the hidden-`MainWindow` bridge can be deleted.

---

## Problem

`trackerkit track` has two execution paths, chosen by `TrackerCliSession.supports_direct_run()`
(`src/hydra_suite/trackerkit/cli_config.py:103`):

- **Direct path** (`trackerkit/headless_tracking.py`) — taken when pose extraction is off *and*
  identity is `none`/`none_disabled`. Uses only `QCoreApplication` + `QThread`/`Signal`.
  `QCoreApplication` never loads a QPA platform plugin, so this path already needs no display.
- **Bridge path** (`trackerkit/cli.py:184-195`) — taken for **every** pose or identity session.
  Constructs a real hidden `MainWindow` and drives it, which requires a `QApplication`, a
  platform plugin, and therefore `QT_QPA_PLATFORM=offscreen`.

Verified empirically: running `trackerkit track tools/equivalence/fixtures/clips/fly_obb.mp4`
with `QT_QPA_PLATFORM=definitely_not_a_platform` completes detection, forward and backward
tracking, and both post-process passes without error. If a platform plugin were being loaded,
Qt would abort at startup. The offscreen requirement therefore comes from the bridge alone.

The bridge exists because the post-tracking pipeline lives in the GUI orchestrators and reads
its configuration directly off widgets. For example `_show_session_summary`
(`trackerkit/gui/orchestrators/tracking.py:4586`) reads
`self._panels.postprocess.enable_postprocessing.isChecked()` and
`self._panels.setup.file_line.text()`. There is no Qt-free way to ask those questions today.

Scale of the coupling:

| File | Lines | `self._mw.` refs | `_panels.` refs | `QMessageBox` |
|---|---|---|---|---|
| `gui/orchestrators/tracking.py` | 4716 | 514 | 93 | 34 |
| `gui/orchestrators/session.py` | 2353 | 565 | 91 | 14 |
| `gui/orchestrators/config.py` | 4384 | 186 | 777 | 46 |

### Motivation

1. **Cloud/container deploy.** `docs/superpowers/specs/2026-07-23-cloud-gpu-inference-design.md:123`
   currently assumes `QT_QPA_PLATFORM=offscreen` inside the container. That assumption should not exist.
2. **Cluster/batch jobs.** Qt is awkward and heavyweight on HPC nodes.
3. **GUI/CLI drift.** Two code paths for the same pipeline already diverge — see the
   `TRAJECTORY_COLORS` defect below.
4. **Architecture debt.** A dependency-direction violation left over from the migration.

### Success criterion

Mirror the no-Qt-in-core principle already established by the `TrackingEngineCore` split
(`docs/superpowers/plans/done/2026-07-19-trackingworker-qt-in-core-split.md`): a Qt-free core
class plus a thin Qt wrapper, with both the GUI and the CLI consuming the same core. The
executable definition of done is a test that runs `trackerkit track` on a real clip with
`PySide6` blocked from import and completes successfully.

---

## Defect found during design

The GUI and CLI generate track overlay colors with different random generators, producing
different colors for the same config:

- GUI, `gui/orchestrators/config.py:1912`: `np.random.seed(42)` then `np.random.randint(0,255,(N,3))`
  → `(102,179,92), (14,106,71), (188,20,102), …`
- CLI, `cli_config.py:485`: `np.random.default_rng(42).integers(0,255,size=(N,3))`
  → `(22,197,166), (111,110,218), (21,177,51), …`

Verified divergent. `TRAJECTORY_COLORS` is used only for annotated-video overlays, so tracking
CSVs are unaffected — which is why the equivalence harness, which compares CSVs, never caught it.

**Decision:** the GUI's legacy `np.random.seed(42)` + `randint` form becomes the single shared
implementation, moved into the params builder used by both paths. Rationale: GUI-rendered videos
are the output users have existing baselines for; CLI-rendered video colors change, and nobody has
a baseline for those. Resolved in Slice 1.

---

## Architecture

One new Qt-free unit, `TrackingSessionCore`, owns everything from "tracking worker finished" to
"session complete". It is constructed with the config dicts, output paths, and a callbacks bundle;
it runs the post-tracking stages inline and returns a `SessionResult`. It never imports Qt and
never touches a widget.

```
                    config dicts (shared vocabulary)
                       ↙                      ↘
   GUI: widgets → build_config_dict()   CLI: cli_config.py → TrackerCliSession
                       ↘                      ↙
                  TrackingSessionCore  (core/tracking/session.py, Qt-free)
                  ├── merge / post-process
                  ├── pose source merge + quality post-pass
                  ├── identity post-pass
                  ├── interpolated crops
                  ├── rich export CSV
                  ├── final media export
                  └── dataset generation
                       ↙                      ↘
   GUI: orchestrator renders callbacks    CLI: headless_tracking logs/prints
```

### The config seam

The seam is the pair of dicts both paths **already** build. This is not a new representation:

1. **The config JSON dict** — lowercase keys (`enable_pose_extractor`, `identity_method`,
   `video_output_enabled`, `interpolation_method`, …). This is exactly the vocabulary in
   `tools/equivalence/fixtures/configs/*.json`. It is what the policy predicates read.
   - GUI builds it in `save_config()` (`gui/orchestrators/config.py:1401`).
   - CLI reads it in `load_tracker_cli_config()` / `load_tracker_cli_session()`.
2. **The params dict** — uppercase keys (`FPS`, `RESIZE_FACTOR`, `END_FRAME`,
   `TRAJECTORY_COLORS`, …) consumed by `TrackingEngineCore`.
   - GUI builds it in `get_parameters_dict()` (`gui/orchestrators/config.py:1906`).
   - CLI builds it as `TrackerCliSession.params`.

Because both paths already produce both dicts, and the GUI's save/load round-trip already proves
the config dict captures widget state, no new schema is introduced. Converting the seam to a typed
dataclass is deferred to Simplification Sprint Slice 2, once every consumer goes through one entry point.

`save_config()` currently interleaves widget reads with save-path prompting and an atomic write.
Slice 1 extracts its pure widget→dict body as `build_config_dict()`; `save_config()` becomes a
caller of it. This is a prerequisite, not an optional cleanup — without it the GUI has no non-interactive
way to hand the service a config.

### Placement

`src/hydra_suite/core/tracking/session.py`, alongside `worker.py`'s `TrackingEngineCore`.
Consistent with existing practice: `core/` already imports `data/` and `training/`
(`TagObservationCache`, `load_torchvision_classifier`, `load_tiny_classifier`), which is what the
export and dataset stages need. `core/` imports nothing from any app layer today and must not
start.

Self-contained stage functions land in `core/post/` and `core/identity/`, next to the code they
already call.

### Prerequisite move

`trackerkit/gui/model_utils.py` is stdlib-only (`json`, `logging`, `os`, `shutil`) and is already
imported by `cli_config.py:22` — an app-layer GUI import from a CLI module. It moves to
`core/inference/model_paths.py` in Slice 1, removing the existing violation and letting the
service resolve model paths without reaching upward.

### What stays in the GUI

Preview rendering, live frame display, stats panels, progress bars, `QMessageBox`, the RefineKit
prompt, widget enable/disable — concretely `start_full`, `stop_tracking`, `_request_qthread_stop`,
`on_progress_update`, `on_tracking_warning`, `show_gpu_info`, `clear_detection_caches`,
`on_stats_update`, `on_new_frame`, `_finalize_tracking_session_ui`, `_show_session_summary`'s
dialog half, `start_tracking*`, and the cache-id helpers that read widgets. The orchestrator
becomes a consumer: commit pending edits → `build_config_dict()` → construct the service → run it
on a `BaseWorker` → render callbacks.

The moved and retained methods are interleaved throughout the file rather than split at a line
boundary, so each slice extracts by method name and leaves the retained methods in place.

---

## Component inventory

**Note on line references:** methods belonging to different stages are interleaved in
`tracking.py` — the ranges below are indicative anchors, not contiguous blocks to cut. For
example `on_merge_finished` (1567) and `on_merge_error` (1541) sit between media-export methods,
and `_log_rich_export_summary` (1119) sits between interpolated-crop methods. Extract by method
name, and re-derive line numbers with `grep -n "    def "` before starting each slice, since
other merges will have shifted them.

| Stage | Current home | Treatment | New home |
|---|---|---|---|
| Policy predicates: `_is_individual_pipeline_enabled`, `_should_export_final_media_videos`, `_should_export_final_canonical_images`, `_should_run_interpolated_postpass`, `_is_individual_image_save_enabled`, `_workflow_mode_key`, `_is_pose_inference_enabled`, `_is_headtail_compute_enabled` | `session.py:896-985` | pure functions over config dict | `core/tracking/session_policy.py` |
| Merge + post-process chaining: `_handle_forward_tracking_done` (2414), `_handle_backward_tracking_done` (2497), `on_tracking_finished` (2619), `_start_postprocess_worker` (2672), `on_postprocess_finished` (2716), `on_postprocess_error` (2738), `merge_and_save_trajectories` (850), `on_merge_progress` (961), `on_merge_error` (1541), `on_merge_finished` (1567), `_finish_tracking_session` (3469) | `tracking.py` (interleaved) | coupled → service methods | `TrackingSessionCore` |
| `MergeWorker` merge logic | `gui/workers/merge_worker.py:1-140` | pure functions | `core/post/merge.py` |
| Pose source merge + quality post-pass: `_check_pose_export_sources`, `_merge_pose_sources_into_df`, `_apply_pose_quality_postprocessing` | `tracking.py:2759-3124` | pure functions (DataFrame in, DataFrame out) | `core/post/pose_merge.py` |
| Identity post-pass: `_apply_identity_postprocessing_to_df` | `tracking.py:3137-3262` | pure function | `core/identity/postprocess_df.py` |
| Rich export: `_build_rich_export_dataframe`, `_export_rich_csv`, `_relink_and_export_rich_csv`, `_write_rich_export_csv`, `_drop_empty_rich_export_columns`, `_rich_export_path`, `_remove_legacy_rich_exports` | `tracking.py:87-137`, `3263-3417` | pure functions | `core/post/rich_export.py` |
| Interpolated crops: `_generate_interpolated_individual_crops` (3611), `_store_interpolated_{pose,tag,cnn,headtail}_result` (979-1038), `_log_interpolated_postpass_summary` (1039), `_count_augmented_pose_rows` (1083), `_count_interpolated_cnn_rows` (1096), `_on_interpolated_crops_finished` (1217) | `tracking.py` (interleaved) | coupled → service methods | `TrackingSessionCore` |
| Final media export: `_generate_final_media_export` (1284), `_start_pending_final_media_export` (1418), `_on_final_media_export_finished` (1441), `_on_final_media_export_error` (1521), `_get_video_draw_params` (1604), `_get_pose_column_info` (1674), `_preextract_traj_arrays` (1737), label/color builders (1802-2011), `_draw_*` (2012-2154), `_render_annotated_video_frames` (2155), `_open_video_cap_and_writer` (2251), `_compute_video_frame_range` (2272), `_generate_video_from_trajectories` (2288), `_load_video_trajectories` (3418), `_run_pending_video_generation_or_finalize` (3436), `_scale_trajectories_to_original_space` (743), `save_trajectories_to_csv` (772) | `tracking.py` (interleaved) | pure functions + one export entry point | `core/post/media_export.py` |
| Dataset generation: `_generate_training_dataset`, `on_dataset_finished` | `tracking.py:4301-4476` | delegates to existing `training/` service | `TrackingSessionCore` |
| Session summary: `_build_session_summary_lines` | `tracking.py:4507-4578` | pure function over config + result | `core/tracking/session_summary.py` |
| Empty-output guard: `_enforce_nonempty_forward` | `headless_tracking.py:157` | moves into the service | `TrackingSessionCore` |

The `_mw`/`_panels` reads inside moved code resolve one of three ways: a config-dict lookup (the
majority — they read widget values that are already config keys), a field on the service's own
state, or an injected callback (the `QMessageBox` sites).

---

## Interfaces

```python
# core/tracking/session.py
@dataclass
class SessionCallbacks:
    progress: Callable[[int, str], None] = _noop2
    status: Callable[[str], None] = _noop1
    warning: Callable[[str, str], None] = _noop2      # replaces QMessageBox.information/.warning
    stage_changed: Callable[[str], None] = _noop1     # drives GUI widget enable/disable
    should_stop: Callable[[], bool] = _never

@dataclass
class SessionResult:
    success: bool
    final_csv_path: str | None
    rich_export_path: str | None
    media_paths: list[str]
    dataset_result: dict | None
    summary_lines: list[str]
    error: str | None

class TrackingSessionCore:
    def __init__(self, *, video_path, config, params, paths, callbacks=SessionCallbacks()): ...
    def run_post_tracking(self, forward_trajectories, backward_trajectories=None) -> SessionResult: ...
```

Callbacks are constructor-injected with no-op defaults, matching `TrackingEngineCore`'s six
injected callbacks. Defaults being no-ops means the CLI can pass nothing and get silent operation.

---

## Data flow

**GUI:** commit pending edits → `build_config_dict()` + `get_parameters_dict()` → construct
`TrackingSessionCore` → run on a single `BaseWorker` → callbacks drive progress bars, status,
message boxes, and stage-based widget enable/disable → `SessionResult` feeds the summary dialog
and the RefineKit prompt.

**CLI:** `load_tracker_cli_session()` → same two dicts → construct the service → run inline on the
main thread → format `SessionResult.summary_lines` to the log, return exit 0/1.

Both paths produce an identical `SessionResult` for identical config.

---

## Error handling

Today failure is reported three ways: `QMessageBox.critical` (GUI),
`_mw._headless_session_error` (bridge), and raised `RuntimeError` (`headless_tracking.py`). The
service collapses these into one model:

- **Fatal** — raise `TrackingSessionError` (new, `core/tracking/errors.py`, following the
  `core/identity/classification/errors.py` precedent). The service aborts; the caller decides
  presentation. GUI shows `QMessageBox.critical`; CLI logs and returns exit 1.
- **Non-fatal** — `callbacks.warning(title, message)`. Every `QMessageBox.information`/`.warning`
  site maps here. GUI shows a box; CLI logs at WARNING.
- **Stage-level degradation** — stages that currently swallow an exception and continue (e.g. the
  rich-export trajectory count) keep doing so, unchanged.

This removes the reason `_suppress_message_boxes()` exists: the CLI currently monkeypatches four
`QMessageBox` static methods (`cli.py:24-56`) so the bridge does not block on a modal dialog.

`_enforce_nonempty_forward` — which refuses to emit an empty CSV when the detection cache contains
detections — moves into the service, so the GUI gains a guard that today protects only the CLI.

---

## Cancellation

The service takes `callbacks.should_stop()`, checked between stages and inside the long loops
(video render, crop extraction) — the same idiom `TrackingEngineCore` already uses for its
`_stop_requested` flag. The GUI passes a lambda over its stop flag; the CLI passes a
signal-handler-backed flag so Ctrl-C stops cleanly instead of leaving a half-written CSV.
Between-stage widget enable/disable stays in the GUI, driven by `callbacks.stage_changed(name)`.

**Known semantic change:** today each post-tracking stage runs on its own `QThread` and can be
killed mid-flight; inline stages instead run to their next `should_stop()` check. Visible only on
user-initiated stops.

---

## Implementation slices

One spec, four sequential plan slices. Each slice is gated independently so the equivalence
harness attributes any behavior change to a specific slice.

### Slice 1 — Policy and prerequisites
Move `model_utils` → `core/inference/model_paths.py`. Extract `build_config_dict()` out of
`save_config()`. Extract the policy predicates and `_build_session_summary_lines` as pure config
functions; GUI methods become one-line delegations. Unify `TRAJECTORY_COLORS` generation on the
GUI's legacy RNG. No service yet, no behavior change.

This slice exists to falsify the design's central assumption early: that every widget the moved
code reads has a config-dict equivalent. If a predicate reads widget state with no config key, it
surfaces here, in the smallest slice, rather than in the largest.

### Slice 2 — Analysis chain
Create `TrackingSessionCore` covering merge → post-process → pose merge → identity post-pass →
interpolated crops → rich export. Extract `MergeWorker`'s merge logic to `core/post/merge.py`. GUI
orchestrator delegates. CLI unchanged.

### Slice 3 — Export chain
Media export and dataset generation into the service. After this the service is at full parity
with the bridge. If media export proves unwieldy (~1100 lines of drawing helpers) it may split
into 3a (canonical crops + dataset) and 3b (annotated video) without disturbing other slices.

### Slice 4 — CLI cutover
`headless_tracking.py` drives the service directly. `supports_direct_run()` returns `True`
unconditionally. Delete `_run_bridge_tracking_session`, `_run_one_tracking_session`,
`_prepare_video_session`, `_ensure_qapplication`, `_suppress_message_boxes` (`cli.py`), and the
three `_headless_*` hooks (`main_window.py:390-392`, `tracking.py:2604`, `2746`, `4590-4612`).
Replace the remaining `QThread`/`QEventLoop`/`QCoreApplication` in `headless_tracking.py` with
plain threading — `TrackingEngineCore` is already Qt-free, `CSVWriterThread` is already a
`threading.Thread`, and `PostProcessWorker`/`MergeWorker` reduce to direct calls once their logic
is in `core/`.

---

## Testing and acceptance

### Equivalence gate (mandatory, per slice)
Each slice runs `tools/equivalence/run_matrix.sh` against the same baseline before and after, on
**both** MPS (`hydra-mps`, this box) and CUDA (mehek, `hydra-cuda`), across all 7 fixture clips.
Acceptance: positions p99 ≈ 0, θ max ≈ 0, identical row counts, 0 unmatched, on both
`_forward.csv` and `_tracking_final.csv`. Known noise floor is the bistable head/tail π-flip on
head/tail clips only.

Per-slice runs are what provide attribution. Conda must be active for any pose/SLEAP clip or the
CSVs come out empty and falsely compare EQUIVALENT — verify row counts > 1 before trusting a pass.

### CLI parity tests
For each fixture clip, run `trackerkit track` and assert outputs match the GUI-driven equivalent.
Slice 4 additionally asserts the four clips that go through the bridge today —
`ant_pose_headtail`, `ant_obb_sleap`, `emi_obb_identity`, `ant_cnn_identity` — produce identical
output via the direct path.

### Qt-free guard tests
Mirroring `tests/test_tracking_engine_core_qtfree.py`:

1. `grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/` stays empty.
2. A subprocess test that runs `trackerkit track` on a real clip with `PySide6` blocked from import
   (a `sys.meta_path` finder raising `ImportError` on `PySide6`) and asserts the session completes
   and writes a non-empty CSV. **This test is the executable definition of done.**

### Unit tests
One per extracted pure function. For the policy functions, a table test asserting each pure
predicate agrees with the widget-reading method it replaced, across all 7 fixture configs.

---

## Risks

1. **Config-dict coverage.** The design assumes every widget the moved code reads has a config
   key. Slice 1 is structured to falsify this in the smallest slice.
2. **Cancellation semantics drift.** Per-stage threads become one thread; see Cancellation above.
3. **Slice 3 size.** Media export drags in ~1100 lines of drawing helpers; split path documented.
4. **`get_parameters_dict` side effects.** It calls `np.random.seed(42)`, mutating global NumPy
   state as a side effect of building params. Unifying colors in Slice 1 must preserve the
   resulting color values exactly while confining the seeding, or overlay colors shift.

---

## Out of scope

- Converting the config seam to a typed dataclass (deferred to Simplification Sprint Slice 2).
- Decomposing `config.py` (4384 lines) beyond extracting `build_config_dict()`.
- Any change to tracking math, detection, or the inference runtime.
- The remaining `session.py` UI-state code, which is legitimately GUI-only.
