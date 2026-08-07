# Slice 5 — TrackerKit GUI Post-Tracking Cutover to `TrackingSessionCore`

**Date:** 2026-08-04
**Status:** Design approved; ready for implementation plan.
**Program:** Final slice of the Qt-free headless refactor (spec `2026-07-24-headless-qt-free-session-service-design.md`). This is the deferred Slice-2 **Task 9**, re-planned after Slice 4 landed.
**Branch:** continue on `feat/headless-qt-free`; the whole program merges to `main` only after this slice.
**Worktree of record for line numbers:** `.worktrees/headless-qt-free` @ `ba9ed24b`. All `tracking.py` line numbers below are anchors from that commit — re-derive against live source before editing (the file is 3788 lines and will shift as methods are deleted).

## Problem

After Slices 2–4, the Qt-free `TrackingSessionCore.run_post_tracking` (`core/tracking/session.py`) owns the entire post-tracking pipeline (postprocess → merge → rich-export → interpolated-crops → media-export → dataset-generation → summary), and the CLI (`trackerkit/headless_tracking.py`) drives it. But the **GUI still runs its own duplicate** of that whole chain in `gui/orchestrators/tracking.py` (~45 post-tracking methods + 11 thin wrappers in `main_window.py`). This duplication is the last major item of the program: it is ~2000 lines of GUI code that re-implements what `core/post/*` already does, and it means GUI behavior can silently drift from the CLI (Slice 4 proved how sharp such drift is — a 100-key param-builder gap).

## Goal

Make the GUI drive `run_post_tracking` (via a new `SessionWorker`) for all post-tracking, delete the duplicated orchestrator methods, and verify the GUI's output is byte-identical to the CLI's. Non-goal: changing `core/` (the CLI equivalence gate must stay valid without a re-gate) and non-goal: the shared param-builder unification (documented as the immediate follow-up, §Follow-up).

## Key design decisions (approved)

1. **Disk-based contract, no core change.** `run_post_tracking(forward, backward)` ignores its DataFrame arguments and re-reads the raw `_forward.csv`/`_backward.csv` from disk (`session.py:460-479`; `_postprocess_csv` at `session.py:184-193` calls `process_trajectories_from_csv(csv_path, ...)`). The GUI's two `TrackingWorker` passes **already write** exactly those raw CSVs (they are what the current `PostProcessWorker` reads — `_start_postprocess_worker` picks `{base}_backward{ext}` / `{base}_forward{ext}` / `raw_csv_path` at `tracking.py:1834-1841`). So the GUI mirrors the CLI: keep the two raw tracking passes, delete the per-pass `PostProcessWorker` + `MergeWorker`, and let `run_post_tracking` do all post-tracking from the raw CSVs on disk. **Zero `core/` change → GUI output ≡ CLI output ≡ pre-cutover bridge output by construction** (both drive the identical service on identical disk inputs).
2. **Scope = post-tracking cutover only.** The GUI keeps using its own `get_parameters_dict()` (the 259-key reference param derivation) and `build_config_dict()`; those are correct and untouched. The CLI/GUI param-builder duplication is the follow-up.

## Architecture — the new GUI flow

The GUI keeps its two **async** `TrackingWorker` QThread passes (the UI must stay responsive with progress/preview; `run_post_tracking` cannot and must not subsume the tracking passes). Only the post-tracking work is delegated.

```
forward TrackingWorker  → writes raw CSV
                          ({base}_forward{ext} when backward enabled, else raw_csv_path)
  └─ on_tracking_finished(is_backward_mode=False):
        if backward_enabled → start_backward_tracking()        [KEEP: async 2nd pass]
        else                → _run_session_worker()
backward TrackingWorker → writes raw {base}_backward{ext}
  └─ on_tracking_finished(is_backward_mode=True):
        → _run_session_worker()
_run_session_worker():
    core = TrackingSessionCore(video_path=…, config=build_config_dict(),
                               params=get_parameters_dict(),
                               paths={raw_csv_path, final_csv_path, detection_cache_path},
                               callbacks=SessionCallbacks(…, should_stop=<cancel flag>))
    worker = SessionWorker(core)         # BaseWorker(QThread)
    worker.finished_result → _on_session_finished(result: SessionResult)
    worker.start()
_on_session_finished(result):
    if not result.success → error dialog (reuse on_postprocess_error's QMessageBox path)
    else → populate summary from result.summary_lines
    → _finalize_tracking_session_ui()    [KEEP: batch continuation + summary dialog]
```

### Component: `SessionWorker(BaseWorker)` — new, `gui/workers/session_worker.py`

- Inherits `BaseWorker` (`widgets/workers.py:6`; subclasses implement `execute()`; signals `progress(int)`, `status(str)`, `error(str)`, Qt `finished`).
- Constructed with a fully-built `TrackingSessionCore` (or the pieces to build one). In `execute()` it calls `self._core.run_post_tracking(None, None)` (DataFrames are ignored by design) and emits the returned `SessionResult` via a new `finished_result = Signal(object)`.
- Bridges `SessionCallbacks` (`session.py:141-147`: `progress(pct,msg)`, `status(msg)`, `warning(title,msg)`, `stage_changed(name)`, `should_stop()`) to Qt: callbacks fire on the worker thread and emit queued signals to the GUI thread. `progress`→`self.progress.emit(int(pct))` (+ status text), `warning`→a `warning = Signal(str, str)`, `stage_changed`→`self.status.emit`.
- **Cancellation:** the GUI's existing cancel action sets a thread-safe stop flag (e.g. `threading.Event`); `SessionCallbacks.should_stop = event.is_set`. `run_post_tracking`'s stages already poll `should_stop`. `BaseWorker.run()` wraps `execute()` in try/except → `error`; `run_post_tracking` also catches `TrackingSessionError` internally and returns `SessionResult(success=False, …)`, so `execute()` normally returns a result even on failure.

### Anchor rewrite: `_finish_tracking_session` (`tracking.py:2602`)

Replace its body (which calls `_export_rich_csv`, `_generate_training_dataset`, `_generate_interpolated_individual_crops`, `_relink_and_export_rich_csv`, `_start_pending_final_media_export`, `_run_pending_video_generation_or_finalize` — `tracking.py:2614-2643`, all deleted) with `_run_session_worker()`. **Keep** `_finalize_tracking_session_ui` (`tracking.py:2645`) as the worker's finished callback.

### Retained GUI methods (do NOT delete)

`on_tracking_finished` (slimmed to fps-accumulate + forward→backward-vs-finish sequencing), `_accumulate_session_fps` (`tracking.py:1748`), `start_backward_tracking` (`tracking.py:2862`), `start_tracking_on_video` (`tracking.py:3335`), `_finalize_tracking_session_ui` (`tracking.py:2645`), `_collect_worker_props_path` (`tracking.py:1705`, subject to §Reconciliation), and all pre-tracking setup / preview / dialog code.

## Deletion set

**In `tracking.py` (~45 methods; verify each has no surviving non-deleted caller before removing):** the post-tracking chain enumerated in the cutover map — rich-export helpers (`_rich_export_path`, `_write_rich_export_csv`, `_drop_empty_rich_export_columns`, `_remove_legacy_rich_exports`, `_build_rich_export_dataframe`, `_export_rich_csv`, `_relink_and_export_rich_csv`, `_log_rich_export_summary`), save/scale/merge (`save_trajectories_to_csv`, `_scale_trajectories_to_original_space`, `merge_and_save_trajectories`, `on_merge_progress`, `on_merge_error`, `on_merge_finished`), interpolated-crops (`_store_interpolated_{pose,tag,cnn,headtail}_result`, `_log_interpolated_postpass_summary`, `_count_augmented_pose_rows`, `_count_interpolated_cnn_rows`, `_on_interpolated_crops_finished`, `_generate_interpolated_individual_crops`), postprocess workers (`_start_postprocess_worker`, `on_postprocess_progress`, `on_postprocess_finished`), pose-source merge + identity postprocess (`_check_pose_export_sources`, `_merge_pose_sources_into_df`, `_apply_pose_quality_postprocessing`, `_resolve_current_tag_cache_path`, `_apply_identity_postprocessing_to_df`), media/video (`_generate_final_media_export`, `_start_pending_final_media_export`, `_on_final_media_export_worker_thread_finished`, `_on_final_media_export_finished`, `_on_final_media_export_error`, `_generate_video_from_trajectories`, `_load_video_trajectories`, `_run_pending_video_generation_or_finalize`), dataset-gen (`_generate_training_dataset`, `on_dataset_progress`, `on_dataset_finished`, `on_dataset_error`), summary (`_build_session_summary_lines`, `_clear_session_summary_state`), and the two pass-completion handlers (`_handle_forward_tracking_done`, `_handle_backward_tracking_done`) whose retained decision logic folds into `on_tracking_finished`. `_show_session_summary` is retained but rewired to consume `SessionResult.summary_lines`.

**In `main_window.py` (11 thin wrappers):** `merge_and_save_trajectories` (2659), `_store_interpolated_{pose,tag,cnn,headtail}_result` (2667/2671/2675/2679), `_on_interpolated_crops_finished` (2685), `_generate_final_media_export` (2735), `on_merge_finished` (2759), `_finish_tracking_session` (2815, → keep a wrapper pointing at the rewritten orchestrator method or inline), `_generate_interpolated_individual_crops` (2823), `_generate_training_dataset` (2927).

**Method:** delete in dependency order (leaves first), and after each removal grep `src/` **and** `tests/` for the symbol — a Slice-3 lesson: deleting GUI methods breaks tests in other files. Removing the now-unused `PostProcessWorker`/`MergeWorker` imports/instantiation from the tracking flow is part of the cutover; the worker classes themselves stay in `gui/workers/` if still referenced elsewhere (verify).

## Reconciliation points (the plan must enumerate and verify each)

`run_post_tracking` must receive, via `paths`/`params`/`config`, everything the deleted GUI chain fed it implicitly:
- **Raw CSV naming.** Confirm the GUI's forward/backward `TrackingWorker` passes write raw to exactly `{base}_forward{ext}` / `{base}_backward{ext}` (backward-enabled) and `raw_csv_path` (forward-only) — matching `session.py:471/479`. This is believed true (it is what `_start_postprocess_worker` reads today) but must be asserted.
- **Detection-cache path.** `paths["detection_cache_path"]` — the GUI has it from the tracking worker; the CLI passes it. Supply it.
- **Individual-properties cache path.** The GUI captures per-pass caches via `_collect_worker_props_path` (`tracking.py:1705`) → `current_individual_properties_cache_path`. `run_post_tracking`'s interpolated-crops / rich-export stages need this. Trace how the CLI supplies it to `run_post_tracking` (via `params`/`config`/`paths`) and make the GUI supply the same value; if the CLI does not currently supply it, that is a gap to close in this slice (and to note for the smoke, since it affects the rich-export stage).
- **`final_csv_path`.** Pass the GUI's session final-CSV path (the service writes `{base}_final{ext}` — reconcile with the GUI's expected output path so downstream GUI features find the file).
- **fps / frames / timing** for the summary: the GUI accumulates `_session_fps_list`/`_session_frames_processed`; `SessionResult.summary_lines` is built inside `run_post_tracking` from `build_session_summary_lines`. Decide whether the GUI appends its fps line (as the CLI driver does) or the service is given the fps via config/params.

## Error handling & cancellation

- Failure: `SessionResult.success == False` → show the existing "Post-Processing Error" dialog (the retained `on_postprocess_error` path) with `result.error`, then `_finalize_tracking_session_ui`. A raised exception in `execute()` → `BaseWorker.error` → same dialog.
- Cancel mid-post: cancel flag → `SessionCallbacks.should_stop` → stages stop cooperatively; treat as a non-success finish (no partial-artifact promotion).
- The two tracking passes keep their existing cancel/stop handling (unchanged).

## Testing / verification

The CLI equivalence gate **cannot** catch a GUI regression (it runs the CLI). Verification has three parts:

1. **Rewrite `tests/test_trackerkit_tracking_orchestrator_dialogs.py`.** It is the only test file that directly patches the deleted methods (functions at lines 892, 989, 1024, 1121, 1187, 1236, 1290, 1311, 1337 per the map). Re-point each at the core stage that now owns the behavior (`core/post/*` already has unit coverage: `test_session_core_run.py`, `test_session_export_chain.py`, etc.) or delete the ones that only tested now-deleted GUI plumbing. Judge against the file's known pre-existing failures (delta, not absolute).
2. **New GUI launch smoke** (`tests/test_gui_session_cutover.py`, offscreen Qt `QT_QPA_PLATFORM=offscreen`): construct a real `MainWindow`, run (a) a forward-only session and (b) a backward-enabled session on fixture clips to completion via the real signal/slot flow, asserting: no `PostProcessWorker`/`MergeWorker` is constructed (the deleted path is gone), the output CSV is non-empty, and it is **byte-identical to the CLI's output** (`run_tracking_cli`) for the same clip/config. Include at least one identity clip (`ant_cnn_identity`) so the pose/identity/rich-export/interp stages are exercised. This is the executable proof that GUI ≡ CLI. Pose/SLEAP clips need the `sleap` conda env; skip when fixtures/models absent, pass when present.
3. **Slice-4 CLI equivalence gate** re-run once (MPS at minimum) as a sanity check that `core/` is genuinely untouched. Not expected to change (no core edit); a divergence would mean an accidental core change.

## Verification of "done"

- GUI launch smoke green (forward-only + backward-enabled, GUI CSV == CLI CSV) on MPS.
- Orchestrator-dialogs tests green (delta vs baseline).
- Targeted suite green; no dangling references to deleted symbols in `src/` or `tests/`.
- CLI gate unchanged (sanity MPS run).
- Whole 4-slice-plus program merges to `main`.

---

## Follow-up (execute immediately after this slice): Shared Qt-free param-builder

**Problem.** There are TWO param derivations producing the engine `params` dict: the GUI's `get_parameters_dict()` (`gui/orchestrators/config.py:2135-2523`, ~259 keys, the reference) and the CLI's `build_tracking_parameters()` (`trackerkit/cli_config.py`). Slice 4 found the CLI missing **100** of the bridge's keys and fixed the tracking-relevant subset across three commits — identity-analysis/classifier (`8ea7b340`), the full identity-in-tracking block incl. `ENABLE_IDENTITY_ONLINE_DECODER`/`ASSOCIATION_IDENTITY_HINT_SCALE` (`a1bde2ba`), and the `POSE_*` block (`d46bd7af`). **~81 keys still diverge** (proven inert for the 7 gate clips by a runtime param-diff, but a real drift hazard): `DATASET_*`, `SLICE_*`, `FINAL_MEDIA_EXPORT_*`, `INDIVIDUAL_*`, `YOLO_OBB_SEG_*`/`YOLO_OBB_DIRECT_TASK`/`YOLO_OBB_FIXED_ANGLE_DEG`, `SUPPRESS_FOREIGN_OBB_*`, `METRIC_*`, `ENABLE_PROFILING`, `MAX_BRIDGE_GAP_FRAMES`, `FRAGMENT_SPATIAL_VETO_THRESHOLD`, `COLOR_TAG_*`, legacy singular `CNN_CLASSIFIER_MODEL_PATH/CONFIDENCE/LABEL/BATCH_SIZE`, `IDENTITY_METHOD` edge, and the two pose Minors (`POSE_SLEAP_BATCH` should equal `POSE_BATCH_SIZE` not read its own field; `_coerce_pose_keypoint_tokens` should emit plain strings like the bridge's `_selected_pose_group_keypoints`).

**Solution.** Extract one pure, Qt-free `build_engine_params(config: dict, …) -> dict` (in `core/` or a shared util) that derives the full engine param set from a config dict, and have BOTH callers use it: `get_parameters_dict()` becomes a thin wrapper that adds the few genuinely-GUI-only keys (display toggles `SHOW_*`, `zoom_factor`, live-viz stride), and `build_tracking_parameters()` calls it directly. This deletes the duplication and makes future drift impossible. The GUI's `_selected_pose_group_keypoints` / `_pose_config` / `_identity_config` derivations are the reference implementations to lift into the pure function (they read persisted config fields, not live widgets — confirmed in Slice 4).

**Verification.** Because this changes the params both paths feed the engine, it needs its own full byte-identity re-gate of the CLI (MPS + CUDA, discounting the mehek cupy environmental pose-clip failures per the program's documented acceptance) PLUS the GUI launch smoke from this slice. The Slice-4 B/B2/B3 commits are the exact template for the per-key derivation and the parity-safety argument (a key inert for a clip must stay inert — e.g. `identity_weight=0.0` gated by `alpha>0` at `hungarian.py:239-240`).

**Why separate from Slice 5.** It touches the gated param path (needs a re-gate), it is large (259-key extraction), and the ~81 keys are proven inert for the gate clips — so it is safely sequenced *after* the post-tracking cutover rather than folded in (which would double the risk surface of one slice).
