# Design: land direct detect/segment + runtime fixes on the retired-detector codebase

**Date:** 2026-07-21
**Status:** design — approved, ready for plan.
**Branch:** `feature/obb-direct-detect-segment` (in `.worktrees/obb-direct-detect-segment`)
**Runs after:** the detector retirement (Plans 1–4) has fully landed on `main`.

## Context (verified against `main`, not predicted)

The legacy-detector retirement is complete on `main` (`b5a9b10`):

- `src/hydra_suite/core/detectors/` **no longer exists**.
- `_direct_obb_runtime.py` now lives at `src/hydra_suite/core/inference/direct_executors.py`.
- `TrackingWorker._build_inference_config_from_params` is **deleted**; the builder is now
  the module function `core/inference/config.py::build_inference_config_from_params`.
- Preview runs through `InferenceRunner.run_realtime`; the ~1000-line preview clone,
  its fake `RuntimeContext`, and the `_advanced_config_value` reach-ins are gone.
- `InferenceRunner.detect_batch` and `build_obb_only_config` exist.

**Critical fact:** the retirement was cut from a *pre-segment* baseline. `main`'s
`OBBDirectConfig` has only `model_path`/`confidence_floor`/`confidence_threshold`/`auto_export`,
and `direct_executors.py` has **no** `create_direct_segment_executor` / `DirectTensorRTSegmentExecutor`.
The entire detect/segment-direct feature exists only on this branch. So the merge
re-applies the feature onto the relocated structure; the runtime fixes layer on top.

## Merge reconciliation map (main → branch)

| Branch touched | Retirement did on `main` | Conflict | Resolution |
|---|---|---|---|
| `core/detectors/_direct_obb_runtime.py` (+ segment executors, TRT segment decode) | `git mv` → `core/inference/direct_executors.py` | rename/modify | Replay `create_direct_segment_executor` + `DirectTensorRTSegmentExecutor` + segment decode into `direct_executors.py`; add to its exports and `create_direct_*` dispatch. |
| `config.py::OBBDirectConfig` (+ `model_task`, `fixed_angle_deg`, `seg_num_angles/crop_size/pad_ratio/mask_threshold`; kept deprecated `compute_runtime`) | added `auto_export`; **removed** `compute_runtime` | both edited dataclass | Union the fields onto `main`'s version. Do **not** reintroduce `compute_runtime` (retirement removed it); confirm `_dict_to_config`/`to_dict` round-trip tolerates its absence in old presets. |
| `worker._build_inference_config_from_params` (+ `YOLO_OBB_SEG_*` clamp via `_clamped_int/_clamped_float`, lines ~4528–4557) | **deleted** the method → `config.build_inference_config_from_params` | delete/modify | **Re-home the clamp + seg-param plumbing into `config.build_inference_config_from_params`.** This is the load-bearing merge step. |
| `preview_worker.py` `_preview_*` (added `_preview_build_yolo_params`, `_preview_direct_model_task`, `_extract_obb_from_masks` call) | **deleted** the clone → `run_realtime` | delete/modify | **Discard the branch's preview edits.** Parity is now structural: preview builds one `InferenceConfig` via `build_inference_config_from_params` and calls `run_realtime`, so `YOLO_OBB_SEG_*` flow identically to the tracking run. |
| `runtime_artifacts.py` (docstring, `DirectExecutorAdapter`, imgsz) | docstring + import repoint | textual | Merge; imgsz/task-guard fixes stay here, retargeted at `direct_executors`. |
| `stages/obb.py` (segment extraction, `_assert_direct_task_matches_checkpoint`) | unchanged by retirement | none | Keep; task-guard fix lands here. |
| `tracking_cache.py` (7 seg keys, `raw_detection_cache_version=4`) | unchanged | none | Keep. |
| `orchestrators/config.py`, `panels/detection_panel.py` (task selector, fixed-angle UI) | unchanged | none | Keep. |
| `tests/test_detectors_engine.py` | **deleted** | — | Do not carry branch edits; new coverage lands in `tests/test_inference_*`. |

## The five fixes, expressed against post-retirement locations

1. **Runtime tiers — document, don't rewire.** `runtime_to_compute_runtime`
   (`core/inference/runtime.py`) already tiers cpu→`cpu` / gpu→`cuda`,`mps` /
   gpu_fast→`tensorrt`,`coreml`. After retirement, the ONNX direct executors'
   only remaining callers are diag/benchmark tooling — the legacy `yolo_detector`
   caller is gone. **Decision:** keep the ONNX executors (cheap, used by the diag
   tools), and add a comment on `_direct_runtime_name` recording that the inference
   stage only ever emits `tensorrt`, so the `"onnx"` branch is tool-only. No stage
   wiring for ONNX.

2. **TRT `iou_threshold` unreachable.** Thread `config.iou_threshold` from
   `load_obb_executor` → `DirectExecutorAdapter` → `create_direct_{obb,detect,segment}_executor`
   → `_decode_*_predictions` in `direct_executors.py`, replacing the hardcoded
   `iou_thres = 1.0 if is_end2end else 0.5`. Preserve `end2end`→1.0 (head does its own NMS).

3. **TRT task-mismatch guard silently skipped.** Set `DirectExecutorAdapter.task`
   from the requested `model_task` so `_assert_direct_task_matches_checkpoint`
   (`stages/obb.py`) no longer no-ops on TRT. Documented limit: catches
   config/adapter disagreement, not a genuinely mislabeled `.engine` (no task metadata).

4. **imgsz-640 hardcode for user-supplied artifacts.** `runtime_artifacts.py` resolves
   imgsz from the artifact/config (reuse `_resolve_imgsz`) instead of defaulting to 640;
   fail loudly if unresolvable rather than silently letterboxing wrong.

5. **Real detect-task test coverage.** Real `yolo11n.pt` detect run through the
   production `load_obb_models`/`run_obb` path, parametrized over `mps` + `cuda`
   (skip only the genuinely-absent device), synthetic input constructed to yield
   ≥1 detection so assertions fire. Lands in the `tests/test_inference_*` suite.
   Tighten the segment smoke test's zero-detection-passes and its broad
   export-error `except` swallow.

## Verification gates

- `grep -rn "core.detectors\|core/detectors" src/ tests/` → empty (retirement gate stays green post-merge).
- **Config-parity test:** `build_inference_config_from_params` with `YOLO_OBB_SEG_*` set
  yields an `OBBDirectConfig` carrying the clamped kernel params — proving preview
  (via `run_realtime`) and the tracking run read identical values, closing the original
  parity bug structurally.
- Preset round-trip: old preset without `compute_runtime` deserializes cleanly.
- Full `pytest -m "not benchmark"`; and the retirement's frame-by-frame parity run
  on a real video against a pre-merge baseline.

## Out of scope

- No preview-clone re-plumbing (retirement deleted it; parity is structural now).
- No new ONNX stage runtime; ONNX stays tool-only.
- No benchmarking changes (retirement's external CLI owns that).
