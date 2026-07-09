# Plan B Task 2 Report — Port preview YOLO OBB branch onto `load_obb_executor`

(Note: this file previously held an unrelated report from a different plan's
"Task 2" — TensorRT/CoreML batch-size threading. That report is superseded
here per this session's explicit instruction to write to this path for Plan
B's Task 2, "Port `_preview_run_yolo_branch` (direct + sequential OBB) onto
`load_obb_executor`".)

## Summary

Ported `_preview_run_yolo_branch` / `_preview_run_yolo_raw_detection` (and the
sequential stage-1 visualization helper) in
`src/hydra_suite/trackerkit/gui/workers/preview_worker.py` off the legacy
`YOLOOBBDetector` class onto the production `load_obb_executor` factory
(`hydra_suite.core.inference.runtime_artifacts`), following Task 1's design
note. `_collect_preview_detection_context`
(`src/hydra_suite/trackerkit/gui/panels/detection_panel.py`) now resolves an
`obb_compute_runtime` via `resolve_compute_runtime(tier, platform,
stage="obb")` instead of `derive_detection_runtime_settings`, added as a new
context key alongside (not replacing) the existing `compute_runtime`,
`headtail_runtime`, `cnn_runtime` keys.

## Key design decisions

1. **Class-id preservation (design note finding #1).** Production `OBBResult`
   has no `class_ids` field. Rather than using `run_obb`/`load_obb_models`
   (which discard `cls` via `_extract_obb_result`), the new code calls
   `load_obb_executor(...)` directly and reuses
   `OBBGeometryMixin._extract_raw_detections(obb_data, return_class_ids=True)`
   — via the already-existing lightweight `hydra_suite.core.detectors.DetectionFilter`
   mixin instance (a throwaway `.params`-only class, exactly the "lightweight
   instance" the design note suggested) — which reads `obb_data.cls` directly
   off the raw ultralytics `Results.obb`. This works uniformly for both the
   plain-`YOLO` torch/mps/cuda branch and the `DirectExecutorAdapter`
   (tensorrt/coreml) branch, since both construct a standard 7-column
   `[x,y,w,h,angle,conf,cls]` OBB tensor.

2. **GUI-only filtering (finding #2) reused, not reimplemented.** `filter_raw_detections`
   and the custom OBB-IOU NMS (`_filter_overlapping_detections`/
   `_compute_obb_iou_batch`) already live in `OBBGeometryMixin`
   (`core/detectors/_obb_geometry.py`), which only depends on `self.params`.
   `DetectionFilter` (in `core/detectors/detection_filter.py`) is exactly the
   pre-existing "lightweight instance" the design note asked for — no new
   filtering code was written; `preview_worker.py` now calls
   `DetectionFilter(yolo_params).filter_raw_detections(...)` in place of
   `detector.filter_raw_detections(...)`.

3. **Head-tail (finding #5).** Added `_preview_load_headtail_model` (calls
   `core/inference/stages/headtail.py::load_headtail_model`) and
   `_preview_run_headtail` (calls `run_headtail`), preserving the exact
   legacy ordering: cheap candidate pre-filter
   (`_preview_select_headtail_candidate_indices`, a direct port of
   `YOLOOBBDetector._select_headtail_candidate_indices`, which only ever
   depended on `params`) → head-tail model call on that raw candidate subset
   (wrapped in a throwaway `OBBResult`) → hints scattered back to the full raw
   index space, then trimmed by the later `filter_raw_detections` call exactly
   as before.

   `load_headtail_model`/`run_headtail` take a `RuntimeContext` argument but
   never read it for anything except deriving the compute-runtime string at
   load time (`runtime_to_compute_runtime`) — so a new helper
   `_preview_runtime_context_for(compute_runtime_str)` synthesizes a minimal
   `RuntimeContext` whose fields round-trip through `runtime_to_compute_runtime`
   back to the requested string, rather than constructing a full
   `InferenceConfig`.

4. **Custom OBB-IOU suppression (finding #3 / #4a).** Confirmed
   `_compute_obb_iou_batch` only reads `self.params` (no other instance
   state) — it is exercised transparently via the `DetectionFilter` instance's
   inherited `filter_raw_detections` → `_filter_overlapping_detections` call,
   with no extraction needed.

5. **`_preview_yolo_sequential_stage1_viz` (finding #4a).** Signature changed:
   the `detector` positional arg was replaced with `detect_model_names` (a
   pre-normalized `dict[int, str]`), since the only thing it read off
   `detector` was `detector.detect_model.{model.,}names` for stage-1 class
   labels. The call site in `_preview_run_yolo_branch` now passes
   `_normalize_preview_model_names(getattr(executors.get("detect"), "names", None))`.

6. **Sequential mode (finding #4).** New `_preview_run_sequential_raw_detection`
   loads a stage-1 "detect" executor and a stage-2 "obb" (crop) executor via
   two `load_obb_executor(...)` calls (mirroring Plan A's `bench_sequential`
   in `trackerkit/benchmarking.py`), then:
   - runs stage-1 `.predict([frame], ...)`,
   - builds crops via `core/inference/stages/obb.py::_build_crops` (imported
     directly — not reimplemented — using a small `_PreviewSeqCropSpec` shim
     exposing the three attributes `_build_crops` actually reads:
     `crop_pad_ratio`, `min_crop_size_px`, `enforce_square_crop`),
   - resizes crops via `core/inference/stages/obb.py::_resize_crops_for_stage2`,
   - runs stage-2 `.predict(...)` in `YOLO_SEQ_INDIVIDUAL_BATCH_SIZE` chunks,
   - merges/rescales per-crop detections back into full-frame coordinates and
     re-sorts/truncates to `MAX_TARGETS*2`, mirroring legacy
     `_seq_accumulate_crop_detections`/`_seq_sort_and_return` byte-for-byte
     (new `_preview_accumulate_crop_detections`/`_preview_sort_merged_detections`
     helpers), including class-id carry-through via
     `extractor._extract_raw_detections(result.obb, return_class_ids=True)`
     per crop.
   - This preview-local sequential path only covers the CPU-numpy branch
     (single BGR frame, no NVDec CUDA-tensor / direct-GPU-crop path) since
     preview only ever processes one in-memory frame — the GPU-tensor
     fast-paths in legacy `_run_sequential_raw_detection` are realtime-tracking
     -only optimizations, out of scope for a single-frame preview.

7. **Dead legacy params removed (finding #6).** `YOLO_DEVICE`,
   `ENABLE_TENSORRT`, `ENABLE_ONNX_RUNTIME`, `TENSORRT_MAX_BATCH_SIZE` reads
   removed from `_preview_build_yolo_params`; a new `OBB_COMPUTE_RUNTIME` key
   (sourced from the context's new `obb_compute_runtime`, falling back to
   `compute_runtime`) added instead and threaded into
   `_preview_load_obb_executors` as the `compute_runtime` argument to
   `load_obb_executor`.

8. **`_collect_preview_detection_context` (`detection_panel.py`).** Per the
   task's scope-boundary correction: only the `derive_detection_runtime_settings`
   call was removed (replaced with
   `tier = self._main_window._selected_runtime_tier(); platform =
   detect_platform(); obb_compute_runtime = resolve_compute_runtime(tier,
   platform, stage="obb")`); `derive_pose_runtime_settings`/`runtime_pose`/
   `pose_backend_family` (Task 3's scope) were left untouched. Note: the
   brief's Step-3 snippet names `self._main_window._current_runtime_tier()`,
   but that method lives on the session orchestrator mixin
   (`orchestrators/session.py`), not on `MainWindow` itself — `MainWindow`'s
   equivalent public method is `_selected_runtime_tier()`, which is what was
   actually called (verified against `test_trackerkit_panels_smoke.py`'s
   `test_preview_detection_context_keeps_identity_overlays_without_master_toggle`,
   which failed with `AttributeError` against the brief's literal name and
   passes with the corrected one).

   The `yolo_device`/`enable_gpu_background`/`enable_tensorrt`/
   `enable_onnx_runtime`/`tensorrt_max_batch_size` context-dict entries (which
   were solely derived from `derive_detection_runtime_settings`) were removed
   along with the now-dead `trt_batch_size` local. `"compute_runtime"`,
   `"headtail_runtime"`, `"cnn_runtime"` keys were left exactly as they were.
   One residual, low-risk side effect: `_preview_run_pose_overlay`
   (Task 3's area, in `preview_worker.py`, line ~770) reads
   `context.get("yolo_device", "cpu")` — since that key no longer exists in
   the context dict, it now always defaults to `"cpu"` rather than the tier-derived
   device string. This is a Task-3-owned code path (pose runtime resolution is
   independently driven by `pose_runtime_flavor`/`compute_runtime` already);
   flagging for Task 3's implementer to confirm/clean up, since the default
   fallback makes this safe (no crash) but is a minor behavior narrowing.

## Files changed

- `src/hydra_suite/trackerkit/gui/workers/preview_worker.py`
- `src/hydra_suite/trackerkit/gui/panels/detection_panel.py`
- `tests/test_trackerkit_preview_worker.py`

## Tests

- Added `test_preview_run_yolo_branch_uses_load_obb_executor_not_legacy_detector`
  (Step 1's failing test — monkeypatches `hydra_suite.core.detectors.YOLOOBBDetector`
  to raise on construction, and monkeypatches
  `hydra_suite.core.inference.runtime_artifacts.load_obb_executor` to a fake
  executor, then asserts `_preview_run_yolo_branch` calls the fake executor's
  `.predict()` and never constructs the legacy detector). Confirmed this test
  fails against the pre-change code (it imports/constructs `YOLOOBBDetector`
  unconditionally and never calls `load_obb_executor`), then passes after the
  implementation.
- Rewrote `test_preview_run_yolo_branch_uses_filtered_headtail_hints` and
  `test_preview_raw_detection_prefilters_headtail_candidates` (both
  exercised the old detector-based API directly and needed updating to the
  new `executors`/`headtail_state`-based signatures — old detector mocks
  removed, replaced with `load_obb_executor`/`_preview_run_direct_raw_detection`/
  `_preview_select_headtail_candidate_indices`/`_preview_run_headtail` mocks
  that preserve each test's original intent).
- All 7 tests in `tests/test_trackerkit_preview_worker.py` pass.
- `tests/test_trackerkit_panels_smoke.py` (40 tests, incl. detection-panel
  wiring and the preview-context test) and `tests/test_compute_runtime.py`
  pass.
- Ran the full repo test suite (`pytest tests/ --ignore=tests/test_identity_postprocess.py
  --ignore=tests/test_bg_parameter_helper.py`) — both ignored files have
  pre-existing, unrelated failures confirmed present on `HEAD` (pre-change)
  too (`test_identity_postprocess.py` fails at collection with an
  unrelated `AttributeError`; `test_bg_parameter_helper.py`'s slider-scrub
  test fails identically before and after this change, confirmed via
  `git stash`).
- `make format` (black/isort — reformatted the 3 changed files; an
  incidental isort touch to an unrelated file, `refinekit/gui/dialogs/merge_wizard.py`,
  was reverted since it's outside this task's scope) and `make lint`
  (moderate flake8) produced no new findings in the changed files (all
  reported issues are pre-existing, in files this task didn't touch).

## Manual smoke test (Step 5)

**Not performed** — this environment has no GUI/display available, so
launching `trackerkit`, loading a real video + YOLO OBB model, and running
"Preview Detection" across CPU/GPU/GPU-Fast tiers (direct and sequential OBB
modes) could not be exercised interactively. This is flagged per this
repo's guidance for honesty about untested UI paths: **a human must run the
manual smoke test described in the brief's Step 5 before merging**,
particularly to verify:
- detections draw with correct class labels and confidences in both direct
  and sequential OBB modes,
- head-tail arrows/labels render correctly when a head-tail model is
  configured,
- no crash across CPU / GPU / GPU-Fast tiers,
- TensorRT/CoreML (gpu_fast) direct-executor paths actually export/load
  correctly on real hardware (this session only exercised the code paths via
  monkeypatched fakes — the real `load_obb_executor` TensorRT/CoreML export
  machinery was not exercised end-to-end).

---

## Code-review fix-up (post-`c54afd7`)

A code reviewer found two Important issues on top of the above. Both are now
fixed in a follow-up commit.

### Finding 1 — Missing test coverage for the sequential-mode path

Added two new tests to `tests/test_trackerkit_preview_worker.py`:

- `test_preview_run_sequential_raw_detection_merges_sorts_and_truncates` —
  calls `_preview_run_sequential_raw_detection` directly with fake stage-1
  (detect) / stage-2 (crop-OBB) executors. Stage-1 returns 4 candidate boxes;
  stage-2 returns non-pre-sorted per-crop detections (confidences 0.4, 0.9,
  0.6) plus a genuinely empty result for the 4th crop, and `max_det=2`.
  Asserts: the empty-result crop is skipped by
  `_preview_accumulate_crop_detections`, the merged detections come back
  confidence-descending and truncated to 2 by
  `_preview_sort_merged_detections` (0.9, 0.6 — the 0.4 one is dropped),
  class ids travel with their detection through the sort/truncate, and
  corners are well-formed `(4, 2)` arrays.
- `test_preview_run_yolo_branch_sequential_mode_uses_two_executors` — full
  end-to-end test through `_preview_run_yolo_branch` in sequential mode,
  monkeypatching `runtime_artifacts.load_obb_executor` to return one of two
  distinct fake executors keyed off the `task=` kwarg (`"detect"` vs.
  `"obb"`), following the exact same monkeypatching pattern the pre-existing
  direct-mode test in this file uses. Also monkeypatches
  `detectors_pkg.YOLOOBBDetector` to raise, confirming the legacy class is
  never touched. Asserts `load_obb_executor` was called once for each task
  with the expected model paths, and that the final
  `_preview_draw_obb_annotations` call receives 2 well-shaped,
  confidence-ordered detections (discovered along the way: the GUI's final
  `MAX_TARGETS` cap in `filter_raw_detections` truncates by detection *size*
  descending, not confidence — the test fixture's box sizes were set to
  correlate with confidence so the expected top-2 is unambiguous either way).

Both new tests pass; `_preview_yolo_sequential_stage1_viz`'s new
`detect_model_names` signature is exercised transitively through the
second test's `stage1_result`/`names` fixture (the sequential branch of
`_preview_run_yolo_branch` calls it whenever `stage1_result is not None`).

### Finding 2 — Dropped pre-crop confidence sort/truncate: investigation

Legacy `YOLOOBBDetector._run_sequential_raw_detection`
(`src/hydra_suite/core/detectors/yolo_detector.py:1594-1596`) explicitly did
`order = np.argsort(det_conf)[::-1]; order = order[:max_det]` on stage-1
detections before building crops. The new `_preview_run_sequential_raw_detection`
does not — it builds crops directly off `boxes` as returned by
`detect_executor.predict(...)`.

**Investigated whether this is safe, rather than leaving it an unstated
assumption.** Confirmed, by reading the actual source in this repo's active
conda env (`hydra-mps`, ultralytics 8.4.34) and this repo's own direct-executor
code, that `detect_executor`'s `.predict()` for both backends is guaranteed
to return confidence-descending + `max_det`-capped output:

- **Torch cpu/mps/cuda** (`_load_torch_executor`) and **CoreML** (which also
  loads via `_load_torch_model`, `runtime_artifacts.py:534`) run ultralytics'
  own `Results` postprocessing, which calls
  `ultralytics.utils.nms.non_max_suppression(..., max_det=max_det)`
  (`.../site-packages/ultralytics/utils/nms.py`). Inside it: NMS keep-indices
  come from `scores.argsort(descending=True)` (`TorchNMS.nms`, nms.py:265;
  `TorchNMS.fast_nms`, nms.py:218) or `torchvision.ops.nms` (also
  score-descending per its own docs), and the result is capped via
  `i = i[:max_det]` (nms.py:157) **after** that descending sort — so the
  surviving boxes are already confidence-descending and capped before
  `Results.boxes` is even constructed.
- **The direct TensorRT/ONNX executor**
  (`src/hydra_suite/core/detectors/_direct_obb_runtime.py`, `_postprocess`,
  ~line 266) calls that *exact same*
  `ultralytics.utils.nms.non_max_suppression(preds, ..., max_det=max_det, ...)`
  function — not a separate TensorRT-side NMS implementation — so the
  guarantee is byte-identical across backends.

This also matches current (non-legacy) production code: `_run_sequential` in
`src/hydra_suite/core/inference/stages/obb.py:444-509` (the real
tracking-time sequential-OBB pipeline) builds crops directly off stage-1
`boxes` with **no** explicit re-sort either (`_build_crops(frame, boxes, seq,
runtime)` at line 469) — for the same reason.

**Conclusion: left the sort/truncate out, as verified-safe rather than an
unstated assumption**, and added a code comment in
`_preview_run_sequential_raw_detection`
(`src/hydra_suite/trackerkit/gui/workers/preview_worker.py`, just before crop
building) documenting this confirmation in detail (with the exact file/line
citations above) so a future reader doesn't have to re-derive it or wonder
if it's an oversight.

### Minor finding — `DetectionFilter(yolo_params)` double instantiation

Fixed: `_preview_run_yolo_raw_detection` now accepts an optional `extractor`
parameter (falls back to constructing its own `DetectionFilter` when
omitted, so its other caller/tests are unaffected); `_preview_run_yolo_branch`
now constructs a single `DetectionFilter(yolo_params)` up front and passes it
in via `extractor=extractor`, reusing it for the later
`filter_raw_detections` call instead of constructing a second instance.
Updated the one existing test that monkeypatched
`_preview_run_yolo_raw_detection` with a fixed lambda signature to accept the
new `extractor=None` keyword.

### Files changed (this fix-up)

- `src/hydra_suite/trackerkit/gui/workers/preview_worker.py`
- `tests/test_trackerkit_preview_worker.py`

### Verification

```
python -m pytest tests/test_trackerkit_preview_worker.py -v
# 9 passed

python -m pytest tests/test_trackerkit_preview_worker.py tests/test_trackerkit_panels_smoke.py tests/test_compute_runtime.py -q
# 41 passed, 1 skipped
```

`make format` (black/isort — reformatted the two changed files only; the
incidental isort touch to `refinekit/gui/dialogs/merge_wizard.py` was
reverted again, as it's outside this fix-up's scope) and
`flake8 --config=.flake8.moderate` on both changed files individually
produced no findings. `make lint` (full-repo) shows only pre-existing
findings in files this fix-up did not touch.
