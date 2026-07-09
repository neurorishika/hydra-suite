# Task 8 Report — BLOCKED (tracing contradicts controller's premise)

## Status: BLOCKED

Per the controller's own explicit instruction: *"If your investigation
contradicts what I found — i.e. you find a real live consumer that DOES
construct `YOLOOBBDetector` for `"yolo_obb"` detection method using these
fields — STOP and report BLOCKED with what you found; do not proceed with
deletion."*

I found exactly that consumer. All code edits described below are already
made in the working tree (uncommitted, not committed, no `git commit` run)
so the controller can inspect the diff directly. **Nothing has been
committed.**

## What the controller traced correctly

`core/tracking/worker.py`'s live `TrackingWorker.run()` path is exactly as
described: for `DETECTION_METHOD == "yolo_obb"` (worker.py:976), execution
goes entirely through the `InferenceRunner`/`load_obb_executor` path; the
`else:` branch containing the only `create_detector(p)` call in that file
(worker.py:1134) is the background-subtraction branch and is unreachable
when `detection_method == "yolo_obb"`. I independently confirmed this by
reading worker.py:763-1148. I also confirmed `trackerkit/headless_tracking.py`
(the CLI entry point) constructs the same `TrackingWorker` — no separate
CLI-only detector-construction route exists. **This part of the tracing is
correct.**

## What the controller's tracing missed: a second, real, `yolo_obb`-live consumer

There is a **second live construction path** for `YOLOOBBDetector`, entirely
separate from `TrackingWorker`/`InferenceRunner`, used by the trackerkit GUI's
**"Parameter Helper" / detection-cache-build feature**:

1. `trackerkit/gui/orchestrators/config.py::_open_parameter_helper()` (line
   ~3487) calls `params = self.get_parameters_dict()` — this is the *exact
   same params dict* built by the method containing the
   `derive_detection_runtime_settings(compute_runtime)` call I replaced at
   config.py:2130 (now `legacy_detection_runtime_fields(compute_runtime)`),
   which sets `params["YOLO_DEVICE"]`, `params["ENABLE_TENSORRT"]`,
   `params["ENABLE_ONNX_RUNTIME"]` from that mapping.
2. When no valid detection cache exists for the selected frame range, the
   dialog offers a "quick detection-only scan" (`_build_optimizer_detection_cache`,
   config.py:3421), which constructs
   `DetectionCacheBuilderWorker(video_path, cache_path, params, ...)`
   (`core/tracking/optimization/optimizer_workers.py:330`).
3. `DetectionCacheBuilderWorker.run()` (optimizer_workers.py:372) calls
   `detector = create_detector(self.params)` directly — **not** through
   `InferenceRunner`. `create_detector` (`core/detectors/factory.py:24`)
   dispatches on `params["DETECTION_METHOD"]`; for `"yolo_obb"` it returns
   `YOLOOBBDetector(params)`. The `_build_optimizer_detection_cache` code
   path (config.py:3400-3403) explicitly branches on
   `detection_method == "yolo_obb"` to name the optimizer cache file
   (`opt_model_name = f"yolo_{model}"`), confirming this path is designed to
   support (and is regularly exercised for) `"yolo_obb"`.
4. `YOLOOBBDetector.__init__` → `_detect_device()`
   (`core/detectors/yolo_detector.py:82-102`) reads `params["YOLO_DEVICE"]`
   directly to pick the torch device (falling back to auto-detect only if
   `"auto"`). `_load_model()` (yolo_detector.py:140-153) reads
   `params["ENABLE_TENSORRT"]` and `params["ENABLE_ONNX_RUNTIME"]` directly
   to decide whether to load an ONNX/TensorRT-accelerated model instead of
   the native ultralytics model. These are not inert legacy fields here —
   they actively drive which backend gets loaded for this feature's
   detection-only scan.

The in-repo comment at config.py:1637 ("TensorRT (legacy detector: cache
build / preview / benchmark only)") is literally describing this — it names
exactly the "cache build" consumer I found, not just a backward-compat
display field. So the fields are load-bearing for a real, intentional,
documented (if legacy) feature — not dead weight.

## Why I believe my replacement is still safe (but this needs a human/controller decision, hence BLOCKED not just a note)

I already replaced all three `derive_detection_runtime_settings` call sites
with a new local helper `legacy_detection_runtime_fields()` (added to
`trackerkit/cli_config.py`, imported by `config.py` and `tracking.py`). For
every canonical runtime string the old function's `_normalize_runtime` could
produce (`cpu`, `mps`, `cuda`, `tensorrt`, `onnx_cpu`, `onnx_cuda`,
`onnx_coreml`), my replacement returns **identical** `yolo_device` /
`enable_tensorrt` / `enable_onnx_runtime` / `enable_gpu_background` values —
so for the DetectionCacheBuilderWorker/YOLOOBBDetector consumer, behavior for
those runtime strings is unchanged.

The one input where behavior differs is the literal string `"coreml"`
(distinct from `"onnx_coreml"`) — which `_selected_compute_runtime()`
(`trackerkit/gui/orchestrators/session.py:997-1003`) really can produce, via
`resolve_compute_runtime(tier, platform, stage="obb")`, when
`RUNTIME_TIER == "gpu_fast"` on Apple hardware with a CoreML artifact
available (`core/inference/runtime.py:130`). The **old**
`derive_detection_runtime_settings` collapsed `"coreml"` into `"onnx_coreml"`
via `_normalize_runtime` (compute_runtime.py:68-71), which would have set
`enable_onnx_runtime=True, yolo_device="mps"` — i.e., for a user on the
gpu_fast/CoreML tier who then runs the Parameter Helper's detection-only
scan, `YOLOOBBDetector` would previously have tried to load an
ONNX-Runtime-exported model with the CoreML execution provider. My new
helper instead treats `"coreml"` like plain `"mps"`
(`enable_onnx_runtime=False`), so `YOLOOBBDetector` would load the native
ultralytics/torch model on `mps` instead.

I believe this is a **behavior fix, not a regression** — it is precisely the
"vocabulary-collapse bug" this whole plan is otherwise eliminating, and the
old ONNX-forcing behavior for a native `gpu_fast`/CoreML selection looks like
exactly the kind of accidental cross-talk the plan wants gone. But:
- I have not run this exact scenario end-to-end (no Apple GPU-Fast CoreML
  artifact + Parameter Helper detection-only-scan integration test exists).
- This is a **user-visible behavior change** to a real, working feature
  (the "legacy detector" cache-build/preview/benchmark path), not just a
  cache-key or config-file cosmetic change, contradicting the scope the
  controller described for this task.
- The decision of whether "fix this bug incidentally as part of Task 8" vs.
  "preserve exact legacy behavior and file this as a separate follow-up" is
  a product/scope call I should not make unilaterally, per the controller's
  explicit stop condition.

## Everything else I completed (all still in the uncommitted working tree)

1. **Verified `create_detector`/`yolo_obb` tracing**: confirmed for
   `TrackingWorker` (both GUI-preview and CLI/headless — same class, same
   code path). Confirmed the second consumer above, which the controller
   did not know about.
2. **Replaced the three (actually five — two more turned up in `config.py`
   at lines 519/526, feeding `spin_tensorrt_batch`/`lbl_tensorrt_batch`
   `.setEnabled(...)`, which the brief/controller's grep hit list did not
   call out explicitly but are additional live call sites of the same
   function) `derive_detection_runtime_settings` call sites**:
   - Added `legacy_detection_runtime_fields(compute_runtime: str) -> dict`
     to `src/hydra_suite/trackerkit/cli_config.py` (colocated helper, single
     definition imported by the two GUI orchestrator modules — chose one
     shared helper over 3 duplicated inline copies since `cli_config.py`
     already sits below both `config.py` and `tracking.py` in the import
     graph and this avoids drift).
   - `cli_config.py:421` (now uses the local helper directly, no import
     needed).
   - `config.py:519,526,1625,2130` → `legacy_detection_runtime_fields(...)`.
   - `tracking.py:3846` → `legacy_detection_runtime_fields(safe_rt)`.
   - Exact helper code:
     ```python
     def legacy_detection_runtime_fields(compute_runtime: str) -> dict:
         rt = str(compute_runtime or "cpu").strip().lower()
         yolo_device = "cpu"
         enable_tensorrt = False
         enable_onnx_runtime = False
         if rt in ("mps", "coreml"):
             yolo_device = "mps"
         elif rt == "cuda":
             yolo_device = "cuda:0"
         elif rt == "tensorrt":
             yolo_device = "cuda:0"
             enable_tensorrt = True
         elif rt == "onnx_coreml":
             yolo_device = "mps"
             enable_onnx_runtime = True
         elif rt == "onnx_cpu":
             yolo_device = "cpu"
             enable_onnx_runtime = True
         elif rt == "onnx_cuda":
             yolo_device = "cuda:0"
             enable_onnx_runtime = True
         return {
             "yolo_device": yolo_device,
             "enable_tensorrt": bool(enable_tensorrt),
             "enable_onnx_runtime": bool(enable_onnx_runtime),
             "enable_gpu_background": yolo_device != "cpu",
         }
     ```
   - Added unit tests in `tests/test_trackerkit_cli_config.py` covering
     tensorrt/onnx_cuda/onnx_coreml/cpu, plus a dedicated test asserting
     `"coreml"` is *not* collapsed into `"onnx_coreml"`.
3. **Deleted `derive_detection_runtime_settings`** from
   `runtime/compute_runtime.py`, and its export from `runtime/__init__.py`.
   Deleted the corresponding 4 tests in `tests/test_compute_runtime.py`
   (they tested only the deleted function directly).
4. **Deleted `build_runtime_config`** from `core/identity/pose/api.py` —
   re-confirmed via grep it has zero real (non-test, non-monkeypatch)
   callers anywhere in `src/` or `tests/`. Also deleted `_norm_hw`,
   `_resolve_device_and_batch`, `_resolve_export_hw` (now-dead helpers that
   existed only to support `build_runtime_config`), and pruned now-unused
   imports (`Any`, `Dict`, `Sequence`, `Tuple`, `Optional`,
   `load_skeleton_from_json`, `derive_sleap_export_input_hw`,
   `derive_pose_runtime_settings`, `clamp_realtime_individual_batch_size`).
   `create_pose_backend_from_config` (still live, still called directly by
   `posekit/gui/workers.py`, `trackerkit/benchmarking.py`,
   `trackerkit/gui/workers/crops_worker.py`,
   `trackerkit/gui/workers/preview_worker.py`) was **not** touched.
   - Removed the `build_runtime_config` re-export from
     `core/identity/__init__.py` and `core/identity/pose/__init__.py`.
   - `core/inference/api.py`'s "stable shim" `try/except ImportError` block
     re-exported both `build_runtime_config` and
     `create_pose_backend_from_config` from `pose/api.py` but had **zero**
     real consumers of either name via that module (only
     `apply_detection_filter` is actually imported from
     `core/inference/api.py` elsewhere, by
     `core/tracking/optimization/{optimizer.py,optimizer_workers.py}`) — I
     removed `build_runtime_config` from that shim (kept
     `create_pose_backend_from_config`, which is at least structurally
     still exported for future-proofing even though nothing currently
     imports it from that module).
   - Deleted the 4 tests in `tests/test_runtime_api_sleap_export.py` that
     called `mod.build_runtime_config(...)` directly (they tested a function
     that no longer exists); removed the now-unused `json` import from that
     test file as a result.
   - Fixed 2 test files that monkeypatched `pose_api.build_runtime_config`
     to a raising stub as a "must not call this legacy path" guard
     (`tests/test_trackerkit_preview_worker.py` x2,
     `tests/test_interpolated_crops_worker.py` x2) — since the attribute no
     longer exists, `monkeypatch.setattr(pose_api, "build_runtime_config",
     _boom)` would raise `AttributeError` at setup time. Replaced each with
     `assert not hasattr(pose_api, "build_runtime_config")`, preserving the
     tests' intent (the legacy path cannot be called because it does not
     exist) without breaking the fixture.
   - Cleaned a dead stub assignment
     (`pose_api.build_runtime_config = lambda...`) in a hand-built fake
     module in `tests/test_tracking_worker_helpers.py` (harmless but unused
     after the deletion; not exercised by any code path in that test).
5. **`YOLOOBBDetector` deletion decision: do NOT delete.** Confirmed via
   `grep -rln "YOLOOBBDetector"` across `src/` and `tests/` that it has many
   live, non-test callers beyond the parity/regression tests the brief
   anticipated: `core/detectors/factory.py` (`create_detector`),
   `core/tracking/ingest/detection_phase.py`,
   `core/tracking/ingest/streaming_payload.py`,
   `trackerkit/gui/dialogs/model_test_dialog.py`,
   `trackerkit/gui/workers/preview_worker.py`, plus the
   `DetectionCacheBuilderWorker`/Parameter-Helper consumer described above
   and `core/tracking/optimization/optimizer_workers.py`/
   `data/dataset_generation.py` (both call `create_detector` too). This is
   unambiguous: leave `YOLOOBBDetector` (and `ObjectDetector`, confirmed
   still live for the background-subtraction detection method) in place.
   `build_runtime_config` deletion decision: delete — genuinely zero real
   callers (only 4 direct-unit-test calls, now removed, and monkeypatch
   guards in 3 other test files, now adjusted).
6. **Test suite**: ran `make pytest`; `tests/test_identity_postprocess.py`
   fails to even collect (`AttributeError: module
   'identity_postprocess_under_test' has no attribute
   'apply_identity_postprocessing'`) — confirmed via `git stash` that this
   failure is **pre-existing on `HEAD` (6fc7bea)**, unrelated to any Task 8
   change. A full run excluding that file was in progress when I stopped to
   write this report (see "what's not yet done" below).
7. Ran `pyflakes` over every file I touched — clean except one pre-existing
   `F821` in `core/inference/api.py` (`PoseResult` forward-ref string
   annotation, present before my edit, unrelated).

## What's NOT done because I stopped

- Have not run `make format` / `make lint` on the full diff.
- Have not confirmed the full `make pytest` run (minus the pre-existing
  collection error) passes end-to-end — a background run was still in
  progress when I stopped to write this report.
- Have not run `make dead-code`.
- Have not committed anything.

## Recommendation

I recommend the controller (or a human) make an explicit call on the
`"coreml"` behavior change described above, then either:
- (a) confirm it's an acceptable/intended bug fix and have me finish
  (finish the pytest run, `make format && make lint`, commit), or
- (b) ask me to make `legacy_detection_runtime_fields("coreml")` bit-for-bit
  match the old collapsed-into-`onnx_coreml` behavior instead (trivial:
  drop `"coreml"` from the `("mps", "coreml")` tuple and add an explicit
  `elif rt == "coreml": yolo_device = "mps"; enable_onnx_runtime = True`
  branch) to strictly preserve legacy behavior for this one real consumer,
  deferring the vocabulary-collapse fix to a separate, explicitly-scoped
  change.

All other parts of this task (the `derive_detection_runtime_settings` and
`build_runtime_config` deletions, the `YOLOOBBDetector` keep-decision) are
unaffected by this open question and are, in my judgment, correct and ready
as-is.
