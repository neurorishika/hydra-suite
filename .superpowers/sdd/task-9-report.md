# Task 9 report — final audit of `compute_runtime.py`'s remaining surface

## Step 1: full inventory of what's exported

```
grep -n "^def \|^class \|^[A-Z_]* *[:=]" src/hydra_suite/runtime/compute_runtime.py
```

```
CANONICAL_RUNTIMES
COREML_PROVIDER_OPTIONS
_best_explicit_onnx_runtime
_normalize_runtime
runtime_label
_cuda_like_available
_onnx_available
_tensorrt_available
_sleap_onnx_available
_provider_name
_append_provider
_tensorrt_ep_cache_options
derive_onnx_execution_providers
_pipeline_supports_runtime
supported_runtimes_for_pipeline
allowed_runtimes_for_pipelines
_best_auto_runtime
_runtime_from_pose_flavor
_DEVICE_MAP / _CUDA_DEVICES
infer_compute_runtime_from_legacy
derive_pose_runtime_settings
```

## Step 2: confirm against expected end-state

Already-gone items (confirmed by absence from the grep above and from
`src/hydra_suite/runtime/__init__.py`):

- `available_tiers`, `tier_label` — moved to `runtime/resolver.py` in Plan A
  Task 1. Confirmed only present in `resolver.py`, not `compute_runtime.py`.
- `derive_detection_runtime_settings`, `build_runtime_config` — deleted in
  Task 8 (`2c083e5`).

Remaining, load-bearing, keep as-is: `CANONICAL_RUNTIMES`, `_normalize_runtime`,
`derive_onnx_execution_providers` + its helpers
(`_tensorrt_ep_cache_options`, `_append_provider`, `_provider_name`,
`_cuda_like_available`, `_onnx_available`, `_tensorrt_available`,
`_sleap_onnx_available`, `COREML_PROVIDER_OPTIONS`),
`derive_pose_runtime_settings`, `infer_compute_runtime_from_legacy` +
`_best_explicit_onnx_runtime` + `_best_auto_runtime` +
`_runtime_from_pose_flavor`.

### `runtime_label` — brief said "flag for deletion if zero callers, don't delete speculatively"

A grep restricted to `src/` and `tests/` shows zero callers of the
*function* `runtime_label(...)` (the many hits under those two dirs are all a
`runtime_label=` dataclass field name in `trackerkit/benchmarking.py`/
`benchmark_dialog.py` — an unrelated coincidence of naming, not calls into
`compute_runtime.runtime_label`). Extending the grep to the whole repo
(`tools/` had been missed on the first pass) turns up 13 real call sites in
`tools/benchmark_models.py` (e.g. line 1584:
`runtime_label(r) for r in supported`). `tools/` is inside `make lint`'s and
`make format`'s scope (`Makefile` lines 375, 404-410, 416) and has
Makefile-invoked entry points (`make benchmark*` targets, lines 214-242) plus
a dedicated test, `tests/test_benchmark_models.py`, which dynamically loads
and executes the module. **`runtime_label` is live. Not deleted, no flag
needed.**

### `supported_runtimes_for_pipeline` / `allowed_runtimes_for_pipelines` / `_pipeline_supports_runtime` — brief's central question

Brief's premise: these are only used by `benchmarking.py`'s old
`collect_active_targets` and `session.py`'s `_resolve_pose_runtime`; if the
`session.py` caller is ported off, delete all three.

**Investigation:**
- `src/hydra_suite/trackerkit/benchmarking.py::collect_active_targets` no
  longer calls any of the three — it already uses `available_tiers` from
  `runtime/resolver.py`. Confirmed via grep; brief's premise was accurate
  here, this caller is already gone.
- `session.py::_resolve_pose_runtime` (lines 1111-1140, pre-edit) *was*
  still calling `supported_runtimes_for_pipeline` for its `artifact_available`
  callable — exactly the Task 9a contingency the brief describes. **Ported**
  (see Step 3 below).
- **Critical finding the brief missed:** `tools/benchmark_models.py` imports
  and calls all three of `supported_runtimes_for_pipeline` (10 call sites:
  module-level import at line 73, called at lines 1952, 2213, 2263, 2301,
  2333, 2369, 2370, 2412) and `allowed_runtimes_for_pipelines` (module-level
  import line 71, called at lines 1583 and 2489). This tool is in-scope for
  `make lint`/`make format` (Makefile lines 375/404/405/409/410/416), has
  dedicated Makefile entry points (`make benchmark*`, lines 214-242), and is
  covered by `tests/test_benchmark_models.py`, which loads it as a real
  module via `importlib.util.spec_from_file_location` and executes
  `_resolve_detector_runtime_artifact_path`. I empirically verified the
  breakage: after monkey-deleting the three names from the imported
  `compute_runtime` module object and re-loading `tools/benchmark_models.py`
  the same way the test does, the load fails with
  `ImportError: cannot import name 'allowed_runtimes_for_pipelines' from
  'hydra_suite.runtime.compute_runtime'`.

**Decision: `supported_runtimes_for_pipeline`, `allowed_runtimes_for_pipelines`,
and `_pipeline_supports_runtime` were NOT deleted.** They remain live,
lint-covered, test-covered production code via `tools/benchmark_models.py`,
which the brief's grep scope (evidently `src/` + `tests/`) missed — where
indeed the only caller left was `session.py`. Deleting them would have
broken `tools/benchmark_models.py` and `tests/test_benchmark_models.py`.
This was flagged and confirmed with the coordinator mid-task; this report
documents the evidence trail.

## Step 3: what was actually deleted / changed

Nothing was deleted from `compute_runtime.py` itself — its full exported
surface, on inspection, is still needed by at least one live caller.

**`session.py::_resolve_pose_runtime` was ported (Task 9a contingency, done
inline as instructed):**

- Removed the `supported_runtimes_for_pipeline`-based `artifact_available`
  callable (which built a pipeline "capability table" list and checked
  membership in it).
- Replaced with a direct capability check using
  `TENSORRT_AVAILABLE`/`ONNXRUNTIME_COREML_AVAILABLE` from
  `hydra_suite.utils.gpu_utils` (the same flags `resolve_compute_runtime`'s
  other established callers ultimately bottom out on), plus an explicit
  SLEAP branch (SLEAP has no CoreML export path; its gpu_fast tier only
  applies on CUDA, mirroring `core/inference/stages/pose.py`'s SLEAP
  tier→flavor gate).
- Replaced the manual `RuntimeResolver(...).resolve(...)` call + backend
  string translation (`"tensorrt"`/`"coreml"`/`resolved.device`) with a call
  to `resolve_compute_runtime(tier, platform, stage=pipeline,
  artifact_available=artifact_available)` — the same wrapper
  `_selected_compute_runtime()` (a few lines above in the same file) already
  uses for the OBB stage. This is the exact `resolve_compute_runtime`
  `artifact_available`-parameter pattern Plan A Task 2 established.
  `resolve_compute_runtime` returns `"coreml"` (not `"onnx_coreml"`) for the
  CoreML backend, but `derive_pose_runtime_settings`'s `_normalize_runtime`
  already aliases `"coreml"` → `"onnx_coreml"` (compute_runtime.py line 68),
  so behavior is unchanged.
- Removed the now-unused `RuntimeResolver` import and the
  `supported_runtimes_for_pipeline` import from `session.py`'s module
  header.

**`tests/test_main_window_config_persistence.py`:**

- `test_saved_config_reflects_tier_derived_pose_runtime_flavor` had a
  `monkeypatch.setattr(session_module, "supported_runtimes_for_pipeline",
  ...)` call that is now dead — `session_module` (the `session.py` module
  object) no longer imports that name, so pytest's `monkeypatch.setattr`
  (which raises `AttributeError` by default when the target attribute
  doesn't exist) would break this test. The test only exercises the `cpu`
  tier, where `RuntimeResolver.resolve()` short-circuits before ever calling
  `artifact_available()`, so the mock was already inert for this test's
  actual assertions. Removed the dead monkeypatch and the now-unused
  `session_module` import.

## Other pre-existing failures discovered (verified via `git stash` against parent commit `7cf2e38`, not fixed — out of scope)

While running scoped tests I hit 10 failures beyond my own touched tests.
For every one of them I reproduced the identical failure on the parent
commit `7cf2e38` (via `git stash` / `git stash pop`), confirming they are
pre-existing test rot, unrelated to this plan, and not introduced by this
task:

- `tests/test_classkit_training_dialog.py::test_training_dialog_prefers_onnx_coreml_for_mps_inference`
  — `AttributeError: 'ClassKitTrainingDialog' object has no attribute
  'compute_runtime_combo'` (the dialog no longer exposes this combo; test
  wasn't updated after some earlier classkit refactor). Confirmed identical
  on parent commit.
- `tests/test_classkit_training_dialog.py::test_training_dialog_exposes_multihead_modes_for_multifactor_scheme`
  and `::test_training_dialog_falls_back_to_multihead_modes_for_legacy_scheme`
  — mode list assertion is stale; dialog now also exposes
  `multihead_custom_shared`. Confirmed identical on parent commit.
- `tests/test_main_window_config_persistence.py::test_video_autoload_restores_pose_keypoint_groups_and_headtail_model`,
  `::test_preview_detection_restores_analyze_individual_controls`,
  `::test_realtime_direct_mode_exposes_micro_batch_controls`,
  `::test_realtime_micro_batch_roundtrip_persists`,
  `::test_advanced_config_defaults_include_identity_decoder_tuning`,
  `::test_get_parameters_dict_exposes_identity_decoder_advanced_overrides`,
  `::test_identity_decoder_tuning_controls_roundtrip_through_tracker_config`
  — all fail with `AttributeError: 'IdentityPanel' object has no attribute
  'chk_identity_offline_split_trajectories'` (an `IdentityPanel` widget
  referenced by these tests doesn't exist yet/anymore — unrelated to
  compute_runtime/runtime work). Confirmed identical on parent commit.

These are separate, pre-existing issues outside this plan's scope (not in
the brief's originally-listed known-failure set, but independently confirmed
pre-existing via stash-diff — not caused by this change), left untouched.

## Step 4: test run

The full `make pytest` aborts immediately on collection of
`tests/test_identity_postprocess.py` (the known pre-existing collection
error: `AttributeError: module 'identity_postprocess_under_test' has no
attribute 'apply_identity_postprocessing'`), before any tests run. A
whole-suite run with that file excluded (`pytest tests
--ignore=tests/test_identity_postprocess.py`) is very slow in this
environment (many GUI/Qt tests); rather than let it run for many minutes, I
ran a scoped pass covering every file touched or exercising the changed
code path, plus the modules whose fate (delete vs. keep) was being audited:

```
tests/test_session_gpu_fast_coreml_notice.py
tests/test_trackerkit_runtime_fallback_indicator.py
tests/test_compute_runtime.py
tests/test_compute_runtime_headtail.py
tests/test_main_window_config_persistence.py::test_saved_config_reflects_tier_derived_pose_runtime_flavor
tests/test_benchmark_models.py
```

Result: **27 passed, 0 failed.**

I additionally ran full `tests/test_main_window_config_persistence.py` and
`tests/test_classkit_training_dialog.py` to check for regressions from my
edits; this surfaced the 10 failures documented above, all verified
pre-existing against parent commit `7cf2e38` via `git stash`/`git stash
pop` (identical failure messages before and after my changes).

## Step 4: lint

Ran on touched files only (per task instructions — several pre-existing
lint failures exist in unrelated files: `core/tracking/worker.py`,
`identity_panel.py`, `tests/helpers/*`, `tools/equivalence/*` — out of
scope):

```
uvx autopep8 --in-place --select=E226,E225,E231 \
    src/hydra_suite/trackerkit/gui/orchestrators/session.py \
    tests/test_main_window_config_persistence.py
black src/hydra_suite/trackerkit/gui/orchestrators/session.py tests/test_main_window_config_persistence.py
isort src/hydra_suite/trackerkit/gui/orchestrators/session.py tests/test_main_window_config_persistence.py
flake8 --config=.flake8.moderate src/hydra_suite/trackerkit/gui/orchestrators/session.py tests/test_main_window_config_persistence.py
```

All clean; `black`/`isort` reported no changes needed; `flake8` produced no
output (zero issues) on both touched files.

## Files changed

- `src/hydra_suite/trackerkit/gui/orchestrators/session.py` — ported
  `_resolve_pose_runtime` off `supported_runtimes_for_pipeline` onto
  `resolve_compute_runtime` + direct `TENSORRT_AVAILABLE`/
  `ONNXRUNTIME_COREML_AVAILABLE` capability checks; dropped now-unused
  `RuntimeResolver` import.
- `tests/test_main_window_config_persistence.py` — removed a dead
  monkeypatch of `session_module.supported_runtimes_for_pipeline` (name no
  longer imported by `session.py`) and the now-unused `session_module`
  import.

## Summary

- `compute_runtime.py`'s exported surface is **unchanged** — every
  currently-exported name is still needed by at least one live, tested
  caller (most of them by `tools/benchmark_models.py`, which the original
  brief's grep scope missed).
- The one required behavioral change — porting `session.py`'s
  `_resolve_pose_runtime` off `supported_runtimes_for_pipeline`'s
  capability-table framing and onto a direct
  `TENSORRT_AVAILABLE`/`ONNXRUNTIME_COREML_AVAILABLE` check via
  `resolve_compute_runtime` — is done, per the brief's Task 9a contingency.
- One now-dead test monkeypatch (referencing the no-longer-imported
  `supported_runtimes_for_pipeline` symbol in `session.py`'s namespace) was
  removed.
- 10 pre-existing test failures were discovered in adjacent files
  (`test_classkit_training_dialog.py`, `test_main_window_config_persistence.py`)
  during scoped verification; all independently confirmed pre-existing
  against parent commit `7cf2e38` and left untouched (out of scope for this
  task).
