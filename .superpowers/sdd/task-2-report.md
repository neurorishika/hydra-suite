# Task 2 Report: `InferenceConfig.runtime_tier` + Hard-Cutover Legacy Migration

---

## TDD RED/GREEN Evidence

### RED (Step 2)

```
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python -m pytest tests/test_inference_config_tier_migration.py -v

ERROR collecting tests/test_inference_config_tier_migration.py
ImportError: cannot import name 'migrate_runtime_to_tier' from 'hydra_suite.core.inference.config'
1 error
```

Confirmed fail as expected — function did not exist yet.

### GREEN (Step 4)

```
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python -m pytest tests/test_inference_config_tier_migration.py tests/test_inference_config.py -v

tests/test_inference_config_tier_migration.py::test_cpu_maps_to_cpu PASSED
tests/test_inference_config_tier_migration.py::test_cuda_and_mps_map_to_gpu PASSED
tests/test_inference_config_tier_migration.py::test_onnx_and_tensorrt_map_to_gpu_fast PASSED
tests/test_inference_config_tier_migration.py::test_mixed_takes_highest_tier PASSED
tests/test_inference_config_tier_migration.py::test_empty_defaults_to_gpu PASSED
tests/test_inference_config.py::test_from_json_round_trip PASSED
tests/test_inference_config.py::test_round_trip_with_headtail PASSED
tests/test_inference_config.py::test_round_trip_with_cnn_phases PASSED
tests/test_inference_config.py::test_runtime_validation_rejects_cuda_cpu_mix PASSED
tests/test_inference_config.py::test_runtime_validation_accepts_cuda_group PASSED
tests/test_inference_config.py::test_runtime_validation_accepts_cpu_group PASSED
tests/test_inference_config.py::test_from_json_validates_on_load PASSED
tests/test_inference_config.py::test_sequential_config_round_trip PASSED

13 passed in 2.20s
```

---

## Files Changed

- **Modified**: `src/hydra_suite/core/inference/config.py`
  - Added `import logging` and `from hydra_suite.runtime.resolver import RuntimeTier`
  - Added module-level `migrate_runtime_to_tier(runtimes: set[str]) -> RuntimeTier` — maps legacy runtime strings to tier; empty set → "gpu"
  - Added module-level `_collect_legacy_runtime_strings(d: dict) -> set[str]` — reads raw dict sub-keys (obb.direct, obb.sequential, headtail, cnn_phases, pose.yolo, pose.sleap)
  - Added `runtime_tier: RuntimeTier = "gpu"` field to `InferenceConfig` dataclass (after `pipeline_depth`)
  - Wired `_dict_to_config` to derive `runtime_tier` from legacy per-stage runtimes when absent, with one-line warning log

- **Created**: `tests/test_inference_config_tier_migration.py`
  - 5 tests: cpu, cuda/mps, onnx/tensorrt, mixed, empty-set

---

## `test_inference_config.py` Adjustments

No adjustments required. All 8 existing tests passed without modification. The new `runtime_tier` field has a default of `"gpu"`, so round-trip tests that write configs (which include `runtime_tier` in the dict) load back correctly. The `_dict_to_config` migration path is exercised transparently.

---

## Self-Review

- `migrate_runtime_to_tier(set())` returns "gpu" (special-cased with early return before any set intersection logic).
- `migrate_runtime_to_tier` priority order: gpu_fast > gpu > cpu (consistent with the brief).
- Migration only logs when legacy runtimes are actually present (not for empty-set / new configs).
- Per-stage `compute_runtime` fields are preserved intact (Task 3 will remove them).
- `RuntimeTier` imported from `hydra_suite.runtime.resolver` — no local redefinition.
- pre-commit hooks (black, ruff, flake8, isort) all passed.

---

## Concerns

None. All 13 tests pass, no regressions.

---

## Commit

`633ddab feat(config): add runtime_tier with hard-cutover legacy migration`

---

## Review Fix: Pose Sub-Dict Mutation Before Legacy Runtime Collection

### Root Cause

In `_dict_to_config`, `pose_d.pop("yolo", None)` and `pose_d.pop("sleap", None)` were called at lines 324–325 — before the `_collect_legacy_runtime_strings(d)` call at line 342. Because `_collect_legacy_runtime_strings` reads `d["pose"]["yolo"]` and `d["pose"]["sleap"]` from the raw dict, those keys were already gone by the time it ran. A legacy config with `pose.yolo.compute_runtime="tensorrt"` would therefore miss the pose runtime, see only `{'cpu'}` (from the OBB stage), and yield `runtime_tier='cpu'` instead of `'gpu_fast'`.

### Fix Applied

Moved the `runtime_tier` derivation block (the `_collect_legacy_runtime_strings(d)` call + `migrate_runtime_to_tier` + warning log) to BEFORE the `pose_d.pop()` mutations in `_dict_to_config`. The `apriltag` construction was left in its original relative position after the pose block. No logic was changed — only execution order.

**File**: `src/hydra_suite/core/inference/config.py` (`_dict_to_config`)

### RED Evidence (new tests, before fix)

```
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python -m pytest tests/test_inference_config_tier_migration.py -v

FAILED tests/test_inference_config_tier_migration.py::test_pose_yolo_tensorrt_migrates_to_gpu_fast
  AssertionError: Expected 'gpu_fast', got 'cpu'
  WARNING: Migrated legacy per-stage runtimes {'cpu'} -> runtime_tier='cpu'

FAILED tests/test_inference_config_tier_migration.py::test_pose_sleap_onnx_cuda_migrates_to_gpu_fast
  AssertionError: Expected 'gpu_fast', got 'cpu'
  WARNING: Migrated legacy per-stage runtimes {'cpu'} -> runtime_tier='cpu'

2 failed, 5 passed
```

### GREEN Evidence (after fix)

```
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python -m pytest tests/test_inference_config_tier_migration.py tests/test_inference_config.py -v

tests/test_inference_config_tier_migration.py::test_pose_yolo_tensorrt_migrates_to_gpu_fast PASSED
tests/test_inference_config_tier_migration.py::test_pose_sleap_onnx_cuda_migrates_to_gpu_fast PASSED
tests/test_inference_config_tier_migration.py::test_cpu_maps_to_cpu PASSED
tests/test_inference_config_tier_migration.py::test_cuda_and_mps_map_to_gpu PASSED
tests/test_inference_config_tier_migration.py::test_onnx_and_tensorrt_map_to_gpu_fast PASSED
tests/test_inference_config_tier_migration.py::test_mixed_takes_highest_tier PASSED
tests/test_inference_config_tier_migration.py::test_empty_defaults_to_gpu PASSED
tests/test_inference_config.py::test_from_json_round_trip PASSED
tests/test_inference_config.py::test_round_trip_with_headtail PASSED
tests/test_inference_config.py::test_round_trip_with_cnn_phases PASSED
tests/test_inference_config.py::test_runtime_validation_rejects_cuda_cpu_mix PASSED
tests/test_inference_config.py::test_runtime_validation_accepts_cuda_group PASSED
tests/test_inference_config.py::test_runtime_validation_accepts_cpu_group PASSED
tests/test_inference_config.py::test_from_json_validates_on_load PASSED
tests/test_inference_config.py::test_sequential_config_round_trip PASSED

15 passed in 2.28s
```

### Commit

`a8e36b5 fix(config): collect legacy pose runtimes before pose_d mutation in _dict_to_config`
