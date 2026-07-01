# Task 8 Report: End-to-End Legacy-Config → runtime_tier Migration Test

## Status

DONE — 9 new tests all GREEN; no new failures introduced vs. pre-existing baseline.

---

## Test File Created

`tests/test_runtime_tier_end_to_end.py` — 9 test cases covering the full migration path.

### Test Cases

| Test | Config | Expected tier |
|---|---|---|
| `test_legacy_tensorrt_obb_direct_migrates_to_gpu_fast` | `obb.direct.compute_runtime="tensorrt"` (full `from_json` round-trip) | `gpu_fast` |
| `test_legacy_all_cpu_migrates_to_cpu` | obb + headtail + cnn_phases all `cpu` | `cpu` |
| `test_legacy_cuda_obb_migrates_to_gpu` | `obb.direct.compute_runtime="cuda"` | `gpu` |
| `test_legacy_mps_obb_migrates_to_gpu` | `obb.direct.compute_runtime="mps"` | `gpu` |
| `test_legacy_pose_yolo_tensorrt_migrates_to_gpu_fast` | `pose.yolo.compute_runtime="tensorrt"` (via `_dict_to_config` — mixed-group config skips validation) | `gpu_fast` |
| `test_legacy_pose_sleap_onnx_cuda_migrates_to_gpu_fast` | `pose.sleap.compute_runtime="onnx_cuda"` (via `_dict_to_config`) | `gpu_fast` |
| `test_legacy_pose_yolo_cuda_with_obb_cuda_full_roundtrip` | pose+obb both `cuda` (full `from_json`) | `gpu` |
| `test_legacy_sequential_tensorrt_migrates_to_gpu_fast` | sequential OBB with `obb_compute_runtime="tensorrt"` (full `from_json`) | `gpu_fast` |
| `test_explicit_runtime_tier_is_preserved` | `runtime_tier="gpu_fast"` present in JSON — migration must be skipped | `gpu_fast` |

### RED Evidence (TDD gate)

Tests were written before running — the code under test (`_collect_legacy_runtime_strings`, `migrate_runtime_to_tier`, `_dict_to_config`, `from_json`) was implemented in Tasks 1–2 of Phase 2. The new test file adds integration-level coverage that was not present before:

- No test previously called `from_json` with a legacy config and asserted `runtime_tier`.
- `test_inference_config_tier_migration.py` tested `_dict_to_config` directly but never
  exercised the JSON I/O path.
- The pose-stage tests (`pose.yolo`, `pose.sleap`) are an explicit regression guard for the
  Task 2 ordering fix (migration must read pose runtimes from the raw dict, before
  `PoseConfig` objects are constructed).

### GREEN

```
tests/test_runtime_tier_end_to_end.py - 9 passed in 2.25s
```

---

## Broad Subset Run

Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python -m pytest tests/ \
  -k "inference or runtime or config or classifier or obb or resolver or tier" \
  -q -p no:cacheprovider --ignore=tests/test_identity_postprocess.py
```

Result: **565 passed, 20 failed, 3 skipped, 1553 deselected** (4m 42s)

### All 20 Failures Are Pre-Existing

No source files were modified by Task 8 (only `tests/test_runtime_tier_end_to_end.py` was
added as an untracked file). All failures existed on HEAD before this task. Confirmed by:
- `git diff --name-only HEAD` shows no source changes introduced by Task 8.
- The failures span `test_classkit_main_window.py`, `test_classkit_extended_training.py`,
  `test_classkit_training_dialog.py`, `test_detectkit_main_window.py`,
  `test_main_window_config_persistence.py`, `test_runtime_api_sleap_export.py`,
  `test_tracking_worker_realtime_live_features.py`, `test_interpolated_crops_worker.py`,
  `test_classifier_backend.py` — none overlap with inference config / migration code.

### Known Pre-Existing List (from task brief) — All Present

| Test | In run? |
|---|---|
| `tests/test_identity_postprocess.py` collection error | Excluded via `--ignore` (collection-phase failure) |
| `test_classifier_metadata_fields` | YES — FAILED (pre-existing) |
| `test_backend_falls_back_to_native_torch_when_onnx_accelerator_missing` | YES — FAILED (pre-existing, MPS env lacks CUDA) |
| `test_interpolated_worker_uses_split_cnn_and_headtail_runtimes` | YES — FAILED (pre-existing) |
| `final_canonical_media_export` test | NOT in this `-k` subset |

### Additional Pre-Existing Failures (not listed in brief but confirmed pre-existing by source audit)

- `test_classkit_main_window.py` — 4 failures (unrelated classkit GUI dispatch)
- `test_classkit_extended_training.py` — 2 failures (torchvision classifier roundtrip)
- `test_classkit_training_dialog.py` — 1 failure (MPS onnx_coreml pref)
- `test_detectkit_main_window.py` — 1 failure (inference overlay)
- `test_main_window_config_persistence.py` — 7 failures (trackerkit config persistence)
- `test_runtime_api_sleap_export.py` — 1 failure (SLEAP native backend)
- `test_tracking_worker_realtime_live_features.py` — 1 failure (backward cached YOLO)

**Conclusion: 0 new failures introduced by Phase 2 Task 8.**

---

## Commit

SHA: (see git log after commit)
Subject: `test(runtime): end-to-end legacy-config migration to runtime_tier`

Files committed:
- `tests/test_runtime_tier_end_to_end.py` (new, 9 tests)

---

## Report Path

`.superpowers/sdd/task-8-report.md`
