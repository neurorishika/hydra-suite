# Direct Detect/Segment + Runtime Fixes — Post-Retirement Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the completed detector-retirement `main` into `feature/obb-direct-detect-segment`, re-apply the direct detect/segment feature onto the relocated `core/inference` structure, apply the five runtime fixes, and leave the branch merge-ready.

**Architecture:** `main` relocated `_direct_obb_runtime.py` → `core/inference/direct_executors.py`, deleted `TrackingWorker._build_inference_config_from_params` in favor of `config.build_inference_config_from_params`, and migrated preview to `InferenceRunner.run_realtime`. The branch's feature was authored against the *old* locations, so this plan does a controlled `git merge main`, resolves the three structural conflicts by deterministic rules (append segment executors; re-home seg-param plumbing into the config builder; discard the now-deleted preview clone), then layers the fixes on the merged code with TDD.

**Tech Stack:** Python 3, ultralytics/torch/torchvision/numpy, PySide6 (untouched here), pytest. Env: `hydra-mps` (this box) / `hydra-cuda` (mehek).

## Global Constraints

- **No pipeline behavior change** except the four intended fixes. `run_realtime` / `run_batch_pass` / `detect_batch` output for the OBB task must stay byte-identical.
- **No imports of `core/detectors`** anywhere — the package no longer exists. Final gate: `grep -rn "core.detectors\|core/detectors\|_direct_obb_runtime" src/ tests/` → empty.
- **Runtime tiering is fixed:** cpu→`cpu`, gpu→`cuda`/`mps`, gpu_fast→`tensorrt`/`coreml`. No new ONNX *stage* runtime; ONNX direct executors remain tool-only.
- **`seg_*` kernel defaults must stay in sync** across `OBBDirectConfig` (config.py), the `_run_direct` fallbacks (stages/obb.py), and `build_inference_config_from_params` clamps — all mirror `rotated_rect_from_masks` defaults (num_angles=24, crop_size=64, pad_ratio=0.15, mask_threshold=0.5).
- Run `make format` before each commit. Tests: `python -m pytest <path> -q` (benchmarks excluded by default). Use the `hydra-mps` interpreter: `/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python`.
- Commit as the configured git user; **no** `Co-Authored-By: Claude` trailer.
- Pre-existing base-suite failures exist (~7 in `test_main_window_config_persistence.py`, unrelated `IdentityPanel`). Gate on the *delta*, not absolute green.

---

## File Structure

**Merge-resolution targets (Phase 0):**
- `src/hydra_suite/core/inference/direct_executors.py` — append segment executors (main already has OBB + detect).
- `src/hydra_suite/core/inference/runtime_artifacts.py` — add `segment` dispatch to `_create_direct_executor`.
- `src/hydra_suite/core/inference/config.py` — union `OBBDirectConfig` fields; re-home seg-param plumbing into `build_inference_config_from_params`.
- `src/hydra_suite/core/inference/stages/obb.py` — bring the branch's direct-mode `model_task` routing, mask extraction, and torch-path task guard.
- `src/hydra_suite/core/tracking/worker.py` — accept `main`'s method deletion; salvage nothing but the clamp *logic* (moves to config.py).
- `src/hydra_suite/trackerkit/gui/workers/preview_worker.py` — accept `main`'s `run_realtime` migration; discard branch preview edits.
- `src/hydra_suite/trackerkit/tracking_cache.py`, `trackerkit/gui/orchestrators/config.py`, `trackerkit/gui/panels/detection_panel.py`, `src/hydra_suite/utils/obb_from_mask.py` — keep branch versions (no `main` conflict).
- Tests: keep branch's `test_direct_obb_runtime_segment.py`, `test_inference_stages_obb.py`, `test_utils_obb_from_mask.py`, `test_inference_config.py`, `test_coreml_segment_smoke.py`; delete `test_detectors_engine.py` (deleted on `main`).

**Fix targets (Phases 1–2):**
- `direct_executors.py` — thread `iou_threshold` into `_decode_segment_predictions` (L1056 equiv) + the three `is_end2end else 0.5` sites (L265/876/932 equiv).
- `runtime_artifacts.py` — resolve imgsz instead of `_DEFAULT_IMGSZ` fallback; add export-time task validation + unverifiable warning.
- `config.py` / `runtime.py` — tier documentation comment on `_direct_runtime_name`.
- Tests: `tests/test_inference_direct_executors_iou.py` (new), `tests/test_inference_detect_task_smoke.py` (new).

---

## Phase 0 — Merge `main` and resolve conflicts

### Task 0.1: Create integration branch and run the merge

**Files:** none (git operations).

- [ ] **Step 1: Snapshot current branch state**

Run:
```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/obb-direct-detect-segment
git status --short && git rev-parse HEAD && git log -1 --oneline main
```
Expected: clean tree (the design + this plan already committed); note the branch SHA and main SHA.

- [ ] **Step 2: Create an integration branch off the feature branch**

Run:
```bash
git checkout -b integration/obb-direct-postretirement
```
Expected: switched to a new branch. (Keeps `feature/obb-direct-detect-segment` intact as a fallback.)

- [ ] **Step 3: Merge main (expect conflicts)**

Run:
```bash
git merge --no-commit --no-ff main
git status --short | grep -E '^(UU|AA|DU|UD|DD|AU|UA)' | sort
```
Expected: a conflict list. Anticipated conflicted/■delete-modify paths:
`config.py`, `runtime_artifacts.py`, `stages/obb.py`, `core/tracking/worker.py`, `trackerkit/gui/workers/preview_worker.py`, and possibly `direct_executors.py` (rename-follow of `_direct_obb_runtime.py`), plus test files. Record the exact list — later tasks resolve each.

- [ ] **Step 4: Do NOT commit yet**

Leave the merge in progress. Each following task resolves specific paths and stages them. If the merge state is lost, `git merge --abort` and restart from Step 3.

---

### Task 0.2: Resolve `direct_executors.py` — append the segment executors

**Files:**
- Modify (resolve): `src/hydra_suite/core/inference/direct_executors.py`

**Interfaces:**
- Consumes: `_BaseDirectOBBExecutor` (already on main).
- Produces: `create_direct_segment_executor(*, runtime, artifact_path, imgsz, class_names=None, class_count=None)`, class `DirectTensorRTSegmentExecutor`, function `_decode_segment_predictions(preds, protos, *, img_tensor_shape, orig_shape, conf_thres, classes, max_det, nc)`.

- [ ] **Step 1: Confirm what main already has**

Run:
```bash
git show main:src/hydra_suite/core/inference/direct_executors.py | grep -n "class DirectONNXDetectExecutor\|class DirectTensorRTDetectExecutor\|def create_direct_detect_executor\|def create_direct_segment_executor\|class DirectTensorRTSegmentExecutor\|def _decode_segment_predictions"
```
Expected: the two Detect executors + `create_direct_detect_executor` are PRESENT on main; `create_direct_segment_executor`, `DirectTensorRTSegmentExecutor`, `_decode_segment_predictions` are ABSENT. Only the segment trio must be added.

- [ ] **Step 2: Extract the segment code from the branch's old file**

The segment additions live in the pre-merge feature file. Retrieve them verbatim:
```bash
git show feature/obb-direct-detect-segment:src/hydra_suite/core/detectors/_direct_obb_runtime.py | sed -n '1005,1261p' > /tmp/segment_block.py
```
This is `_decode_segment_predictions` (1005–1103), `DirectTensorRTSegmentExecutor` (1106–1235), and `create_direct_segment_executor` (1238–1261).

- [ ] **Step 3: Resolve the file**

If git flagged `direct_executors.py` as conflicted, accept **main's** content as the base (`git checkout --theirs` semantics for the non-segment body), then append the segment block. If git auto-merged it (rename-followed the branch edits cleanly), verify the segment trio is present and correctly placed after `create_direct_detect_executor`. Concretely:
- Ensure `_decode_segment_predictions`, `DirectTensorRTSegmentExecutor`, `create_direct_segment_executor` exist exactly once.
- Ensure they import from within `direct_executors` (no `core/detectors` references).
- Ensure `DirectTensorRTSegmentExecutor` subclasses `_BaseDirectOBBExecutor` (NOT the OBB TRT executor — segment engines have two outputs).

- [ ] **Step 4: Syntax + import check**

Run:
```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -c "import ast,sys; ast.parse(open('src/hydra_suite/core/inference/direct_executors.py').read()); print('parse-ok')"
grep -n "core.detectors\|_direct_obb_runtime" src/hydra_suite/core/inference/direct_executors.py
```
Expected: `parse-ok`; second grep empty.

- [ ] **Step 5: Stage**

```bash
git add src/hydra_suite/core/inference/direct_executors.py
```

---

### Task 0.3: Resolve `runtime_artifacts.py` — segment dispatch

**Files:**
- Modify (resolve): `src/hydra_suite/core/inference/runtime_artifacts.py`

**Interfaces:**
- Consumes: `create_direct_segment_executor` (Task 0.2).
- Produces: `_create_direct_executor(*, runtime, artifact_path, imgsz, class_names=None, task="obb")` dispatching `task=="segment"` → `create_direct_segment_executor`.

- [ ] **Step 1: Inspect both sides of `_create_direct_executor`**

```bash
git show main:src/hydra_suite/core/inference/runtime_artifacts.py | sed -n '217,264p'
git show feature/obb-direct-detect-segment:src/hydra_suite/core/inference/runtime_artifacts.py | sed -n '217,264p'
```
Expected: main dispatches `detect` vs `obb` only; branch adds `elif task == "segment"`.

- [ ] **Step 2: Resolve to the union**

In the merged `_create_direct_executor`: import `create_direct_segment_executor` alongside the other two factories, and add the dispatch arm:
```python
    if task == "detect":
        factory = create_direct_detect_executor
    elif task == "segment":
        factory = create_direct_segment_executor
    else:
        factory = create_direct_obb_executor
```
Keep main's call site (forwards `runtime`, `artifact_path=str(...)`, `imgsz`, `class_names`).

- [ ] **Step 3: Verify no other conflict markers remain in the file**

Run:
```bash
grep -n "^<<<<<<<\|^=======\|^>>>>>>>" src/hydra_suite/core/inference/runtime_artifacts.py
```
Expected: empty.

- [ ] **Step 4: Stage**

```bash
git add src/hydra_suite/core/inference/runtime_artifacts.py
```

---

### Task 0.4: Resolve `OBBDirectConfig` — union the fields

**Files:**
- Modify (resolve): `src/hydra_suite/core/inference/config.py` (`OBBDirectConfig` dataclass, main lines 34–43)

**Interfaces:**
- Produces: `OBBDirectConfig` with fields `model_path`, `confidence_floor=1e-3`, `confidence_threshold=0.25`, `auto_export=True`, `model_task: Literal["obb","detect","segment"]="obb"`, `fixed_angle_deg: float=0.0`, `seg_num_angles: int=24`, `seg_crop_size: int=64`, `seg_pad_ratio: float=0.15`, `seg_mask_threshold: float=0.5`.

- [ ] **Step 1: Write the failing round-trip test**

The branch already ships `tests/test_inference_config.py` with these assertions; run its config-serialization subset first to drive the resolution:
```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_config.py -q -k "model_task or seg" 2>&1 | tail -15
```
Expected (pre-resolution): FAIL/ERROR — the merged `OBBDirectConfig` may lack the new fields, or the merge left markers.

- [ ] **Step 2: Resolve the dataclass to the union**

Take main's `OBBDirectConfig` (which has `auto_export` and dropped the deprecated `compute_runtime`) and add the six branch fields (`model_task`, `fixed_angle_deg`, `seg_num_angles`, `seg_crop_size`, `seg_pad_ratio`, `seg_mask_threshold`) with the defaults above and their explanatory comments. **Do NOT reintroduce `compute_runtime`** — the retirement removed it. Ensure `Literal` is imported (it is, for `OBBConfig.mode`).

- [ ] **Step 3: Guard legacy-preset deserialization**

`_dict_to_config` reconstructs via `OBBDirectConfig(**obb_d["direct"])` (main line 305). A legacy preset that still carries `"compute_runtime"` would raise `TypeError: unexpected keyword`. Add a filter right before reconstruction:
```python
        direct_d = dict(obb_d["direct"])
        direct_d.pop("compute_runtime", None)  # legacy field removed in retirement
        direct = OBBDirectConfig(**direct_d)
```
(Replace the existing `OBBDirectConfig(**obb_d["direct"])` expression.)

- [ ] **Step 4: Run the config test to green**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_config.py -q 2>&1 | tail -15
```
Expected: PASS (round-trip of `model_task` + seg params; default-to-obb; legacy `compute_runtime` ignored).

- [ ] **Step 5: Stage (do not commit — config builder resolved next)**

```bash
git add src/hydra_suite/core/inference/config.py tests/test_inference_config.py
```

---

### Task 0.5: Re-home seg-param plumbing into `build_inference_config_from_params`

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py` (`build_inference_config_from_params`, direct `else:` branch, main lines 405–418 reads + 479–496 construction)
- Modify (resolve): `src/hydra_suite/core/tracking/worker.py` (accept `main`'s deletion of `_build_inference_config_from_params`)
- Test: `tests/test_inference_config_from_params.py` (extend if present on main, else create)

**Interfaces:**
- Consumes: `OBBDirectConfig` (Task 0.4).
- Produces: `build_inference_config_from_params(params)` that reads `YOLO_OBB_DIRECT_TASK`, `YOLO_OBB_FIXED_ANGLE_DEG`, `YOLO_OBB_SEG_NUM_ANGLES`, `YOLO_OBB_SEG_CROP_SIZE`, `YOLO_OBB_SEG_PAD_RATIO`, `YOLO_OBB_SEG_MASK_THRESHOLD` (clamped) and folds them into the direct-mode `OBBDirectConfig`.

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_inference_config_from_params.py` (or append if it exists):
```python
def test_direct_seg_params_flow_into_obbdirectconfig():
    from hydra_suite.core.inference.config import build_inference_config_from_params
    params = {
        "YOLO_OBB_MODE": "direct",
        "YOLO_MODEL_PATH": "m.pt",
        "YOLO_OBB_DIRECT_TASK": "segment",
        "YOLO_OBB_SEG_NUM_ANGLES": 48,
        "YOLO_OBB_SEG_CROP_SIZE": 128,
        "YOLO_OBB_SEG_PAD_RATIO": 0.25,
        "YOLO_OBB_SEG_MASK_THRESHOLD": 0.6,
    }
    cfg = build_inference_config_from_params(params)
    d = cfg.obb.direct
    assert d.model_task == "segment"
    assert (d.seg_num_angles, d.seg_crop_size) == (48, 128)
    assert abs(d.seg_pad_ratio - 0.25) < 1e-9
    assert abs(d.seg_mask_threshold - 0.6) < 1e-9

def test_direct_seg_params_are_clamped_and_default():
    from hydra_suite.core.inference.config import build_inference_config_from_params
    params = {
        "YOLO_OBB_MODE": "direct", "YOLO_MODEL_PATH": "m.pt",
        "YOLO_OBB_DIRECT_TASK": "segment",
        "YOLO_OBB_SEG_NUM_ANGLES": 9999,   # over max 180 -> default 24
        "YOLO_OBB_SEG_PAD_RATIO": "nope",  # non-finite -> default 0.15
    }
    d = build_inference_config_from_params(params).obb.direct
    assert d.seg_num_angles == 24
    assert abs(d.seg_pad_ratio - 0.15) < 1e-9

def test_direct_task_defaults_to_obb_when_unset():
    from hydra_suite.core.inference.config import build_inference_config_from_params
    d = build_inference_config_from_params(
        {"YOLO_OBB_MODE": "direct", "YOLO_MODEL_PATH": "m.pt"}).obb.direct
    assert d.model_task == "obb"
    assert abs(d.fixed_angle_deg) < 1e-9
```

- [ ] **Step 2: Run it — expect failure**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_config_from_params.py -q -k "direct_seg or direct_task" 2>&1 | tail -15
```
Expected: FAIL — the direct branch on main doesn't set `model_task`/seg fields yet.

- [ ] **Step 3: Add the clamp helpers + reads (ported from the deleted worker method)**

In `config.py`, ensure `import math` is present. Inside `build_inference_config_from_params`, in the direct-mode `else:` branch (before the `OBBDirectConfig(...)` construction), insert:
```python
        model_task = str(params.get("YOLO_OBB_DIRECT_TASK", "obb")).strip().lower()
        if model_task not in {"obb", "detect", "segment"}:
            model_task = "obb"
        fixed_angle_deg = float(params.get("YOLO_OBB_FIXED_ANGLE_DEG", 0.0) or 0.0)

        def _clamped_int(raw, default, lo, hi):
            try:
                v = int(raw)
            except (TypeError, ValueError):
                return default
            return v if lo <= v <= hi else default

        def _clamped_float(raw, default, lo, hi):
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return default
            return v if math.isfinite(v) and lo <= v <= hi else default

        seg_num_angles = _clamped_int(params.get("YOLO_OBB_SEG_NUM_ANGLES", 24), 24, 4, 180)
        seg_crop_size = _clamped_int(params.get("YOLO_OBB_SEG_CROP_SIZE", 64), 64, 16, 256)
        seg_pad_ratio = _clamped_float(params.get("YOLO_OBB_SEG_PAD_RATIO", 0.15), 0.15, 0.0, 1.0)
        seg_mask_threshold = _clamped_float(params.get("YOLO_OBB_SEG_MASK_THRESHOLD", 0.5), 0.5, 0.05, 0.95)
```
Then extend the `OBBDirectConfig(...)` constructor in that branch to pass:
```python
            direct=OBBDirectConfig(
                model_path=direct_model_path,
                confidence_floor=1e-3,
                confidence_threshold=yolo_conf,
                model_task=model_task,
                fixed_angle_deg=fixed_angle_deg,
                seg_num_angles=seg_num_angles,
                seg_crop_size=seg_crop_size,
                seg_pad_ratio=seg_pad_ratio,
                seg_mask_threshold=seg_mask_threshold,
            ),
```

- [ ] **Step 4: Resolve `worker.py` — accept the deletion**

The branch modified `worker._build_inference_config_from_params` (a delete/modify conflict since main deleted it). Accept main's deletion entirely (the logic now lives in config.py):
```bash
grep -n "_build_inference_config_from_params" src/hydra_suite/core/tracking/worker.py
```
If any conflict markers or a residual method body remain, remove them so the method is gone and the one call site uses `build_inference_config_from_params(p)` (main's form). Verify:
```bash
grep -n "_build_inference_config_from_params\|build_inference_config_from_params" src/hydra_suite/core/tracking/worker.py
grep -n "^<<<<<<<\|^=======\|^>>>>>>>" src/hydra_suite/core/tracking/worker.py
```
Expected: only the module-function call `build_inference_config_from_params(...)` remains; no markers.

- [ ] **Step 5: Run the parity test to green**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_config_from_params.py -q 2>&1 | tail -15
```
Expected: PASS.

- [ ] **Step 6: Stage**

```bash
git add src/hydra_suite/core/inference/config.py src/hydra_suite/core/tracking/worker.py tests/test_inference_config_from_params.py
```

---

### Task 0.6: Resolve `stages/obb.py` — bring the direct-mode routing + torch task guard

**Files:**
- Modify (resolve): `src/hydra_suite/core/inference/stages/obb.py`

**Interfaces:**
- Consumes: `config.direct.model_task`, `config.direct.fixed_angle_deg`, `config.direct.seg_*`; `rotated_rect_from_masks` (utils).
- Produces: `_run_direct` routing on `model_task` (obb/detect/segment); `_extract_obb_from_masks(result, frame_idx, raw_detection_cap=0, *, num_angles=24, crop_size=64, pad_ratio=0.15, mask_threshold=0.5)`; `_assert_direct_task_matches_checkpoint(model, model_task, model_path)`.

- [ ] **Step 1: Confirm main has no direct-mode routing**

```bash
git show main:src/hydra_suite/core/inference/stages/obb.py | grep -n "model_task\|_extract_obb_from_masks\|_assert_direct_task"
```
Expected: empty on main — all of this is branch-new and must be present after resolution.

- [ ] **Step 2: Resolve to include the branch's direct-mode code**

Ensure the merged `stages/obb.py` contains, from the branch: `_run_direct` reading `model_task = config.direct.model_task if config.direct else "obb"` and dispatching detect/segment/obb; `_extract_obb_from_masks` + `_extract_raw_tensors_from_masks`; `_extract_obb_from_boxes` (detect fixed-angle); `_assert_direct_task_matches_checkpoint` and its call in `load_obb_models`'s direct branch (loading with `task=config.direct.model_task`). Keep main's sequential-mode code unchanged (stage-1 `task="detect"`). If main and branch both edited `load_obb_models`, take the union: main's structure + the branch's `task=`/guard lines.

- [ ] **Step 3: Verify markers gone + kernel-default sync**

```bash
grep -n "^<<<<<<<\|^=======\|^>>>>>>>" src/hydra_suite/core/inference/stages/obb.py
grep -n "num_angles: int = 24\|crop_size: int = 64\|pad_ratio: float = 0.15\|mask_threshold: float = 0.5" src/hydra_suite/core/inference/stages/obb.py
```
Expected: no markers; the `_run_direct` seg fallbacks and `_extract_obb_from_masks` signature use the canonical defaults (24/64/0.15/0.5).

- [ ] **Step 4: Stage**

```bash
git add src/hydra_suite/core/inference/stages/obb.py
```

---

### Task 0.7: Resolve `preview_worker.py` and tests

**Files:**
- Modify (resolve): `src/hydra_suite/trackerkit/gui/workers/preview_worker.py`
- Delete: `tests/test_detectors_engine.py`
- Keep: branch inference tests

- [ ] **Step 1: Accept main's preview migration; discard branch clone edits**

`main` replaced the preview YOLO branch with `InferenceRunner.run_realtime` and deleted the clone functions the branch had edited (delete/modify conflict). Accept main's version wholesale:
```bash
git checkout main -- src/hydra_suite/trackerkit/gui/workers/preview_worker.py
grep -n "_advanced_config_value\|_preview_run_yolo_raw_detection\|_extract_obb_from_masks\|core.detectors" src/hydra_suite/trackerkit/gui/workers/preview_worker.py
```
Expected: run_realtime-based file; grep empty (no clone, no seg reach-ins, no legacy imports). Parity is now structural — preview builds the same `InferenceConfig` and calls `run_realtime`, so `YOLO_OBB_SEG_*` reach the kernel identically to the tracking run.

- [ ] **Step 2: Delete the retired detector test**

```bash
git rm -f tests/test_detectors_engine.py 2>/dev/null || git rm tests/test_detectors_engine.py
```
Expected: staged deletion (matches main). If the branch also edited it, this resolves the delete/modify by taking the deletion.

- [ ] **Step 3: Repoint any file-path test imports**

Some branch tests import direct executors by the old path. Fix them:
```bash
grep -rln "core/detectors/_direct_obb_runtime\|core.detectors._direct_obb_runtime\|detectors import" tests/
```
For each hit, repoint to `hydra_suite.core.inference.direct_executors`. Re-run the grep; expected empty.

- [ ] **Step 4: Stage**

```bash
git add src/hydra_suite/trackerkit/gui/workers/preview_worker.py tests/
```

---

### Task 0.8: Complete the merge and checkpoint tests

**Files:** none new.

- [ ] **Step 1: Verify no conflict markers anywhere**

```bash
grep -rn "^<<<<<<<\|^=======\|^>>>>>>>" src/ tests/ | grep -v Binary
git diff --name-only --diff-filter=U
```
Expected: both empty (all conflicts resolved/staged).

- [ ] **Step 2: Legacy-reference gate**

```bash
grep -rn "core.detectors\|core/detectors\|_direct_obb_runtime" src/ tests/
```
Expected: empty.

- [ ] **Step 3: Commit the merge**

```bash
make format
git add -A
git commit --no-edit
```
Expected: merge commit created.

- [ ] **Step 4: Run the affected suites**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest \
  tests/test_inference_config.py tests/test_inference_config_from_params.py \
  tests/test_inference_stages_obb.py tests/test_utils_obb_from_mask.py \
  tests/test_direct_obb_runtime_segment.py tests/test_inference_obb_artifacts.py -q 2>&1 | tail -25
```
Expected: PASS (segment/detect routing, config round-trip, mask kernel). Any failure here is a merge-resolution defect — fix before proceeding. Do NOT continue to fixes until this is green.

---

## Phase 1 — Fix: thread `iou_threshold` through TensorRT decode

### Task 1.1: Make the direct executors honor the configured IoU

**Files:**
- Modify: `src/hydra_suite/core/inference/direct_executors.py` (the four hardcoded sites; main L265/876/932 + segment L1056-equiv)
- Modify: `src/hydra_suite/core/inference/runtime_artifacts.py` (`DirectExecutorAdapter.predict` already receives `iou`; thread it to the executor)
- Test: `tests/test_inference_direct_executors_iou.py` (new)

**Interfaces:**
- Consumes: existing `executor.predict(frames, *, conf_thres, classes, max_det)` — extend with `iou_thres`.
- Produces: `_decode_segment_predictions(..., iou_thres: float = 0.5)` and each `_postprocess` honoring a passed-through `iou_thres`, preserving the `is_end2end → 1.0` override.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_direct_executors_iou.py
"""iou_thres passed by the caller must reach non_max_suppression, not the
hardcoded 0.5. Uses a monkeypatched NMS to capture the kwarg (no GPU/engine)."""
import types
import numpy as np
import hydra_suite.core.inference.direct_executors as de


def test_segment_decode_uses_caller_iou(monkeypatch):
    captured = {}

    def fake_nms(pred, *args, **kwargs):
        captured["iou_thres"] = kwargs.get("iou_thres", args[2] if len(args) > 2 else None)
        # Return an empty detection set to short-circuit the rest.
        import torch
        return [torch.zeros((0, 6 + 32))]

    monkeypatch.setattr(de.nms, "non_max_suppression", fake_nms)
    import torch
    preds = torch.zeros((1, 32 + 4 + 1, 10))
    protos = torch.zeros((1, 32, 8, 8))
    de._decode_segment_predictions(
        preds, protos, img_tensor_shape=(1, 3, 64, 64), orig_shape=(64, 64),
        conf_thres=0.25, classes=None, max_det=100, nc=1, iou_thres=0.33,
    )
    assert abs(captured["iou_thres"] - 0.33) < 1e-9
```

- [ ] **Step 2: Run — expect failure**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_direct_executors_iou.py -q 2>&1 | tail -15
```
Expected: FAIL — `_decode_segment_predictions` has no `iou_thres` param (TypeError) or still passes 0.5.

- [ ] **Step 3: Add `iou_thres` to the segment decoder**

In `_decode_segment_predictions`, add `iou_thres: float = 0.5` to the signature and replace the hardcoded `iou_thres=0.5` in the `non_max_suppression(...)` call with the parameter.

- [ ] **Step 4: Thread it through the detect/OBB `_postprocess` sites**

For `_BaseDirectOBBExecutor._postprocess`, `DirectONNXDetectExecutor._postprocess`, `DirectTensorRTDetectExecutor._postprocess`, and `DirectTensorRTSegmentExecutor._postprocess`: accept an optional `iou_thres` (plumbed from `predict`), and change `iou_thres = 1.0 if is_end2end else 0.5` to `iou_thres = 1.0 if is_end2end else (caller_iou if caller_iou is not None else 0.5)`. Add an `iou_thres: float | None = None` kwarg to each executor's `predict(...)`.

- [ ] **Step 5: Thread from the adapter**

In `runtime_artifacts.py` `DirectExecutorAdapter.predict`, forward the received `iou` kwarg into `self._executor.predict(..., iou_thres=iou)`.

- [ ] **Step 6: Run to green**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_direct_executors_iou.py tests/test_direct_obb_runtime_segment.py -q 2>&1 | tail -15
```
Expected: PASS (new test + existing segment decode tests still pass — the default remains 0.5 when no caller value).

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/core/inference/direct_executors.py src/hydra_suite/core/inference/runtime_artifacts.py tests/test_inference_direct_executors_iou.py
git commit -m "fix(inference): honor configured iou_threshold on TensorRT direct decode"
```

---

## Phase 2 — Remaining fixes

### Task 2.1: Resolve imgsz for user-supplied artifacts instead of defaulting to 640

**Files:**
- Modify: `src/hydra_suite/core/inference/runtime_artifacts.py` (`_load_direct_executor`, the `imgsz = _DEFAULT_IMGSZ` fallback at main L596)
- Test: `tests/test_inference_obb_artifacts.py` (extend)

**Interfaces:**
- Consumes: `_resolve_imgsz(pt_path)` (existing) and `imgsz_override` (existing param on `load_obb_executor`).
- Produces: `_load_direct_executor` raising `ArtifactExportError` when imgsz is unresolvable for a prebuilt `.engine`/`.onnx` and no override is given, instead of silently using 640.

- [ ] **Step 1: Write the failing test**

```python
def test_prebuilt_engine_without_imgsz_raises_not_defaults_640(monkeypatch, tmp_path):
    from hydra_suite.core.inference import runtime_artifacts as ra
    from hydra_suite.core.inference.runtime_artifacts import ArtifactExportError
    engine = tmp_path / "m.engine"
    engine.write_bytes(b"stub")
    # No .pt sibling, no override, no resolvable imgsz -> must raise, not letterbox at 640.
    import pytest
    with pytest.raises(ArtifactExportError):
        ra._load_direct_executor(
            model_path=str(engine), compute_runtime="tensorrt",
            imgsz_override=None, task="obb",
        )
```
(Adjust the call to `_load_direct_executor`'s real signature confirmed in Task 0; keep the intent: prebuilt artifact + unresolvable imgsz ⇒ raise.)

- [ ] **Step 2: Run — expect failure**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_obb_artifacts.py -q -k "prebuilt_engine_without_imgsz" 2>&1 | tail -15
```
Expected: FAIL — currently returns a 640-letterbox executor instead of raising.

- [ ] **Step 3: Replace the silent default**

In `_load_direct_executor`, where `imgsz = _DEFAULT_IMGSZ` is used for the prebuilt-artifact path: prefer `imgsz_override`, then `_resolve_imgsz` (for a `.pt` source), and if the artifact is a prebuilt `.engine`/`.onnx` with no resolvable size and no override, `raise ArtifactExportError(f"Cannot resolve imgsz for prebuilt artifact {model_path!r}; pass imgsz_override.")`. Keep `_DEFAULT_IMGSZ` only for the auto-export-from-.pt path where `_resolve_imgsz` genuinely applies.

- [ ] **Step 4: Run to green**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_obb_artifacts.py -q 2>&1 | tail -15
```
Expected: PASS (new test + existing artifact tests).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/runtime_artifacts.py tests/test_inference_obb_artifacts.py
git commit -m "fix(inference): resolve imgsz for prebuilt artifacts instead of silent 640 default"
```

---

### Task 2.2: Honest task validation for direct artifacts

> **Design refinement (flag for reviewer):** the original design proposed setting `DirectExecutorAdapter.task` from the requested `model_task`. That is tautological — the same `task` string drives both executor construction and the check, so it can never detect a mismatch. The valuable, honest fix is: validate the requested task against the **source `.pt` checkpoint** at export/load time (where `.task` truly exists — the auto-export path loads the `.pt` anyway), and emit a **one-time warning** when only a prebuilt `.engine`/`.mlpackage` is supplied and the task cannot be verified. This supersedes the design's fix #3 wording.

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py` (`load_obb_models` direct branch — call the existing torch guard against the source checkpoint before/around export)
- Modify: `src/hydra_suite/core/inference/runtime_artifacts.py` (emit the unverifiable-task warning on the prebuilt path)
- Test: `tests/test_inference_stages_obb.py` (extend the existing task-mismatch tests)

**Interfaces:**
- Consumes: `_assert_direct_task_matches_checkpoint(model, model_task, model_path)` (present from Task 0.6).
- Produces: a `logging.warning(...)` on the prebuilt-artifact path stating the requested task cannot be verified against a task-less artifact.

- [ ] **Step 1: Write the failing test**

```python
def test_prebuilt_artifact_task_unverifiable_warns(monkeypatch, caplog):
    # When a .engine (no .task) is loaded for a direct segment config, we cannot
    # verify the task; assert we warn rather than silently proceed.
    from hydra_suite.core.inference import runtime_artifacts as ra
    import logging
    caplog.set_level(logging.WARNING)
    # Drive _load_direct_executor with a stub executor whose object has no .task,
    # with task="segment"; expect a warning mentioning "cannot verify".
    # (Wire via the same monkeypatch pattern used elsewhere in this file.)
    ...
    assert any("verify" in r.message.lower() and "task" in r.message.lower()
               for r in caplog.records)
```
(Fill the `...` using this file's existing executor-stub monkeypatch pattern, confirmed while resolving Task 0.6.)

- [ ] **Step 2: Run — expect failure**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_stages_obb.py -q -k "unverifiable" 2>&1 | tail -15
```
Expected: FAIL — no warning emitted today.

- [ ] **Step 3: Implement**

- In `load_obb_models`' direct branch, keep the existing `_assert_direct_task_matches_checkpoint` call for the torch path (source `.pt` has `.task`).
- On the prebuilt-artifact direct path (`.engine`/`.onnx`/`.mlpackage`, executor lacks `.task`), emit `logger.warning("Direct %s artifact %r carries no task metadata; cannot verify it matches configured model_task=%r.", runtime, model_path, model_task)`.
- Keep the existing torch-path mismatch `ValueError` behavior unchanged.

- [ ] **Step 4: Run to green**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_stages_obb.py -q 2>&1 | tail -20
```
Expected: PASS (existing mismatch accept/reject tests + the new warning test).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/stages/obb.py src/hydra_suite/core/inference/runtime_artifacts.py tests/test_inference_stages_obb.py
git commit -m "fix(inference): warn when direct artifact task is unverifiable; keep torch-path guard"
```

---

### Task 2.3: Document the runtime-tier → executor-runtime boundary

**Files:**
- Modify: `src/hydra_suite/core/inference/runtime_artifacts.py` (`_direct_runtime_name`, main L264–270)

- [ ] **Step 1: Add the clarifying comment**

Above `_direct_runtime_name`, add:
```python
# Maps a resolved compute_runtime to the direct-executor runtime name. The
# inference stage only ever reaches this with the gpu_fast tier, so it emits
# "tensorrt" exclusively; the "onnx" direct executors exist for the diagnostic
# tools (tools/diag_*, compare_runtimes.py) and are never selected by the
# stage pipeline. Runtime tiers: cpu->cpu, gpu->cuda/mps, gpu_fast->tensorrt/coreml.
```

- [ ] **Step 2: Sanity check imports still load**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -c "import hydra_suite.core.inference.runtime_artifacts; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/hydra_suite/core/inference/runtime_artifacts.py
git commit -m "docs(inference): clarify runtime-tier -> direct-executor mapping and ONNX tool-only scope"
```

---

### Task 2.4: Real detect-task smoke test (MPS + CUDA) with synthetic data

**Files:**
- Create: `tests/test_inference_detect_task_smoke.py`
- Modify: `tests/test_coreml_segment_smoke.py` (tighten the zero-detection pass + broad export-error swallow)

**Interfaces:**
- Consumes: `hydra_suite.core.inference.config.build_obb_only_config`, `hydra_suite.core.inference.stages.obb.load_obb_models`/`run_obb` (or `InferenceRunner.detect_batch`).

- [ ] **Step 1: Write the device-parametrized real test**

```python
# tests/test_inference_detect_task_smoke.py
"""Real detect-task YOLO run through the production load/run path on each
available device. Synthetic image built to reliably yield >=1 detection so
assertions actually fire (not skipped on empty output)."""
import numpy as np
import pytest

pytest.importorskip("ultralytics")
import torch


def _available_devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def _synthetic_frame_with_objects():
    # White field with three dark filled rectangles -> a COCO detect model
    # ("person"/"cell phone"-agnostic) reliably fires on high-contrast blobs.
    img = np.full((640, 640, 3), 255, np.uint8)
    for (x, y) in [(120, 120), (400, 300), (250, 480)]:
        img[y:y + 90, x:x + 60] = (20, 20, 20)
    return img


@pytest.mark.parametrize("device", _available_devices())
def test_detect_task_direct_runs_end_to_end(device, tmp_path):
    from ultralytics import YOLO
    from hydra_suite.core.inference.config import build_obb_only_config
    from hydra_suite.core.inference.runner import InferenceRunner

    # Download a real detect checkpoint (cached by ultralytics).
    _ = YOLO("yolo11n.pt")

    tier = "cpu" if device == "cpu" else "gpu"
    cfg = build_obb_only_config("yolo11n.pt", compute_runtime=device, mode="direct")
    # Force detect-task direct mode + a permissive confidence so blobs register.
    cfg.obb.direct.model_task = "detect"
    cfg.obb.direct.confidence_threshold = 0.05
    cfg.obb.confidence_threshold = 0.05

    runner = InferenceRunner(cfg)
    try:
        results = runner.detect_batch([_synthetic_frame_with_objects()], frame_indices=[0])
    finally:
        runner.close()

    assert len(results) == 1
    res = results[0]
    # Real detections OR a well-formed empty result; but on this high-contrast
    # frame with conf=0.05 we expect at least one box on cpu.
    assert res.centroids.shape[1] == 2
    assert np.isfinite(res.centroids).all()
    if device == "cpu":
        assert res.centroids.shape[0] >= 1, "expected >=1 detection on synthetic blobs"
```
(Confirm `build_obb_only_config`'s `compute_runtime`/`runtime_tier` argument names against Task 0's reference and adjust; if `detect_batch`'s `OBBResult` uses `.angles`/`.shapes`/`.corners`, assert those shapes instead.)

- [ ] **Step 2: Run — expect real execution (not skip)**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_detect_task_smoke.py -q 2>&1 | tail -20
```
Expected: at least the `cpu` (and on this box `mps`) parametrization runs and PASSES; `cuda` param is deselected via the device list, not silently skipped inside the test. If it fails, treat as a real bug in the detect-direct path (systematic-debugging), not a test-only issue.

- [ ] **Step 3: Tighten the segment smoke test**

In `tests/test_coreml_segment_smoke.py`: (a) make the zero-detection case assert a well-formed empty `OBBResult` (correct shapes/dtypes) rather than passing vacuously; (b) narrow the export-error `except` so it only swallows genuinely-environmental failures (missing coremltools / Xcode toolchain) and re-raises anything else, so a real export regression surfaces.

- [ ] **Step 4: Run the segment smoke test**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_coreml_segment_smoke.py -q 2>&1 | tail -15
```
Expected: PASS on Apple Silicon (this box); the environmental-skip path still skips cleanly off-platform.

- [ ] **Step 5: Commit**

```bash
make format
git add tests/test_inference_detect_task_smoke.py tests/test_coreml_segment_smoke.py
git commit -m "test(inference): real detect-task smoke on mps/cuda; tighten segment smoke assertions"
```

---

## Phase 3 — Verification & merge preparation

### Task 3.1: Full delta gate + parity run

**Files:** none.

- [ ] **Step 1: Legacy-reference and marker gates**

```bash
grep -rn "core.detectors\|core/detectors\|_direct_obb_runtime" src/ tests/ ; echo "exit=$?"
grep -rn "^<<<<<<<\|^=======\|^>>>>>>>" src/ tests/ ; echo "exit=$?"
```
Expected: both greps print nothing.

- [ ] **Step 2: Format + lint**

```bash
make format && make lint-moderate 2>&1 | tail -20
```
Expected: clean format; no new moderate lint errors vs base.

- [ ] **Step 3: Full suite (delta gate)**

```bash
/Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest -m "not benchmark" -q 2>&1 | tail -25
```
Expected: only the known ~7 pre-existing `test_main_window_config_persistence.py` failures (IdentityPanel). No new failures. If new failures appear, debug before merge.

- [ ] **Step 4: Equivalence/parity run (byte-identical OBB tracking)**

Per CLAUDE.md's equivalence harness, run the fast smoke clips (`fly_obb`, `worm_bgsub`) baseline-vs-current on MPS, confirming the merge introduced no OBB-path drift:
```bash
conda activate hydra-mps
bash tools/equivalence/fixtures/fetch_fixtures.sh   # once
git worktree add --detach ../.equiv-legacy legacy/main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/../.equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_integ RUNTIME=mps bash tools/equivalence/run_matrix.sh fly_obb worm_bgsub
git worktree remove --force ../.equiv-legacy && git worktree prune
```
Expected: EQUIVALENCE at/near the DETERMINISM floor for both clips (positions p99≈0, θ max≈0, identical row counts). Verify CSV row counts > 1 before trusting an EQUIVALENT.

- [ ] **Step 5: CUDA cross-check (mehek)**

Run the detect-task smoke + the equivalence smoke clips on the CUDA box so TensorRT/CUDA paths (the ones the iou/imgsz/segment fixes most affect) are exercised:
```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin && git checkout integration/obb-direct-postretirement
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
python -m pytest tests/test_inference_detect_task_smoke.py -q
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_integ RUNTIME=cuda bash tools/equivalence/run_matrix.sh fly_obb worm_bgsub
```
Expected: detect smoke passes on `cuda`; equivalence at floor.

---

### Task 3.2: Prepare the merge

**Files:** none.

- [ ] **Step 1: Update the branch**

Fast-forward the feature branch to the verified integration branch (or open the PR from the integration branch directly — user's call):
```bash
git checkout feature/obb-direct-detect-segment
git merge --ff-only integration/obb-direct-postretirement
```
Expected: fast-forward. (Skip if the user prefers to PR the integration branch as-is.)

- [ ] **Step 2: Open the PR**

```bash
git push -u origin feature/obb-direct-detect-segment
gh pr create --base main --title "Direct detect/segment OBB detection + runtime fixes (post-retirement)" \
  --body "$(cat <<'EOF'
Adds YOLO detect- and segment-task models as direct OBB detection sources on the
post-retirement InferenceRunner path, plus four runtime fixes.

## Feature
- `model_task ∈ {obb, detect, segment}` on `OBBDirectConfig`, threaded via
  `build_inference_config_from_params` (seg kernel params clamped) into the OBB stage.
- Segment: GPU rotated-rect-from-mask kernel; TensorRT segment executor.
- Detect: fixed-angle OBB from axis-aligned boxes.
- Detection-cache fingerprint covers all seg/task params (cache version 4).

## Fixes
- TensorRT direct decode now honors the configured `iou_threshold` (was hardcoded 0.5).
- Prebuilt `.engine`/`.onnx` artifacts resolve `imgsz` or fail loudly (was silent 640).
- Direct artifacts without task metadata emit an explicit unverifiable-task warning;
  torch-path task guard preserved.
- Runtime-tier → direct-executor mapping documented; ONNX direct executors are tool-only.

## Verification
- Full suite green modulo ~7 pre-existing IdentityPanel failures.
- Real detect-task smoke on cpu/mps/cuda; CoreML segment smoke tightened.
- OBB equivalence at determinism floor (MPS + CUDA) on fly_obb/worm_bgsub.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR opened against `main`.

- [ ] **Step 3: Request review**

Invoke `superpowers:requesting-code-review` (or `/code-review`) on the branch diff before asking a human to merge.

---

## Self-Review

- **Spec coverage:** merge-reconciliation map → Phase 0 (Tasks 0.2–0.7); re-home config builder → Task 0.5; fix #1 tiers/ONNX docs → Task 2.3; fix #2 TRT iou → Task 1.1; fix #3 task guard → Task 2.2 (re-scoped, flagged); fix #4 imgsz → Task 2.1; fix #5 detect tests → Task 2.4; verification gates → Phase 3. All covered.
- **Design refinement flagged:** Task 2.2 supersedes the design's tautological "set adapter.task" with export-time source-checkpoint validation + unverifiable warning. Reviewer should confirm.
- **Assumptions to confirm during execution (Task 0):** exact `build_obb_only_config` arg names (`compute_runtime` vs `runtime_tier`), `OBBResult` field names (`centroids`/`angles`/`shapes`/`corners`), and `_load_direct_executor`'s real signature — the reference sheets give current line numbers; re-confirm before writing the two fix tests that call them.
- **Kernel-default sync (Global Constraint):** enforced at Task 0.4 (config), 0.5 (builder clamps), 0.6 (stage fallbacks).
