# Inference Acceleration Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the production accelerations the inference redesign dropped — cross-frame batching for all individual-analysis stages, GPU-native crop residency, foreign-region suppression, NVDEC decode, and TensorRT/ONNX auto-export — inside the clean `core/inference/` module, with output that is reproducible by construction.

**Architecture:** Pure batch-native stage functions operate on a cross-frame `CropBatch`. A single `Pipeline` orchestrator streams fixed frame windows through the stages with a bounded, configurable `pipeline_depth` (1 = synchronous/parity, 2 = double-buffer default, >2 = deep). Batch membership is a pure function of frame index, so all depths produce byte-identical caches. One stream-sync chokepoint guards cross-thread GPU handoffs; one async `CacheWriter` decouples disk I/O.

**Tech Stack:** Python, PyTorch (CUDA/MPS/CPU), NumPy, OpenCV, Ultralytics YOLO, SLEAP (subprocess), PyNvVideoCodec (NVDEC), ONNX Runtime / TensorRT, pytest.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-30-inference-acceleration-restoration-design.md`. Every task serves it.
- **Reproducibility invariant:** batch membership + per-item math depend only on frame index, never on wall-clock/thread scheduling. `pipeline_depth` changes *when* work runs, never *what* it computes.
- **Stage rules (preserve):** stage functions do **no I/O, no mode branching, no device detection**. Device/stream decisions live in `RuntimeContext`; orchestration lives in `Pipeline`.
- **Parity gate compares like-for-like:** same `(decode-path, executor, depth)`. NVDEC≠CPU-decode and TRT/FP16≠PyTorch-FP32 are expected, documented, not regressions.
- **Foreign suppression:** honor existing `PoseConfig.suppress_foreign_regions`, default **on**.
- **`pipeline_depth` default = 2.** depth=1 reserved for parity/debug.
- **Per-stage runtime independence and GPU-memory scheduling are OUT of scope.**
- **Environment to run anything in this worktree:** `cd <worktree>; PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest ...`. A bare `python`/`pytest` resolves the **main** worktree's editable install, not this code.
- **Commits:** commit as the configured git user; **omit** any `Co-Authored-By: Claude` trailer. Pre-commit hooks fail in-sandbox (write to `~/.cache/pre-commit`), so commit with `--no-verify` after manually running `black`/`isort` on touched files.
- **Equivalence harness** (`tools/equivalence/`) must stay at parity on the 4 passing clips at depth=1 after each phase.
- All new public types/functions live under `src/hydra_suite/core/inference/`.

### Current signatures this plan builds on (verbatim from the tree)

```
core/inference/result.py        : OBBResult, HeadTailResult, CNNFactorPrediction,
                                  CNNDetectionPrediction, CNNResult, PoseResult,
                                  AprilTagResult, FrameResult   (all @dataclass)
core/inference/runtime.py       : @dataclass RuntimeContext(cuda_mode, device, use_nvdec,
                                  tensor_on_cuda); RuntimeContext.from_config(config)
core/inference/config.py        : InferenceConfig(... detection_batch_size=1, realtime=False,
                                  use_cache=True ...); PoseConfig(... suppress_foreign_regions,
                                  background_color ...)
core/inference/runner.py        : InferenceRunner.__init__, caches_all_valid, run_realtime,
                                  run_batch_pass(L501), _run_batch(L563), load_frame(L687)
core/inference/stages/crops.py  : extract_canonical_crops(L20), extract_aabb_crops(L51),
                                  _extract_canonical_cpu(L79), extract_classifier_crops(L119),
                                  _extract_canonical_gpu(L200), _extract_canonical_gpu_legacy(L247)
core/inference/stages/obb.py    : run_obb(L157), load_obb_models(L141), _load_yolo(L416),
                                  materialize_tensors(L428)
core/inference/stages/headtail.py: run_headtail(L69), load_headtail_model(L32)
core/inference/stages/cnn.py    : run_cnn(L38), load_cnn_model(L25)
core/inference/stages/pose.py   : run_pose(L145), load_pose_model(L63)
core/inference/stages/apriltag.py: run_apriltag(L48)
core/inference/cache/store.py   : DetectionCacheHandle, HeadTailCacheHandle, CNNCacheHandle,
                                  PoseCacheHandle, AprilTagCacheHandle (write_frame/read_frame/
                                  is_valid/close; written_frames/covers_frame_range)
```

### Legacy source to port from (kept in tree, do not import — copy/adapt cleanly)

```
NVDEC decode      : core/tracking/ingest/detection_phase.py:124-224
                    (_nvdec_frame_to_cuda_tensor, _should_use_nvdec, _try_open_nvdec, _read_nvdec_batch)
TRT/ONNX export   : core/detectors/_runtime_artifacts.py:543-737 (_try_load_onnx_model,
                    _try_load_tensorrt_model, _maybe_enable_direct_obb_executor)
Direct OBB exec   : core/detectors/_direct_obb_runtime.py (create_direct_obb_executor)
Foreign crop mask : core/canonicalization/crop.py:559 (_apply_foreign_mask_canonical),
                    :220 (canonical_crop_with_foreign_mask)
Foreign keypoints : utils/geometry.py:172 (filter_keypoints_by_foreign_obbs)
GPU crop batch    : core/canonicalization/crop.py:309 (gpu_canonical_crop_batch)
Pose CUDA         : core/identity/pose/backends/sleap.py:844 (predict_batch_cuda)
Double-buffer ref : core/tracking/pose/pose_pipeline.py:261-674
```

---

## Phase 1 — Batch-native stages + unified GPU crop extraction (behavior-preserving, depth=1)

Goal: every individual stage consumes one cross-frame `CropBatch`, crops are extracted once on-device and shared. No throughput/concurrency change yet; output identical to today.

### Task 1: Add `CropBatch` result type

**Files:**
- Modify: `src/hydra_suite/core/inference/result.py`
- Test: `tests/test_inference_cropbatch.py` (create)

**Interfaces:**
- Produces: `CropBatch(crops: torch.Tensor, detection_ids: np.ndarray, frame_index: np.ndarray, obb_by_frame: dict[int, OBBResult], native_sizes: np.ndarray)`; method `frames() -> list[int]` (sorted unique frame indices); `select_frame(frame_idx) -> tuple[np.ndarray, ...]` returning the row indices into the batch for that frame.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_cropbatch.py
import numpy as np
import torch
from hydra_suite.core.inference.result import CropBatch, OBBResult


def _obb(frame_idx, n):
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((n, 2), np.float32),
        angles=np.zeros(n, np.float32),
        sizes=np.ones(n, np.float32),
        shapes=np.ones((n, 2), np.float32),
        confidences=np.ones(n, np.float32),
        corners=np.zeros((n, 4, 2), np.float32),
        detection_ids=np.array([frame_idx * 10000 + s for s in range(n)], np.int64),
    )


def test_cropbatch_indexes_rows_by_frame():
    batch = CropBatch(
        crops=torch.zeros(3, 3, 8, 8),
        detection_ids=np.array([0, 1, 10000], np.int64),
        frame_index=np.array([0, 0, 1], np.int64),
        obb_by_frame={0: _obb(0, 2), 1: _obb(1, 1)},
        native_sizes=np.array([[8, 8], [8, 8], [8, 8]], np.int64),
    )
    assert batch.frames() == [0, 1]
    rows = batch.select_frame(1)
    assert list(rows) == [2]
    rows0 = batch.select_frame(0)
    assert list(rows0) == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_cropbatch.py -v`
Expected: FAIL — `ImportError: cannot import name 'CropBatch'`.

- [ ] **Step 3: Implement `CropBatch`**

Append to `result.py` (after `OBBResult`, since it references it):

```python
@dataclass
class CropBatch:
    """Cross-frame canonical crops shared read-only by head-tail / CNN / pose.

    Row order is detection-id order (frame-index-derived), so batch membership
    is a pure function of frame index — the reproducibility invariant.
    """
    crops: "torch.Tensor"            # (N, C, H, W) device-resident or CPU
    detection_ids: np.ndarray        # (N,) int64
    frame_index: np.ndarray          # (N,) int64
    obb_by_frame: dict               # frame_idx -> OBBResult
    native_sizes: np.ndarray         # (N, 2) int64 — pre-pad crop h,w

    def frames(self) -> list:
        return sorted({int(f) for f in self.frame_index.tolist()})

    def select_frame(self, frame_idx: int) -> np.ndarray:
        return np.nonzero(self.frame_index == int(frame_idx))[0]
```

Add `import numpy as np` / `from dataclasses import dataclass` if not present (they are — confirm). Keep `torch` import lazy/`TYPE_CHECKING` to match the file's existing import style.

- [ ] **Step 4: Run test to verify it passes**

Run: same command as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/result.py tests/test_inference_cropbatch.py
git commit --no-verify -m "feat(inference): add CropBatch cross-frame result type"
```

### Task 2: Unified GPU-native batched crop extraction → `CropBatch`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/crops.py`
- Test: `tests/test_inference_extract_crops_batch.py` (create)

**Interfaces:**
- Consumes: `OBBResult`, `RuntimeContext`, `HeadTailConfig`/`PoseConfig` crop params, `gpu_canonical_crop_batch` (CUDA) / `_extract_canonical_cpu` (CPU/MPS).
- Produces: `extract_crops(frames: list, obb_results: list[OBBResult], *, canonical_margin: float, canonical_aspect_ratio: float, out_size: tuple[int,int], runtime: RuntimeContext) -> CropBatch`. Crops are device-resident when `runtime.cuda_mode`. Detections concatenated across the window in detection-id order.
- Removes: dead `_extract_canonical_gpu_legacy` (crops.py:247).

- [ ] **Step 1: Write the failing test** (CPU path — runs on MPS/CI)

```python
# tests/test_inference_extract_crops_batch.py
import numpy as np
import torch
from hydra_suite.core.inference.stages.crops import extract_crops
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.result import OBBResult


def _runtime_cpu():
    return RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False, tensor_on_cuda=False)


def _obb(frame_idx, n):
    cx = np.linspace(20, 40, n).astype(np.float32)
    corners = np.stack([
        np.stack([cx - 5, np.full(n, 15, np.float32)], -1),
        np.stack([cx + 5, np.full(n, 15, np.float32)], -1),
        np.stack([cx + 5, np.full(n, 25, np.float32)], -1),
        np.stack([cx - 5, np.full(n, 25, np.float32)], -1),
    ], axis=1).astype(np.float32)
    return OBBResult(
        frame_idx=frame_idx, centroids=np.stack([cx, np.full(n, 20, np.float32)], -1),
        angles=np.zeros(n, np.float32), sizes=np.full(n, 100, np.float32),
        shapes=np.ones((n, 2), np.float32), confidences=np.ones(n, np.float32),
        corners=corners,
        detection_ids=np.array([frame_idx * 10000 + s for s in range(n)], np.int64),
    )


def test_extract_crops_concatenates_window_in_detection_id_order():
    frames = [np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8)]
    obbs = [_obb(0, 2), _obb(1, 1)]
    batch = extract_crops(frames, obbs, canonical_margin=1.3,
                          canonical_aspect_ratio=2.0, out_size=(32, 32),
                          runtime=_runtime_cpu())
    assert batch.crops.shape[0] == 3
    assert list(batch.detection_ids) == [0, 1, 10000]
    assert list(batch.frame_index) == [0, 0, 1]
    assert batch.frames() == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_extract_crops_batch.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_crops'`.

- [ ] **Step 3: Implement `extract_crops`**

Add to `crops.py`. It loops frames, computes per-detection canonical crops using the **existing** primitives (`gpu_canonical_crop_batch` when `runtime.cuda_mode`, else `_extract_canonical_cpu`), concatenates results, and records `detection_ids`/`frame_index`/`native_sizes`. Reuse the affine math already in `extract_canonical_crops`/`_extract_canonical_gpu`; do not reimplement it. Concatenate in input order (frames ascending, detections in OBBResult order = detection-id order). Pad all crops to `out_size` (the warp target), recording `native_sizes` before padding so pose can undo it (mirror current `stages/pose.py` padding recovery at L186-203).

```python
def extract_crops(frames, obb_results, *, canonical_margin, canonical_aspect_ratio,
                  out_size, runtime):
    crops_list, det_ids, frame_idx_list, native_sizes = [], [], [], []
    for frame, obb in zip(frames, obb_results):
        if obb.detection_ids.shape[0] == 0:
            continue
        # reuse the existing single-frame canonical extraction (GPU or CPU)
        frame_crops, sizes = _extract_canonical_window(
            frame, obb, canonical_margin, canonical_aspect_ratio, out_size, runtime)
        crops_list.append(frame_crops)               # (n_i, C, H, W)
        det_ids.append(obb.detection_ids)
        frame_idx_list.append(np.full(obb.detection_ids.shape[0], obb.frame_idx, np.int64))
        native_sizes.append(sizes)
    if not crops_list:
        empty = torch.zeros((0, 3, out_size[1], out_size[0]),
                            device=runtime.device if runtime.cuda_mode else "cpu")
        return CropBatch(empty, np.zeros(0, np.int64), np.zeros(0, np.int64),
                         {o.frame_idx: o for o in obb_results}, np.zeros((0, 2), np.int64))
    return CropBatch(
        crops=torch.cat(crops_list, dim=0),
        detection_ids=np.concatenate(det_ids),
        frame_index=np.concatenate(frame_idx_list),
        obb_by_frame={o.frame_idx: o for o in obb_results},
        native_sizes=np.concatenate(native_sizes),
    )
```

Implement `_extract_canonical_window` as a thin wrapper that calls the existing GPU/CPU canonical routine for one frame and returns `(tensor, native_sizes_array)`. Delete `_extract_canonical_gpu_legacy` (crops.py:247) and its only caller comment.

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Run the full crops test module + black/isort, then commit**

```bash
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_extract_crops_batch.py tests/test_inference_crops.py -v
black src/hydra_suite/core/inference/stages/crops.py && isort src/hydra_suite/core/inference/stages/crops.py
git add src/hydra_suite/core/inference/stages/crops.py tests/test_inference_extract_crops_batch.py
git commit --no-verify -m "feat(inference): unified cross-frame GPU-native crop extraction (extract_crops -> CropBatch); drop dead legacy gpu path"
```

### Task 3: Batch-native head-tail / CNN / pose + `scatter`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/headtail.py`, `stages/cnn.py`, `stages/pose.py`
- Create: `src/hydra_suite/core/inference/stages/assemble.py` (the `scatter` reassembly)
- Test: `tests/test_inference_batch_stages.py` (create)

**Interfaces:**
- Produces:
  - `run_headtail_batch(batch: CropBatch, model, config, runtime) -> dict[int, HeadTailResult]` (keyed by frame_idx)
  - `run_cnn_batch(batch: CropBatch, model, config, runtime) -> dict[int, CNNResult]`
  - `run_pose_batch(batch: CropBatch, model, config, runtime) -> dict[int, PoseResult]`
  - `scatter(obb_by_frame, headtail, cnns, pose, apriltag, config) -> list[FrameResult]` — per-frame `FrameResult` with heading resolution + pose-keypoint foreign suppression (suppression added in Task 5).
- Consumes: existing `run_headtail`/`run_cnn`/`run_pose` per-frame internals (refactor them to operate on a crop sub-tensor + that frame's OBBResult, then call per frame inside the `_batch` wrappers). The per-frame functions stay (used by `run_realtime`); the `_batch` wrappers slice `CropBatch` by frame and delegate, so numerics are identical.

- [ ] **Step 1: Write the failing test** (uses fake models returning deterministic outputs)

```python
# tests/test_inference_batch_stages.py
import numpy as np, torch
from hydra_suite.core.inference.result import CropBatch, OBBResult
from hydra_suite.core.inference.stages.headtail import run_headtail_batch
from hydra_suite.core.inference.runtime import RuntimeContext


class _FakeHTBackend:
    # returns label "up" + conf 0.9 for every crop
    def predict_batch(self, crops):
        n = len(crops)
        return [{"label": "up", "confidence": 0.9} for _ in range(n)]


class _FakeHTModel:
    backend = _FakeHTBackend()
    class_names = ["up", "down", "left", "right"]


def _obb(frame_idx, n):
    return OBBResult(frame_idx=frame_idx, centroids=np.zeros((n, 2), np.float32),
        angles=np.zeros(n, np.float32), sizes=np.ones(n, np.float32),
        shapes=np.ones((n, 2), np.float32), confidences=np.ones(n, np.float32),
        corners=np.zeros((n, 4, 2), np.float32),
        detection_ids=np.array([frame_idx*10000+s for s in range(n)], np.int64))


def test_run_headtail_batch_keys_by_frame_and_matches_counts():
    batch = CropBatch(crops=torch.zeros(3, 3, 16, 16),
        detection_ids=np.array([0, 1, 10000], np.int64),
        frame_index=np.array([0, 0, 1], np.int64),
        obb_by_frame={0: _obb(0, 2), 1: _obb(1, 1)},
        native_sizes=np.array([[16,16]]*3, np.int64))
    rt = RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False, tensor_on_cuda=False)
    out = run_headtail_batch(batch, _FakeHTModel(), config=None, runtime=rt)
    assert set(out) == {0, 1}
    assert out[0].heading_hints.shape[0] == 2
    assert out[1].heading_hints.shape[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_batch_stages.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_headtail_batch'`.

- [ ] **Step 3: Implement the `_batch` wrappers**

For each of head-tail/CNN/pose: add a `run_*_batch(batch, model, config, runtime)` that runs the backend **once** over `batch.crops` (the whole window — the perf win), then splits per-frame results using `batch.select_frame(f)` and assembles the existing per-frame result type for each frame. Reuse the existing per-detection assembly logic from the current `run_headtail`/`run_cnn`/`run_pose` (extract it into a helper that takes `(rows, obb_for_frame, raw_predictions)` so both the per-frame and batch paths share it — DRY). For pose, undo per-crop padding using `batch.native_sizes` exactly as `stages/pose.py:186-203` does today, and call `predict_batch_cuda` when `batch.crops.is_cuda` and the backend provides it (mirror `pose_pipeline.py:692-695`).

Create `stages/assemble.py` with `scatter(...)` that, for each frame, takes that frame's `OBBResult` + the per-frame head-tail/CNN/pose/apriltag results and builds a `FrameResult` with `resolved_headings` (pose → head-tail → OBB axis priority, copied from the current assembly in `runner._run_batch`).

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black src/hydra_suite/core/inference/stages/{headtail,cnn,pose,assemble}.py
isort src/hydra_suite/core/inference/stages/{headtail,cnn,pose,assemble}.py
git add src/hydra_suite/core/inference/stages/ tests/test_inference_batch_stages.py
git commit --no-verify -m "feat(inference): batch-native head-tail/CNN/pose stages + scatter reassembly"
```

---

## Phase 2 — Foreign-region suppression

### Task 4: Foreign-mask canonical crops in `extract_crops`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/crops.py`
- Test: `tests/test_inference_foreign_mask.py` (create)

**Interfaces:**
- `extract_crops(..., suppress_foreign: bool = False, background_color=(0,0,0))` — when true, neighbor OBB polygons within the same frame are blacked out in each crop via the existing `_apply_foreign_mask_canonical` (`core/canonicalization/crop.py:559`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_foreign_mask.py
import numpy as np, torch
from hydra_suite.core.inference.stages.crops import extract_crops
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.result import OBBResult


def _two_adjacent_obbs():
    # two boxes; box B overlaps box A's canonical crop region
    corners = np.array([
        [[10,10],[30,10],[30,30],[10,30]],
        [[28,10],[48,10],[48,30],[28,30]],
    ], np.float32)
    return OBBResult(frame_idx=0, centroids=np.array([[20,20],[38,20]],np.float32),
        angles=np.zeros(2,np.float32), sizes=np.full(2,400,np.float32),
        shapes=np.ones((2,2),np.float32), confidences=np.ones(2,np.float32),
        corners=corners, detection_ids=np.array([0,1],np.int64))


def test_foreign_mask_blacks_out_neighbor_pixels():
    frame = np.full((64, 64, 3), 200, np.uint8)
    obb = _two_adjacent_obbs()
    rt = RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False, tensor_on_cuda=False)
    masked = extract_crops([frame], [obb], canonical_margin=1.5,
        canonical_aspect_ratio=1.0, out_size=(32,32), runtime=rt,
        suppress_foreign=True, background_color=(0,0,0))
    plain = extract_crops([frame], [obb], canonical_margin=1.5,
        canonical_aspect_ratio=1.0, out_size=(32,32), runtime=rt,
        suppress_foreign=False)
    # masking must zero strictly more pixels than the unmasked crop
    assert (masked.crops[0] == 0).sum() > (plain.crops[0] == 0).sum()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_foreign_mask.py -v`
Expected: FAIL — `extract_crops() got an unexpected keyword argument 'suppress_foreign'`.

- [ ] **Step 3: Implement foreign masking**

Thread `suppress_foreign`/`background_color` into `extract_crops` → `_extract_canonical_window`. For each target detection, pass the *other* detections' corners in that frame as `foreign_corners` to the canonical extractor and apply `_apply_foreign_mask_canonical`. On CUDA, rasterize the foreign polygons into a mask tensor and multiply (do not round-trip to CPU); on CPU/MPS use the existing `cv2.fillPoly` path. Reuse `canonical_crop_with_foreign_mask` (`crop.py:220`) for the CPU path rather than reimplementing.

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black src/hydra_suite/core/inference/stages/crops.py && isort src/hydra_suite/core/inference/stages/crops.py
git add src/hydra_suite/core/inference/stages/crops.py tests/test_inference_foreign_mask.py
git commit --no-verify -m "feat(inference): wire foreign-region suppression into canonical crop extraction"
```

### Task 5: Foreign-OBB pose-keypoint suppression in `scatter`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/assemble.py`
- Test: `tests/test_inference_keypoint_foreign.py` (create)

**Interfaces:**
- `scatter(...)` calls `filter_keypoints_by_foreign_obbs` (`utils/geometry.py:172`) on each detection's frame-space keypoints, zeroing confidence for keypoints inside other animals' OBBs, gated on `suppress_foreign_regions`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_keypoint_foreign.py
import numpy as np
from hydra_suite.core.inference.stages.assemble import suppress_foreign_keypoints


def test_keypoint_inside_foreign_obb_is_zeroed():
    # keypoint at (40,20) lands inside the foreign box [30..50]x[10..30]
    kpts = np.array([[[15.0, 20.0, 0.9], [40.0, 20.0, 0.9]]], np.float32)  # (1 det, 2 kpts, 3)
    target_corners = np.array([[10,10],[25,10],[25,30],[10,30]], np.float32)
    foreign = [np.array([[30,10],[50,10],[50,30],[30,30]], np.float32)]
    out = suppress_foreign_keypoints(kpts[0], target_corners, foreign)
    assert out[0, 2] == 0.9     # inside target → kept
    assert out[1, 2] == 0.0     # inside foreign → zeroed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_keypoint_foreign.py -v`
Expected: FAIL — `ImportError: cannot import name 'suppress_foreign_keypoints'`.

- [ ] **Step 3: Implement**

Add `suppress_foreign_keypoints(keypoints, target_corners, foreign_corners_list)` to `assemble.py` delegating to `hydra_suite.utils.geometry.filter_keypoints_by_foreign_obbs`. Call it inside `scatter` for each detection after keypoints are in frame space, gated on the config flag.

- [ ] **Step 4: Run test to verify it passes** — same command, Expected: PASS.

- [ ] **Step 5: Re-run the equivalence harness at depth=1 (expect dense-clip improvement) and commit**

```bash
black src/hydra_suite/core/inference/stages/assemble.py && isort src/hydra_suite/core/inference/stages/assemble.py
git add src/hydra_suite/core/inference/stages/assemble.py tests/test_inference_keypoint_foreign.py
git commit --no-verify -m "feat(inference): foreign-OBB pose-keypoint suppression in scatter"
# Then run tools/equivalence per its README on the 4 passing clips + the 2 dense clips; record results in PARITY_AUDIT.md
```

---

## Phase 3 — `Pipeline` orchestrator (depth=1) + depth-invariance harness

### Task 6: Add `pipeline_depth` to config

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py`
- Test: `tests/test_inference_config_pipeline_depth.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_config_pipeline_depth.py
from hydra_suite.core.inference.config import InferenceConfig, OBBConfig, OBBDirectConfig


def _min_cfg(**kw):
    return InferenceConfig(obb=OBBConfig(mode="direct",
        direct=OBBDirectConfig(model_path="m.pt")), **kw)


def test_pipeline_depth_defaults_to_2_and_roundtrips():
    assert _min_cfg().pipeline_depth == 2
    d = _min_cfg(pipeline_depth=1).to_dict() if hasattr(_min_cfg(), "to_dict") else None
    cfg = InferenceConfig.from_dict({**_min_cfg().__dict__, "pipeline_depth": 4}) \
        if hasattr(InferenceConfig, "from_dict") else _min_cfg(pipeline_depth=4)
    assert cfg.pipeline_depth == 4
```

(If the config uses `from_json`/`to_json` rather than dict helpers, adapt the round-trip half to write/read a temp JSON file; keep the default assertion.)

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`pipeline_depth` unknown / default not 2).

- [ ] **Step 3: Implement** — add `pipeline_depth: int = 2` to `InferenceConfig` (config.py:144 block) and to the `from_json`/`from_dict` reader (mirror `detection_batch_size` at config.py:266) with default 2. Add a validation: `pipeline_depth >= 1` else `InferenceConfigError`.

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/config.py tests/test_inference_config_pipeline_depth.py
git commit --no-verify -m "feat(inference): add pipeline_depth config (default 2)"
```

### Task 7: `Pipeline` (depth=1) and move the runner onto it

**Files:**
- Create: `src/hydra_suite/core/inference/pipeline.py`
- Modify: `src/hydra_suite/core/inference/runner.py` (`run_batch_pass` L501, remove `_run_batch` L563 inner per-frame loop + intra-frame ThreadPool)
- Test: `tests/test_inference_pipeline_depth1.py` (create)

**Interfaces:**
- Produces: `class Pipeline(stages, runtime, cache_writer, *, depth=1, queue_bound=None)`; `Pipeline.run(frame_source, frame_range) -> InferencePassResult`. `stages` is a small struct of the loaded models + config needed to call `run_obb`, `extract_crops`, the `_batch` stage fns, `run_apriltag`, `scatter`.
- A `BatchWindow` = `frames[k*W:(k+1)*W]`; W = `config.detection_batch_size`.
- Consumes: Tasks 2–5 stage functions; `CacheWriter` (Task 10 — until then, write synchronously inline via the existing cache handles).

- [ ] **Step 1: Write the failing test** (depth=1 produces the same per-frame results as the old path on a fake-model stub)

```python
# tests/test_inference_pipeline_depth1.py
# Build a Pipeline with fake stage callables that echo deterministic FrameResults,
# run over 5 synthetic frames with W=2, assert: (a) 5 FrameResults in frame order,
# (b) windows are [0,1],[2,3],[4] — i.e. batch boundaries are frame-indexed.
from hydra_suite.core.inference.pipeline import Pipeline, BatchWindow


def test_windows_are_frame_indexed_not_arrival_indexed():
    seen = []
    def fake_stage(window):
        seen.append([f.index for f in window.frames])
        return [object() for _ in window.frames]
    pipe = Pipeline.for_test(window_size=2, depth=1, stage=fake_stage)
    results = pipe.run_frames(range(5))
    assert seen == [[0, 1], [2, 3], [4]]
    assert len(results) == 5
```

(Provide a `Pipeline.for_test(window_size, depth, stage)` classmethod + a tiny `run_frames(range)` shim used only by tests, so the orchestration logic is testable without real models. The production `run()` calls the same internal `_iter_windows` + `_process_window`.)

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `Pipeline` (depth=1 only)**

Synchronous loop: `_iter_windows(frame_range)` yields `BatchWindow`s of W frames by index; `_process_window(window)` runs OBB → `extract_crops` → `run_*_batch` (sequentially) → `run_apriltag` → `scatter` → hand each `FrameResult` to the cache writer (synchronous for now). No threads. Then rewrite `runner.run_batch_pass` to construct a `Pipeline(depth=config.pipeline_depth)` and call `.run(...)`; delete the per-frame `_run_batch` inner loop and the `ThreadPoolExecutor` block. `run_realtime` keeps the per-frame stage functions (single-frame `CropBatch` of one window).

- [ ] **Step 4: Run to verify it passes** + run the existing runner/stage suite

```bash
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_pipeline_depth1.py tests/test_inference_runner.py tests/test_inference_stages.py -v
```
Expected: PASS (pre-existing unrelated failures noted in PARITY_AUDIT excepted).

- [ ] **Step 5: Commit**

```bash
black src/hydra_suite/core/inference/pipeline.py src/hydra_suite/core/inference/runner.py
isort src/hydra_suite/core/inference/pipeline.py src/hydra_suite/core/inference/runner.py
git add src/hydra_suite/core/inference/pipeline.py src/hydra_suite/core/inference/runner.py tests/test_inference_pipeline_depth1.py
git commit --no-verify -m "feat(inference): Pipeline orchestrator (depth=1); runner runs on it; remove per-frame loop + intra-frame ThreadPool"
```

### Task 8: Depth-invariance test harness

**Files:**
- Create: `tests/test_inference_depth_invariance.py`
- Create fixture helper: `tests/helpers/tiny_clip.py` (a 6-frame synthetic clip + stub models that exercise OBB→crops→headtail→cnn→pose→cache)

**Interfaces:**
- `run_pipeline_to_caches(tmp_path, depth) -> dict[str, bytes]` — runs the full pipeline at the given depth over the tiny clip and returns `{cache_filename: sha256}`.

- [ ] **Step 1: Write the test (depth=1 only for now — asserts determinism of a single depth)**

```python
# tests/test_inference_depth_invariance.py
from tests.helpers.tiny_clip import run_pipeline_to_caches


def test_depth1_is_deterministic_across_runs(tmp_path):
    a = run_pipeline_to_caches(tmp_path / "a", depth=1)
    b = run_pipeline_to_caches(tmp_path / "b", depth=1)
    assert a == b
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`ImportError` on helper).

- [ ] **Step 3: Implement the helper** — synthetic clip + stub models (deterministic outputs) wired through the real `Pipeline`/cache handles; hash each written `.npz`.

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_inference_depth_invariance.py tests/helpers/tiny_clip.py
git commit --no-verify -m "test(inference): depth-invariance harness (depth=1 determinism)"
```

---

## Phase 4 — Stream-sync + depth=2 double-buffer + async CacheWriter

### Task 9: Stream-sync chokepoint on `RuntimeContext`

**Files:**
- Modify: `src/hydra_suite/core/inference/runtime.py`
- Test: `tests/test_inference_stream_sync.py` (create)

**Interfaces:**
- `RuntimeContext.handoff(tensor) -> tensor` — on CUDA, records an event on the current stream and returns the tensor tagged for the consumer to wait on; provides `RuntimeContext.await_handoff(tensor)` consumer-side. On CPU/MPS both are identity no-ops.

- [ ] **Step 1: Write the failing test** (CPU no-op contract — CUDA path validated on `mehek`)

```python
# tests/test_inference_stream_sync.py
import torch
from hydra_suite.core.inference.runtime import RuntimeContext


def test_handoff_is_identity_on_cpu():
    rt = RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False, tensor_on_cuda=False)
    t = torch.arange(6)
    assert rt.await_handoff(rt.handoff(t)) is t
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`AttributeError: handoff`).

- [ ] **Step 3: Implement** — `handoff`/`await_handoff`. CUDA: `torch.cuda.Event`, record on `torch.cuda.current_stream()` in `handoff`, `wait_event` in `await_handoff`. Non-CUDA: return the tensor unchanged.

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/runtime.py tests/test_inference_stream_sync.py
git commit --no-verify -m "feat(inference): single CUDA stream-sync chokepoint on RuntimeContext"
```

### Task 10: Async, frame-ordered `CacheWriter`

**Files:**
- Create: `src/hydra_suite/core/inference/cache/writer.py`
- Test: `tests/test_inference_cache_writer.py` (create)

**Interfaces:**
- `class CacheWriter(handles: dict[str, CacheHandle], *, async_mode: bool)`; `submit(frame_result)`; `flush()`; `close()`. Writes per-frame in **ascending frame order** even if `submit` is called out of order (internal min-ordered buffer that emits a frame only once all lower frames are written). `async_mode=False` writes inline.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_cache_writer.py
from hydra_suite.core.inference.cache.writer import CacheWriter


class _RecordingHandle:
    def __init__(self): self.frames = []
    def write_frame(self, frame_idx, **kw): self.frames.append(frame_idx)
    def close(self): pass


def test_writes_in_frame_order_despite_out_of_order_submit():
    h = _RecordingHandle()
    w = CacheWriter({"detection": h}, async_mode=False)
    for fr in [_fr(2), _fr(0), _fr(1)]:  # _fr builds a FrameResult-like with .frame_idx
        w.submit(fr)
    w.flush(); w.close()
    assert h.frames == [0, 1, 2]
```

(Provide `_fr(i)` in the test building a minimal object with `.frame_idx=i` and the per-type payload attributes the handles read; for `async_mode=False` ordering is enforced by the buffer, not by threads.)

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement** — ordered buffer keyed by frame_idx, a `next_expected` cursor; `submit` inserts and drains all contiguous ready frames to the handles. `async_mode=True` runs the drain on a single worker thread fed by a `queue.Queue`; `close` joins. Map each `FrameResult` field to the right handle's `write_frame(**kwargs)` (detection→`result=obb`, headtail, cnn per phase label, pose, apriltag) per the current `store.py` `write_frame` signatures.

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/cache/writer.py tests/test_inference_cache_writer.py
git commit --no-verify -m "feat(inference): frame-ordered CacheWriter (sync + async modes)"
```

### Task 11: depth=2 double-buffer in `Pipeline`

**Files:**
- Modify: `src/hydra_suite/core/inference/pipeline.py`
- Modify: `tests/test_inference_depth_invariance.py`

**Interfaces:**
- depth=2: a producer thread runs decode+OBB for window k+1 while the main thread runs crops+individual+scatter for window k; results go through `CacheWriter(async_mode=True)`. GPU handoff of the OBB output tensor uses `RuntimeContext.handoff`/`await_handoff`. Stop checked at window boundaries.

- [ ] **Step 1: Extend the invariance test to assert 1 ≡ 2**

```python
def test_depth1_equals_depth2(tmp_path):
    a = run_pipeline_to_caches(tmp_path / "d1", depth=1)
    b = run_pipeline_to_caches(tmp_path / "d2", depth=2)
    assert a == b
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (depth=2 path not implemented / not byte-identical).

- [ ] **Step 3: Implement depth=2** — one-window producer/consumer with a bounded `queue.Queue(maxsize=1)` handing `(window, obb_results)` from producer to consumer; consumer does crops→stages→scatter→cache. Supervisor: on exception, set a stop flag, drain/join producer with timeout, `cache_writer.flush()/close()`, re-raise. Ensure batch boundaries + chunking unchanged so output matches depth=1.

- [ ] **Step 4: Run to verify it passes** — Expected: PASS (`a == b`).

- [ ] **Step 5: Commit**

```bash
black src/hydra_suite/core/inference/pipeline.py
git add src/hydra_suite/core/inference/pipeline.py tests/test_inference_depth_invariance.py
git commit --no-verify -m "feat(inference): depth=2 double-buffer pipeline; assert depth-invariance 1==2"
```

---

## Phase 5 — Deep pipeline (depth>2)

### Task 12: bounded-queue deep pipelining

**Files:**
- Modify: `src/hydra_suite/core/inference/pipeline.py`
- Modify: `tests/test_inference_depth_invariance.py`

- [ ] **Step 1: Extend the invariance test to 1 ≡ 2 ≡ 4**

```python
def test_depths_1_2_4_byte_identical(tmp_path):
    h = {d: run_pipeline_to_caches(tmp_path / f"d{d}", depth=d) for d in (1, 2, 4)}
    assert h[1] == h[2] == h[4]
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (depth=4 not supported).

- [ ] **Step 3: Implement** — generalize the producer/consumer to `depth` windows in flight via bounded queues (`maxsize` derived from depth); `CacheWriter` ordered buffer already tolerates out-of-order completion. No per-item math change.

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/pipeline.py tests/test_inference_depth_invariance.py
git commit --no-verify -m "feat(inference): configurable deep pipeline (depth>2); assert 1==2==4"
```

---

## Phase 6 — NVDEC frame source

### Task 13: `FrameSource` abstraction + `CpuFrameReader`

**Files:**
- Create: `src/hydra_suite/core/inference/sources.py`
- Modify: `src/hydra_suite/core/inference/pipeline.py` (consume a `FrameSource`)
- Test: `tests/test_inference_frame_source_cpu.py` (create)

**Interfaces:**
- `class FrameSource(ABC)`: `__iter__ -> Iterator[tuple[int, frame]]` (frame = numpy HWC uint8 on CPU path; CUDA tensor on NVDEC path); `frame_count: int`; `close()`.
- `CpuFrameReader(video_path)` — cv2-based, mirrors current frame read in `runner`/`detection_phase`.

- [ ] **Step 1: Write the failing test** — open a tiny generated mp4/avi fixture, assert `CpuFrameReader` yields `(idx, ndarray)` for each frame in order with correct count.

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `FrameSource` + `CpuFrameReader`**; refactor `Pipeline` to pull frames from a `FrameSource` instead of an inline reader.

- [ ] **Step 4: Run to verify it passes**; re-run depth-invariance + a runner smoke test.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/sources.py src/hydra_suite/core/inference/pipeline.py tests/test_inference_frame_source_cpu.py
git commit --no-verify -m "feat(inference): FrameSource abstraction + CpuFrameReader"
```

### Task 14: `NvdecFrameReader` (CUDA, with CPU fallback)

**Files:**
- Modify: `src/hydra_suite/core/inference/sources.py`
- Test: `tests/test_inference_frame_source_nvdec.py` (create — skips when no CUDA/PyNvVideoCodec)

**Interfaces:**
- `NvdecFrameReader(video_path, device)` — zero-copy decode → CUDA tensor, ported from `detection_phase.py:124-224` (`_nvdec_frame_to_cuda_tensor`, `_try_open_nvdec`, `_read_nvdec_batch`). `make_frame_source(video_path, runtime)` returns `NvdecFrameReader` when `runtime.use_nvdec` and the decoder imports successfully, else `CpuFrameReader` (logged).

- [ ] **Step 1: Write the test** — guarded by `pytest.importorskip("PyNvVideoCodec")` + `torch.cuda.is_available()`; asserts NVDEC yields CUDA tensors of correct count. Also a CPU-side test that `make_frame_source` falls back to `CpuFrameReader` when `use_nvdec=False`.

- [ ] **Step 2: Run to verify it fails** (the fallback test runs on MPS) — Expected: FAIL (`ImportError`/`AttributeError`).

- [ ] **Step 3: Implement** — port the NVDEC helpers cleanly into `NvdecFrameReader`; implement `make_frame_source` with the try/except fallback + log.

- [ ] **Step 4: Run the fallback test (MPS)** — Expected: PASS. NVDEC test → SKIPPED locally; runs on `mehek`.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/sources.py tests/test_inference_frame_source_nvdec.py
git commit --no-verify -m "feat(inference): NvdecFrameReader (zero-copy CUDA decode) + make_frame_source fallback"
```

---

## Phase 7 — TensorRT / ONNX auto-export + direct executor

### Task 15: OBB loader gains artifact auto-export + direct executor

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py` (`_load_yolo` L416, `load_obb_models` L141)
- Create: `src/hydra_suite/core/inference/runtime_artifacts.py` (cleanly ported `_try_load_onnx_model`/`_try_load_tensorrt_model`/direct-executor selection from `core/detectors/_runtime_artifacts.py` + `_direct_obb_runtime.py`)
- Modify: `src/hydra_suite/core/inference/config.py` (add `auto_export: bool` to OBB runtime config; default True on CUDA)
- Test: `tests/test_inference_obb_artifacts.py` (create — CUDA/TRT guarded; CPU asserts selection logic via a fake exporter)

**Interfaces:**
- `load_obb_executor(model_path, compute_runtime, *, auto_export) -> executor` — returns a PyTorch model (cpu/mps/cuda), or an ONNX/TRT direct executor when `compute_runtime in {onnx_*, tensorrt}`, auto-exporting `.onnx`/`.engine` from `.pt` on first load. `run_obb` calls through this; the existing OBB geometry extraction (`_extract_obb_result`/`materialize_tensors`) is unchanged.

- [ ] **Step 1: Write the failing test** — with a fake exporter injected, assert: (a) `compute_runtime="cuda"` returns the torch model unchanged; (b) `compute_runtime="tensorrt"` with `auto_export=True` and a missing `.engine` triggers the export hook exactly once and returns the direct executor; (c) `auto_export=False` + missing engine raises a clear error (not silent PyTorch fallback).

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`ImportError`/`AttributeError`).

- [ ] **Step 3: Implement** — port the artifact logic into `runtime_artifacts.py` (no import from `core/detectors/`); add `auto_export` to config; wire `_load_yolo`/`load_obb_models` to `load_obb_executor`. Preserve the square-letterbox preprocessing parity from `_maybe_enable_direct_cuda_obb_executor`.

- [ ] **Step 4: Run to verify it passes** (CPU selection-logic tests; TRT/ONNX export tests SKIP locally) — Expected: PASS/SKIP.

- [ ] **Step 5: Commit**

```bash
black src/hydra_suite/core/inference/stages/obb.py src/hydra_suite/core/inference/runtime_artifacts.py src/hydra_suite/core/inference/config.py
isort src/hydra_suite/core/inference/stages/obb.py src/hydra_suite/core/inference/runtime_artifacts.py src/hydra_suite/core/inference/config.py
git add src/hydra_suite/core/inference/stages/obb.py src/hydra_suite/core/inference/runtime_artifacts.py src/hydra_suite/core/inference/config.py tests/test_inference_obb_artifacts.py
git commit --no-verify -m "feat(inference): OBB TRT/ONNX auto-export + direct CUDA executor (H4)"
```

---

## Phase 8 — CUDA performance benchmark

### Task 16: Benchmark script + run instructions for `mehek`

**Files:**
- Create: `tools/equivalence/perf_benchmark.py`
- Create: `tools/equivalence/PERF_BENCHMARK.md` (run instructions)

**Interfaces:**
- `perf_benchmark.py --video <path> --config <path> --depths 1,2,4 --nvdec on,off --trt on,off` → prints a table of frames/sec for each combination, plus a baseline row running the legacy precompute path (`core/tracking/...`) on the same clip. Exits non-zero if the best new-pipeline config is slower than the legacy baseline (the "to par" gate).

- [ ] **Step 1: Write the script** — argument matrix, time `runner.run_batch_pass` per combo, time the legacy path once, tabulate, compute the gate.

- [ ] **Step 2: Write `PERF_BENCHMARK.md`** — exact commands to run on `mehek` (env activation, dataset path, the `perf_benchmark.py` invocation), and how to paste results back. Note the sandbox cannot reach the host, so the user runs it (or whitelists the host / uses `! <cmd>`).

- [ ] **Step 3: Local smoke** — run `perf_benchmark.py --help` and a depth=1 CPU pass on the tiny clip to verify it executes (no CUDA assertions).

- [ ] **Step 4: Commit**

```bash
git add tools/equivalence/perf_benchmark.py tools/equivalence/PERF_BENCHMARK.md
git commit --no-verify -m "feat(equivalence): CUDA perf benchmark + run instructions for to-par gate"
```

- [ ] **Step 5: Hand off to user** — run on `mehek`, capture the table, confirm the "to par" gate passes; record results in `PARITY_AUDIT.md`.

---

## Self-Review

**Spec coverage:**
- Cross-frame batching (OBB already; individual stages) → Tasks 2–3, 7. ✓
- GPU-native residency + unified crops → Tasks 2, 9. ✓
- Foreign-region suppression (crop + keypoint) → Tasks 4–5. ✓
- NVDEC → Tasks 13–14. ✓
- TRT/ONNX auto-export + direct executor (H4) → Task 15. ✓
- Reproducibility invariant / depth-invariance → Tasks 6–8, 11–12. ✓
- Configurable depth (default 2) → Tasks 6, 7, 11, 12. ✓
- Reliability / cancellation / stream-sync → Tasks 9, 11. ✓
- Async frame-ordered cache writer → Task 10. ✓
- Verification: gate 1 (depth-invariance) Tasks 8/11/12; gate 2 (equivalence) Task 5 step 5; gate 3 (CUDA perf) Task 16. ✓
- Non-goals (per-stage runtime, memory scheduler) → not implemented. ✓

**Placeholder scan:** Large ports (NVDEC, TRT/ONNX) cite exact legacy source + define the target adapter signature + a test — concrete port instructions, not placeholders. Test code is literal. No "TBD"/"handle edge cases".

**Type consistency:** `CropBatch` fields/methods (`frames`, `select_frame`, `detection_ids`, `frame_index`, `native_sizes`) consistent across Tasks 1–5. `run_*_batch` return `dict[int, *Result]` consistent in Task 3 and consumed by `scatter`. `pipeline_depth` name consistent Tasks 6–12. `RuntimeContext.handoff/await_handoff` consistent Tasks 9, 11. `CacheWriter.submit/flush/close` consistent Tasks 10–12. `make_frame_source`/`FrameSource` consistent Tasks 13–14. `load_obb_executor` consistent Task 15.
