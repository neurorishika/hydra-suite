# Task 2 Report: Unified Cross-Frame GPU-native Crop Extraction

## TDD RED/GREEN Evidence

### RED (Step 2)
```
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE .../python -m pytest tests/test_inference_extract_crops_batch.py -v
ERROR: ImportError: cannot import name 'extract_crops' from 'hydra_suite.core.inference.stages.crops'
```
Confirmed fail as expected.

### GREEN (Step 4)
```
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE .../python -m pytest tests/test_inference_extract_crops_batch.py tests/test_inference_stages_crops.py -v
tests/test_inference_extract_crops_batch.py::test_extract_crops_concatenates_window_in_detection_id_order PASSED
tests/test_inference_stages_crops.py::test_extract_canonical_crops_returns_tensor PASSED
tests/test_inference_stages_crops.py::test_extract_canonical_crops_empty_obb PASSED
tests/test_inference_stages_crops.py::test_extract_aabb_crops_returns_list PASSED
tests/test_inference_stages_crops.py::test_extract_aabb_crops_empty_obb PASSED
tests/test_inference_stages_crops.py::test_canonical_and_aabb_same_count PASSED
tests/test_inference_stages_crops.py::test_onnx_cuda_uses_cpu_path PASSED
tests/test_inference_stages_crops.py::test_canonical_crops_dtype_normalized PASSED
8 passed in 4.03s
```

## Files Changed

- **Modified**: `src/hydra_suite/core/inference/stages/crops.py`
  - Added import of `CropBatch` from `..result`
  - Removed unused `math` and `torch.nn.functional as F` imports
  - Added `_extract_canonical_window(frame, obb, margin, aspect_ratio, out_size, runtime) -> (tensor, native_sizes_array)` helper
  - Added `extract_crops(frames, obb_results, *, canonical_margin, canonical_aspect_ratio, out_size, runtime) -> CropBatch`
  - Deleted `_extract_canonical_gpu_legacy` (dead code, 45 lines)

- **Created**: `tests/test_inference_extract_crops_batch.py`
  - `test_extract_crops_concatenates_window_in_detection_id_order`: verifies 2+1 detections across 2 frames, correct shape, detection_ids, frame_index, and frames() output

## Brief Deviation

The brief's `_runtime_cpu()` factory omitted the required `default_runtime` field of `RuntimeContext`. Fixed in the test file by adding `default_runtime="cpu"` (consistent with how existing `test_inference_stages_crops.py` constructs the object).

## Self-Review

- Reproducibility invariant satisfied: concatenation is frames-ascending, detections in OBBResult input order.
- GPU path delegates to `_extract_canonical_gpu` primitives (`compute_native_crop_dimensions` + `compute_alignment_affine` + `gpu_canonical_crop_batch`).
- CPU/MPS path delegates to `_warp_canonical_crop` per detection; pads to `out_size` after recording native sizes.
- Empty-window case returns zero-row CropBatch on correct device.
- No affine math reimplemented.
- Dead `_extract_canonical_gpu_legacy` removed cleanly.
- `black` and `isort` applied (file unchanged — already compliant).

## Concerns

None. All tests pass, no regressions in existing crops tests.

## Commit

`2e6ee24 feat(inference): unified cross-frame GPU-native crop extraction (extract_crops -> CropBatch); drop dead legacy gpu path`

---

## Fix pass (review findings)

### Test command and output

```
PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE /Users/neurorishika/miniforge3/envs/hydra-mps/bin/python -m pytest tests/test_inference_extract_crops_batch.py tests/test_inference_crops.py -v

============================= test session starts ==============================
platform darwin -- Python 3.13.12, pytest-8.x, pluggy-1.x
collected 2 items

tests/test_inference_extract_crops_batch.py::test_extract_crops_concatenates_window_in_detection_id_order PASSED [ 50%]
tests/test_inference_crops.py::test_oversize_native_crop_is_resized_not_truncated PASSED [100%]

============================== 2 passed in 3.40s ===============================
```

### Finding 1 (DRY) — what changed

Extracted two private helpers from the duplicated code in `_extract_canonical_window`'s CPU branch:

- `_frame_as_hwc_numpy(frame)` — the tensor→numpy CHW→HWC conversion that was copy-pasted into every function that accepts a `np.ndarray | torch.Tensor` frame.
- `_warp_crops_for_obb(arr, obb, aspect_ratio, padding_fraction)` — the per-detection `_warp_canonical_crop` loop that previously existed independently in `_extract_canonical_cpu` and `_extract_canonical_window`.

Both `_extract_canonical_cpu` and `_extract_canonical_window` CPU branch now call these helpers. No warp/affine math was reimplemented.

### Finding 2 (Correctness — CPU/GPU divergence on oversize crops) — what changed

**GPU path semantics confirmed:** `gpu_canonical_crop_batch` uses PyTorch `F.affine_grid` + `F.grid_sample` with `mode="bilinear"` and `padding_mode="border"` to warp each detection directly to `(out_w, out_h)`. Crops larger than `out_size` are downscaled; crops smaller than `out_size` have their border pixels replicated. No hard-crop, no zero-padding.

**CPU fix:** Replaced the old `crop[:out_h, :out_w]` hard-crop (which silently discarded pixels for oversize crops and recorded the wrong native dimensions) with `cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)`. This matches the GPU path's bilinear resize-to-target behavior. `native_sizes` continues to record the true native pre-resize `(h, w)` in both paths, consistent with the GPU path.

### New test

`tests/test_inference_crops.py::test_oversize_native_crop_is_resized_not_truncated`

Constructs a 120×60 px OBB in a 256×256 frame (native crop >> 32×32 `out_size`), runs `_extract_canonical_window` on the CPU path, and asserts:
1. Output shape is exactly `(1, 3, 32, 32)`.
2. `native_sizes` records dimensions strictly larger than `out_size` (proving a resize occurred).
3. The crop is not all-zero (content was preserved).
4. Last row and column are not all-zero (no hard-crop/zero-pad artifact).
