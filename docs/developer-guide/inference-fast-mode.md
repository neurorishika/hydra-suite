# Inference Fast Mode

This page documents the **shipped** fast-mode tiers for the HYDRA Suite inference
pipeline.  Phase 1 established determinism-preserving GPU cleanups; Phase 3 adds
native accelerator export for both CUDA (TensorRT) and Apple Silicon (CoreML).

---

## Fast-Mode Tiers

| Platform | Tier | Format | Status |
|---|---|---|---|
| NVIDIA GPU | TensorRT fp32 | `.engine` | Shipped (Phase 2/3) |
| Apple Silicon | CoreML | `.mlpackage` | Shipped (Phase 3) |
| CPU / other | Native PyTorch | in-memory | Always available (best-effort) |

### CUDA → TensorRT

On CUDA hardware the pipeline auto-exports the detector (OBB model) to a TensorRT
`.engine` file on first run.  The artifact is cached next to the source `.pt` file
and reused on subsequent runs.

* Format: TensorRT FP32 engine
* Precision: fp32 (numerically close to the PyTorch baseline but NOT bit-identical
  — kernel fusion reorders ops; see the Determinism note below)
* Trigger: `compute_runtime = "tensorrt"` in TrackerKit / ClassKit config
* Artifact path: `<model>.engine` adjacent to the source `.pt`

### Apple Silicon → CoreML

On Apple Silicon (MPS) the pipeline auto-exports both the OBB detector and all
classifier families to CoreML `.mlpackage` bundles on first run.

* Format: CoreML `.mlpackage` (directory bundle)
* Compute units: `ALL` (Neural Engine + CPU + GPU)
* Minimum deployment target: macOS 13
* Trigger: `compute_runtime = "coreml"` in TrackerKit / ClassKit config
* Artifact path: `<model>.mlpackage` adjacent to the source `.pt` / `.pth`

---

## Coverage Matrix

The following table shows which classifier families have a shipped CoreML exporter.
All families are covered — no family is left without a CoreML export path (spec §7
hard requirement).

| Family | Exporter function | Test |
|---|---|---|
| TinyClassifier | `training.tiny_model.export_tiny_to_coreml` | `test_fastmode_coverage_matrix.py::test_tinyclassifier_family_exports_coreml` |
| Torchvision (resnet18, efficientnet\_b0, mobilenet\_v3\_small, …) | `training.torchvision_model.export_torchvision_to_coreml` | `…::test_torchvision_resnet18_family_exports_coreml` |
| TIMM (timm/\*) | `training.torchvision_model.export_torchvision_to_coreml` (same path) | `…::test_timm_family_exports_coreml` |
| YOLO / OBB (ultralytics) | `ultralytics YOLO.export(format="coreml")` | `…::test_yolo_obb_family_exports_coreml` |

All four families export a real `.mlpackage` directory bundle on Apple Silicon.  The
coverage-matrix test (`tests/test_fastmode_coverage_matrix.py`) is skipped
automatically on non-Mac platforms and when `coremltools` is absent.

---

## Best-Effort Native-GPU Fallback Contract

When a CoreML or TensorRT artifact cannot be produced (missing dependency, export
error, unsupported model variant), the pipeline falls back to native PyTorch GPU
inference rather than crashing:

1. The runtime resolver detects that `coreml` / `tensorrt` is requested.
2. If the `.mlpackage` / `.engine` artifact is missing and `auto_export=True`, an
   export is attempted.
3. If the export fails, a warning is logged and the resolver downgrades to the next
   available runtime (`mps` → `cpu` on Apple, `cuda` → `cpu` on NVIDIA).
4. The fallback is surfaced in the TrackerKit status bar (Phase 3, Task 7).

This contract means the pipeline is never hard-blocked by an export failure — it
degrades gracefully at the cost of losing the accelerated code path.

---

## Determinism Note

CoreML and TensorRT inference is **not bit-identical** to the PyTorch baseline:

* CoreML: the Neural Engine may use reduced-precision arithmetic internally even
  for fp32 models.  Run-to-run results on the same hardware are **deterministic**
  (same input → same output), but the exact values may differ from a PyTorch CPU
  or MPS run by small floating-point rounding errors.
* TensorRT fp32: also deterministic run-to-run but may differ slightly from PyTorch
  due to kernel fusion and cuDNN algorithm selection.

Identity classification thresholds and tracking assignment logic are robust to these
small numerical differences.  The determinism test suite (`test_coreml_determinism.py`)
verifies run-to-run reproducibility on Apple Silicon.

---

## What Was Intentionally Not Shipped

The table below lists levers that were evaluated and deferred:

| Lever | Reason not shipped |
|---|---|
| `channels_last` (CUDA) | No-op on real 96×96 classifier crops; breaks OBB inference (`fuse_conv_and_bn` `.view()` error on channels-last tensors) |
| `channels_last` (MPS) | ~54% regression at batch 64 |
| TensorRT fp16 | Non-bit-identical (−4.5% accuracy risk on small-crop classifiers); deferred to an explicit `GPU-Fast-fp16` sub-option |
| TF32 matmul | Breaks fp32 determinism guarantee |
| `cudnn.benchmark` | Run-to-run nondeterminism |

Phase 1 shipped only safety-preserving cleanups (`inference_mode`, pinned/non-blocking
H2D uploads) because the real pipeline was already well-optimized and the only
remaining exact lever (`channels_last`) proved unsafe.
