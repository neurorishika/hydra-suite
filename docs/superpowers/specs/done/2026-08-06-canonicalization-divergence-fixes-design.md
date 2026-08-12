# Canonicalization Divergence Fixes — Design

**Date:** 2026-08-06
**Status:** Approved (design), pending implementation plan
**Scope:** Follow-up to `feat/global-canonicalization` (merge `0ca40789`). Closes the
device- and format-dependent divergences that an adversarial review found in the shipped
two-layer canonicalization, without weakening any of its seven goals.

## Background

The global-canonicalization refactor produces every animal crop in two layers:

- **Layer 1** — `canonical_affine` builds a rigid affine (rotation + translation, *no
  scale*) that warps a detection onto a project-fixed canvas. Verified geometrically
  exact; body size survives as signal (Reqs 3, 4 hold).
- **Layer 2** — `fit_to_model_input` + `apply_fit` isotropically letterbox the canvas
  crop into a model's input size. Verified single-scale, no anisotropy (Req 2 holds on the
  CPU path).

An adversarial, from-scratch review (2026-08-06) judged the merged code against seven
requirements: (1) one canonicalization, (2) no distortion, (3) size preserving,
(4) uniform output, (5) clip-not-rescale, (6) training and inference produce the same
model input, (7) no unnecessary round trips. Reqs 1–5 and 7 largely hold. The failures
cluster on the **GPU pixel path**, one **backend double-resample**, and **guards/formats
that silently corrupt training data**.

### Findings this design closes

| ID | Severity | Summary | Verified |
| --- | --- | --- | --- |
| **F1** | Major | On CUDA GPU tiers, crops are produced with `F.grid_sample` (Layer 1) + `F.interpolate` bilinear (Layer 2); training crops are produced with `cv2.warpAffine` + `cv2.resize` (INTER_AREA/LINEAR). The kernels differ (measured: downscale max Δ 71/255, mean 3.0, ~20% of pixels off by >2). Because training is *always* cv2, retraining on cv2 data does **not** converge them — a permanent train/infer domain shift on NVIDIA+NVDEC. | Yes (numeric) |
| **F3** | Major | ViTPose applies a **redundant** `apply_fit` letterbox on the non-CUDA branch (`pose.py:451`) that CUDA and training both skip; ViTPose's own `box2cs`/`top_down_affine` already *is* Layer 2. Result: MPS/CPU ViTPose ≠ CUDA ViTPose ≠ training. | Yes |
| **F2** | Real crash, narrow | Two CUDA classifier fallback paths call `self.predict_batch(numpy_crops, input_is_bgr=...)` but `predict_batch` (`backend.py:1338`) has no such parameter → guaranteed `TypeError`. | Yes |
| **F4** | Major (data) | Crop-dataset (`generator.py`) and oriented-video (`oriented_video.py`) exporters accumulate clip counters but only write them to `metadata.json`, which nothing reads or surfaces. Req 5 ("that fact REPORTED") unmet on the two paths that build training data. | Yes |
| **F5** | Real, format-gated | Crop-dataset export honors `jpg`; training then reads DCT-recompressed crops while inference uses uncompressed canonical crops. Breaks Req 6 whenever the operator selects jpg. | Yes |
| **F6** | Latent | Interpolated-crops `apply_fit`-failure branch (`interpolated_crops.py:575`) appends a raw canvas crop yet still composes `fit_m` into back-projection → geometrically incoherent. Effectively dead, but a silently-wrong second path. | Yes |
| **F7** | Structural | (a) params→geometry derivation duplicated in three places; (b) three hardcoded `from_reference(20,2,1.3)` fallbacks fire on a missing `geometry=`; (c) ClassKit eval forces square `CanonicalFitTransform((sz,sz))`; (d) `canonical_margin` absent from `default.json`. | Yes |

Test-adequacy review additionally found the only GPU pixel-parity test is **square-only and
too-loose** (`mean-abs<0.03`), no test asserts CUDA train==infer, and the inference/export
clip guards have no driving test — so F1/F2/F3/F4 would all ship green.

## Decisions

Two forks were resolved with the user:

1. **Resampling strategy → unify on one torch kernel.** Retire `cv2` from the crop path.
   A single torch resampler (grid_sample + antialiased `F.interpolate`) is used for
   training-data generation *and* for inference on every device. Rationale: a numeric test
   confirmed `torch antialias ≠ cv2 INTER_AREA` (max 53/255, ~86% of pixels differ), so two
   implementations cannot be made to agree — the only coherent options are one kernel or
   model-invariance. One kernel makes "one canonicalization" true *in code*, keeps the
   on-GPU no-round-trip path (Req 7), and the accepted retrain absorbs the cost.
2. **Robustness augmentation → Moderate.** Training-only perturbation covering resample
   kernel, sub-pixel warp jitter, and mild blur/JPEG, so models are invariant to residual
   CPU/CUDA/MPS float noise *and* to ONNX/TensorRT/CoreML internal resampling.

Three design-detail confirmations:

- Crop-dataset (model-input) export becomes **lossless-only**; jpg is removed there. (F5)
- `geometry` becomes a **hard-required** argument on crop/stage helpers; the hardcoded
  fallback geometries are deleted (fail-loud, not silently-divergent). (F7b)
- ViTPose's internal `top_down_affine` **stays cv2** — it is model-internal preprocessing,
  identical between training and inference on every device, so it is not a "second
  canonicalization." Only *our* Layer 2 is unified.

## Non-goals

- No change to Layer 1 geometry math (verified correct).
- No re-derivation of `REFERENCE_BODY_SIZE` or the Kalman/Hungarian knobs.
- No change to the tracking/assignment/post-processing algorithms.
- No attempt to make crops byte-identical *across* devices — within-device determinism
  (reproducible reruns) is preserved; cross-device parity is delivered via augmentation,
  not bit-equality.

## Architecture

### One resampler, one seam

New module `src/hydra_suite/core/canonicalization/resample.py` exposes the *only* warp and
resize used anywhere on the crop path:

```
canonical_warp(frame_t, M_align, geometry) -> crop_t          # Layer 1, grid_sample
canonical_warp_batch(frame_t, M_aligns, geometry) -> crops_t  # batched Layer 1
letterbox_fit(crop_t, model_wh) -> tensor                     # Layer 2, F.interpolate(antialias=True)+zero-pad
letterbox_fit_params(source_wh, model_wh) -> FitResult        # pure arithmetic (unchanged from fit.py)
```

- All operate on `torch` tensors on whatever device the tensor already lives on. There is
  no CPU/GPU branch — device is a property of the input, not a code path.
- `cv2.warpAffine` and `cv2.resize` are removed from the crop path. The following collapse
  onto the seam:
  - `crop.py`: `extract_canonical_crop`, `gpu_canonical_crop`, `gpu_canonical_crop_batch`
    → thin wrappers over `canonical_warp*` (numpy-in/numpy-out variants convert once).
  - `fit.py`: `apply_fit` → wraps `letterbox_fit`; `fit_to_model_input`/`fit_affine`
    retained (pure arithmetic, kernel-independent).
  - `stages/crops.py`: `apply_fit_gpu`, `_extract_canonical_cpu`, `_extract_canonical_gpu`,
    `extract_classifier_crops*` (cpu/gpu pairs) unify into single device-agnostic
    functions. The `apply_fit_batch` thread-pool stays only where a numpy disk-export path
    needs it.
  - `training/canonical_transform.py`: `CanonicalFitTransform` calls `letterbox_fit` on a
    CPU tensor and returns numpy uint8 (interface unchanged for downstream PIL/torchvision
    augmentations).

- **Foreign masking** is exact polygon rasterization (fill regions, no resampling), so it
  is already device-independent. It is applied as an on-device tensor multiply after the
  warp. Unchanged in behavior.

- **`letterbox_fit`** uses `mode="bilinear", align_corners=False, antialias=True`.
  `antialias` is a no-op on upscale in torch (verified — no error, ignored when
  `scale >= 1`), matching the intent of the old INTER_LINEAR-up branch, and applies the
  area-like anti-aliasing filter only on downscale. Zero padding, centered offset, identical
  arithmetic to the current `FitResult`.

### Train==infer contract

The **un-augmented** transform is byte-identical between training-data generation and
inference: both call `canonical_warp` + `letterbox_fit` with the same arguments on the same
(CPU) inputs during dataset export, and the same functions at inference. Augmentation (see
below) is an explicit, opt-in *training-time* perturbation layered on top — it is a
deliberate divergence for robustness, not an unaccounted one. A test pins the clean-path
identity (§Testing).

### Consequences accepted

- **Within-device determinism holds** (grid_sample/interpolate are deterministic on a fixed
  device); reproducible reruns and the DETERMINISM floor of the equivalence harness are
  unaffected.
- **Cross-device** crops differ by sub-gray-level float noise — never bit-equal, as was
  already true for the pre-existing grid_sample path. Covered by augmentation.
- **Equivalence harness re-baseline (one-time, both platforms):** controls `fly_obb`,
  `worm_bgsub` (no crop consumers) stay byte-identical; crop-consuming clips change
  cv2→torch **by design** and are re-baselined. This is the same re-baseline discipline the
  original merge used.
- **CPU-tier performance:** `grid_sample` on CPU for many small crops may be slower than
  `cv2.warpAffine`. Mitigated by the existing batched warp (one `grid_sample` call per
  frame). Validated against `PERF_TOLERANCE` on both platforms; if a real regression
  appears, the batched path is the lever.

## Per-finding resolution

- **F1** — Resolved by the seam: training data and inference call identical torch
  functions. No cv2/torch split remains for classifiers, SLEAP-exported, or YOLO.
- **F3** — ViTPose (and any backend whose model does its own crop preprocessing) declares
  an **identity Layer-2 fit on all devices** via the existing `preferred_input_size==0`
  mechanism SLEAP-native uses. The non-CUDA `apply_fit` at `pose.py:451` is removed; both
  branches feed the raw canonical canvas. ViTPose's `box2cs`/`top_down_affine` is the one
  resample, identical train vs infer vs device. Tests `test_pose_model_input_wh_non_square`
  and `test_vitpose_backend_geometry` are rewritten to assert the identity-fit contract
  instead of pinning the redundant resample.
- **F2** — `predict_batch(self, crops, input_is_bgr: bool = True)`: honor the flag (skip
  the BGR→RGB flip when already RGB), matching `predict_batch_cuda`/`_preprocess_cuda`
  semantics. Call sites `backend.py:1264/1284` now resolve. A test drives both fallback
  branches (a factor lacking a CUDA forward; an unknown execution backend).
- **F4** — `generator.py` and `oriented_video.py` route their per-detection overflow
  through the shared `ClippingStats`. `finalize()` (and the oriented-video writer) call
  `ClippingStats.summary()` → `logger.warning` and include the line in the export summary.
  A test asserts an overflowing OBB both increments the counter and surfaces the warning.
- **F5** — Crop-dataset export (model input) is **lossless-only**, defaulting to PNG (the
  current default); the jpg option is removed from that path. Oriented-video export
  (human-facing, not a model input) keeps jpg.
- **F6** — The interpolated-crops `apply_fit`-failure branch **skips the detection**
  (returns `None`, matching the degenerate-OBB path) rather than feeding a raw crop with a
  composed `fit_m`. The incoherent second path is deleted.
- **F7** — (a) All params→geometry derivations call the single
  `canonical_geometry_from_params`; `inference/config.py` and the dataclass fallback stop
  re-inlining `from_reference`. (b) `geometry` is a required argument on the crop/stage
  helpers; the hardcoded `from_reference(20,2,1.3)` fallbacks in `cnn.py`, `headtail.py`
  (stage), and `classification/headtail.py` are deleted — a missing geometry raises.
  (c) ClassKit eval builds `CanonicalFitTransform((input_h, input_w))`. (d) `default.json`
  (and shipped presets) gain `canonical_margin` and `reference_body_size`.

## Robustness augmentation (Moderate)

A training-only transform `CanonicalAug`, applied to the canonical crop *before*
`letterbox_fit`, shared by every crop-consuming trainer (CNN identity, head-tail, ViTPose,
SLEAP, YOLO-pose, YOLO-classify):

- **Resample kernel** ∼ `U{torch-bilinear, torch-antialias, cv2-AREA, cv2-LINEAR}` — the
  crop is resized to the model input with a randomly chosen kernel each sample.
- **Sub-pixel warp jitter**: `dx, dy ∼ U(-0.5, 0.5)` px, `dθ ∼ U(-1, 1)°` folded into
  `M_align` before the warp.
- **Mild degradation**: with `p = 0.3`, a small gaussian blur or JPEG round-trip
  (`q ∈ [85, 100]`).

Off by default at inference and in the clean train==infer test. Enabled via the trainer's
augmentation config. Kept intentionally mild to avoid degrading fine-feature accuracy
(tags, bristles); the Heavy photometric option was explicitly declined.

## Testing / guards

New or rewritten tests (targeting the exact blind spots the review named):

1. **Non-square GPU pixel-parity** — replace the square 128×128 / `mean-abs<0.03` test with
   a non-square canvas → non-square model, asserting CPU-torch vs (CUDA-torch, run on the
   CUDA box) agree to a tight float tolerance *and* a landmark-position check (catches
   anisotropy and any H,W↔W,H swap).
2. **CUDA train==infer** — assert the un-augmented training transform tensor equals the
   inference tensor for the same crop on the GPU code path.
3. **Clip-guard surfacing** — `runner`/`pipeline` and both exporters increment
   `ClippingStats` on an overflowing OBB *and* emit the warning in a real run.
4. **ViTPose identity-fit** — assert ViTPose (and SLEAP-native) receive the raw canonical
   canvas (identity Layer-2) on both branches; one resample total.
5. **F2 fallback** — both CUDA classifier fallback branches execute without `TypeError` and
   return correctly-oriented (RGB) results.
6. **F7 fail-loud** — a crop/stage helper called without `geometry` raises, not
   silently-defaults.

Cross-device numeric checks that require CUDA run on the mehek box (`hydra-cuda`), per the
existing equivalence workflow.

## Sequencing

1. **Seam + safe fixes** (no retrain): resampler module; collapse crop.py/fit.py/crops.py
   onto it; F2, F6, F7. Re-baseline the equivalence harness on MPS and CUDA (controls
   byte-identical; crop clips changed-by-design).
2. **F3** ViTPose identity-fit + test rewrites.
3. **F4/F5** export clip surfacing + lossless-only crop datasets; `CanonicalAug`.
4. **Retrain**: regenerate crop datasets (do **not** reuse — the old ones are cv2-baked and
   jpg-lossy) and retrain all crop-consuming models against the unified path with Moderate
   augmentation.

## Acceptance

- Equivalence harness: controls byte-identical, DETERMINISM exact, PERFORMANCE within
  `PERF_TOLERANCE`, on **both** MPS and CUDA. Crop clips re-baselined and documented.
- New tests (1–6 above) pass, including the CUDA-side numeric checks on mehek.
- No `cv2.warpAffine`/`cv2.resize` remaining on the crop path (grep gate).
- Retrained models evaluated on held-out data show no accuracy regression vs the
  pre-refactor baseline on either platform.
