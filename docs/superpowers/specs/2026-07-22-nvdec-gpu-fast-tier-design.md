# NVDEC Decode Confined to the GPU-Fast Tier

**Date:** 2026-07-22
**Status:** Design (approved) — pending implementation plan
**Author:** Rishika Mohanta
**Related:** `2026-07-22-gpu-native-classifier-crop-design.md`, memory `project_runtime_gen2_core_done`
**Scope:** Part 1 of 2. Part 2 (harden classkit training path with decode/crop augmentation) is a separate, deferred spec.

## Problem

NVDEC hardware video decode is a real speedup on decode/detection-bound work, but a
hardware-decoded frame is **not** color-equivalent to a cv2 CPU-decoded frame. For an
**untagged** stream, cv2 hardcodes BT.601 + limited range while NVDEC guesses its own
convention; even after matching to BT.601 limited range on-GPU (see the related spec),
an irreducible **~max-2 / mean-0.48** per-channel residual remains (chroma upsampling +
swscale-vs-cvtColor differences). Verified impact on a colortag identity classifier:
~6% per-frame label flips from decode color + ~4% from the grid_sample vs cv2 crop,
compounding to ~44% final identity divergence.

Today NVDEC activates on **any** CUDA tier (`use_nvdec = _nvdec_available()` whenever
`cuda_mode`, runtime.py:142). That means the **GPU tier** — which users expect to be
byte-identical to CPU/MPS — silently decodes on NVDEC when the codec/resolution allows,
introducing non-determinism vs the CPU-decode path. (It's dormant today only because the
H.264-Level-6 4512² colony clips exceed NVDEC's H.264 4096 limit and fall back to CPU;
HEVC/AV1 or smaller clips would trigger it.)

## Goal

Confine NVDEC to the **GPU-Fast tier only**. The GPU-Fast tier already means "fastest
path, accepts small numerical differences" (it uses exported TensorRT, which is not
byte-identical to native). NVDEC's small color residual fits that contract exactly.

- **GPU tier:** CPU decode (`CpuFrameReader`) → **byte-identical** to CPU/MPS. Never NVDEC.
- **GPU-Fast tier:** NVDEC decode when available/decodable; frames flow end-to-end
  through the on-GPU crop path (grid_sample) and `predict_batch_cuda`, with the BT.601
  color conversion already implemented. The small residual is **accepted** here and made
  robust later via Part 2 (training-path hardening) — not a correctness gate for Part 1.
- **CPU / MPS tiers:** unchanged.

Non-goals: the classkit training-path hardening (Part 2); eliminating the residual
(shown irreducible); changing the `gpu` vs `gpu_fast` model-backend selection rules.

## Design

### 1. Gate NVDEC on the tier (runtime.py `RuntimeContext.from_config`)

```
nvdec = _nvdec_available() and config.runtime_tier == "gpu_fast"
```

(was: `_nvdec_available()` for any `cuda_mode`). Consequences:
- GPU tier → `use_nvdec=False` → `make_frame_source` returns `CpuFrameReader` →
  numpy BGR frames → byte-identical decode.
- GPU-Fast tier → `use_nvdec=True` → `NvdecFrameReader` (when the clip is NVDEC-decodable;
  it already falls back to `CpuFrameReader` per-clip on MBCount/codec limits).

### 2. Decouple "frame is on GPU" from "native torch" for crop routing

The GPU grid_sample crop path currently keys off `runtime.tensor_on_cuda`, which is
`cuda_mode AND obb_backend=="torch"`. On GPU-Fast the OBB backend is `tensorrt`, so
`tensor_on_cuda` is False — yet NVDEC frames are genuine CUDA tensors that should take
the on-GPU crop path. Update the crop gate (`crops.frames_on_cuda`) to activate when the
frame is **actually a CUDA tensor** on a GPU tier, independent of `tensor_on_cuda`:

```
frames_on_cuda(runtime, frames) := runtime.requested_gpu
    and frames has a torch CUDA tensor as its first non-None frame
```

`predict_batch_cuda` already covers both classifier execution backends the GPU-Fast tier
produces: native torch (`_forward_torch_cuda`) and ONNX/TensorRT via IOBinding
(`_forward_onnx_iobinding`). The strict load-time capability check (from the related spec)
guarantees a GPU-tier classifier exposes one of these, so a GPU-Fast NVDEC frame never
falls to the RGB-unaware CPU crop path.

### 3. Confirm the OBB + pose stages consume NVDEC frames on GPU-Fast

The OBB stage already letterboxes CUDA HWC RGB tensors (obb.py:74/102) for the native
path; the implementation plan must confirm the **TensorRT** OBB path on GPU-Fast accepts
the same CUDA HWC frame (or normalizes it) — this is the one open detail to verify, since
GPU-Fast OBB is `tensorrt`, not native ultralytics. Pose/classifier GPU crop paths already
handle NVDEC frames (related spec).

### 4. Color handling (already implemented, keep)

NVDEC frames are RGB via the on-GPU BT.601 limited-range conversion
(`sources._nv12_to_rgb_bt601`); the classifier GPU path passes `input_is_bgr=False`.
No change; just confirm it only runs on GPU-Fast now.

## Acceptance / Verification

- **GPU tier byte-identical:** `ant_cnn_identity` + `ant_obb_sleap` on the GPU tier (CUDA)
  are **byte-identical** before vs after this change (pos/θ max 0), confirming NVDEC no
  longer engages there. This is the primary correctness gate.
- **GPU-Fast NVDEC end-to-end:** an NVDEC-decodable clip (lossless-HEVC transcode of
  `ant_cnn_identity`) runs to completion on GPU-Fast, produces valid non-empty output,
  and is deterministic (`new_a == new_b`). The colortag identity residual vs CPU decode
  is **reported, not gated** (Part 2 closes it). Confirm NVDEC actually engaged (no
  `falling back to CpuFrameReader` for that clip) and measure the decode speedup.
- **MPS/CPU unchanged:** byte-identical.

## Testing

- **Unit:** `RuntimeContext.from_config` sets `use_nvdec` True only for `gpu_fast` (and
  only with `_nvdec_available()`), False for `gpu`/`cpu`; `frames_on_cuda` returns True for
  a CUDA-tensor frame on a GPU tier regardless of `tensor_on_cuda`, False for numpy frames.
- **Integration (mehek):** GPU-tier byte-identity vs `main`; GPU-Fast HEVC run
  (determinism + NVDEC-engaged + perf). MPS byte-identity locally.

## Risks & Open Questions

- **TensorRT OBB + CUDA HWC frame (§3):** the one path not yet exercised with NVDEC frames;
  the plan must verify or add normalization. If TensorRT OBB needs CPU numpy, GPU-Fast may
  do one frame D→H for detection while keeping crops on-GPU — still correct, slightly less
  optimal; acceptable for Part 1.
- **Residual accepted, not fixed:** colortag identity on GPU-Fast NVDEC stays ~44%-divergent
  from CPU decode until Part 2 (training-path hardening) lands. This spec deliberately does
  not gate on it — GPU-Fast is opt-in and documented as "fast, small-delta."
- **Clip decodability:** NVDEC silently falls back to CpuFrameReader for clips it can't
  decode (H.264 >4096, MBCount limits). GPU-Fast then equals GPU-tier decode for those
  clips — correct, just no speedup. Log which reader was used.
