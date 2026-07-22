# GPU-Native Classifier Crop Path (NVDEC → grid_sample → predict_batch_cuda)

**Date:** 2026-07-22
**Status:** Design (approved) — pending implementation plan
**Author:** Rishika Mohanta
**Related memory:** `project_runtime_gen2_core_done`, `project_pose_runtime_golden_rule`

## Problem

On the CUDA box, video frames are decoded on the GPU via NVDEC
(`NvdecFrameReader`, `sources.py:184`; active when `_nvdec_available()` — verified
on mehek: PyNvVideoCodec 2.0.2 + cupy 13.6.0). Frames arrive as CUDA `uint8`
tensors, explicitly **not** bit-identical to cv2 CPU decode.

The **classifier** stages (`run_headtail_batch`, `run_cnn_batch`, and their
per-frame siblings) extract crops through `_frame_as_hwc_numpy`
(`crops.py:100`), which does `frame.cpu().numpy()` — pulling the **entire NVDEC
GPU frame back to CPU** every frame — then warps per crop with `cv2.warpAffine`.
This is:

- **A real roundtrip.** The full frame is copied device→host each frame purely so
  cv2 can run. (The classifier *forward* already runs on GPU via `_forward_torch`
  → `.to(cuda)`, `backend.py:941-947`; the roundtrip is the *crop*, not the
  forward.)
- **Inconsistent.** OBB and pose crops on CUDA already use a GPU `grid_sample`
  path (`_extract_canonical_gpu`, `crops.py:45-52`, gated on
  `runtime.tensor_on_cuda`). Only the classifier crops force a CPU detour.
- **The dominant cost of the dense-colony 1.49× regression** (authoritative
  benchmark 2026-07-21: `ant_cnn_identity` CUDA legacy 71.1s → current 105.8s).
  The cost scales with crop count (thousands of headtail/cnn crops per clip),
  each paying a CPU cv2 warp plus a per-frame frame D→H.

Because NVDEC already makes the CUDA pipeline non-bit-identical to legacy (and to
MPS), byte-identity-to-legacy is **not** an available correctness bar on CUDA.
The fix is to make the classifier crop path GPU-native and consistent with the
pose/OBB path.

## Goal

On a GPU tier with NVDEC active, the classifier crop **and** forward stay on the
GPU end-to-end — NVDEC frame → `grid_sample` crop at the classifier input size →
`predict_batch_cuda` — eliminating the per-frame frame D→H and the per-crop CPU
cv2 warp. On non-CUDA paths (MPS/CPU/NVDEC-disabled) behaviour is unchanged.

Non-goals: AprilTag/AABB crops (`extract_aabb_crops`, niche — stays CPU); the
pose path (already GPU); any change to the `gpu` vs `gpu_fast` golden rule; MPS
behaviour.

## Design

### Platform split (keyed on `runtime.tensor_on_cuda`)

- **`tensor_on_cuda == True`** (CUDA + NVDEC): new pure-GPU classifier path.
- **`tensor_on_cuda == False`** (MPS / CPU / `HYDRA_DISABLE_NVDEC` / onnx-cpu):
  the existing cv2 CPU path, **unchanged**. MPS stays byte-identical to legacy.

This mirrors the existing OBB/pose crop split, so there is one consistent rule
for "crops follow the frame's device."

### Three coupled changes

The GPU crop only pays off if the forward also stays on GPU — a GPU crop handed
to numpy `predict_batch` is just pulled back to CPU. So the three move together:

1. **GPU classifier crop extractor.** A `grid_sample`-based extractor that warps
   the on-GPU frame directly to the classifier's `input_size` (`out_w, out_h`),
   producing an `(N, C, input_h, input_w)` CUDA tensor, BGR channel order,
   uint8-quantized on-GPU (`floor`/`round` to match the cv2 reference's 8-bit
   crops as closely as possible). Parallels `_extract_canonical_gpu` but targets
   the classifier input size rather than the native canonical size.

2. **`predict_batch_cuda` factor-bundle support.** Add `_forward_multi_cuda` in
   `ClassifierBackend` so a `classifier_multihead` / `yolo_multihead` bundle whose
   sub-backends each expose a CUDA forward runs every factor's forward on the
   shared on-GPU batch and assembles per-factor probabilities, mirroring the numpy
   `_forward_yolo_multi` (log→concat→softmax) for output-shape parity. This
   removes the blanket numpy fallback at `backend.py:1151` for CUDA-capable
   factors. (The colortag fixture bundle is two native `.pth` factors.)

3. **Stage routing.** In `run_headtail_batch` / `run_cnn_batch` (and per-frame
   siblings), when `runtime.tensor_on_cuda`: build the crop on-GPU (change 1) and
   call `predict_batch_cuda` (change 2), keeping tensors on-device. Otherwise the
   current cv2 + `predict_batch` path.

### Strict capability check (fail fast, load time)

On a GPU tier with NVDEC active, at classifier **load** verify the backend — and,
for a bundle, every factor — exposes a CUDA-native forward (native torch or ONNX
IOBinding). If any does not (e.g. a YOLO-classifier factor, CoreML on a
non-NVDEC host is N/A), raise a clear error naming the model and stating the
constraint: *the gpu tier requires CUDA-native classifiers under NVDEC.* This
turns an unsupported config into an immediate, legible failure rather than a
mid-batch exception or a silent per-frame CPU fallback.

Rationale for strict (vs best-effort CPU fallback): the stated principle is
"always pure-GPU paths on GPU/GPU-Fast." A silent CPU fallback would reintroduce
exactly the roundtrip this work removes, invisibly. The fixtures in use are all
CUDA-capable, so strict is workable today; YOLO-classifier factor bundles on the
gpu tier become an explicit, documented non-support.

## Acceptance / Verification

- **Determinism:** `new_a == new_b` byte-identical (the current pipeline is
  already deterministic; the GPU crop path must stay so).
- **Agreement vs current CUDA:** on `ant_cnn_identity`, positions within the
  determinism floor and **identity-label agreement ≥ 99%** against today's CUDA
  pipeline. (Byte-identity is not required — `grid_sample` ≠ cv2 — but the
  identity decisions must be materially unchanged.)
- **Performance:** the primary objective — the 1.49× on `ant_cnn_identity`
  (CUDA) closes materially. Report the new ratio.
- **MPS regression guard:** MPS output byte-identical before/after (its path is
  untouched), confirmed on `ant_obb_sleap` + `ant_cnn_identity`.

## Testing

- **Unit:** GPU classifier crop returns the right shape/dtype/device and input
  size; the strict capability check raises on a non-CUDA backend / factor;
  `_forward_multi_cuda` output shape matches numpy `_forward_yolo_multi`.
- **Integration:** the CUDA equivalence run on mehek (determinism + agreement +
  perf); MPS byte-identity locally. Row-count > 1 verified on every CSV (the
  documented empty-CSV trap; the new fail-loud guard from commit `8d1d8ef` backs
  this).

## Risks & Open Questions

- **grid_sample vs cv2 crop divergence** propagates into identity decisions; the
  ≥99% agreement gate bounds acceptable drift. If agreement falls short, revisit
  the on-GPU quantization / sampling convention to better match cv2 INTER_LINEAR.
- **CUDA-only validation.** The GPU path exercises only on mehek (flaky). Mitigate
  with focused unit tests that run anywhere (crop shape/device, strict-check,
  factor-forward shape) so most correctness is provable off-box; reserve the box
  for the final determinism/agreement/perf gate.
- **Observability gap (noted, not in scope):** the current in-memory pipeline does
  not write a SLEAP service log (`posekit/logs`) the way the legacy disk path did.
