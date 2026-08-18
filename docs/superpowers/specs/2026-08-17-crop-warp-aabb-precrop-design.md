# Slice A — AABB Pre-Crop Canonical Warp

**Date:** 2026-08-17
**Status:** Design approved, pending spec review
**Branch:** `perf/crop-warp-aabb-precrop`

## Context

A CUDA profile of the full headless tracker on `ant_pose_headtail` (500 frames,
18 animals/frame) attributed wall-clock as follows (total 214 s under cProfile):

| Cost | Share | Root cause |
|---|---|---|
| Crop canonicalization warp (`canonical_warp_batch`) | **~32%** (69 s) | full-frame `expand(N).contiguous()` + N full-res `grid_sample` |
| Postprocessing (pandas) | ~28% | Slice B |
| OBB detection | ~9% | — |
| Video random-seek | ~6% | Slice C |
| Headtail EfficientNet | ~4.5% | — |
| Pose / SLEAP | ~4.6% | Slice D |

The single largest self-time entry across the whole run was
`torch.Tensor.contiguous()` at **47 s (22%)**, produced by
`resample.py:123`:

```python
frame_expanded = frame_chw.unsqueeze(0).expand(n, -1, -1, -1).float()
crops = F.grid_sample(frame_expanded.contiguous(), grid, ...)
```

For a frame with `N` detections this materializes **N copies of the entire
frame** (e.g. 18 × a 2048² frame) and then runs N full-resolution
`grid_sample`s. Because cv2 decodes to CPU numpy, `extract_canonical_crops`
sets `device="cpu"` (`crops.py:88`), so the whole warp runs **on CPU** — which
is also why GPU utilization is ~0–1% on pose clips.

The canonical canvas is derived from the reference body size
(`reference_body_px 66.94 × √aspect 2.45 × margin ≈ 209×85`). The frame-space
footprint of that canvas for one detection is only ~230² px, so the warp is
sampling ~2048² of frame to fill an ~209×85 canvas, and doing it N times.

### Measurement provenance
Harnesses (mehek, `hydra-cuda`): `sleap_service_roundtrip_bench.py`,
`pose_share_e2e.py`, `profile_pipeline.py` (scratchpad). Profile saved at
mehek:`/tmp/prof/pipeline.prof`. See memory `project-sleap-roundtrip-audit`
(this slice) and `project-pipeline-perf-slices-bcd` (the other three).

## Goal

Eliminate the ×N full-frame replication (and the 47 s `.contiguous()`) by
sampling only the small frame region each canonical crop actually maps to.
**Byte-identical output**, same public interface, device-agnostic (wins on both
the cv2/CPU decode path and the NVDEC/CUDA path).

Non-goals: changing crop geometry/values; the `suppress_foreign` mask path;
running the warp on GPU when frames are CPU (a possible later add-on, explicitly
out of scope here).

## Approach (A1)

For each detection, compute the axis-aligned bounding box of the canvas
footprint in frame space, slice that sub-region, adjust the affine for the
sub-region origin, then run a single batched `grid_sample` over the stacked
small regions.

### Data flow (per detection `i`)
1. `m_inv = cv2.invertAffineTransform(m_align_i)` (canvas → frame).
2. Map the 4 canvas corners `(0,0), (cw-1,0), (0,ch-1), (cw-1,ch-1)` through
   `m_inv` → 4 frame-space points → AABB `(x0, y0, x1, y1)`.
3. Pad the AABB by `PAD = 1` px on every side (bilinear neighbor coverage),
   then clamp to `[0, W) × [0, H)`. `x0=floor(min-PAD)`, `x1=ceil(max+PAD)`.
4. `sub = frame_chw[:, y0:y1, x0:x1]` — a view, no copy.
5. Adjusted affine maps sub-region input coords `(u, v)` (= frame `(u+x0,
   v+y0)`) to canvas: translation becomes `M[:, :2] @ [x0, y0] + M[:, 2]`,
   linear part unchanged.

### Batching
- `Hmax, Wmax = max sub height/width over the frame's detections`.
- Allocate `batch = zeros(N, C, Hmax, Wmax)`; copy each `sub` top-left aligned.
- Build `theta_i` normalized to `(Wmax, Hmax)` with the sub-origin offset.
- One `F.grid_sample(batch, grid, mode="bilinear", padding_mode="zeros",
  align_corners=True)` → `(N, C, canvas_h, canvas_w)`.

The batched buffer's zero-pad region is never sampled: each detection's grid
covers only its own AABB by construction, so padding rows/cols beyond `sub` are
outside `[-1, 1]` for that grid or map to the clamped-out (zeros) area.

## Components / Interfaces

All changes confined to `src/hydra_suite/core/canonicalization/resample.py`.
The public functions keep their **exact signatures and output**, so callers in
`stages/crops.py` (`extract_canonical_crops`, `extract_classifier_crops`) are
untouched.

- `_canvas_footprint_aabb(m_align, geometry, frame_hw) -> tuple[int,int,int,int]`
  — new pure helper (corners → `m_inv` → padded, clamped AABB).
- `_theta_for_subregion(m_align, x0, y0, canvas_wh, pad_wh) -> np.ndarray`
  — generalizes `_theta_from_m_align` to a sub-origin and padded-input
  normalization. `_theta_from_m_align` is retained (or expressed as the
  `x0=y0=0, pad_wh=(w_in,h_in)` special case).
- `canonical_warp` (N=1) and `canonical_warp_batch` (N>1) — reworked internals.

## Acceptance bar (amended 2026-08-17 after implementation)

Implementation proved the AABB path **cannot be bitwise `torch.equal`** to the
full-frame path: `affine_grid` normalizes the sampling grid to the input
tensor's extent, so the sub-region (normalize by `pad-1`) and the full frame
(normalize by `w_in-1`) are two float32 quantizations of the *same* real-valued
map, differing by **max |Δ| ≈ 6e-5** on [0,1] pixels (diffuse, not a geometry
bug — footprint/theta helpers and the exact-reduction test all pass).

**Ruling (human-approved):** the acceptance bar is the project's established
standard — the equivalence harness at its determinism floor (positions p99 ≈ 0,
θ at floor), NOT bitwise crop equality. This matches precedent: the prior
`cv2.warpAffine → grid_sample` canonicalization migration was likewise not
bitwise-identical to legacy yet accepted at the floor. Rationale: the
classifier/head-tail crops are quantized to uint8 downstream
(`(crops*255).round()`), where a 6e-5 delta rounds away; only the pose (float)
path carries it, far below SLEAP precision. The unit oracle is relaxed to
`max|Δ| < 1e-3` (>15× the observed noise, <¼ the uint8 step 3.9e-3); **Task 6's
MPS+CUDA equivalence harness is the true acceptance gate.**

## Correctness Argument (real-number / noise-floor)

Output matches within the noise floor because:
- **Same source pixels.** Grid values are the same canvas→frame mapping; the
  only change is which tensor those coords index (a sub-view vs the full frame),
  with the affine translation compensated for the sub-origin.
- **Out-of-frame → zeros.** The AABB is clamped to the frame; canvas pixels
  mapping outside the frame map to sub-coords `< 0` or `≥ sub` → outside
  `[-1, 1]` → `grid_sample` returns 0, exactly as `padding_mode="zeros"` did on
  the full frame.
- **Boundary neighbors present.** The `PAD = 1` margin (clamped) guarantees the
  bilinear neighbors of any in-frame sampled coordinate are inside `sub`.

### Edge cases
- OBB partly/fully outside frame: footprint clamps; the missing area fills with
  zeros — matches current behavior.
- Degenerate/zero-area AABB (empty after clamp): emit a zero crop for that
  detection.
- `n == 0`: unchanged early return.
- `suppress_foreign`: operates on the *output* `(N,C,ch,cw)` crops in
  `stages/crops.py`; unaffected by this change.
- `native_sizes` rows stay `[canvas_h, canvas_w]`.

## Verification (drives the TDD plan)

1. **Unit byte-identical test** (new, `tests/`): for random frames × random OBBs
   including rotated, near-border, fully-out-of-frame, single, many, and
   degenerate detections, assert `max|old − new| == 0` on both CPU and (if
   available) CUDA tensors. This is the primary correctness gate; write it
   first (red) against the new implementation.
2. **Microbench** (scratchpad or `tools/equivalence/`): old vs new
   `canonical_warp_batch` on the fixture frame with realistic N, on CPU and
   CUDA — record wall-clock and peak memory to quantify the win.
3. **Equivalence harness** (the standing gate): byte-identical tracking output
   vs `legacy/main` on MPS (this box) and CUDA (mehek), full pose-clip subset.
   Positions p99 ≈ 0, θ at determinism floor, identical row counts.

## Risks

- Sub-region affine/normalization math is the only place a subtle off-by-one
  could break byte-identity — the unit test in step 1 is the guard and must be
  written before the rewrite is trusted.
- Building N padded sub-regions adds a small per-window copy (N × ~230²); this is
  negligible against the removed N × full-frame replication, but the microbench
  confirms net win rather than assumes it.

## Follow-ups (out of scope, tracked separately)

Slices B (pandas postprocessing), C (sequential video read), D (pose
connecting-layer cleanups) — see memory `project-pipeline-perf-slices-bcd`.
An on-GPU warp for CPU-decoded frames (upload once, warp small regions on CUDA)
is a possible add-on after A lands.
