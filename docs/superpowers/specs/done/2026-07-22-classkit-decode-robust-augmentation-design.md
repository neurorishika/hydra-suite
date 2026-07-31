# ClassKit Training-Path Hardening: Decode & Resample Augmentation

**Date:** 2026-07-22
**Status:** Design (approved) — pending implementation plan
**Author:** Rishika Mohanta
**Related:** `2026-07-22-nvdec-gpu-fast-tier-design.md` (Part 1), `2026-07-22-gpu-native-classifier-crop-design.md`
**Scope:** Part 2 of 2. Independent of Part 1 — models trained with this can be deployed on any decode path.

## Problem

Classifier crops the model sees at **inference** are not bit-identical to the **training**
crops:

- **Color (decode):** training crops come from cv2 CPU decode (BT.601, limited range).
  At inference, GPU-Fast/NVDEC decodes YCbCr→RGB with a different matrix/range (BT.601 vs
  BT.709, limited vs full), leaving an irreducible ~max-2 / mean-0.48 per-channel residual.
- **Geometry (resample):** training crops use `cv2.warpAffine`/`cv2.resize` (INTER_LINEAR);
  the inference GPU path warps with `torch grid_sample` (bilinear), explicitly *not*
  bit-identical (crops.py:372).

A colortag/identity classifier reads **color as its signal** and was trained on exactly
one decode+resample convention with no augmentation covering these axes. Measured effect:
~6% per-frame colortag flips from decode color + ~4% from grid_sample crop → ~44% final
identity divergence on the NVDEC path.

Chasing bit-exact decode/resample matching is a losing game (every camera, codec, GPU,
and driver differs). The robust, future-proof fix is to **train the classifiers to be
invariant** to the *space* of decode and resample transforms — not to any one NVDEC delta.

## Goal

Add two opt-in augmentations to the ClassKit classifier training path so a model trained
with them is robust to decode-color and crop-resample differences (NVDEC today; other
codecs/cameras/decoders in future) **without collecting decode-specific data**:

1. **Decode-color augmentation** — randomly re-decode each crop through a sampled
   (matrix, range) convention, spanning {BT.601, BT.709} × {limited, full}, plus mild
   photometric jitter.
2. **Resample augmentation** — randomly re-warp each crop with an alternate resampler
   (torch `grid_sample` and/or randomized interpolation + sub-pixel jitter), so the model
   sees both cv2 and grid_sample geometry.

Both default **off** (backward-compatible), recommended **on** for colortag/identity and
head/tail models. Applies across all three in-process trainers (torchvision custom,
multihead-shared, tiny). YOLO-classify (Ultralytics subprocess) is out of scope for these
custom transforms — noted, not blocked.

Non-goals: retraining any specific model now (done when convenient); the YOLO-classify
path; Part 1's tiering.

## Design

### 1. Shared augmentation module

New `src/hydra_suite/training/augmentation.py` with two pure functions operating on an
**RGB uint8 `(H, W, 3)` numpy array** (the common currency: the tiny path is already
cv2/numpy RGB at runner.py:314; the torchvision path can wrap these via a custom transform
around PIL↔numpy). Reusing one implementation keeps the three trainers consistent.

```python
def simulate_decode_color(rgb_u8: np.ndarray, strength: float, rng) -> np.ndarray:
    """Re-decode the crop through a randomly sampled YCbCr<->RGB convention.

    With probability ~strength: RGB -> YCbCr (fixed BT.601, the training origin) ->
    RGB using a RANDOMLY sampled matrix in {BT.601, BT.709} and range in
    {limited(16-235), full(0-255)}, optionally blended by `strength` toward identity.
    This reproduces exactly the decode variation (matrix + range) the model meets at
    inference, without needing NVDEC data. Followed by mild brightness/contrast jitter.
    """

def simulate_resample(rgb_u8: np.ndarray, prob: float, rng) -> np.ndarray:
    """With probability `prob`, re-warp the crop through an alternate resampler:
    torch grid_sample (bilinear, align_corners=False) and/or a randomized
    interpolation (cv2 INTER_LINEAR/CUBIC/AREA) with sub-pixel translation jitter.
    Simulates the cv2.warpAffine vs grid_sample geometry gap (crops.py:372)."""
```

Determinism: both take an explicit `rng` (seeded per-epoch/worker) so training runs stay
reproducible.

### 2. Config surface (`training/contracts.py:AugmentationProfile`, line 124)

Add real dataclass fields (the class is `slots=True`, so no dynamic attrs):

```python
decode_color_sim: float = 0.0   # 0=off; ~0.5 recommended. P(apply) + blend strength.
resample_sim: float = 0.0       # 0=off; ~0.3 recommended. P(apply alternate resampler).
```

Backward-compatible defaults (0.0 = current behavior). Persisted with the run spec via the
existing registry; no migration needed (new fields default off for old specs).

### 3. Injection into the three trainers

- **Torchvision custom** (`_train_custom_classify`, transforms at runner.py:1274-1313) and
  **multihead-shared** (runner.py:1602-1624): insert a custom transform BEFORE
  `ToTensor()`/`Normalize` that applies `simulate_decode_color` + `simulate_resample` on the
  PIL image (convert PIL→numpy RGB, augment, back to PIL). Gate on
  `profile.decode_color_sim > 0` / `profile.resample_sim > 0`. Val transform: **no**
  decode/resample aug (evaluate on the clean cv2 convention).
- **Tiny** (`_apply_tiny_augmentation`, runner.py:327-365): call the same two functions on
  the cv2 RGB numpy crop, gated on the profile fields, alongside the existing flips/jitter.

One shared implementation, three call sites — no logic duplication.

### 4. GUI (`classkit/gui/dialogs/training.py`, augmentation tab `_build_space_and_augmentations_tab`, line 1566)

Add two controls next to the existing jitter spinboxes (training.py:1580-1638): a
"Decode-color robustness" slider/spinbox (→ `decode_color_sim`) and a "Resample robustness"
control (→ `resample_sim`), assembled into the `AugmentationProfile` at the existing
build site (~training.py:1900-1930) and reflected in the augmentation summary text
(~training.py:2044-2068). Add a short tooltip explaining these harden the model against
GPU-Fast/NVDEC and cross-codec decode differences.

## Acceptance / Verification

The metric is **decode/resample robustness measured directly on real crops**, using the
crop-level isolation harness already built for this investigation (run the classifier on
the same detections cropped via cv2-decode+cv2-warp vs NVDEC-decode+grid_sample; report
label agreement).

- **Robustness gain:** a small colortag model trained WITH the augmentation (on the existing
  training set) reaches **≥99%** cv2-vs-NVDEC crop label agreement, up from ~90% without —
  measured with the crop-level harness on held-out detections.
- **No baseline regression:** validation accuracy on the clean cv2 val set is within noise
  of the un-augmented baseline (the augmentation must not hurt in-distribution accuracy).
- **Reproducibility:** two training runs with the same seed + profile produce identical
  weights (the `rng` is seeded, not global).

## Testing

- **Unit:** `simulate_decode_color` / `simulate_resample` return same-shape uint8 RGB, are
  identity at strength 0, produce bounded change (no NaNs/overflow), and are deterministic
  under a fixed `rng`. `AugmentationProfile` round-trips the new fields through the registry.
- **Integration:** a fast end-to-end train of a tiny colortag head with `decode_color_sim`
  and `resample_sim` on, on a small fixture set, then the crop-level agreement harness
  confirms the robustness gain (≥99%) vs an un-augmented control.

## Risks & Open Questions

- **Over-augmentation hurting accuracy:** too-strong decode/resample jitter could blur the
  color signal. Mitigate with conservative default strengths (0.5 / 0.3) and the
  no-regression acceptance gate; expose strength so operators can tune.
- **PIL↔numpy round-trip cost** in the torchvision transform: negligible vs model forward,
  but keep the conversion inside the custom transform (no extra copies).
- **YCbCr round-trip precision:** the decode-sim uses cv2/`numpy` YCbCr conversions; it need
  not be bit-exact to any real decoder — it only needs to span the *space* of conventions,
  which a random {matrix}×{range} sample achieves.
- **Head/tail model:** head/tail is orientation (not color) so it is less color-sensitive,
  but resample jitter still helps; enabling both is safe and recommended.
