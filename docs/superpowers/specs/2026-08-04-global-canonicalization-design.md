# Global canonicalization — design

Date: 2026-08-04. Branch: `feat/global-canonicalization`, off `main` @ `25a18d82`.

Supersedes the per-checkpoint geometry work only in the sense of building on it:
Slice 1 (`2026-08-03-vitpose-per-checkpoint-geometry-design.md`) and Slice 2
(`2026-08-04-vitpose-dataset-auto-sizing-design.md`) are merged prerequisites.

## Why

Today the repo canonicalizes animal crops in three different ways, and none of
them matches what any model is trained on.

`core/canonicalization/crop.py` (inference and tracking) sizes the canvas long
edge from the animal's own major axis and derives the short edge from the
species aspect ratio, while warping the animal's own oriented box onto it. The
two axes therefore get independent scales:

```
scale_x = canvas_w / (major * margin) = 1                # native
scale_y = canvas_h / (minor * margin) = own_AR / ref_AR  # silently resampled
```

Measured on the ant clip: the median animal is 2.44:1 against a configured 2.8,
so it is squashed 13% vertically; about 40% of detections are distorted more
than 20%; a curled 1.3:1 animal is squashed 54%. The distortion varies per
animal and per frame, and the model cannot observe it.

Two other implementations — `identity/dataset/oriented_video.py::_compute_affine`
and `canonicalization/dataset.py::MatMetadataCanonicalizer` — use the animal's
*own* aspect and are isotropic. So training exports and inference crops are
built by different geometry.

A second, subtler problem: because the canvas long edge tracks each animal's own
major axis, **every animal is normalised to the same length before any model
sees it**. Body size — a real biological signal, and a strong identity cue — is
destroyed by construction.

This design replaces all three with one canonicalization, used identically by
inference, crop-dataset generation, oriented-video generation, ClassKit training
and PoseKit training.

Supporting survey: `docs/superpowers/specs/notes/global-canonicalization-research.md`.

## Non-goals

- **No change to `REFERENCE_BODY_SIZE`**: not its meaning (median of the
  per-detection geometric mean of major and minor), not its auto-set formula,
  not any of its existing consumers (Kalman, Hungarian, background, cache key).
  The crop path merely reads it in addition.
- **No new user-facing config knobs.** Canvas geometry is derived from knobs
  that already exist: `REFERENCE_BODY_SIZE`, `RESIZE_FACTOR`,
  `reference_aspect_ratio`, and the canonical margin.
- **No automatic sizing of the canvas to guarantee zero clipping.** Sizing the
  crop so the largest animal fits is the operator's job, via margin and aspect
  ratio. The system reports clipping; it does not silently compensate.
- No full-frame pose mode (SLEAP/DLC only, separate branch).
- No change to detection, filtering, assignment, or post-processing.

---

## 1. The contract

Two layers. Each has exactly one implementation, called by every producer and
every consumer.

### Layer 1 — canonicalization: frame → canonical crop

```
body_px   = REFERENCE_BODY_SIZE * RESIZE_FACTOR      # existing config, unchanged
major_px  = body_px * sqrt(reference_aspect_ratio)   # geometric mean → major axis
canvas_w  = even(major_px * margin)
canvas_h  = even(canvas_w / reference_aspect_ratio)

src rect  = canvas_w x canvas_h frame pixels,
            centred on the OBB centroid,
            rotated so the major axis is horizontal
```

`REFERENCE_BODY_SIZE` is the geometric mean `sqrt(major * minor)`; with
`ar = major / minor` that gives `major = body_px * sqrt(ar)`. This recovers the
major-axis extent from the existing knob without redefining it.

The defining property: **the source rectangle does not depend on the animal's
own dimensions.** The OBB supplies a centre and an angle; nothing else. It
follows that:

- `scale = 1` on both axes. Layer 1 is a **rigid** transform — rotation and
  translation only. There is no scaling, therefore no anisotropy, and the only
  resampling is the bilinear interpolation inherent to rotating.
- Every canonical crop has identical pixel dimensions `(canvas_w, canvas_h)`.
- Every canonical crop has identical scale, padding and centring.
- Body size survives as signal: a larger individual renders larger, a curled
  animal renders shorter.
- The slack around a small animal fills with **real surrounding frame**, not
  synthetic padding — the affine samples the source image. Only foreign-OBB
  masking synthesises fill, using the existing background colour.

`reference_aspect_ratio` now sets the canvas **shape** only. It can no longer be
*wrong*, only wasteful: a poorly matched value costs background pixels on the
short axis. The misconfiguration failure mode that motivated this work
disappears.

**Clipping.** An animal whose `major * margin` exceeds `canvas_w` (or whose
`minor * margin` exceeds `canvas_h`) is clipped at the canvas edge. This is
expected and is the operator's dial: raise `margin` until the largest animal
fits. The system counts clipped detections and reports the count and the worst
observed overflow ratio; it never rescales to compensate, because that would
reintroduce per-animal scale.

### Layer 2 — model fit: canonical crop → model input tensor

```
fit    = min(model_w / canvas_w, model_h / canvas_h)     # isotropic
inner  = (round(canvas_w * fit), round(canvas_h * fit))
output = inner pasted centred into (model_w, model_h),
         remainder filled with the configured background colour
```

Isotropic, centred, letterboxed. Called identically by inference and by training
data loading, in every kit. This is what replaces
`transforms.Resize((sz, sz))`.

Layer 2 applies to any source image, not only Layer 1 output — it needs only the
source dimensions. So a ClassKit project containing the operator's own images,
never produced by MAT, is fitted by exactly the same rule as inference uses.

### Composition and inversion

```
M_total   = M_fit ∘ M_orient ∘ M_align       # frame → model input
M_inverse = invert(M_total)                  # model input → frame
```

`M_align` is Layer 1's rigid transform, `M_orient` the head-tail 0/90/180/270
rotation (`apply_headtail_rotation`), `M_fit` Layer 2's scale-and-translate.
Keypoints back-project through `M_inverse` in one step. Because `M_fit` is
isotropic and `M_align` is rigid, the composite carries no shear — which is the
structural reason the class of bug fixed in `83cf1907` cannot recur.

---

## 2. New module

`src/hydra_suite/core/canonicalization/geometry.py` — pure geometry, no Qt, no
imports from any app layer.

```python
@dataclass(frozen=True)
class CanonicalGeometry:
    """Fixed canonical crop geometry for one project/session."""
    canvas_wh: tuple[int, int]
    margin: float
    aspect_ratio: float

    @classmethod
    def from_reference(
        cls,
        reference_body_px: float,   # REFERENCE_BODY_SIZE * RESIZE_FACTOR
        aspect_ratio: float,
        margin: float,
    ) -> "CanonicalGeometry": ...

    @property
    def canvas_w(self) -> int: ...
    @property
    def canvas_h(self) -> int: ...


def canonical_affine(
    corners: np.ndarray,           # (4, 2) OBB corners, frame coords
    geometry: CanonicalGeometry,
) -> tuple[np.ndarray, float, bool]:
    """Return (M_align 2x3 frame→canvas, major_axis_theta, clipped)."""


@dataclass(frozen=True)
class FitResult:
    tensor_hw: tuple[int, int]     # model input (H, W)
    inner_hw: tuple[int, int]      # scaled content size
    offset_xy: tuple[int, int]     # top-left of content within the canvas
    scale: float


def fit_to_model_input(
    source_wh: tuple[int, int],
    model_wh: tuple[int, int],
) -> FitResult:
    """Isotropic centred letterbox parameters. Pure arithmetic."""


def apply_fit(
    image: np.ndarray | torch.Tensor,
    fit: FitResult,
    background: tuple[int, int, int],
) -> np.ndarray | torch.Tensor: ...


def fit_affine(fit: FitResult) -> np.ndarray:
    """2x3 affine for the fit, for composition into M_total."""
```

`canonical_affine` reports `clipped` rather than raising, so callers can count
without branching on control flow.

---

## 3. What changes, by site

### Core geometry

`core/canonicalization/crop.py`:
- `compute_crop_dimensions`, `compute_native_crop_dimensions`,
  `compute_native_scale_affine` — **deleted**. Canvas dimensions are no longer a
  function of the detection.
- `compute_alignment_affine` — **deleted**, replaced by `canonical_affine`.
- `extract_canonical_crop`, `gpu_canonical_crop`, `gpu_canonical_crop_batch`,
  `extract_and_classify_batch`, `_apply_foreign_mask_canonical` — retained,
  re-pointed at `CanonicalGeometry`.
- `apply_headtail_rotation` — retained; the 90° dimension swap now produces a
  canvas of `(canvas_h, canvas_w)`, which Layer 2 handles like any other source.
- `invert_keypoints` — unchanged.

`core/canonicalization/dataset.py` — **deleted**. `MatMetadataCanonicalizer` and
`get_canon_transform` have no production consumers (tests only), and the
per-image metadata path they implement is subsumed by Layer 1 + Layer 2.
`tests/test_canonicalization.py` and `tests/test_canonicalization_flexible.py`
go with them. The inert `canonicalize_mat` DB columns
(`classkit/core/store/db.py`) are left in place — dropping SQLite columns is not
worth a migration — but are documented as dead.

### Inference

- `stages/crops.py` — all nine entry points produce `(N, C, canvas_h, canvas_w)`.
  The batch-max zero-pad in `_extract_canonical_cpu` (:215-227) is deleted;
  crops are already uniform. `extract_classifier_crops` and its GPU/batch
  variants stop warping straight to model input and instead produce canonical
  crops that the consumer fits via Layer 2. Its unused `aspect_ratio` parameter
  goes away with the rewrite.
- `stages/pose.py` — the slice-back hack `hwc[: ch, : cw]` (:269, and the
  equivalent at :402) is deleted. `M_inverse` becomes `M_total`'s inverse.
- `stages/cnn.py`, `stages/headtail.py` — take a `CanonicalGeometry` instead of
  `aspect_ratio` / `margin` floats; fit via Layer 2.
- `config.py` — `canonical_aspect_ratio` and `canonical_margin` are replaced by
  a single `canonical: CanonicalGeometry`, built in `from_parameters` from
  `REFERENCE_BODY_SIZE`, `RESIZE_FACTOR`,
  `ADVANCED_CONFIG.reference_aspect_ratio` and the canonical margin. This is a
  derived field, not a new knob.
- `cache/keys.py` — the canonical geometry enters the detection-cache key.
  Without this, a stale `.npz` silently mixes conventions.
- `pipeline.py`, `api.py`, `runner.py`, `result.py`,
  `core/tracking/ingest/streaming_payload.py` — thread the geometry object;
  `canonical_crops_cpu` / `canonical_crops_cuda` become uniformly shaped.

### Identity

- `identity/classification/headtail.py` (:517, :594, :748) — three crop sites
  collapse to one Layer 1 call plus Layer 2. The `128 x 128/ref_AR` fallback
  canvas is deleted.
- `identity/dataset/generator.py` — writes Layer 1 crops. Gains a provenance
  sidecar (below).
- `identity/dataset/oriented_video.py::_compute_affine` (:1198-1220) — deleted,
  replaced by Layer 1.

### Training

- `training/runner.py` (:1297, :1337, :1635, :1664) — `transforms.Resize((sz, sz))`
  becomes a Layer 2 transform into the model's input shape. Both train and eval
  transforms; the augmentation pipeline composes after the fit, as today.
- PoseKit ViTPose training (`posekit/core/vitpose_training.py`,
  `posekit/gui/dialogs/training.py`) — the dataset transform fits into
  `PoseGeometry.image_size_wh` via Layer 2, so training and inference share it.
- `training/pose_geometry_measure.py` (Slice 2) — unchanged in formula; its
  suggestion now describes the model input shape, which is what Layer 2 fits
  into.

### Provenance sidecar

`generator.py` and `oriented_video.py` write `.canonical.json` beside each
exported crop directory:

```json
{
  "canvas_wh": [64, 26],
  "margin": 1.5,
  "aspect_ratio": 2.44,
  "reference_body_px": 20.0,
  "resize_factor": 1.0,
  "clipped_count": 3,
  "worst_overflow_ratio": 1.08,
  "hydra_suite_version": "..."
}
```

Consumers (ClassKit, PoseKit) read it to confirm a dataset's geometry and to
warn when mixing datasets built under different geometry. A dataset without a
sidecar is treated as unknown-provenance and fitted by Layer 2 on its raw
dimensions — correct behaviour for the operator's own images.

### Config defects swept in the same change

| ID | Defect | Fix |
|---|---|---|
| D1 | `crops_worker.py:306` reads `REFERENCE_ASPECT_RATIO` (uppercase), a key nothing writes, so the interpolated-crop head-tail analyzer always uses the 2.0 fallback | Read the geometry object |
| D2 | `cli_config.py:299` defaults `reference_aspect_ratio` to 4.0 where every other site defaults to 2.0 | Single default, 2.0 |
| D3 | `extract_classifier_crops` accepts an unused `aspect_ratio` | Removed with the rewrite |

All three are symptoms of the geometry being computed in more than one place;
they cannot recur once it is computed in one.

---

## 4. Testing

**Layer 1 (pure, no Qt, no models):**
- Rigidity: for a fixed geometry, the affine's linear part is a pure rotation —
  `M[:, :2]` is orthogonal with unit singular values, for OBBs of many different
  aspect ratios. This is the direct assertion that the old bug is gone.
- Invariance to animal size: two OBBs at the same centre and angle but different
  extents produce **identical** affines.
- Canvas dims depend only on the geometry, never on the detection.
- `major_px = body_px * sqrt(ar)` round-trips: an OBB built to the reference
  aspect and body size exactly fills `canvas / margin`.
- Clipping is reported, not silently absorbed: an oversized OBB sets
  `clipped=True` with the expected overflow ratio; an undersized one does not.
- Head-tail 90° rotation swaps canvas dims and composes invertibly.

**Layer 2 (pure):**
- Isotropy: `fit.scale` is a single scalar; content aspect is preserved exactly.
- Centring: offsets are symmetric within one pixel on the padded axis.
- Identity case: source dims equal model dims → `scale == 1`, zero offset, no
  padding.
- Both orientations: source wider than model, and taller.
- `fit_affine` composed with its inverse round-trips points to sub-pixel error.

**Composition:**
- A synthetic keypoint at a known frame position, pushed through
  `M_total` and back through `M_inverse`, returns to within sub-pixel error —
  across aspect ratios, head-tail directions, and both fit orientations.

**Train/inference identity (the property the whole design exists for):**
- The same source image, fitted by the inference path and by the training
  transform, produces **byte-identical** tensors. One test per kit
  (ClassKit, PoseKit). This is the regression that would otherwise reappear
  silently.

**Integration:**
- The 10 existing tests that lock current crop geometry are re-baselined
  against the new contract, not deleted:
  `tests/helpers/tiny_clip.py`, `test_canonical_crop.py`,
  `test_gpu_classifier_crop.py`, `test_inference_api_pose.py`,
  `test_inference_cache_keys.py`, `test_inference_crops.py`,
  `test_inference_extract_crops_batch.py`, `test_inference_foreign_mask.py`,
  `test_inference_stages_crops.py`,
  `test_pipeline_pose_batch_canonical_geometry.py`.
- CPU and GPU crop paths agree to within interpolation tolerance under the new
  geometry (existing `test_gpu_classifier_crop.py` pattern).

**Gate:** delta-based. The base suite carries pre-existing failures (6 in
`test_oriented_track_video_export.py`, plus ordering pollution in
`test_vitpose_export.py`); the gate is "no new failures", not an absolute count.

---

## 5. Equivalence and retraining

This change is **intentionally not equivalent**. Every clip that runs a
crop-consuming stage will differ: `emi_obb_identity`, `ant_pose_headtail`,
`ant_obb_sleap`, `ant_obb_sequential`, `ant_cnn_identity`. `fly_obb` and
`worm_bgsub` should be unaffected — to be confirmed, and their continued
equivalence is a useful control that the change is confined to crop consumers.

The harness is therefore used to **re-baseline**, not to pass: run the matrix
before and after on both platforms (MPS here, CUDA on mehek), record the new
baseline, and confirm the two non-crop clips stay byte-identical.

Every existing model was trained on old-convention crops and needs retraining:
head-tail, CNN identity, ViTPose, SLEAP. The operator has accepted this.

Head-tail warrants explicit measurement rather than assumption. The comment at
`stages/crops.py:240-247` records that merely adding a resample step flipped
1-2% of direction decisions against legacy, and head-tail sits upstream of
tracking identity. After retraining, measure direction agreement against the
current model on a held-out clip and report it, rather than inferring health
from tracking output.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Head-tail regression propagating into identity | Explicit direction-agreement measurement post-retrain; head-tail is demonstrably resample-sensitive |
| Clipping unnoticed by the operator | Counted and reported per run and stamped into the sidecar; surfaced in the GUI summary |
| Stale detection caches mixing conventions | Canonical geometry enters the cache key |
| Crop memory grows | The canvas is sized for the reference animal plus margin, not per-detection; fixed-size batches also remove the batch-max padding waste |
| Mixed-provenance training datasets | Sidecar comparison warns on mismatch |
| A consumer quietly keeping its own resize | The byte-identical train/inference test per kit is the structural guard |
