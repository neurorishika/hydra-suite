# Global canonicalization — research findings

Survey of every canonical-crop producer, consumer, and trainer in the repo, in
preparation for a global switch to: **fixed canvas dimensions + fixed global
scale + centred animal + no anisotropy**, consumed identically by inference,
ClassKit training and PoseKit training.

Date: 2026-08-04. Read-only survey of `main` @ `4b542729`.

---

## 1. There are three canonicalizations, not one

| # | Implementation | Canvas dims | Aspect handling | Anisotropic? |
|---|---|---|---|---|
| A | `core/canonicalization/crop.py` | per-detection | forced to `reference_aspect_ratio` | **yes** — `scale_y = own_AR / ref_AR` |
| B | `core/individual/dataset/oriented_video.py::_compute_affine` (:1198) | per-detection | animal's **own** aspect | no |
| C | `core/canonicalization/dataset.py::MatMetadataCanonicalizer` (:92) | per-detection | animal's **own** aspect | no |

A is the tracking/inference path. B is the oriented-track-video exporter. C is
the metadata-driven canonicalizer intended for dataset consumers.

### A — the anisotropy, precisely

`compute_native_crop_dimensions` (crop.py:67):
```
canvas_w = major * margin           # native along the major axis
canvas_h = canvas_w / ar            # derived from the SPECIES aspect
```
`compute_alignment_affine` (crop.py:133) maps a source rectangle of
`w_exp = major*margin` by `h_exp = minor*margin` onto that canvas via
`cv2.getAffineTransform` on three corners, giving independent axis scales:
```
scale_x = canvas_w / w_exp = 1               # native — the stated contract
scale_y = canvas_h / h_exp = own_AR / ar     # silently resampled
```
So the native-scale contract holds on the major axis and is violated on the
minor axis. Measured on the ant clip: own_AR p50 = 2.44 against a configured
2.8 → 13% vertical downsample for the median animal; ~40% of detections
distorted more than 20%; a curled 1.3:1 animal is squashed 54%.

---

## 2. C is dead code

`MatMetadataCanonicalizer` and `get_canon_transform` have **no production
consumers** — only `tests/test_canonicalization.py` and
`tests/test_canonicalization_flexible.py` import them.

ClassKit's database carries a `canonicalize_mat` column
(`classkit/core/store/db.py:271,294,1451,1474,1536,1750`) that is **never read
or written anywhere in `src/`**. The feature was schema'd and never wired.

**Consequence: ClassKit does not canonicalize at all.** It consumes whatever
images the MAT exporter wrote.

---

## 3. ClassKit training squares everything

`training/runner.py:1297, 1337, 1635, 1664` — `transforms.Resize((sz, sz))`,
unconditionally, both train and eval transforms. There is no (H, W) variant;
the ClassKit auto-size feature produces a single `sz`.

This produces an accidental consistency worth understanding before changing
anything:

- **Train:** own padded OBB → A's ref-AR canvas (anisotropic) → PNG →
  `Resize((sz,sz))` (anisotropic again) — net map: own padded OBB → square.
- **Infer:** `extract_classifier_crops` (crops.py:231) warps the own padded OBB
  **straight** to the model's `target_size` — net map: own padded OBB → square.

The two anisotropic steps compose to the same net transform as the one
anisotropic step. **The AR standardisation cancels out for CNN identity.** It
is consistent for the wrong reason: both ends discard the aspect ratio.

Implication: converting only one end (e.g. making the exporter isotropic while
leaving `Resize((sz,sz))`) would *break* a currently-working correspondence.
The CNN identity path must be converted at both ends in the same change.

---

## 4. Pose is where it does not cancel

Pose keypoints are back-projected through `M_inverse` derived from the
canonical canvas (`stages/pose.py:270-271`, `_assemble_pose_result` →
`invert_keypoints`). The anisotropy is therefore baked into output
coordinates rather than cancelled by a symmetric resize. This is the bug class
fixed in `4b542729` (`pipeline.py:363` passing `ar, mg`).

---

## 5. Per-detection canvas dims force machinery that global canonicalization deletes

- `_extract_canonical_cpu` (crops.py:205-228) zero-pads every crop to the batch
  max, **bottom-right**, so a batch tensor mixes scales and carries asymmetric
  padding.
- `run_pose_batch` slices it back off: `hwc[: int(ch), : int(cw)]`
  (pose.py:269), recomputing `cw, ch` to know where to cut.
- `extract_canonical_crops` returns `(N, C, H, W)` where H, W are the batch max,
  not a meaningful canvas.

Under a fixed canvas all of this collapses to a plain stack.

---

## 6. Latent config defects found during the survey

| ID | Defect | Location |
|---|---|---|
| D1 | `crops_worker.py` reads `REFERENCE_ASPECT_RATIO` (**uppercase**) — a key nothing in `src/` or `tests/` ever writes. The interpolated-crop head-tail analyzer therefore **always** uses the 2.0 fallback, ignoring configuration. Same bug family as the one just fixed in `pipeline.py`. | `trackerkit/gui/workers/crops_worker.py:306` |
| D2 | CLI defaults `reference_aspect_ratio` to **4.0**; `default.json`, the GUI spin, `InferenceConfig`, `pose._CANONICAL_ASPECT_RATIO`, `stages/cnn.py`, `stages/headtail.py` all default to **2.0**. | `trackerkit/cli_config.py:299` vs `resources/configs/default.json:45` et al. |
| D3 | `extract_classifier_crops` accepts an `aspect_ratio` parameter and never uses it. | `stages/crops.py:231-280` |
| D4 | `canonicalize_mat` DB columns are inert (see §2). | `classkit/core/store/db.py` |

D1 and D2 mean the AR actually in force today varies by entry point. Any
before/after comparison must pin it explicitly.

---

## 7. `REFERENCE_BODY_SIZE` cannot be redefined

Current meaning: **median of the per-detection geometric mean of major and
minor axes**, at `resize=1.0` (`detection_panel.py:1400`, auto-set button
labelled "Auto-Set Body Size from Median"; `main_window.py:1935`).

Consumers that would silently change behaviour if its meaning moved from
"typical" to "maximum":

| Consumer | Use |
|---|---|
| `core/filters/kalman.py:224` | process-noise scale |
| `core/assigners/hungarian.py:144-148` | gating radius, motion scale |
| `core/assigners/hungarian.py:815, 930` | body-size-relative thresholds |
| `core/background/measure.py:119`, `optimizer.py:39` | expected blob area |
| `core/tracking/worker.py:921, 1310, 2548, 2933, 3552` | assorted size gates |
| `core/inference/cache/keys.py:152` | detection-cache key |
| `core/tracking/optimization/optimizer.py:759, 896` | parameter search |

**Therefore: the canonical scale needs its own knob**, defaulted from
`REFERENCE_BODY_SIZE` but set independently (the user's stated intent is to
size it from the *maximum* animal so nothing clips). Reusing
`REFERENCE_BODY_SIZE` for both would couple crop framing to motion gating.

Note also `RESIZE_FACTOR`: body size is specified at `resize=1.0` and scaled by
`RESIZE_FACTOR` at every consumer. The canonical scale must follow the same
convention or crops will silently differ between resized and unresized runs.

---

## 8. Head-tail is already half-way there

`HeadTailAnalyzer` (`identity/classification/headtail.py:517, 594, 748`) and
`extract_classifier_crops` already use a **fixed** canvas — the model's input
size, or a `128 × 128/ref_AR` fallback (headtail.py:600-605, 755-760). So the
classifier paths already have uniform dims; what they lack is the isotropic
scale. They are the smallest delta.

They are also the most resample-sensitive: the comment at `crops.py:240-247`
records that routing head-tail through a shared native crop + a second
interpolate flipped **1-2% of direction decisions** versus legacy. Any change
here needs a head-tail retrain and a direction-agreement measurement, not just
a parity diff.

---

## 9. Full inventory of sites to change

**Core geometry (the new contract):**
- `core/canonicalization/crop.py` — `compute_crop_dimensions`,
  `compute_native_crop_dimensions`, `compute_native_scale_affine`,
  `compute_alignment_affine`, `extract_canonical_crop`, `gpu_canonical_crop`,
  `gpu_canonical_crop_batch`, `extract_and_classify_batch`,
  `apply_headtail_rotation` (dimension swap on 90° rotation), `invert_keypoints`

**Inference:**
- `core/inference/stages/crops.py` — 9 crop entry points (CPU/GPU/batch × canonical/classifier)
- `core/inference/stages/pose.py:267-271, 402-403` — drop the slice-back hack
- `core/inference/stages/cnn.py:67, 128`, `stages/headtail.py:98, 246` — defaults
- `core/inference/config.py:317-318, 807-812` — `canonical_aspect_ratio`, `canonical_margin`, + new scale field
- `core/inference/cache/keys.py` — the canonical geometry must enter the cache key
- `core/inference/pipeline.py`, `api.py`, `runner.py`, `result.py`
- `core/tracking/ingest/streaming_payload.py` — `canonical_crops_cpu/cuda` shapes become uniform

**Identity:**
- `core/individual/classification/headtail.py:517, 594, 748`
- `core/individual/dataset/generator.py:92, 323-350, 505-520` (ClassKit's source of images)
- `core/individual/dataset/oriented_video.py:1198-1220`

**Dataset canonicalizer:**
- `core/canonicalization/dataset.py` — either delete (dead) or re-point at the new contract

**Training:**
- `training/runner.py:1297, 1337, 1635, 1664` — `Resize((sz,sz))` must become the canonical geometry
- `training/pose_geometry_measure.py` (Slice 2) — extend to measure body size, not just aspect
- PoseKit ViTPose training path (`posekit/core/vitpose_training.py`, `posekit/gui/dialogs/training.py`)

**GUI / config:**
- `trackerkit/gui/panels/detection_panel.py:952-985` — body size + AR spins, auto-set buttons
- `trackerkit/gui/orchestrators/config.py:485, 1577, 2072`
- `trackerkit/cli_config.py:295-300` (D2)
- `trackerkit/gui/workers/crops_worker.py:306` (D1), `:1456`
- `trackerkit/gui/workers/preview_worker.py:388`
- `trackerkit/gui/panels/dataset_panel.py`, `identity_panel.py`
- `classkit/gui/dialogs/training.py` auto-size row

**Tests locking current geometry (10 files):**
`tests/helpers/tiny_clip.py`, `test_canonical_crop.py`, `test_gpu_classifier_crop.py`,
`test_inference_api_pose.py`, `test_inference_cache_keys.py`, `test_inference_crops.py`,
`test_inference_extract_crops_batch.py`, `test_inference_foreign_mask.py`,
`test_inference_stages_crops.py`, `test_pipeline_pose_batch_canonical_geometry.py`
+ `test_canonicalization.py`, `test_canonicalization_flexible.py` (§2 dead code)

**Equivalence fixtures affected** (any clip with head-tail, CNN identity, or pose):
`emi_obb_identity`, `ant_pose_headtail`, `ant_obb_sleap`, `ant_obb_sequential`,
`ant_cnn_identity`. `fly_obb` and `worm_bgsub` should be unaffected if they run
no crop-consuming stage — to be confirmed.

---

## 10. What the new contract is

```
s        = canonical_body_px / (CANONICAL_REFERENCE_SIZE * RESIZE_FACTOR)
canvas_w = canonical_body_px * margin              # fixed, model property
canvas_h = canvas_w / ar                           # fixed, model property
src rect = (canvas_w / s) x (canvas_h / s)         # SAME scale both axes
           centred on the OBB centroid, rotated so the major axis is horizontal
```

Properties:
- identical pixel dimensions for every crop, everywhere
- identical scale for every crop — `s` is a session constant, not per-detection
- no anisotropy anywhere
- animal centred; slack fills with **real surrounding frame**, not padding
  (the affine samples the source image; only foreign-OBB masking synthesises fill)
- body size is preserved as signal instead of being normalised away
- overflow (animal larger than the canvas) is clipped, counted, and reported;
  the operator sizes the canvas from the dataset maximum

Retained meaning of the existing knobs:
- `reference_aspect_ratio` → canvas **shape** only. It can no longer be *wrong*,
  only wasteful (padding on the short axis). This removes the misconfiguration
  failure mode entirely.
- `canonical_margin` → padding, now honoured on both axes.
- new `CANONICAL_REFERENCE_SIZE` → canvas **scale**, independent of
  `REFERENCE_BODY_SIZE` (§7).

---

## 11. Open risks

| Risk | Note |
|---|---|
| Head-tail regression | §8 — 1-2% of decisions already move under a pure resample change; needs retrain + agreement measurement, and it sits upstream of identity |
| CNN identity correspondence | §3 — must convert exporter and `Resize((sz,sz))` together or break a working cancellation |
| Canonical scale misconfiguration | §7 — new load-bearing knob; wrong value = whole-dataset domain shift. Mitigate with measurement + a stamped provenance sidecar |
| Crop memory | fixed canvas sized for the largest animal means every crop pays the worst case |
| Cache invalidation | §9 — geometry must enter the cache key or stale `.npz` caches will silently mix conventions |
| Equivalence gate | intentionally non-equivalent; needs a deliberate re-baseline, not a pass |
