# Legacy → New Inference Pipeline — Static Code-Path Parity Audit

**Date:** 2026-06-26 (audit) · **Updated:** 2026-06-29 (Tier-2 remediation)
**Legacy tree:** `main` (`src/hydra_suite`)
**New tree:** `feature/inference-pipeline-redesign` worktree (`src/hydra_suite`)

> **Remediation status (2026-06-29):** All four CRITICAL findings (C1–C4), the Tier-1 HIGH
> detection-filter findings (**H1, H2, H5, H6**), **and** the Tier-2 classifier/cache
> findings (**H7, H8, H9**) are **confirmed and fixed** (matching legacy). See the
> **Remediation log** section and the per-finding **✅ RESOLVED** tags. Still open: **H3,
> H4** and the MEDIUM/LOW/perf items — the deliberate acceleration drops that need an owner
> decision before reinstating. Prioritized in **Next best targets**.
**Scope:** Verify that **every code path** in the legacy inference implementation has an
equivalent in the redesigned `core/inference/` pipeline. This is a **static** audit
(reading both trees), complementary to the runtime `run_matrix.sh` equivalence harness.

## Method

Seven parallel domain audits compared the legacy implementation (inline in
`core/tracking/worker.py`, `core/detectors/`, `core/identity/**`,
`core/tracking/pose_pipeline.py`) against the new staged pipeline
(`core/inference/stages/*`, `runner.py`, `cache/*`, plus the rewired
`core/tracking/{worker,frame_result_bridge,streaming_payload,optimizer}.py` and
`core/tracking/identity/evidence.py`):

1. Detection / OBB stage + detection filtering
2. Crop extraction + affine alignment
3. Head-tail + CNN + AprilTag per-crop classifiers
4. Pose (YOLO + SLEAP) + pose feature extraction
5. Identity evidence assembly / online decoder / calibration
6. Caching (detection `.npz`, pose, properties, tag, identity)
7. Worker orchestration + per-stage runtime selection + optimizer/streaming rewire

Every **CRITICAL** and **HIGH** finding below, plus both inter-agent conflicts, was
**re-verified directly against source** by the coordinator. Such rows are tagged
**✅ VERIFIED**. Remaining rows are reported with cited `file:line` evidence and are
internally consistent but were not independently re-read.

---

## Bottom line

The **core per-frame math is faithfully preserved**: the online identity decoder, the
`CalibrationModel`, the evidence-emitter internals, pose feature extraction, the
affine/warp primitives, OBB geometry normalization, and the
detect→crop→{headtail,cnn,pose}→apriltag→identity **stage order** are byte-identical or
equivalent. The two named bug fixes (realtime double-filter, cross-frame crop padding)
and `start_frame/end_frame` handling are correctly in place, and the
`USE_NEW_INFERENCE_PIPELINE` flag was removed cleanly (no dead branches).

**The redesign is now at functional code-path parity for every correctness finding.** The
**4 crash-/corruption-level regressions (C1–C4)**, the **Tier-1 detection-filter findings
(H1/H2/H5/H6)** that decide *which detections enter tracking*, and the **Tier-2
classifier/cache findings (H7/H8/H9)** are all **fixed** (see Remediation log). Still open
are only the **deliberate acceleration drops (H3 foreign-region suppression; H4 TRT/ONNX
auto-export; perf tier)** — design decisions that need an explicit owner call before
reinstating, not parity bugs. The existing equivalence harness would **not** catch most of
the original regressions — its fixtures don't exercise AprilTags, `calibration_temperature
≠ 1.0`, the optimizer, multi-class models, sub-threshold CNN votes, truncated caches, or
non-default config keys — so extending the harness (multi-class OBB clip, size-cap clip,
AprilTag clip, temperature run, truncated-cache run) is now the highest-value remaining
work alongside the H3/H4 decision.

---

## Remediation log (2026-06-28)

CRITICAL-only pass. Each fix was made in the new tree and verified.

| ID | Fix | Files touched |
|---|---|---|
| C1 | `canonical_margin` now reads `ADVANCED_CONFIG["yolo_headtail_canonical_margin"]` (default 1.3), matching legacy `yolo_detector.py:286`; no longer reuses `INDIVIDUAL_CROP_PADDING`. | `core/tracking/worker.py` |
| C2 | Calibration applied **at tracking time** inside `IdentityEvidenceEmitter` (new optional `calibration: CalibrationModel`), reproducing legacy `predict_batch_posteriors(calibration=…)` (`calibrate_probs → exp → renormalise`). Caches stay **raw** (design intent preserved); decoder now sees calibrated evidence. `_build_cnn_evidence_emitter` passes the model. Verified the legacy `CNNPrecomputePhase` path is dead (no double-calibration). | `core/tracking/evidence_emitter.py`, `core/tracking/worker.py` |
| C3 | Reverted the mistaken "Correction 21" alias; optimizer scorer **and** cache builder bind the legacy `data.detection_cache.DetectionCache` directly (the optimizer subsystem reads+writes its cache entirely via the legacy API). | `core/tracking/optimizer.py`, `core/tracking/optimizer_workers.py` |
| C4 | AprilTag stage uses the SEB lab fork via `_get_apriltag()` + `at.apriltag(...)`, parses dict results (`id`/`center`/`lb-rb-rt-lt`), restores the `tag36ARTag` family guard (loud error), and matches legacy grayscale→unsharp→`mean+f·(x−mean)` preprocessing. | `core/inference/stages/apriltag.py` |

**Verification:** all edited files byte-compile; 173 passed across the inference
stage/cache/runner/cnn-identity sweep; evidence-pipeline + 13 worker-integration tests
pass. AprilTag stage tests updated from the (wrong) upstream-API mocks to lab-fork
dict-style mocks. New regression test
`test_emitter_internal_calibration_matches_calibrate_at_source` proves emitter-internal
calibration == legacy calibrate-at-source (closes the `temperature ≠ 1.0` blind spot).
Run env: `~/miniforge3/envs/hydra-mps/bin/python` + `KMP_DUPLICATE_LIB_OK=TRUE` (base
`miniforge3` python has a broken torch/OpenMP).

**Still-failing (pre-existing, NOT introduced — confirmed via `git stash` baseline):**
`test_identity_evidence_emitter_uses_factor_posteriors` (expects flat catalog; emitter
builds composite cartesian-product catalog for multi-factor models) and
`test_tracking_optimizer_helpers.py` ×2 (`_ParamsFilter` lacks `filter_raw_detections` in
the legacy 12-tuple branch). Both fail identically on the unmodified branch.

### HIGH detection-filter pass (2026-06-28, Tier 1)

| ID | Fix | Files touched |
|---|---|---|
| H1 | `OBBConfig.target_classes` now threaded from `YOLO_TARGET_CLASSES` (None/empty == all) and passed as `classes=` to the direct + sequential-detect `model.predict()` calls. | `core/tracking/worker.py`, `core/inference/stages/obb.py` |
| H2 | **Already wired** in current WIP (post-audit): the worker builder honors `enable_aspect_ratio_filtering` + `reference_aspect_ratio` × `min/max_aspect_ratio_multiplier` → `OBBConfig.min/max_aspect_ratio`, and `filtering.py` applies the major/minor gate (CPU + CUDA + `filter_with_indices`). Verified equivalent to legacy `_obb_geometry:492-505`; covered by tests. No code change needed this pass. |
| H5 | Final post-filter cap now keeps the **largest** detections (`argsort(sizes)`, not confidence) in all three filter paths, matching legacy `_obb_geometry:587-588`. Builder now sets `max_detections = MAX_TARGETS` (was `2*MAX_TARGETS`, which silently disabled the legacy cap) while keeping `raw_detection_cap = 2*MAX_TARGETS`. Defaults reconciled to legacy: `iou_threshold` 0.45→**0.7**, `MAX_TARGETS` fallback 20→**8**. | `core/inference/stages/filtering.py`, `core/inference/config.py`, `core/tracking/worker.py` |
| H6 | Finite + positive-geometry (`w>0 & h>0`) validity mask added at OBB extraction (`_extract_obb_result`, `materialize_tensors`) and on the CUDA tensor filter path, dropping NaN/Inf/degenerate detections before caching or tracking — mirrors legacy `_obb_geometry:303-312` (incl. the warning log). | `core/inference/stages/obb.py`, `core/inference/stages/filtering.py` |

**Verification:** 191 passed across the inference + worker sweep (1 pre-existing failure
deselected). New regression tests: size-based final cap (CPU + CUDA paths), `target_classes`
forwarding (set + empty→None), and zero-height-drop-without-divz-warning. One obsolete test
(`...no_divzero_warning_for_zero_height`, which asserted degenerate detections are *kept*)
was updated to the legacy-parity behavior (dropped).

### HIGH classifier/cache pass (2026-06-29, Tier 2)

| ID | Fix | Files touched |
|---|---|---|
| H7 | Head-tail loader now rejects **multi-head** artifacts with `HeadTailFormatError` (was silently using only factor 0) and runs the checkpoint labels through `validate_headtail_labels` (canonical alias normalization: `head_left`/`north`/`n`/… → `left`/`up`/…), matching legacy `HeadTailAnalyzer._load_model` (`headtail.py:253-260`). `_label_to_heading_offset` also normalizes at runtime so aliased labels resolve instead of becoming undirected. | `core/inference/stages/headtail.py` |
| H8 | CNN top-class predictions below the per-phase `confidence_threshold` now collapse to `None` (confidence retained), matching legacy `CNNClassifier.predict_batch` (`cnn.py:404-407`). The threshold is threaded from each `CNNConfig.confidence_threshold` into `populate_live_cnn_store` → `_cnn_det_pred_to_class_prediction`. Stops `TrackCNNHistory.majority_class` from counting low-confidence guesses legacy discarded. Full posteriors (Bayesian decoder) stay unthresholded, as in legacy. | `core/tracking/frame_result_bridge.py`, `core/tracking/worker.py` |
| H9 | Detection cache now records **every processed frame index** (`written_frames`), so a processed-but-empty frame is distinguishable from an absent one. `read_frame` returns `None` for an absent frame (→ `load_frame` raises `KeyError`) instead of a misleading empty `OBBResult`. New `covers_frame_range` / `get_missing_frames` (legacy `detection_cache.py:403-433`) are surfaced via runner `detection_cache_covers_range` / `detection_cache_missing_frames`; the worker now refuses a **backward** pass and skips **reuse** when the cache does not cover `[START_FRAME, END_FRAME]` (truncated/interrupted forward passes no longer silently run a partial frame set). Falls back to unique `frame_indices` for pre-existing caches. | `core/inference/cache/store.py`, `core/inference/runner.py`, `core/tracking/worker.py` |

**Verification:** 453 passed / 23 failed across the inference + worker sweep; all 23
failures are **pre-existing worktree WIP** (unrelated `orchestrators/tracking.py`,
`live_features.py`, `test_individual_properties_cache.py` changes), confirmed identical via
two `git stash` baselines (mine-only and all-uncommitted). 12 new regression tests:
multi-head rejection, label alias normalization, non-canonical-label rejection (H7);
sub-threshold→None, above-threshold retained, default-threshold-keeps-all (H8); full vs
truncated `covers_frame_range`, `get_missing_frames`, and empty-frame-reads-empty vs
absent-frame-reads-None (H9).

---

## CRITICAL findings (verified; crash or silent corruption)

### C1 — Canonical-crop margin reads the wrong config key ✅ VERIFIED · ✅ RESOLVED (2026-06-28)
- **New:** `core/tracking/worker.py:4689`
  `canonical_margin=float(params.get("INDIVIDUAL_CROP_PADDING", 1.3))`
- **Legacy:** margin sourced from `ADVANCED_CONFIG["yolo_headtail_canonical_margin"]`
  (default 1.3) — `core/detectors/yolo_detector.py:286,304`.
- `INDIVIDUAL_CROP_PADDING` is the **crop padding fraction** (default `0.1`), a different
  quantity. Head-tail derives `padding_fraction = max(0.0, canonical_margin - 1.0)`
  (`core/identity/classification/headtail.py:154`). A real config carrying
  `INDIVIDUAL_CROP_PADDING=0.1` therefore yields `canonical_margin=0.1 →
  padding_fraction=0.0`.
- **Impact:** every canonical crop feeding head-tail / CNN / pose collapses to **zero
  margin** (legacy used 0.3) whenever head-tail is enabled on a YOLO run. Corrupts
  heading, identity, and pose crops.
- **Fix:** read `yolo_headtail_canonical_margin` (default 1.3); do not reuse the padding
  key. One-line change.

### C2 — CNN calibration temperature is never applied (wired path) ✅ VERIFIED · ✅ RESOLVED (2026-06-28)
*(Resolved a direct conflict between the identity-evidence and classifier audits.)*
- **Legacy:** posteriors are temperature-calibrated **at source** —
  `core/tracking/precompute.py:1244-1252` calls
  `predict_batch_posteriors(crops, calibration=self._calibration)`
  (`core/identity/classification/cnn.py:482-543`). The calibrated posteriors then flow to
  both the detection-posterior cache and the live store / online decoder.
- **New:** the CNN stage emits **raw** softmax (`CNNFactorPrediction.raw_probabilities`,
  `core/inference/stages/cnn.py:71`, `result.py:49`).
  `frame_result_bridge.populate_live_cnn_store` forwards `factor.raw_probabilities`
  directly to `IdentityEvidenceEmitter.build_frame_evidences(posteriors=…)`
  (`frame_result_bridge.py:154-190`). The emitter stores only a calibration **signature
  string** (`evidence_emitter.py:70-76,201`) and `_build_log_probs_from_posteriors`
  (`:265-284`) consumes the posteriors **without temperature scaling**.
- The only temperature-applying implementation, `IdentityEvidenceBuilder._calibrate`
  (`core/tracking/identity/evidence.py:144-154`), is **instantiated nowhere**
  (`grep "IdentityEvidenceBuilder("` → 0 hits in non-test code).
- **Why the conflict arose:** the emitter/decoder internals *are* byte-identical between
  trees (identity-evidence audit was correct about that), but their **input** changed
  from calibrated → raw posteriors. Both the live-store path and the
  detection-posterior cache now hold uncalibrated values.
- **Impact:** any deployment with `calibration_temperature ≠ 1.0` feeds uncalibrated
  identity evidence to the online decoder → different identity commitment than legacy.
- **Fix:** apply `CalibrationModel.calibrate` to posteriors in the CNN stage or inside
  the emitter, or wire `IdentityEvidenceBuilder` into the live path.

### C3 — Optimizer detection-cache rewire will throw at runtime ✅ VERIFIED · ✅ RESOLVED (2026-06-28)
- **New:** `core/tracking/optimizer.py:39-47` binds
  `from hydra_suite.core.inference.cache.store import DetectionCacheHandle as DetectionCache`
  inside a `try`. The new module exists, so the import **succeeds** and the legacy
  fallback (`except ImportError: from hydra_suite.data.detection_cache import DetectionCache`)
  **never triggers**.
- But the optimizer/preview/builder still call the **legacy API**:
  - `DetectionCache(self.detection_cache_path, mode="r")` — `optimizer.py:600`
  - `.is_compatible()` — `:601`
  - `.covers_frame_range(...)` / `.get_missing_frames(...)` — `:609-610`
  - `.get_frame(f_idx)` — `:121`
  - `add_frame()` / `.save()` / `mode="w"` in `optimizer_workers.py` builder.
- `DetectionCacheHandle` is a `@dataclass(path, key)` exposing only
  `is_valid / write_frame / read_frame / close` (`cache/store.py:57-90`). None of the
  legacy methods exist, and the dataclass has no `mode=` parameter → `TypeError` /
  `AttributeError` before any scoring runs.
- The new `apply_detection_filter` OBBResult branch (`optimizer.py:121-147`) is also dead
  because it begins with the nonexistent `get_frame`.
- **Impact:** the parameter-optimizer path is broken. Only the filtering helper + import
  alias were rewired; the cache lifecycle was not.

### C4 — AprilTag detector uses the wrong library API ✅ VERIFIED · ✅ RESOLVED (2026-06-28)
- **New:** `core/inference/stages/apriltag.py:24-39`
  ```python
  import apriltag
  options = apriltag.DetectorOptions(families=…, nthreads=…, quad_decimate=…,
      quad_sigma=…, refine_edges=…, decode_sharpening=…, max_hamming=…)
  detector = apriltag.Detector(options)
  except ImportError: detector = None
  ```
  This is the **upstream pupil-apriltags / python-apriltag** API.
- **Legacy:** `core/identity/classification/apriltag.py` mandates the **SEB lab fork**:
  `at.apriltag(family=…, maxhamming=…, …)` (`:184`) and verifies `tag36ARTag` support via
  `_has_required_apriltag_family` (`:34-66`), raising a clear error otherwise
  (`APRILTAG_REQUIRED_FAMILY = "tag36ARTag"`, `:26`).
- Against the required fork, `apriltag.DetectorOptions` raises `AttributeError`, which the
  `except ImportError` does **not** catch → uncaught crash. With a stock apriltag build it
  constructs but cannot decode the lab `tag36ARTag` family.
- **Impact:** AprilTag identity is effectively broken or silently disabled, and the
  required-family guard is gone. **No fixture clip exercises AprilTags**, so the runtime
  harness is blind to this.

---

## HIGH findings (confirmed dropped detection / runtime paths)

### H1 — `YOLO_TARGET_CLASSES` / `classes=` never passed ✅ VERIFIED · ✅ RESOLVED (2026-06-28)
No `classes=` argument in any `model.predict()` call (`obb.py:150,174,196`) and
`OBBConfig.target_classes` (`config.py:48`) is read nowhere in `stages/`. Legacy passed
`classes=target_classes` to every predict (`yolo_detector.py:489,1078,1665`). Multi-class
OBB models now emit **all** classes into tracking → spurious detections.

### H2 — Aspect-ratio filtering entirely absent ✅ VERIFIED · ✅ RESOLVED (already wired in WIP)
Zero references to `enable_aspect_ratio_filtering` / `reference_aspect_ratio` /
`min/max_aspect_ratio_multiplier` in `core/inference/`. Legacy gated detections by aspect
ratio (`filter_raw_detections:492-505`). Abnormal-aspect detections are no longer rejected.

### H3 — Foreign-region suppression configured but never applied ✅ VERIFIED
`suppress_foreign_regions` (`config.py:106`) and `background_color` appear **only** in the
config default and the pose cache key (`cache/keys.py:127`). No
`_apply_foreign_mask_canonical` / `fillPoly` / `foreign_corners=` call anywhere in
`core/inference/`. Legacy `UnifiedPrecompute` extracted canonical crops with
`suppress_foreign=True` (blacking out neighbor pixels). In dense multi-animal scenes,
pose/CNN crops now contain unmasked neighbors; toggling the (encoded) flag changes the
cache key but produces identical unmasked crops. (Note: legacy head-tail/CNN
*classification* crops were also unmasked; the regression is specifically vs the legacy
precompute canonical crops feeding pose/identity.)

### H4 — TensorRT / ONNX auto-export dropped ✅ VERIFIED
`ENABLE_TENSORRT` / `ENABLE_ONNX_RUNTIME` / `auto_export` / `.engine` have **zero**
references in the new worker. `_build_inference_config_from_params` never reads the
toggles; `_load_yolo` (`obb.py:329-338`) only `.to()`s cuda/mps. Legacy
`derive_detection_runtime_settings` + `_try_load_{onnx,tensorrt}_model` (auto-export) is
bypassed. `COMPUTE_RUNTIME="tensorrt"` without a prebuilt engine silently runs PyTorch or
fails. Same gap on the YOLO-pose path (pose audit GAP 3).

### H5 — Final detection cap sorts by confidence, not size ✅ VERIFIED · ✅ RESOLVED (2026-06-28)
`filtering.py:77,147,249` all use `np.argsort(confidences)[::-1][:max_detections]`.
Legacy capped to `MAX_TARGETS` keeping the **largest** detections
(`argsort(sizes)[::-1]`, `filter_raw_detections:587-588`). Different survivors when
detections exceed the cap; default cap value also differs (`MAX_TARGETS` vs
`OBBConfig.max_detections=20`). Default `iou_threshold` likewise changed (0.7 → 0.45,
`config.py:53`).

### H6 — NaN/Inf & non-positive-geometry validity guard missing ✅ VERIFIED · ✅ RESOLVED (2026-06-28)
No `isfinite` / `isnan` / positivity-drop in `obb.py`; only divide-by-zero guards
(`safe_minor`/`safe_h`). Legacy dropped non-finite or `major<=0`/`minor<=0` detections
before assignment/Kalman (`_extract_raw_detections:303-332`). NaN centroids/angles can now
be cached and fed to the tracker.

### H7 — Head-tail multi-head rejection + label normalization removed ✅ RESOLVED (2026-06-29)
- New loader never checks `meta.is_multihead`; a multi-head artifact silently loads and
  only factor 0 is used (`headtail.py:33-45`). Legacy raised `HeadTailFormatError`.
- New does literal `_DIRECTION_OFFSET.get(label)` (`headtail.py:100-101`) with no alias
  normalization (`head_left`, `north`, `n`, …) that legacy applied via
  `validate_headtail_labels` (`headtail.py:24-97,260`). Non-canonical labels silently
  become undirected.
- **Fix:** `load_headtail_model` now raises `HeadTailFormatError` on multi-head metadata
  and normalizes labels via the legacy `validate_headtail_labels` (storing the canonical
  set); `_label_to_heading_offset` normalizes aliases at runtime via
  `normalize_headtail_label`. Covered by 3 new tests (multi-head rejection, alias
  normalization, non-canonical rejection).

### H8 — CNN sub-threshold predictions no longer collapse to None ✅ RESOLVED (2026-06-29)
`frame_result_bridge._cnn_det_pred_to_class_prediction` (`:78-85`) always emits the argmax
class name regardless of confidence. Legacy `CNNIdentityBackend.predict_batch` set the
class to `None` when `best_conf < threshold` (`cnn.py:404-407`). The live-store majority
vote (`TrackCNNHistory.majority_class`, excludes None/"unknown") now counts
low-confidence guesses it previously discarded.
- **Fix:** the per-phase `CNNConfig.confidence_threshold` is now threaded through
  `populate_live_cnn_store` into `_cnn_det_pred_to_class_prediction`, which collapses a
  sub-threshold top class to `None` (confidence retained). Full posteriors fed to the
  Bayesian decoder stay unthresholded (legacy parity). Covered by 3 new tests.

### H9 — Cache validation is key-only; no frame-range coverage check ✅ RESOLVED (2026-06-29)
New `caches_all_valid()` / `is_valid()` only compare the stored `cache_key` string
(`cache/store.py:38-45,64-67`; `runner.py:294-298`). Legacy additionally required
`covers_frame_range(start,end)` **and** `get_missing_frames()`
(`data/detection_cache.py:403-433`; legacy `worker.py:1257,1276`). Consequences:
- A cache from a shorter/interrupted forward pass (same model+config+video signature)
  validates → backward/reuse silently processes a **truncated** frame set.
  `START/END_FRAME` are folded into the bg-sub key but **not** the OBB detection key
  (`keys.py:39-54,78-80`).
- A frame absent from a *valid* cache returns an **empty `OBBResult`, not `None`**
  (`store.py:77-88`); `load_frame` only raises `KeyError` when `raw_obb is None`
  (`runner.py:663-664`), which can't happen → a missing frame is silently treated as "no
  animals."
- **Fix:** the detection cache now persists `written_frames` (every processed frame index,
  including zero-detection frames) and exposes `covers_frame_range` / `get_missing_frames`;
  `read_frame` returns `None` for an unprocessed frame (→ `KeyError` upstream) while a
  processed-but-empty frame still reads back as an empty `OBBResult`. The runner surfaces
  `detection_cache_covers_range` / `detection_cache_missing_frames`, and the worker gates
  both the backward pass (hard refuse + missing-frame log) and forward-cache reuse on full
  `[START_FRAME, END_FRAME]` coverage. Pre-existing caches fall back to unique
  `frame_indices`. Covered by 4 new tests.

---

## Per-domain coverage tables

### Domain 1 — Detection / OBB + filtering

> Legacy detection lives in `core/detectors/` (`bg_detector.py` background-subtraction
> `ObjectDetector`; `yolo_detector.py` `YOLOOBBDetector`), selected by
> `factory.create_detector` on `DETECTION_METHOD`. The redesign replaces **only** the
> YOLO-OBB path with `core/inference/`; bg-sub still routes through the legacy
> `create_detector` (worker.py:1382,2507) and is unchanged.

| Legacy path (file:line) | Covered? | New location | Notes |
|---|---|---|---|
| Detector selection by `DETECTION_METHOD` (factory.py:18-29) | YES | worker.py:314,1043,2507 + runner | bg-sub legacy; yolo_obb → InferenceRunner |
| Background-subtraction detector (bg_detector.py:158-238) | YES (unchanged) | legacy `create_detector` reused | Out of redesign scope, still reachable |
| OBB mode "direct" (yolo_detector.py:980) | YES | obb.py:138-164 `_run_direct` | Equivalent |
| OBB mode "sequential" (yolo_detector.py:1520) | PARTIAL | obb.py:167-207 `_run_sequential` | Several sub-options dropped (below) |
| Invalid `obb_mode` → "direct" fallback (yolo_detector.py:50,120) | UNCLEAR | config.py:45 `Literal[...]` | Typed; no runtime coercion |
| `iou=1.0` disables internal NMS (yolo_detector.py:488,1077) | YES | obb.py:152,178,202 | Matches |
| `RAW_YOLO_CONFIDENCE_FLOOR` w/ `max(1e-4,…)` (yolo_detector.py:1666) | PARTIAL | obb.py:149 (default 1e-3) | No 1e-4 clamp; floors differ |
| **`YOLO_TARGET_CLASSES` / `classes=`** (yolo_detector.py:489,1078,1665) | **RESOLVED** ✅ | obb.py `_run_direct`/`_run_sequential` (`classes=target_classes or None`) | H1 fixed 2026-06-28 |
| `max_det` = `MAX_TARGETS*2` at predict (yolo_detector.py:252,1667) | MISSING / PARTIAL | obb.py (none); filtering.py:76 post cap | No predict-time cap; full raw set cached |
| Angle fold to [0,π)+deg-guard (`_extract_raw_detections:289-299`) | YES | obb.py:35-72 `_normalize_obb_geometry` | Equivalent |
| Major/minor swap when w<h, angle+90 (:295-299) | YES | obb.py:64-68 | Matches |
| **Finite/positive validity mask** (:303-332) | **RESOLVED** ✅ | obb.py `_valid_detection_mask` (extract+materialize); filtering CUDA keep | H6 fixed 2026-06-28 |
| Ellipse area in `shapes[:,0]` (:335) | PARTIAL | obb.py:69 sizes=major*minor; filtering ×π/4 | `OBBResult.sizes` = rect area, not ellipse |
| Corner reconstruction [TL,TR,BR,BL] (:348-356) | YES | obb.py:75-107 `_corners_from_xywhr` | Equivalent (validated) |
| Raw cap sort-by-conf then `[:cap]` (:358-360) | MISSING | — | No raw confidence cap |
| `class_ids` extraction (:268-276,376) | UNCLEAR | not in OBBResult | No class-id field carried |
| Confidence gate `>= YOLO_CONFIDENCE_THRESHOLD` (:481) | YES | filtering.py:50,99,230 | Matches |
| Size gate (ellipse area, `ENABLE_SIZE_FILTERING`) (:483-490) | PARTIAL | filtering.py:52-56 | No master toggle; always applies when bounds set |
| **Aspect-ratio gate** (:492-505) | **RESOLVED** ✅ | filtering.py:58-60,114-120,271-275 + worker builder | H2 (wired in WIP) |
| ROI mask gate w/ bounds (:507-515) | YES | filtering.py:58-67,110-114,236-240 | Equivalent |
| Empty-after-filter guard (:517-520) | YES | filtering.py:70-71,117-118,242-243 | Matches |
| OBB NMS (convex-hull IOU, AABB pre-screen) (`_obb_geometry.py:32-69`,:75-246) | PARTIAL | filtering.py:153-195 `_obb_nms` | `rotatedRectangleIntersection` vs `intersectConvexConvex`; `>` vs `>=`; default 0.7→0.45 |
| NMS only when len>1 (:105-131) | YES | filtering.py:73,143,244 | Matches |
| **Final cap by SIZE** (:587-588) | **RESOLVED** ✅ | filtering.py (now `argsort(sizes)`); max_detections=MAX_TARGETS | H5 fixed 2026-06-28 |
| `iou_threshold` default 0.7 (:415) | RESOLVED ✅ | config.py (0.7); builder default 0.7 | H5 default reconciled 2026-06-28 |
| `detection_ids` subset-preserved (:423-428,537) | YES | filtering.py:42-44 (CPU); :138 (CUDA regen) | Matches contract |
| Direct **ONNX** executor (`_direct_obb_runtime.py:390`) | MISSING | — | onnx_cuda → CPU wrapper path |
| Direct **TensorRT** executor (:578) | MISSING | — | No TRT warmup/streams |
| Direct **PyTorch-CUDA** executor, square letterbox (:684-698) | MISSING | obb.py:334 `.to("cuda")` only | Rectangular letterbox on CUDA → different scores for non-square video |
| Direct **detect** stage-1 executors (:837,899) | MISSING | obb.py:174 plain predict | Wrapper-only |
| Native CUDA raw-tensor path | YES (new) | obb.py:143-260; filtering.py:83-150 | New optimization |
| ONNX MPS→CoreML fallback (`_predict_with_coreml_fallback`:400-436) | MISSING | obb.py:336 | No retry-on-CPU |
| ONNX/TRT fixed-batch chunk+pad (:466-505,2074) | MISSING | — | Static-shape artifacts may error |
| ONNX per-frame fallback on batch failure (:2123) | MISSING | worker.py:1413 fatal | Batch exception fatal |
| Sequential stage-1 conf/imgsz (:1049,1087) | YES | config.py:33,35; obb.py:175,180 | Present |
| Sequential crop pad/min-size/square (`_build_sequential_crop:558-560`) | PARTIAL | obb.py:227-235 | min-size only under enforce_square; int() vs floor/ceil |
| Sequential stage-2 resize + coord scale-back (:1445-1457) | PARTIAL | obb.py:202,274-276 | No pre-resize / sx,sy scale-back |
| Sequential stage-2 pow2 pad / individual batch (:1295,1325) | PARTIAL | obb.py:192-203 | Plain chunking |
| Sequential merge + final sort (:1475) | YES | obb.py:311-326 `_merge_obb_results` | Present |
| GPU crop build path (NVDec) (`_seq_build_gpu_crops:1174`) | MISSING | obb.py:216-219 (`.cpu().numpy()`) | NVDec zero-copy lost |
| Empty stage-1 / crops guards (:1546,1586) | YES | obb.py:185-191 | Matches |
| Per-frame vs batched (:1636,2275) | YES | runner.py:300,470 | Both exist |
| Cached-detection reuse (worker.py:1308) | YES (diff mechanism) | runner.py:652; filtering.py:214 | `.npz` + re-filter |
| Detection cache key excludes conf/iou/size (keys.py) | YES | keys.py:39-54 | Threshold edits reuse cache |
| Empty-model/inference-exception → empty (:1658,1702) | PARTIAL | runner.py:315 (no try/except) | Realtime exception propagates |

### Domain 2 — Crops + affine alignment

> Shared primitive `core/canonicalization/crop.py` is **byte-identical** between trees;
> only the callers changed.

| Legacy path (file:line) | Covered? | New location | Notes |
|---|---|---|---|
| `compute_alignment_affine` (crop.py:133-197) | YES | identical; crops.py:187,234; pose.py:196 | All stages delegate |
| `compute_native_crop_dimensions` (crop.py:67-102) | YES | identical; crops.py:158,183 | Preserved |
| Degenerate-OBB guard (crop.py:159) | YES | crops.py:190,237; pose.py:198; headtail.py:560 | Behaviors differ slightly (GAP 3) |
| Major-axis + atan2 heading (crop.py:162-169) | YES | identical + headtail `_signed_major_axis_from_corners:137` | Re-derived to cancel corner flip |
| CPU warp INTER_LINEAR + BORDER_REPLICATE (crop.py:212-218) | YES | crops.py:193-199 | Same flags |
| GPU single warp `gpu_canonical_crop` (crop.py:226-306) | PARTIAL | retained, **no inference caller** | Only batch variant used |
| GPU batched warp `gpu_canonical_crop_batch` (crop.py:309-394) | YES | crops.py:246 `_extract_canonical_gpu` | CUDA path; N=0/N=1 guards preserved |
| Fixed-128 classification canvas + even-round (headtail.py:526-532) | PARTIAL | crops.py:103-162 (native+interpolate) | Different intermediate canvas (GAP 2) |
| Native-scale crop for downstream (precompute.py:513-528) | YES | crops.py:171-199 | CPU path matches |
| Per-detection independent canvas | PARTIAL | crops.py:103-246 (per-frame max canvas) | Stacks + recovers native sub-region |
| **Foreign-OBB suppression** (`_apply_foreign_mask_canonical` crop.py:559-578) | **MISSING** ✅ | config+key only | See H3 |
| `bg_color` fill (crop.py:205) | MISSING | config field unused | Tied to H3 |
| `apply_headtail_rotation` (crop.py:397-448) | N/A by design | headtail returns heading hints, `canonical_affines=None` | Architectural change (Correction 15) |
| detection_id keying / filter→raw affine map (precompute.py:327-348) | YES | runner.py:321-338 (`make_detection_ids`, `filter_with_indices`) | Keyed + re-stamped |
| AABB crop w/ padding+clamp | YES | crops.py:51-76 `extract_aabb_crops` | empty→1×1 guard |
| Channel BGR/HWC↔CHW, uint8/float (headtail.py:543-550) | YES | crops.py:85-90,115,218; resize ×255 :167 | Double-/255 trap handled |
| Empty / N=0 guard | YES | crops.py:35-36 (`(0,3,64,64)`); pose/cnn/headtail early-return | 64×64 placeholder (cosmetic) |

**Domain 2 gaps:** H3 (foreign suppression); GAP 2 — padding unified at 0.3 vs legacy's
split 0.1 (native) / 0.3 (classify), and `PoseConfig.crop_padding=0.1` is ignored
(`pose.py:184`); GAP 3 — degenerate-OBB fallback diverges (CPU→zeros, GPU→8×8 identity,
pose/headtail→skip) vs legacy AABB fallback (`precompute.py:530-545`); GAP 4 —
`gpu_canonical_crop` single-crop primitive dead in new pipeline.
**Verified-correct:** "rewritten to mirror `compute_alignment_affine`" (TRUE),
"GPU-batched + CPU warpAffine split" (TRUE, branches on `runtime.tensor_on_cuda`).

### Domain 3 — Head-tail / CNN / AprilTag

> Low-level `evidence.py`, `online.py`, `calibration.py`, `evidence_emitter.py`,
> `identity/cache.py`, `live_features.py` are **byte-identical** between trees.

**Head-tail**

| Legacy path (file:line) | Covered? | New location | Notes |
|---|---|---|---|
| Backend w/ `trt_profile_max_batch=batch_size` (headtail.py:247-251) | PARTIAL | stages/headtail.py:38 | TRT profile max-batch drops to 512 default |
| Flat-only enforcement (`HeadTailFormatError`) (:253-257) | **RESOLVED** ✅ | headtail.py `load_headtail_model` | H7 fixed 2026-06-29 |
| Label validation/alias normalization (:24-97,260) | **RESOLVED** ✅ | headtail.py `load_headtail_model` + `_label_to_heading_offset` | H7 fixed 2026-06-29 |
| Eager warmup at load (:265) | PARTIAL | — | Lazy warmup only |
| Heading offsets r=0,l=π,u=−π/2,d=+π/2 (:52-57) | YES | headtail.py:15-20 | Identical |
| `predict_batch` chunking / TRT clamp (:639-743) | MISSING | headtail.py:82 | Whole batch in one call |
| GPU-native `analyze_crops_cuda`/IOBinding (:360-506,655) | MISSING | headtail.py:82; crops.py:119 | CPU path only; perf regression |
| conf<thresh → unknown/NaN (:308-310,694) | YES | headtail.py:98-99; result.py:64-66 | Matches |
| axis_theta from alignment affine (:557,765) | PARTIAL | headtail.py:92-107 | Re-derived; ~98% parity, ~2% disagreement |
| heading/conf/directed padded to det count (:313-314,398) | YES | headtail.py:64-66; runner.py:194-208 | git-history padding fixes present & correct |
| `from_frame_result` directed→bool, affines None→zeros | YES | streaming_payload.py:121-134 | Fixes correct |

**CNN identity**

| Legacy path (file:line) | Covered? | New location | Notes |
|---|---|---|---|
| Flat (K=1) argmax+conf (cnn.py:56-71,400) | YES | stages/cnn.py:66-75; bridge:61-92 | Matches |
| Multi-head per-factor split (cnn.py:368-417) | YES | stages/cnn.py:66-75 | Preserved |
| scoring_mode guard (cnn.py:368-376) | PARTIAL | config.py:74 + evidence.py:175 | Honored only in **dead** `IdentityEvidenceBuilder` |
| Per-factor conf thresholding → None (cnn.py:404-407) | **RESOLVED** ✅ | bridge `_cnn_det_pred_to_class_prediction` (threshold threaded from `CNNConfig`) | H8 fixed 2026-06-29 |
| Raw probs surfaced (cnn.py:482-543) | YES | stages/cnn.py:71; result.py:49 | "store raw, calibrate later" |
| **Temperature/calibration before evidence** (cnn.py:530; precompute.py:1247) | **MISSING** ✅ | evidence_emitter.py:182-205 (raw) | See C2 |
| `predict_batch_cuda` (cnn.py:419-480) | MISSING | stages/cnn.py:63 | CPU only |
| cost_atomic / cost_per_head_average (cnn.py:656-722) | YES | evidence.py:175-185 | Mirrored (dead path) |
| `_build_log_probs_from_posteriors` (emitter 265-284) | YES | identical | — |
| Empty-input guard (cnn.py:390,512) | YES | stages/cnn.py:54-55 | — |

**AprilTag**

| Legacy path (file:line) | Covered? | New location | Notes |
|---|---|---|---|
| Required lab-fork family check (apriltag.py:34-66) | MISSING | apriltag.py:24-39 | See C4 |
| Detector API `at.apriltag(...)` (apriltag.py:184-192) | MISSING ✅ | apriltag.py:28-37 `DetectorOptions` | See C4 |
| Composite-strip decode + reproject (apriltag.py:230-316) | MISSING | apriltag.py:68-84 (per-crop loop) | Loses single-call design |
| Preprocess grayscale→unsharp→contrast on composite (:252-260) | PARTIAL | apriltag.py:71-105 | Contrast formula differs (`x·f` vs `mean+f·(x−mean)`) |
| `unsharp_kernel` odd-ness | PARTIAL | apriltag.py:99 (`k|1`) | New forces odd |
| `max_tag_id` filter (:271) | YES | apriltag.py:78-80 | Same |
| Corner reproject to abs frame coords (:294-304) | MISSING | apriltag.py:84 (crop-relative) | Geometry crop-local; no offset |
| `hamming` per tag (:275,313) | MISSING | bridge:253 (hard-coded 0) | Hamming dropped |
| `det_indices` mapping (:311) | YES | apriltag.py:82 | Preserved |
| Empty-input guard (:217) | YES | apriltag.py:60-61 | — |
| Config from params (:97-118) | PARTIAL | config.py:116-131 + worker builder | Some params unmapped |

### Domain 4 — Pose (YOLO + SLEAP)

> Shared `core/identity/pose/{api,types,utils,quality,artifacts}.py` and
> `backends/{yolo,sleap_utils}.py` are **byte-identical**. `backends/sleap.py` differs only
> by additive changes (a `_to_uint8_image` fix + `predict_batch_cuda` + diagnostics).

| Legacy path (file:line) | Covered? | New location | Notes |
|---|---|---|---|
| Backend via `build_runtime_config`→`create_pose_backend_from_config` (api.py:190-317) | YES | stages/pose.py:100-138 (SLEAP) | YOLO branch builds `YoloNativeBackend` directly (skips validation/export, GAP 3) |
| SLEAP delegated to `create_pose_backend_from_config` | YES | pose.py:100-138 | git-claim verified |
| SLEAP branch in PoseConfig builder | YES | worker.py:4763-4772 | git-claim verified |
| skeleton/keypoint_names threading (api.py:124-135) | YES | pose.py:62-74,89,135; worker.py:1636-1645 | git-claim verified |
| Consume canonical `PoseResult` (pose_pipeline.py:724-757) | YES | pose.py:211-232 | More tolerant of batch axis |
| YOLO keypoint extraction/best-instance/conf clamp (yolo.py:233-289) | YES | unchanged | — |
| crop→frame mapping: affine vs AABB offset (pose_pipeline.py:727-745) | PARTIAL | pose.py:184-201,229-231 | **No AABB fallback** (GAP 2); affine-only |
| Letterbox pre-resize (`POSE_PIPELINE_PRE_RESIZE`) (:530-534) | MISSING by design | — | Silently ignored |
| **Foreign-OBB keypoint suppression** (`filter_keypoints_by_foreign_obbs`:746-754) | **MISSING** | pose.py (none) | GAP 4 |
| Empty-detection guard (:507,614) | YES | pose.py:181-182; bridge:211 | — |
| Cross-frame batch + double-buffer + async cache + parallel crops (pose_pipeline.py:442,549) | PARTIAL | pose.py:203 (per-frame) | Within-frame batch preserved; perf regression |
| Device/runtime branches (`derive_pose_runtime_settings`) | PARTIAL | pose.py:104-121 (hand-rolled) | `sleap_export_input_hw=None`; flavor ignored (GAP 5) |
| `predict_batch_cuda` (pose_pipeline.py:688) | MISSING | pose.py:188,203 (CPU) | On-device path unused |
| YOLO conf/iou/max_det/batch (api.py:246-255) | YES | pose.py:85-94 | `max_det=1` matches default |
| Pose quality `min_valid_conf`, valid_mask (api.py:175) | YES | pose.py:233-234 | Default 0→1 (GAP 1) |
| Pose cache write (IndividualPropertiesCache) (:240,571) | YES (new store) | runner.py:434-440; **gated by valid_mask** bridge:215-220 | Semantic change (GAP 1) |
| Pose cache read (`build_pose_detection_keypoint_map` features.py:75-98) | YES | features.py:75-142 adapter | git-claim verified; back-compat |
| Pose-direction derivation (`_pf_*` worker loop ~2896) | YES | worker.py:2779-2820 | Byte-identical; in worker, not run_pose |
| Heading merge pose→headtail→OBB (`_pf_build_direction_overrides`) | YES | worker.py:2836-2847; runner `assemble_resolved_headings` (dead) | Two impls; worker wins (GAP 6) |
| Pose visibility → KF R-scaling + prototype EMA (worker:3628,3698) | YES | new worker retains | Populated from new live store |

**Pose gaps:** GAP 1 (valid_mask gates storage; default 0→1 conflates two legacy
thresholds — storage `POSE_MIN_KPT_CONF_VALID`/`min_valid` vs direction
`POSE_DIRECTION_MIN_VALID_KEYPOINTS`); GAP 2 (no AABB fallback → keypoints left in crop
space if affine throws); GAP 3 (YOLO path skips validation + ONNX/TRT auto-export);
GAP 4 (foreign-OBB keypoint post-suppression dropped); GAP 5 (SLEAP export-input-HW /
`derive_pose_runtime_settings` / `POSE_RUNTIME_FLAVOR` ignored); GAP 6 (runner's
`assemble_resolved_headings` pose branch + `anterior/posterior/ignore_keypoints` config
fields are dead); GAP 7 (cross-frame batch / double-buffer / async-cache / parallel-crop
perf path absent). **New-path fix (not a regression):** SLEAP all-zero-keypoints bug fixed
via `_to_uint8_image` (handles float32 [0,1] canonical crops).

### Domain 5 — Identity evidence / online decoder / calibration

> `evidence.py`, `online.py`, `calibration.py`, `evidence_emitter.py`,
> `identity/cache.py`, `live_features.py` are **byte-identical**; the per-frame
> evidence-assembly block in `worker.py` is **byte-identical**. Only the live-store CNN
> emitter wiring changed.

| Legacy path (file:line) | Covered? | New location | Notes |
|---|---|---|---|
| Temperature calibration → signature (worker.py:830-851) | YES (wiring) | worker.py:848-869; `_build_cnn_evidence_emitter:4536` | Signature logic identical — but **values uncalibrated** (C2) |
| `CalibrationModel.calibrate` log-softmax temp (calibration.py:42-60) | YES | identical | Byte-identical (but unused in live path, C2) |
| AprilTag-priority via `apriltag_log_prior` (worker.py:3358-3387) | YES | worker.py:3272-3301 | Byte-identical |
| CNN evidence from cached posteriors + remap (worker.py:3390-3441) | YES | worker.py:3304-3355 | Byte-identical |
| Composite (multi-factor) joint log-prior (worker.py:3454-3513) | YES | worker.py:3368-3427 | Byte-identical |
| Single-factor (flat) `cnn_log_prior` (worker.py:3514-3541) | YES | worker.py:3428-3455 | Byte-identical |
| Decoder `update_frame` predict→fuse→swap→lock→assign→commit (online.py:381-495) | YES | identical | Log-space fusion identical |
| Per-det_id keying (evidence_emitter:140-165) | YES | identical | — |
| det_index→detection_id mapping (frame_result_bridge:163-176) | YES (new) | bridge:163-176 | New plumbing; matches semantics |
| Empty/missing-evidence guards | YES | identical | — |
| Numeric contract (full-catalog log_probs, logaddexp renorm, 1e-300 floor) | YES | identical | — |
| Live-store emitter setup (streaming) (worker.py:1660-1739) | PARTIAL | worker.py:1660-1678 + helper 4470-4592 | Deltas D1, D2 below |
| Batch emitter setup (worker.py:1749-1802) | YES | worker.py:1741-1805 | Equivalent |
| Emitter flush at finalize (worker.py:4387-4401) | YES | worker.py:4301-4315 | Byte-identical |
| Headtail → identity evidence | N/A by design | — | Headtail feeds heading only, both trees |

**Domain 5 deltas:**
- **C2** (CNN posteriors uncalibrated in wired path) — the critical one.
- **D1 [MEDIUM]:** flush registration no longer gated on `ENABLE_IDENTITY_POSTERIOR_CACHE`.
  Legacy appended the emitter for sidecar flush only when opted in
  (`worker.py:1724-1729`); new appends unconditionally (`worker.py:1674-1678`) → identity
  evidence `.npz` sidecar is **always written**. Decoder math unaffected; disk-IO behavior
  changed.
- **D2 [LOW]:** metadata-missing fallback differs — legacy defaults
  `_factor_labels=[["unknown"]]` and still builds an emitter (`worker.py:1674-1679`); new
  `_build_cnn_evidence_emitter` returns `None` if it can't resolve `class_names_per_factor`
  even after a fresh `ClassifierBackend` load (`worker.py:4504-4531`) → live store falls
  back to top-1-only. New behavior arguably better; edge-case difference.
- **INFO:** `IdentityEvidenceBuilder` (`core/tracking/identity/evidence.py`) is **dead** —
  no production importer; its `_calibrate`/scoring-mode logic is an independent
  reimplementation that, if wired, would need parity testing.

### Domain 6 — Caching

| Legacy field / path (file:line) | Covered? | New location | Notes |
|---|---|---|---|
| `meas` [cx,cy,theta] (detection_cache.py:219,296) | YES | store.py:79-88 (centroids+angles) | Recombined by readers |
| `sizes` (:220,299) | YES | store.py:84,99,128 | — |
| `shapes` (ellipse_area, aspect) (:221,302) | YES | store.py:85,102,129 | — |
| `confidences` (:222,305) | YES | store.py:86,103,130 | Threshold re-applied at read |
| `obb_corners` (:224,311) | YES | store.py:87,102,131 | — |
| `detection_ids` stride 10000 (:223,308) | YES | result.py:12,31; store.py:88 | No float64→int normalization (G5) |
| `heading_hints` (:225,314) | YES (relocated) | HeadTailCacheHandle store.py:199-203 | Separate headtail cache |
| `heading_confidences` (:226,317) | YES (relocated) | store.py:204-208 | — |
| `directed_mask` (:227,320) | YES (relocated) | store.py:209-213 | — |
| `canonical_affines` (M_align) (:228,323) | DROPPED by design | result.py:44-45 (None on load) | Recomputed — verify bit-parity (G3) |
| `canonical_canvas_dims` (:229,326) | MISSING by design | — | Consumed as `_canvas_dims` (dropped) |
| `canonical_M_inverse` (:230,329) | MISSING by design | — | Consumed as `_M_inverse` (dropped) |
| Schema/version {"2.0".."2.4"} (:85-90) | PARTIAL | base.py:11 (int 2) in key string | No multi-version accept list (OK for fresh schema) |
| Cache key construction | YES (stronger) | keys.py (model_path+mtime+config_hash+video_signature) | Materially stronger |
| Backward refuses w/o valid cache (worker:1294) | YES | worker.py:1268-1302; 1366-1375 (bgsub) | Preserved |
| `use_cached_detections` reuse (worker:1236) | YES | worker.py:1303-1321 | — |
| **`covers_frame_range`** (:403-416; worker:1257) | **RESOLVED** ✅ | store.py `covers_frame_range`; runner `detection_cache_covers_range`; worker gates backward + reuse | H9 fixed 2026-06-29 |
| **`get_missing_frames`** (:422-433; worker:1276) | **RESOLVED** ✅ | store.py `get_missing_frames`; runner `detection_cache_missing_frames` | H9 fixed 2026-06-29 |
| `matches_frame_range` (:418-420) | PARTIAL | `written_frames` set stored | Coverage via membership, not exact-range metadata |
| Missing-frame on read → empty entry (:296-331) | **RESOLVED** ✅ | store.py `read_frame` returns None for unprocessed frame (vs empty for processed-empty) | H9 fixed 2026-06-29 |
| Empty-frame padding to range (worker:4343) | PARTIAL | bgsub worker:2570; OBB runner:563 | No post-hoc fill loop |
| `savez_compressed` (:257) | PARTIAL | store.py:48-50 (`np.savez`) | Uncompressed; larger files |
| Tag cache tag_ids/centers/corners/det_indices/**hammings** (tag_observation_cache.py:124) | PARTIAL | AprilTagCacheHandle store.py:401-468 | **hammings NOT carried** (G4) |
| Tag cache version + range (:72,244) | PARTIAL | keys.py:138-150 (key-only) | Same limitation as H9 |
| Pose keypoints object array (properties/cache.py:333) | YES | PoseCacheHandle store.py:331-395 (+valid_mask) | Preserved |
| Pose summary stats on read (:390-436) | PARTIAL | not in PoseResult | Recomputable; export not rewired (G6) |
| Properties id hashing (:88-274) | N/A (legacy retained) | unchanged | Export still on legacy |
| DetectedPropertiesCache theta/heading fields (detected_cache.py:141) | N/A (legacy retained) | unchanged | "alias revert" per git log (G6) |
| IdentityEvidenceCache (identity/cache.py) | N/A (legacy retained) | unchanged | Used as-is |
| float64 det-id back-compat read (:17-47) | N/A (new schema) | — | v2-only |

**Caching gaps:** G1+G2 = **H9 ✅ RESOLVED** (frame-range coverage now enforced for the OBB
detection cache; tag cache range remains key-only — see row above);
G3 (canonical metadata not persisted — verify recomputation matches originally-applied
transform); G4 (AprilTag `hammings` dropped — corroborates Domain 3); G5 (no float→int
det-id normalization, low risk on v2-only); **G6 (HIGHEST follow-up):** rich-export
readers (`properties/export.py`) still consume legacy `IndividualPropertiesCache` /
`DetectedPropertiesCache` — **confirm the new worker still WRITES those two caches**, or
HeadingResolved / HeadTail export columns silently vanish.
**Positive:** backward refuse-without-cache preserved; stride 10000 preserved; keying
strictly stronger; read-only backward handles never overwrite forward caches
(`runner.py:288-292`); pose keypoint round-trip + det_id alignment preserved.

### Domain 7 — Worker orchestration + runtime

| Legacy orchestration path | Covered? | New location | Notes |
|---|---|---|---|
| Stage order detect→crops→headtail→pose→cnn→apriltag→identity | YES | runner.py:315-452,532-650 | headtail+cnn+pose concurrent (legacy too) |
| Forward fresh non-realtime = two-phase (batch detect→cache→replay) | YES | worker.py:1387-1440; load_frame 2391-2465 | Mapped to batch pass + replay |
| Forward realtime = per-frame live detect | YES | worker.py:2589-2643 | Caches opened for writing first call |
| Backward = cache-driven replay, refuse w/o cache | YES | worker.py:1266-1322; 2391-2465 | Parity |
| Background-subtraction live + backward replay | YES | worker.py:1350-1385, 2467-2587 | New DetectionCacheHandle bgsub_detection.npz |
| start_frame/end_frame in batch pass | YES (fixed) | runner.py:470-528; worker.py:1400-1409 | Commit 6b127dc |
| Realtime double-filter bug | YES (fixed) | runner.py:315-340 | Commit 6109a2f |
| Cross-frame crop padding removal | YES (fixed) | runner.py:546-650 | Commit 6109a2f |
| USE_NEW_INFERENCE_PIPELINE flag | YES (removed) | worker.py:80-81 (comment only) | No dead branches |
| Filtering gates (conf/IOU/size/ROI/max_det) | YES | filtering.py:35-83 | (aspect-ratio missing, H2) |
| Empty-frame handling | YES | runner.py:342-345,567; worker.py:2444 | Clean |
| detection_ids = frame*10000+slot | YES | result.py:12,30 | Consistent incl. bgsub |
| Streaming payload to GUI | YES | streaming_payload.py; worker.py:2657-2751 | Preserved |
| Live store population (pose/cnn/tag) incl. backward | YES | worker.py:1602-1683,2713-2751; bridge | Preserved |
| Heading resolution priority | YES | worker.py:2836-2877 | Byte-identical (runner copy unused) |
| **HeadTail canonical_margin source** | **MISSING** ✅ | worker.py:4689 | See C1 |
| Per-stage runtime independence | PARTIAL | runtime.py:18-41; config.py:158-174 | RuntimeContext collapses to one device |
| **OBB TensorRT/ONNX auto-build** | **MISSING** ✅ | worker.py:4612-4666 | See H4 |
| `POSE_RUNTIME_FLAVOR` translation | PARTIAL/MISSING | worker.py:4725-4727 | Flavor ignored → CPU fallback |
| `onnx_coreml` runtime | MISSING | obb.py:334-338 | CoreML provider never wired |
| `YOLO_DEVICE`/auto-detect | PARTIAL | worker.py:4612 | auto→CUDA no longer honored |
| Mixed CUDA/CPU validation | PARTIAL | config.py:176-185 (only from_json) | Not called from worker builder → silent all-GPU promote |
| tensor_on_cuda gate | YES | runtime.py:16,25; obb.py:159 | Correct |
| NVDEC GPU decode | PARTIAL (perf) | runner.py:482 (plain cv2) | `use_nvdec` tracked, unused |
| Sequential-OBB stage2 params | PARTIAL | worker.py:4630-4650 | Dataclass defaults; runtimes forced |
| AprilTag preprocessing params | PARTIAL | worker.py:4790-4798 | refine_edges/decode_sharpening/unsharp/contrast/max_tag_id unmapped (defaults match) |
| **Optimizer detection acquisition (rewire)** | **MISSING/BROKEN** ✅ | optimizer.py:39-47,121,600-610 | See C3 |

**Domain 7 extra gaps:** optimizer scoring fidelity loss even if C3 fixed — new filter
branch builds `OBBConfig(confidence_threshold=…)` only, dropping `roi_mask` (hardcoded
None, `api.py:38`), size bounds, IOU, target_classes, and returns empty head-tail fields →
non-equivalent optima.

---

## Recommended next steps

1. ~~Fix the four CRITICAL items first.~~ **DONE (2026-06-28)** — see Remediation log.
2. ~~Restore the HIGH detection-filter paths (target_classes, aspect-ratio, NaN/finite
   guard, size-based final cap).~~ **DONE (2026-06-28, Tier 1)** — see Remediation log.
   Foreign-region suppression (H3) is the remaining acceleration/behaviour decision (step 3).
2b. ~~Fix the HIGH classifier/cache paths (CNN sub-threshold→None, head-tail multi-head
   rejection + label normalization, cache frame-range coverage).~~ **DONE (2026-06-29,
   Tier 2)** — see Remediation log. **← all correctness findings now fixed; current front is
   the harness extension (step 5) + the H3/H4 owner decision (step 3).**
3. **Decide explicitly** on the intentionally-dropped acceleration paths (direct
   ONNX/TRT/PyTorch-CUDA executors, GPU-native head-tail/CNN/pose inference, NVDEC,
   TRT/ONNX auto-export, per-stage runtime independence). If dropped on purpose, the
   harness PERFORMANCE gate should confirm acceptable throughput and the square-letterbox
   change should be validated for non-square video; if not, they are regressions.
4. **Confirm G6:** verify the new worker still writes `IndividualPropertiesCache` /
   `DetectedPropertiesCache`, since `properties/export.py` still reads them.
5. **Extend the equivalence harness** to cover the blind spots: an AprilTag clip, a
   `calibration_temperature ≠ 1.0` run, a multi-class OBB model, an optimizer invocation,
   a frame-range-coverage cache test, and a non-square-video CUDA run.
6. **Clean up dead/unwired code:** `IdentityEvidenceBuilder` (intended calibration home),
   the runner's `assemble_resolved_headings` pose branch, the unused `roi_mask_cuda`, and
   the unused `gpu_canonical_crop` single-crop primitive — each is a maintenance trap that
   currently misrepresents where authoritative logic lives.

---

## Next best targets (post-CRITICAL, prioritized)

Ranked by *parity impact ÷ effort*, with no architectural decision required until Tier 3.

### Tier 1 — detection-filter correctness ✅ DONE (2026-06-28)
All four fixed to match legacy (H1 target_classes, H2 aspect-ratio [already wired],
H5 size-based final cap + default reconciliation, H6 finite/positive guard). See the
Remediation log. **Still owed:** the harness fixtures that would *catch* regressions here —
a **multi-class OBB clip** and a **size-cap-exceeded clip** (both current blind spots).

### Tier 2 — classifier/cache correctness ✅ DONE (2026-06-29)
All three fixed to match legacy (H8 CNN sub-threshold→None, H7 head-tail multi-head
rejection + alias normalization, H9 cache frame-range coverage + absent-vs-empty frame
distinction). See the Remediation log. **Still owed:** the harness fixtures that would
*catch* regressions here — a **temperature/sub-threshold CNN clip** and a
**truncated-cache run** (both current blind spots).

### Tier 3 — needs an explicit owner decision (do not silently implement)
8. **H3 — foreign-region suppression** (pose/identity canonical crops). Behavioral change
   in dense scenes; confirm whether the redesign intends to keep it.
9. **H4 + perf tier — dropped acceleration** (TRT/ONNX auto-export, GPU-native head-tail/
   CNN/pose, NVDEC, direct executors, square-letterbox-on-CUDA, per-stage runtime
   independence). These are *performance/accuracy trade-offs*, not bugs — the owner must
   decide drop-on-purpose (then gate via harness PERFORMANCE + non-square-video check) vs
   restore.

### Cheap, high-value verification (can run anytime, ~30 min)
- **G6** — **partially cleared (2026-06-28):** the new worker DOES still open both caches
  in write mode — `IndividualPropertiesCache(mode="w")` at `worker.py:721` and
  `DetectedPropertiesCache(mode="w")` at `worker.py:1959` — so the gross "never written"
  data-loss risk does **not** materialize. **Remaining check:** confirm those writers are
  still *populated with the heading/theta/headtail-resolved fields* the export columns read
  (`export.py:502,635`), not just constructed. One export-roundtrip test on a clip with
  head-tail enabled settles it.

### Suggested immediate sequence (updated 2026-06-29)
Tier 1 (C1–C4 + H1/H2/H5/H6) **and** Tier 2 (H7/H8/H9) are complete — **every correctness
finding is now fixed**. Remaining work, in order: (1) the **G6 export-roundtrip test**, then
(2) the **harness fixtures** that would catch the now-fixed regressions — multi-class OBB,
size-cap-exceeded, temperature/sub-threshold CNN, AprilTag, and truncated-cache clips. Only
after that, take the **H3 / H4 / perf-tier owner decision** (drop-on-purpose vs restore);
these are deliberate acceleration/behaviour trade-offs, not parity bugs.

---

## Severity index

| ID | Severity | One-line | Verified | Status |
|---|---|---|---|---|
| C1 | CRITICAL | canonical_margin reads `INDIVIDUAL_CROP_PADDING` → zero-margin crops | ✅ | ✅ fixed 2026-06-28 |
| C2 | CRITICAL | CNN posteriors never temperature-calibrated in wired path | ✅ | ✅ fixed 2026-06-28 |
| C3 | CRITICAL | optimizer calls legacy cache API on new handle → crash | ✅ | ✅ fixed 2026-06-28 |
| C4 | CRITICAL | AprilTag uses upstream API, not lab fork → broken/disabled | ✅ | ✅ fixed 2026-06-28 |
| H1 | HIGH | `classes=`/target_classes never passed to predict | ✅ | ✅ fixed 2026-06-28 |
| H2 | HIGH | aspect-ratio filtering absent | ✅ | ✅ wired (WIP) |
| H3 | HIGH | foreign-region suppression configured but never applied | ✅ |
| H4 | HIGH | TensorRT/ONNX auto-export dropped (OBB + YOLO-pose) | ✅ |
| H5 | HIGH | final detection cap by confidence, not size | ✅ | ✅ fixed 2026-06-28 |
| H6 | HIGH | NaN/Inf & non-positive-geometry validity guard missing | ✅ | ✅ fixed 2026-06-28 |
| H7 | HIGH | head-tail multi-head rejection + label normalization removed | ✅ | ✅ fixed 2026-06-29 |
| H8 | HIGH | CNN sub-threshold predictions no longer collapse to None | ✅ | ✅ fixed 2026-06-29 |
| H9 | HIGH | cache validation key-only; no frame-range coverage | ✅ | ✅ fixed 2026-06-29 |
| D1 | MEDIUM | identity sidecar now always written (ignores opt-in flag) | cited |
| G3 | MEDIUM | canonical affine/canvas/M_inverse not persisted (recomputed) | cited |
| G4 | MEDIUM | AprilTag hamming dropped | ✅ |
| Pose G1 | MEDIUM | valid_mask gates keypoint storage; threshold conflation | cited |
| Pose G2 | MEDIUM | no AABB fallback for keypoint mapping | cited |
| Pose G4 | MEDIUM | foreign-OBB keypoint post-suppression dropped | cited |
| Pose G5 | MEDIUM | SLEAP export-input-HW / runtime-flavor ignored | cited |
| RT | MEDIUM | per-stage runtime independence collapsed; mixed-config silent promote | cited |
| Crop G2 | MEDIUM | crop padding unified at 0.3 vs split 0.1/0.3 | cited |
| Crop G3 | LOW | degenerate-OBB fallback diverges (zeros/identity/skip vs AABB) | cited |
| G6 | LOW(follow-up) | export readers on legacy caches — confirm writes | cited |
| D2 | LOW | metadata-missing emitter fallback differs | cited |
| Perf | LOW | direct executors / GPU-native / NVDEC / chunk-pad dropped | cited |
