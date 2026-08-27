# Identity Subsystem Repair — Design

**Status:** approved for planning (2026-08-27). Plan: `docs/superpowers/plans/2026-08-27-identity-subsystem-repair.md`.

## 1. Problem (diagnosed, adversarially reviewed)

Three runs on `DEMO/ID` (25 colour-tagged *O. biroi*, frames 9300–10000): identity OFF tracks well (117 final trajectories, median 28 frames); OFFLINE and ONLINE identity produce 689 trajectories, median 7 frames. Root causes, each independently reproduced by a second model:

| # | Finding | Evidence |
|---|---|---|
| R1 | **Train/inference preprocessing drift.** Classifier `20260429-105036_…colortag` was trained (April) with `transforms.Resize((128,128))` — an anisotropic *squash* of wide (~90×250, aspect 2.0–3.2) crops on light-grey substrate. Inference today letterboxes the 148×66 canonical canvas onto a **zero (black) 128×128** (`fit.apply_fit` → `resample.letterbox_fit`), leaving 71/128 rows black. | Rebuilt pipeline input reproduces cached probabilities **bit-exactly** (max diff 0.0). Same weights, same detections: black letterbox → joint max-prob median 0.141, 6.7 distinct labels/frame; squash → **0.946**, 18.6 labels/frame. BGR/RGB, ImageNet norm, MPS numerics, rotation, margin all refuted. |
| R2 | **Nothing detects R1.** Model has no `.canonical_meta.json`; `warn_on_geometry_mismatch` treats "unstamped" as fine. Every pre-Aug-2026 checkpoint silently bypasses the only guard. The head/tail orientation model is from the same April era. | `canonical_meta.read_canonical_meta` → `None` → silent. |
| R3 | **PELT splitter over-segments ~8×.** `detect_identity_changepoints` z-scores each label column *per trajectory* with a `col_std < 1e-8 → 1.0` guard; trajectories with constant posteriors (std ~1e-4) are rescaled to unit variance and PELT splits float noise. | Real smoothed evidence: z-scored pen=3 → 679 changepoints; raw pen=3 → 83; true argmax switches 87. On *good* evidence z-scored still 8× over. A BIC penalty does not fix it (422/318/198). |
| R4 | **PELT cuts are permanent.** Relink (`processing.py:3098`) rejects `gap < 1`; PELT produces `gap == 0`. 607/689 OFFLINE fragments abut another at `end+1`. | `788 → 689` is relink chaining real gaps; no PELT cut is ever undone. |
| R5 | **`min_fragment_frames` deletes rows.** `split_trajectories_at_changepoints` drops post-split remnants shorter than the floor instead of merging them. | `offline.py:269`. Latent data loss. |
| R6 | **`unknown` can never win; `identity_unknown_prior` is dead.** `substrate._factor_log_prob` floors each factor at 1e-6 → joint `unknown` = 1e-12 exactly on every detection. | 15 464/15 464 detections. |
| R7 | **Smoother saturates.** Forward–backward over conditionally-independent emissions turns 0.12 evidence into 0.92 in 10 frames, 0.9999 in 50, so *systematically* wrong evidence becomes confidently wrong. Nothing checks evidence quality before it drives structure. | Measured on traj 0 (700/700 frames same wrong label). |
| R8 | **Classifier crops are not head/tail-oriented** while the catalog is ordered (`pink_yellow` ≠ `yellow_pink`). Orientation comes from YOLO OBB corner order; `headtail.npz` (present in cache) is unused by the CNN stage. Stable on 10/11 tracks; one track at 0.47 ordered agreement. | `stages/cnn.py` → `canonical_affine(corners)`. |

Not defects (verified): offline identity does not touch the tracking pass (OFF/OFFLINE forward+backward CSVs md5-identical); ONLINE positions identical because `identity_weight=0.0` short-circuits at `hungarian.py:391` — but with weight > 0 the garbage evidence would drive rejoin and KF position resets, so R1 is the top priority.

## 2. Goals / non-goals

**Goals**
1. A model is always preprocessed the way it was trained: a `fit_policy` carried by the artifact, honoured by every classifier consumer (identity CNN, head/tail), stamped by training, and loud (warning) when inferred for legacy artifacts.
2. Fragment solver structural decisions are calibrated: changepoints ≈ real label switches; no cut is irreversible; no rows are dropped.
3. Evidence can express ignorance (`unknown` prior live) and a source that is demonstrably uninformative cannot restructure trajectories.
4. Classifier crops are head-first when the head/tail stage is confident, on both train (already true: `dataset/generator.py` resolves head-tail > motion > OBB) and inference.

**Non-goals**: retraining the user's models (they become usable without retrain via `fit_policy=squash`); the ONLINE-vs-OFFLINE 14-fragment partition difference in conflict resolution (follow-up); TensorRT/CoreML export changes; new GUI knobs.

## 3. Design

### 3.1 `fit_policy` (R1, R2)
- `ClassifierMetadata.fit_policy: Literal["letterbox","squash","native"]`.
  - torch checkpoints: read `ckpt["fit_policy"]`; **absent → `"squash"`** with a one-time `WARNING` naming the artifact ("pre-2026-08-05 training used anisotropic Resize; assuming squash — re-publish to stamp"). Rationale: every checkpoint produced before `3a2163ac` was squash-trained; that is the only population that lacks the key.
  - multihead manifests: top-level `"fit_policy"`; absent → same rule.
  - YOLO classifiers → `"native"` (ultralytics applies its own transform; unchanged).
- Layer 2 becomes policy-driven in one shared function: `canonicalization/fit.py: fit_crops_for_model(crops, geometry, metadata) -> list[np.ndarray]` (numpy) and `resample.fit_batch_for_model(crops_chw, model_wh, policy)` (torch, CUDA branch). `squash` = antialiased bilinear resize to `(in_w,in_h)` (torch `F.interpolate(antialias=True)`, matching PIL's antialiased `Resize` far better than cv2 `INTER_LINEAR`); `letterbox` = existing zero-fill letterbox (unchanged, byte-identical).
- Training (`training/runner.py`) writes `fit_policy="letterbox"` into every checkpoint dict and multihead manifest it publishes. `scripts/stamp_fit_policy.py <artifact> --policy {letterbox,squash}` stamps existing artifacts (for models trained after `3a2163ac` but before this change).
- `warn_on_geometry_mismatch` unchanged; the fit-policy warning is the new loud path for unstamped models.

### 3.2 Head-first classifier crops (R8)
- `extract_classifier_crops` / `extract_classifier_crops_batch_np` gain optional `heading_hints: np.ndarray | None`, `directed_mask: np.ndarray | None`. When `directed_mask[i]` and `heading_hints[i]` finite, the crop affine uses `resolve_directed_angle(theta, hint, True)` (head → +x), i.e. `canonical_affine` rotated by π when the OBB axis points away from the head. Undirected detections keep today's behaviour byte-for-byte.
- Pipeline passes the frame's `HeadTailResult` into `run_cnn_batch` (head/tail already runs before CNN). Head/tail stage itself stays undirected (it *produces* direction).

### 3.3 PELT (R3, R4, R5)
- Remove per-column z-scoring; PELT runs on raw known-label probabilities in [0,1]. `CHANGEPOINT_PENALTY` default 3.0 unchanged (calibrated: 83 vs 87 switches on real data).
- Remnants shorter than `MIN_FRAGMENT_FRAMES` are merged into the preceding segment (or following, for a leading remnant) — never dropped.
- After `_iterative_assign`, adjacent fragments sharing `OriginalTrajectoryID` whose final labels are equal (including both unknown) are re-merged under one `TrajectoryID`. The solver owns undoing its own cuts; relink semantics unchanged.

### 3.4 Evidence honesty (R6, R7)
- `substrate.map_cnn_to_catalog(..., unknown_prior: float)`: after the product-over-factors, known mass is scaled to `1 - unknown_prior` and `unknown` gets `unknown_prior` (config `identity_unknown_prior`, default 0.05). Evidence cache `evidence_schema_version` 2 → 3 so stale sidecars are rebuilt.
- Evidence-quality circuit breaker in `run_fragment_solver`: over all frames with evidence, compute `q_conf = frac(max known posterior ≥ 0.5)` and `q_div = mean distinct argmax labels per frame / min(catalog_known, mean detections per frame)`. If `q_conf < 0.10` **or** `q_div < 0.30`: log `ERROR` with both numbers, skip PELT, and write `IdentityFinalSource = NONE` for all rows (labels still annotated as `IdentityFinalSmoothedLabel` for inspection). Thresholds are constants in `offline.py`, not GUI knobs.

### 3.5 Verification
- Unit tests per task (see plan).
- Equivalence matrix (MPS here, CUDA on mehek): non-identity clips byte-identical; identity clips (`ant_cnn_identity`, `emi_obb_identity`) **expected** to diverge in identity columns (and positions if their configs use `identity_weight > 0`) — record the delta explicitly, do not paper over.
- Manual acceptance on `DEMO/ID/OFFLINE`: rerun post-processing; expect final trajectory count within 1.3× of OFF (≤ ~150), ≥ 15 distinct labels/frame in `IdentityFinalSmoothedLabel`, and the visually verified tracks (30 blue_blue, 36 orange_yellow, 58 orange_green/green_orange) labelled correctly.

> **Note (updated 2026-08-27, post-Task-9 fix wave — this section is now known-stale relative to the measured gate; not rewritten in place per docs-lifecycle convention, corrected here instead):**
>
> **Equivalence-divergence scope was wider than stated above.** The measured MPS
> equivalence matrix (see `tools/equivalence/notes/2026-08-27-identity-repair-gate.md`)
> found **5** clips diverge, not 2: the 2 named here (`ant_cnn_identity`,
> `emi_obb_identity`) plus `ant_pose_headtail`, `ant_obb_sleap`, and
> `ant_obb_sequential`. All 5 share one mechanism, root-caused (with an honest
> correlation-not-causation caveat — the exact unstamped checkpoint no longer exists
> to re-run a counterfactual) in the gate record: they all point at the same shared
> head/tail orientation checkpoint
> (`classification/orientation/20260429-104937_efficientnet_b0_obiroi_train1.pth`),
> which Step 1 of the same gating session re-stamped with `fit_policy=squash`. Once
> stamped, Tasks 1-3's fit-policy-aware Layer-2 dispatch computes head/tail
> orientation slightly differently from the `main` baseline on borderline frames,
> producing π-magnitude θ flips that cascade into small forward/final row-count
> deltas. Positions remain byte-identical (p99 = 0.0) on every clip, confirming the
> Kalman/assignment/detection geometry itself is untouched. Full per-clip evidence:
> the gate record above.
>
> **Measured outcome (2026-08-27): the DEMO/ID acceptance thresholds stated above
> were not fully met.** Final trajectory count came in at **171**, not ≤~150 (a real
> 4.0x improvement over OFFLINE's original 689, but short of the plan's specific
> numeric bar); the ≥15 distinct-labels/frame threshold was met (17.09 measured);
> the 30/36/58 track spot-check came in **1/3 correct** (track 58 only — tracks 30
> and 36 both mislabel, converging on the same wrong value `blue_pink`, a possible
> residual identity-confusion pattern rather than scattered noise). This is
> attributed to residual classifier accuracy on the DEMO/ID colortag catalog, not
> to a mechanism within this plan's scope (fit-policy correctness, PELT calibration,
> evidence honesty, crop orientation) — retraining the classifier, or a dedicated
> ground-truth accuracy evaluation, is a follow-up outside this repair. See the gate
> record for the full measured breakdown and reasoning.
