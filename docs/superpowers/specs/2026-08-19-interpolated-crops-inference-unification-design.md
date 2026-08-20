# Interpolated-Crop Inference Unification — Design Spec

> **Status:** APPROVED, ready for planning.
> **Decided:** 2026-08-19. Revised 2026-08-19 after adversarial spec review (6 factual
> corrections, expected-difference registry, and load-bearing decisions added — see
> "Adversarial review corrections" callouts throughout).

## Goal

Route `core/post/interpolated_crops.py`'s pose/CNN/AprilTag/head-tail inference on
occlusion-fill crops through the same `InferenceRunner`/`Pipeline` code that real
detections use, instead of its own hand-rolled backend loading. In the process: fix the
geometry divergence (interpolated-crop positions ignoring `interpolation_method`), unify
provenance into one explicit convention across all four signal types, fix two confirmed
bugs (CNN silently gated behind pose; the post-pass trigger not checking CNN/AprilTag/
head-tail enablement), and remove dead/orphaned code.

## Motivation (verified on main, via adversarial audit)

- **Hand-rolled duplicate inference path.** `interpolated_crops.py` loads its own
  `pose_backend`/`cnn_backends`, builds its own canonical affine, and batches its own
  pose/CNN calls, entirely bypassing `core/inference/runner.py`. This is the exact
  divergence pattern this codebase has already retired elsewhere (detector retirement,
  bgsub InferenceRunner unification, direct-OBB unification) — see CLAUDE.md's Active
  Refactoring Context.
- **Confirmed bug: CNN inference is gated behind pose being enabled.**
  `interpolated_crops.py:1066-1084` nests the entire crop-extraction call inside
  `if pose_backend is not None:`, and CNN crops are only appended if a pose crop was
  produced. A user with CNN classifiers configured but pose extraction disabled gets zero
  interpolated CNN rows, silently.
- **Confirmed bug: the post-pass trigger doesn't check CNN/AprilTag/head-tail.**
  `session_policy.should_run_interpolated_postpass` (`session_policy.py:56-65`) only ORs
  canonical-image export, pose export, and final-media export — so a user who wants only
  interpolated CNN/tag/head-tail predictions never triggers `run_interpolated_crops` at
  all.
- **Confirmed inconsistency: interpolated-crop geometry is always linear and independently
  re-derived, ignoring `interpolation_method`.** `run_interpolated_crops` runs after
  mechanism (1)'s trajectory interpolation has already filled `X`/`Y`/`Theta` into the CSV
  (possibly via Cubic/Spline + heading-flip correction), but `_process_occluded_run`
  recomputes `cx`/`cy` from scratch, always linearly, with its own bespoke `_interp_angle`
  — ignoring whatever the CSV already contains for those frames. It also has no `max_gap`
  cap, so it synthesizes crops (and runs full inference) for frames where the exported
  CSV's position is legitimately `NaN`. Downstream, `oriented_video.py:653-656` compounds
  this: on a sidecar-lookup miss it drops the frame from the rendered video entirely
  instead of falling back to the CSV's own geometry.
- **Confirmed inconsistency: two incompatible provenance conventions.** Pose/CNN coalesce
  interpolated results into the original columns (recoverable only via `DetectionID`
  absence); AprilTag/head-tail write to separate `Interp*` columns. `rich_export.py`'s
  `count_augmented_pose_rows`/`count_interpolated_cnn_rows` exist to support a
  detected-vs-interpolated summary but have zero callers — the rich-export summary never
  reports a CNN split at all.
- **Confirmed dead code:** `tag_identity.py::build_tag_only_trajectories`/
  `_interpolate_segment_rows` has zero callers anywhere in the codebase, and carries a
  third, incompatible provenance scheme (explicit `Interpolated` bool, no `DetectionID`
  column, `State` hardcoded to `"active"`).

## Key architectural finding (verified during brainstorming)

The real forward/backward tracking pass — the one that produces the CSV
`interpolated_crops.py` post-processes — does **not** go through `run_realtime`. It goes
through `Pipeline`/`run_batch_pass` (`worker.py:1209-1229`), which calls its own **batched**
stage functions: `extract_canonical_crops_batch` + `run_pose_batch` (`stages/pose.py:382`),
`run_cnn_batch` (`stages/cnn.py:129`), `run_headtail_batch` (`stages/headtail.py:245`), and
`run_apriltag` looped per-frame (`pipeline.py:392-398`, no batch variant exists for tags).
`run_realtime`'s singular closures (`_do_ht`/`_do_cnn`/`_do_pose`/`_do_at`,
runner.py:917-968) are a **separate, deliberately-maintained, verified-equivalent (not
byte-identical) implementation** used only for live/realtime-preview tracking, confirmed by
direct call-site tracing (`worker.py:1209-1229` → `run_batch_pass`/`Pipeline` for the real
forward/backward pass; `worker.py:2325` → `run_realtime` only in the separate live-preview
branch). (Correction: the earlier draft cited `run_cnn_batch`'s docstring, `cnn.py:145-162`,
as documenting this split explicitly — it doesn't; that docstring documents bit-identity
with the per-frame `run_cnn` path and a CPU/CUDA identity-agreement gate, not the
realtime-vs-Pipeline split. The split itself is real and confirmed by the call-site tracing
above, just not by that docstring.) This is a pre-existing, accepted split and out of scope
here.

**Consequence:** the correct unification target is the `*_batch` functions `Pipeline`
actually uses, not `run_realtime`. This means the new adapter needs **no extraction from
`runner.py` and no changes to `runner.py`/`Pipeline` at all** — it becomes a new caller of
already-batched, already-verified functions. The stage functions themselves were confirmed
to be pure `(frame, OBBResult, model, config, runtime, geometry) -> Result` functions with
no `DetectionID`/cache coupling and no cross-call mutable state (the one stateful stage,
bg-sub, is structurally unreachable from a synthetic-OBB caller, since the adapter never
calls `run_obb`/`run_bgsub`). Precedent for a non-live caller driving stage functions
directly already exists: `detect_batch` (`runner.py:1158`), used by dataset generation.

## Architecture

```
core/post/synthetic_detections.py          (NEW — small, single-purpose)
    build_synthetic_obb_result(interp_rows) -> OBBResult
        # ellipse_to_obb_corners per row (existing helper, reused)
        # detection_ids via naming.py::synthetic_interpolated_det_id (existing scheme, reused)

core/post/interpolated_crops.py             (MODIFIED)
    - keeps: gap scanning, geometry decision, sidecar CSV/image writing
    - removed: hand-rolled pose_backend/cnn_backends loading, hand-built canonical affine,
      hand-rolled batching (_flush_pose_batch/_flush_cnn_batch internals)
    - added: calls into extract_canonical_crops_batch + run_pose_batch, run_cnn_batch,
      run_headtail_batch, run_apriltag — the SAME functions Pipeline calls for real
      detections, via a synthetic OBBResult from synthetic_detections.py
```

No changes to `core/inference/runner.py`, `Pipeline`, or any cache-write logic. This keeps
the change additive and low-risk: the only thing synthetic about the interpolated path is
the geometry input, not the inference call.

**Model-loading glue (decided, adversarial-review G4).** The `*_batch` stage functions take
already-constructed `PoseModel`/`CNNModel`/`HeadTailModel`/`AprilTagModel` + `InferenceConfig`
as arguments — they don't load models themselves. `_load_all_models` (`runner.py:253`), the
function that does that orchestration (bundle loading, `_warn_geometry_mismatch`,
warmup/close lifecycle, SLEAP service ownership), is private to `runner.py`. Decision:
**`synthetic_detections.py` calls `_load_all_models` directly** (an intra-package private
import from `core/post/` to `core/inference/`, not a new public API) rather than
re-implementing bundle-loading orchestration in the adapter. This is a deliberate exception
to "no changes to runner.py" in spirit only — no lines in `runner.py` change, but the adapter
does reach into it. Lifecycle: `interpolated_crops.py` owns the loaded-model bundle's
lifetime and is responsible for closing it (mirroring the existing
`_cleanup_backends`-on-exit pattern already used for `pose_backend`/`cnn_backends` today),
not `InferenceRunner`, which never sees this bundle.

## Geometry sourcing (decided)

`_process_occluded_run` changes its sourcing priority per occluded frame:

1. If the incoming `final_csv` row already has non-NaN `X`/`Y`/`Theta` — i.e. mechanism
   (1)'s trajectory interpolation already filled it, respecting the user's
   `interpolation_method` and heading-flip correction — **use that value directly.**
2. Otherwise (row is `NaN`), fall back to independent linear position interpolation +
   `_interp_angle`'s ±180° heading disambiguation, exactly as today.

(Correction: the rule is **NaN-triggered**, not "beyond `max_gap`" as the earlier draft said.
`max_gap` is one common reason a row is `NaN` (mechanism (1) refuses to fill gaps longer than
`max_gap`), but it isn't the only one: with `interpolation_method: "None"` — the GUI's
default (`postprocess_panel.py:385`) — mechanism (1) never fills *any* occluded row
regardless of gap length, so priority 2 (today's linear fallback) is what actually runs for
most users out of the box. This is expected and matches current behavior; the fix here is
only that the CSV's own value is respected in the (Cubic/Spline-configured) case where it's
present.)

Size (`w`, `h`) is unaffected — it continues to source from the OBB detection-cache
endpoints as today; this was not found to have a divergence issue.

## Provenance (decided)

Standardize all four signal types on **coalesce into original columns + explicit
`*Source` column**, replacing both prior conventions.

**(Correction: the four merge functions live in `core/individual/properties/export.py`
(`merge_interpolated_pose_df` l.1002, `merge_interpolated_apriltag_df` l.1150,
`merge_interpolated_cnn_df` l.1209, `merge_interpolated_headtail_df` l.1285) — not
`pose_merge.py`, which contains a different function, `merge_pose_sources_into_df`. The
earlier draft misattributed these.)**

**(Correction: the real-detection column names below were wrong in the earlier draft — AprilTag's
real column is `DetectedTagID`, not `TagID`; head-tail has no single `HeadingRad` column, it
has five: `HeadingResolved`, `HeadingMethod`, `HeadingIsDirected`, `HeadTailAngleRad`, and a
classifier-confidence column. The table below uses the real schema.)**

| Signal | Real-detection columns (unchanged) | Interpolated columns retired | New source column |
|---|---|---|---|
| Pose | `PoseKpt_*` | (coalesce, no separate columns today) | `PoseSource` (`'real'`\|`'interp'`) |
| CNN | `CNN_<label>_Class`, `CNN_<label>_Conf` | (coalesce, no separate columns today) | `CNN_<label>_Source` |
| AprilTag | `DetectedTagID`, `DetectedTagLabel`, `DetectedTagConf` | `InterpTagID`, `InterpTagHamming`, `InterpTagConf` | `TagSource` |
| Head-tail | `HeadingResolved`, `HeadingMethod`, `HeadingIsDirected`, `HeadTailAngleRad` | `InterpHeadingRad`, `InterpHeadingConf`, `InterpHeadingDirected` | `HeadingSource` |

For AprilTag and head-tail, an interpolated value now coalesces directly into
`DetectedTagID`/`HeadTailAngleRad` (etc.) instead of writing a separate `Interp*` column, and
`HeadingResolved`/`HeadingMethod`/`HeadingIsDirected` gain a `"headtail_interp"`-style
`HeadingMethod` value (or equivalent) so the existing 4-way `HeadingMethod` vocabulary
(`"headtail"`/`"pose"`/`"velocity"`/`"default"`) can represent an interpolated head-tail
result without a parallel bool. `DetectionID` itself is unaffected — it stays `NaN` for
interpolated rows as today; `*Source` is now the authoritative, explicit signal,
`DetectionID`-absence becomes a derivable (not primary) fact.

**Identity-evidence consequence (final-review addendum, not called out in the original
draft).** Because interpolated AprilTag ids now coalesce into `DetectedTagID`, they are also
read by `identity_postprocess.derive_unique_identity_key_series` as apriltag identity
evidence — feeding the relink identity-veto logic for rows that used to be excluded (when the
interpolated id lived only in the separate, unconsumed `InterpTagID` column). This is accepted
as intentional: it is consistent with the already-shipped CNN coalesce precedent (CNN's
interpolated values already flowed into identity evidence the same way via coalescing into
their original columns), and it achieves the same real-detection-parity goal this refactor
targets. No gating logic was added to distinguish real vs. interpolated tag ids in the identity
path.

**Consumer migration (adversarial-review G5, not enumerated in the earlier draft).** The
`Interp*` columns being retired are read outside the merge functions themselves — the plan
must update every one of these call sites, not just the writers:
- `core/individual/postprocess_df.py:288-290` — `has_apriltag = DetectedTagID.notna() |
  InterpTagID.notna()`; becomes `DetectedTagID.notna()` alone once `InterpTagID` coalesces
  into `DetectedTagID` (the `TagSource` column is now separately available if the
  real-vs-interp distinction is still needed downstream).
- `core/post/rich_export.py:176-189` — reads `InterpTagID`/`InterpHeadingRad` directly to
  compute fill counts; replace with `count_by_source(df, "TagSource")` /
  `count_by_source(df, "HeadingSource")` (see below).
- `tests/test_identity_evidence_vectorized.py`, `tests/test_properties_export.py` — assert
  against the current `Interp*` schema; must be updated to the new columns as part of this
  change, not left to fail.

**CSV schema-compatibility stance (adversarial-review G6).** This is a **breaking change** to
the exported rich CSV's column set: `Interp*` columns disappear, `*Source` columns are new,
and AprilTag/head-tail interpolated values move into the real-detection columns. No
backward-compatible aliasing is kept — the earlier draft didn't state this. Downstream
consumers to check/update: RefineKit's proofreading UI (if it reads `Interp*` for
visualization) and any identity-postprocess code beyond `postprocess_df.py` above. Document
the schema change in the CSV format changelog if one exists; this is not silently absorbed by
a version bump alone.

`rich_export.py`'s dead `count_augmented_pose_rows`/`count_interpolated_cnn_rows` (confirmed
zero callers) are replaced by one generic `count_by_source(df, source_col) -> {real: N,
interp: N}`, used uniformly for all four signal types and wired into `log_rich_export_summary`
so CNN finally gets a detected-vs-interpolated line in the summary, matching pose/AprilTag/
head-tail.

## Bug fixes folded in

1. **CNN/pose decoupling — subsumed by the architecture, not a standalone patch.**
   (Correction: the earlier draft described this as "widen the gate at
   `interpolated_crops.py:1066` to `pose_backend is not None or cnn_backends`" — that alone
   doesn't fix anything, because in the current hand-rolled code the CNN crop *is* the pose
   crop (`_pending_cnn_crops.append(pose_crop)`, l.1082): CNN input is never built
   independently, so widening the `if` still yields zero CNN crops with pose disabled. The
   real fix is that `run_cnn_batch` builds its own classifier crops
   (`extract_classifier_crops_batch_np`, independent of any pose crop) — so this bug is fixed
   *by* the unification itself, not by a preparatory one-line change to the old code.)
2. **Postpass trigger completeness** — `session_policy.py:56-65`: add CNN-classifier /
   AprilTag / head-tail enablement terms to `should_run_interpolated_postpass`'s OR-list,
   using the **same predicates the loader already uses** so this doesn't recreate the bug in
   miniature with a mismatched condition (adversarial-review G7): CNN enablement is
   `bool(params.get("CNN_CLASSIFIERS", []))` (matches `worker.py:809` and
   `config.py:676/949`); AprilTag enablement is `params.get("USE_APRILTAGS", False)`
   (matches `config.py:1074`); head-tail enablement reuses the existing
   `is_headtail_compute_enabled` predicate (`session_policy.py:35-41`), which the OR-list
   omits today despite already existing in the same file.
3. **Oriented-video fallback** — `oriented_video.py:653-656`: on an `interp_lookup` miss,
   fall back to the row's own CSV `X`/`Y`/`Theta` (converted to an OBB via
   `REFERENCE_BODY_SIZE`) instead of dropping the frame from the rendered video.

**AprilTag/foreign-suppression decisions (adversarial-review G8/G9, unaddressed in the
earlier draft).** Today's hand-rolled AprilTag detection foreign-masks other synthetic tasks'
corners via `SUPPRESS_FOREIGN_OBB_REGIONS`/`INDIVIDUAL_BACKGROUND_COLOR`
(`_detect_apriltags_in_frame` l.714-726, using `pose_pipeline.extract_one_crop`), then calls
`detect_in_crops(crops, offsets, det_indices=...)`. `Pipeline`'s real-detection path instead
calls `run_apriltag(aabb_crops, obb, model, cfg.apriltag)` with `extract_aabb_crops`
(`stages/crops.py:110-135`), which has **no foreign-suppression parameter at all** — it only
masks within a synthetic `OBBResult`'s own detections via `extract_canonical_crops_batch`'s
`suppress_foreign` flag (`crops.py:421/450`), used by pose, not by the AABB/AprilTag path.
Decision: the adapter passes `suppress_foreign=True` to `extract_canonical_crops_batch` for
the pose call (masking other *interpolated* tasks in the same frame, matching today's
intra-synthetic-batch masking), but the AprilTag call goes through `extract_aabb_crops`
un-suppressed, matching `Pipeline`'s real-detection behavior — meaning **interpolated
AprilTag crops lose foreign-suppression of other interpolated tasks** relative to today. This
is a deliberate, registered behavior change (see the expected-difference list under
Testing), not an oversight: it makes the interpolated path match what real detections already
get, rather than preserving a suppression behavior that was itself an interpolated-path-only
artifact. Padding source: use `cfg.apriltag.crop_padding` (`Pipeline`'s convention), not the
old `APRILTAG_CROP_PADDING` param — confirm during planning that the two padding values are
configured identically today, since a mismatch would itself be visible in the golden diff.

## Dead code removed

`tag_identity.py::build_tag_only_trajectories` and `_interpolate_segment_rows` are deleted
outright — **zero production callers**, confirmed by grep. (Correction: the earlier draft
said "zero callers anywhere in the codebase," which overlooked that
`tests/test_tag_identity.py:16,159,170,184` imports and directly tests
`build_tag_only_trajectories` (2 test functions), and `build_tag_only_trajectories` is
explicitly whitelisted in `pyproject.toml:255` (`[tool.deadcode] ignore-names`) and
documented as dead in `docs/schematics/trackerkit_pipeline.md`.) The deletion must also:
delete the 2 tests in `test_tag_identity.py`, remove the `pyproject.toml:255` whitelist
entry, and update/remove the `docs/schematics/trackerkit_pipeline.md` reference — otherwise
`make dead-code` and the test suite both break on an already-deleted symbol. If a tag-only
tracking mode is wanted later, it should be designed fresh against the unified `*Source`
provenance scheme, not resurrected from this orphaned implementation.

## Error handling

Unchanged defensive posture, applied at the new call sites — but the mechanism moves
(adversarial-review G2/G3, unaddressed in the earlier draft):

- **Degenerate-OBB pre-filter is load-bearing, not defensive, and must live in the adapter.**
  `extract_canonical_crops`/`_batch` (`stages/crops.py`) does **not** raise or skip on a
  degenerate OBB — it fudges one with an identity affine (`crops.py:97-98`). Today's
  loud-skip-and-tally behavior comes entirely from `interpolated_crops.py`'s own
  `_compute_frame_corners_and_affines` (l.964-995) computing the canonical affine itself and
  checking the `1e-3` epsilon (`canonicalization/geometry.py:106`) *before* calling any
  extractor. The stage layer has **no `ClippingStats` plumbing at all** — `_clipped` is
  discarded at every call site (`crops.py:94,206,393`; `pose.py:318,450`). Decision: the
  adapter (`synthetic_detections.py` / the modified `interpolated_crops.py`) must
  pre-compute each synthetic OBB's canonical affine itself, apply the same `1e-3` epsilon
  check, and drop+tally degenerate rows into `ClippingStats` **before** calling
  `extract_canonical_crops_batch`/`run_pose_batch`/etc. — exactly as it does today, just
  positioned before the `*_batch` call instead of inside a hand-rolled extractor. If this
  pre-filter is skipped, a degenerate synthetic OBB silently produces a garbage
  identity-affine crop instead of today's loud skip — strictly worse than today, not neutral.
- Missing/corrupted detection-cache size lookups at gap endpoints keep the existing
  `REFERENCE_BODY_SIZE`-derived fallback (`_get_detection_size`, unchanged).
- If the CSV genuinely lacks `X`/`Y` for a frame (the fallback-to-independent-interpolation
  branch — see the NaN-triggered rule above, not a `max_gap` threshold), behavior matches
  today's uncapped synthesis exactly.

## Testing

1. **Unit/integration tests** for `synthetic_detections.py`'s `OBBResult` construction
   (corner geometry, negative-ID assignment) and the new `*Source` provenance columns
   (correct value under real/interpolated/mixed rows, coalesce-not-overwrite behavior
   preserved).
2. **Characterization golden, against a pre-registered expected-difference list
   (adversarial-review G1 — the earlier draft's "anything not deliberately changed should be
   bit-identical" bar was unachievable as stated, because the unification changes several
   inference *inputs*, not just plumbing).** Capture `interpolated_pose.csv` /
   `interpolated_cnn_<label>.csv` / `interpolated_tags.csv` / `interpolated_headtail.csv`
   from current `main` on an occlusion-heavy fixture *before* the change. Diff field-by-field
   against the overhaul's output. The following differences are **expected and must be
   pre-registered**, not treated as regressions when the golden diff surfaces them:
   - **CNN crop identity.** Today, CNN classifies the same foreign-masked pose crop used for
     keypoints (`_pending_cnn_crops.append(pose_crop)`, l.1082, masked via
     `_extract_pose_crop`'s masking, l.1122-1133). `run_cnn_batch` builds independent,
     unmasked classifier crops (`extract_classifier_crops_batch_np` takes no suppression
     parameter). On any frame with ≥2 simultaneous interpolated tasks, CNN inputs — and
     therefore `CNN_<label>_Class`/`_Conf` outputs — change. This is a fix toward
     real-detection parity (CNN already sees unmasked crops for real detections), not a
     regression, but the diff will show it.
   - **AprilTag crop masking.** Per the suppress-foreign decision above, interpolated
     AprilTag crops go from foreign-masked to unmasked — tag detections on frames with ≥2
     simultaneous interpolated tasks may change.
   - **Pose crop LSB rounding.** Today's masking rounds to `uint8` then masks
     (`crop.py:144-156`); the batch path truncates, masks, then converts back
     (`_apply_foreign_mask_canonical_batch`, `crops.py:399`) — expect ±1 LSB pixel
     differences (and correspondingly tiny keypoint-coordinate differences) on frames where
     suppression is active, not exact byte-identity.
   - **Head-tail crop construction.** Switches from `HeadTailAnalyzer.analyze_crops`
     (today, l.762) to `run_headtail_batch` — a materially different crop-construction path.
     Verify equivalence empirically (compare outputs on the golden fixture) rather than
     assuming byte-identity; this is a plan-level verification item, not a given.
   Anything **not** on this list — geometry-sourcing priority (deliberate), CNN-gating fix
   (deliberate), new `*Source`/retired `Interp*` columns (deliberate schema change, see
   Provenance) — should be understood as intentional and cross-checked against the design
   decisions above, not just "not flagged as a regression."
3. **Golden-fixture sourcing (adversarial-review G10).** The existing equivalence fixtures
   (`fly_obb`, `worm_bgsub`, etc., `tools/equivalence/fixtures/`) are the fast/smoke set and
   are not documented as occlusion-heavy with CNN+AprilTag+head-tail signals active
   simultaneously. Confirm during planning whether any existing fixture qualifies, or budget
   creating a new occlusion-heavy fixture (or a synthetic occluded-CSV harness that doesn't
   require a full clip) as explicit planned work — this was unbudgeted in the earlier draft.
4. **Standard equivalence harness** (`tools/equivalence/run_matrix.sh`, MPS here + CUDA on
   mehek per `CLAUDE.md`) run on the surrounding pipeline to confirm no collateral effect on
   non-interpolated tracking output.

## Out of scope

- Any change to `run_realtime`, `Pipeline`, or cache I/O.
- Reconciling `run_realtime` vs. `Pipeline`'s batch-function split (pre-existing, verified,
  deliberately accepted — not part of this design).
- Redesigning a tag-only tracking mode (the orphaned code is deleted, not replaced).
