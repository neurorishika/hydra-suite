# Interpolated-Crop Inference Unification — Design Spec

> **Status:** APPROVED, ready for planning.
> **Decided:** 2026-08-19.

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
byte-identical) implementation** used only for live/realtime-preview tracking —
`run_cnn_batch`'s docstring (`cnn.py:145-162`) documents this explicitly. This is a
pre-existing, accepted split and out of scope here.

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

## Geometry sourcing (decided)

`_process_occluded_run` changes its sourcing priority per occluded frame:

1. If the incoming `final_csv` row already has non-NaN `X`/`Y`/`Theta` — i.e. mechanism
   (1)'s trajectory interpolation already filled it, respecting the user's
   `interpolation_method` and heading-flip correction — **use that value directly.**
2. Only fall back to independent linear position interpolation + `_interp_angle`'s ±180°
   heading disambiguation for frames beyond `max_gap`, where mechanism (1) legitimately
   left `NaN`.

Size (`w`, `h`) is unaffected — it continues to source from the OBB detection-cache
endpoints as today; this was not found to have a divergence issue.

## Provenance (decided)

Standardize all four signal types on **coalesce into original columns + explicit
`*Source` column**, replacing both prior conventions:

| Signal | Real-detection columns (unchanged) | New source column |
|---|---|---|
| Pose | `PoseKpt_*` | `PoseSource` (`'real'`\|`'interp'`) |
| CNN | `CNN_<label>_Class`, `CNN_<label>_Conf` | `CNN_<label>_Source` |
| AprilTag | `TagID` | `TagSource` |
| Head-tail | `HeadingRad` | `HeadingSource` |

Each merge function in `pose_merge.py` (`merge_interpolated_pose_df`,
`merge_interpolated_cnn_df`, `merge_interpolated_apriltag_df`,
`merge_interpolated_headtail_df`) sets its `*Source` column explicitly instead of relying
on implicit `DetectionID` absence or a separate `Interp*` column. `DetectionID` itself is
unaffected — it stays `NaN` for interpolated rows as today; `*Source` is now the
authoritative, explicit signal, `DetectionID`-absence becomes a derivable (not primary)
fact.

`rich_export.py`'s dead `count_augmented_pose_rows`/`count_interpolated_cnn_rows` are
replaced by one generic `count_by_source(df, source_col) -> {real: N, interp: N}`, used
uniformly for all four signal types and wired into `log_rich_export_summary` so CNN
finally gets a detected-vs-interpolated line in the summary, matching pose/AprilTag/
head-tail.

## Bug fixes folded in

1. **CNN/pose decoupling** — `interpolated_crops.py:1066`: gate crop extraction on
   `pose_backend is not None or cnn_backends`, not `pose_backend is not None` alone.
2. **Postpass trigger completeness** — `session_policy.py:56-65`: add CNN-classifier /
   AprilTag / head-tail enablement terms to `should_run_interpolated_postpass`'s OR-list.
3. **Oriented-video fallback** — `oriented_video.py:653-656`: on an `interp_lookup` miss,
   fall back to the row's own CSV `X`/`Y`/`Theta` (converted to an OBB via
   `REFERENCE_BODY_SIZE`) instead of dropping the frame from the rendered video.

## Dead code removed

`tag_identity.py::build_tag_only_trajectories` and `_interpolate_segment_rows` are deleted
outright (zero callers, confirmed by grep). If a tag-only tracking mode is wanted later, it
should be designed fresh against the unified `*Source` provenance scheme, not resurrected
from this orphaned implementation.

## Error handling

Unchanged defensive posture, applied at the new call sites:

- A degenerate synthetic OBB (zero/near-zero edge, per the existing `1e-3` epsilon check in
  `canonicalization/geometry.py:106`) is dropped from the batch before it reaches
  `extract_canonical_crops_batch`, not fudged — tallied in `ClippingStats` exactly as
  today, just recorded before the `*_batch` call instead of inside a hand-rolled extractor.
- Missing/corrupted detection-cache size lookups at gap endpoints keep the existing
  `REFERENCE_BODY_SIZE`-derived fallback (`_get_detection_size`, unchanged).
- If the CSV genuinely lacks `X`/`Y` for a frame beyond `max_gap` (the fallback-to-
  independent-interpolation branch), behavior matches today's uncapped synthesis exactly.

## Testing

1. **Unit/integration tests** for `synthetic_detections.py`'s `OBBResult` construction
   (corner geometry, negative-ID assignment) and the new `*Source` provenance columns
   (correct value under real/interpolated/mixed rows, coalesce-not-overwrite behavior
   preserved).
2. **Characterization golden.** Capture `interpolated_pose.csv` /
   `interpolated_cnn_<label>.csv` / `interpolated_tags.csv` / `interpolated_headtail.csv`
   from current `main` on an occlusion-heavy fixture *before* the change. Diff field-by-field
   against the overhaul's output to separate intentional changes (new geometry-sourcing
   priority, CNN-gating fix, provenance columns) from accidental divergence — anything not
   deliberately changed by this design should be bit-identical.
3. **Standard equivalence harness** (`tools/equivalence/run_matrix.sh`, MPS here + CUDA on
   mehek per `CLAUDE.md`) run on the surrounding pipeline to confirm no collateral effect on
   non-interpolated tracking output.

## Out of scope

- Any change to `run_realtime`, `Pipeline`, or cache I/O.
- Reconciling `run_realtime` vs. `Pipeline`'s batch-function split (pre-existing, verified,
  deliberately accepted — not part of this design).
- Redesigning a tag-only tracking mode (the orphaned code is deleted, not replaced).
