# Identity Final-Output Consistency — Design

**Status:** Shipped — merged to main (14 task commits + 5-commit final fix wave,
scoped re-review ALL_FINDINGS_ADDRESSED, 2026-08-27). Follows
`2026-08-27-identity-subsystem-repair-design.md` (merged `f2d4ca36`). Plan:
`docs/superpowers/plans/done/2026-08-27-identity-final-consistency.md`.

## 1. Problem (audited on `DEMO/ID/ONLINE`, run 2026-08-27 15:09)

Run: 25 colour-tagged *O. biroi*, frames 9300–10000 @10 fps, online identity on
(`identity_weight=0.0`), fragment solver on, PELT on, relink on. Output
`ant_tracking_final_with_individual.csv`: 15 849 rows, 171 trajectories. The user
reports empty X/Y, empty identities, tracks carrying several identities, and
labels flickering in the video. Every one of those is real and has a mechanical
cause. Numbers below were computed from the CSV and independently recomputed by
the adversarial reviewer; the solver mechanisms were reproduced by re-running
`run_fragment_solver` offline on the run's own evidence cache and params (the
re-run reproduces the *phenomena* — dead base assignment, long tracks unknown,
labels against evidence — not the byte-exact shipped output, because the shipped
solve ran on the 129-trajectory pre-relink frame).

| # | Symptom in the CSV | Count | Root cause (verified) |
|---|---|---|---|
| S1 | `X/Y/Theta` NaN | 222 rows, all `State=occluded`, in runs of 6–11 frames; 221 mid-track + one **leading** 7-frame run (t85) | `interpolate_trajectories` caps at `interpolation_max_gap_seconds`=0.5 s → 5 frames, but the tracker coasts occluded slots up to `max_occlusion_gap_seconds`=1.0 s → 10 frames (+1). Runs of 1–5 are filled (258 rows), 6–11 are left NaN. The interpolated-crops pass then *does* linearly fill those same rows to cut crops (`interpolated_crops.py:322-346`, "NaN-triggered, not max_gap-triggered"), so **all 222** carry a `CNN_colortag_*_Class` (`CNN_colortag_Source="interp"`) and a `HeadTailAngleRad` while having no position — evidence with no geometry. A leading run can never be interpolated. |
| S2 | Frames missing entirely inside tracks | 15 gaps / 36 rows (sizes 1–10) | Relink joins fragments across a gap of 1–`MAX_OCCLUSION_GAP` frames (`_score_relink_candidate` gap gate) and `_assemble_relinked_chains` concatenates; nothing re-indexes or fills. Stitching (`STITCH_MAX_GAP_FRAMES`) and overlap-merge can create gaps the same way. Relink also rewrites the base `_final.csv`, so this is not rich-CSV-only. |
| S3 | `IdentityFinalSource` empty | 5 683 rows = exactly the `unknown` rows | `IdentityFinalSource.NONE = ""` (`columns.py:60`) — "no source" is serialised as an empty cell, indistinguishable from "never written". |
| S4 | `IdentityFinalSmoothedLabel` empty, confidence 0.0 | 1 776 rows = 1 296 active (all with a DetectionID) + 480 occluded (none with a DetectionID) | **Two causes.** (a) 1 296: `_annotate_smoothed_labels` blanks the label and zeroes the confidence whenever the smoothed max-posterior < `IDENTITY_DISPLAY_THRESHOLD` (config `identity_display_threshold`=0.95) — a realtime *display* knob applied to a *record* column, destroying the confidence that exists to gate it. (b) 480: rows with no `DetectionID` never join the evidence cache (`load_trajectory_evidence` joins on `(FrameID, DetectionID)`), so they have no smoothed posterior at all, although the crop pass wrote a per-row classifier output on them. |
| S5 | One track, several identities | 8 tracks (t39, t52, t57, t60, t76, t77, t82, t140), all with >1 `OriginalTrajectoryID`; e.g. t60 `green_orange`(87)→`blue_orange`(17) | Post-solver relink (`relink_and_export_rich_csv`: solver → relink → write) gates on the *dominant raw per-row CNN key* (`_fragment_unique_identity_sources` on `UniqueIdentityKey`, `identity_sources_conflict`), never on `IdentityFinalLabel`, and `_assemble_relinked_chains` concatenates rows without reconciling identity columns. |
| S6 | Long, evidence-consistent tracks are `unknown`; the label sits on short fragments | 21 fully-unknown tracks = 5 436 rows, incl. **t111 (701 fr, `orange_yellow` mean-posterior 0.99999, stability 1.0)**, t110 (701 fr), t115 (589 fr, `blue_blue` 1.0); `blue_blue` is held by t0–t4 (69/15/26/1/15 fr), `orange_yellow` by t88 whose own evidence is `pink_yellow` at 0.99999. In the offline re-run 9 of the 14 fragments ≥300 frames are unknown. | Three compounding solver defects, §1.1. |
| S7 | Labelled tracks that contradict their own evidence | 33 of 150 labelled tracks have a label ≠ the mode of their own smoothed label (a further 44 have no smoothed label to compare); 67/185 fragments in the re-run contradict their `MeanCNNProbs` argmax — e.g. a 452-frame fragment with `blue_green` at 0.99998 assigned `green_green` | Same as S6: with support flattened, top-3 candidates are near-ties and spatial term + length factor pick among them. |
| S8 | Labels switch on/off in the video; CSV disagrees with the video | realtime label present on a median 1.7 % of a track's rows; 41 on/off runs on t110; `IdentityRealtimeLabel == IdentityFinalLabel` on 28.8 % of rows where both exist | `ant_tracking.mp4` is rendered post-hoc by `media_export.render_annotated_video` (log 15:13:03, `session._run_annotated_video`) from the **rich CSV**, and `build_video_track_label_array` / `build_video_track_color_key_array` (`media_export.py:135-186`) prefer **`UniqueIdentityKey`** (raw per-frame classifier evidence) over `IdentityFinalLabel`. The flicker is the classifier's per-frame output, not the solver's decision. (The live in-GUI overlay separately draws the realtime decoder's commits, `worker.py:4035`; not what the user watched.) |
| S9 | Solver runs twice per run (identical output, 42.8 s wasted) | log 15:12:18 and 15:13:00 | `run_post_tracking` → `_export_rich` (solver #1) → `_run_interp_crops` → `_relink_export_rich` (`build_rich_export_dataframe` = solver #2, then relink). Pass 1's rich CSV is overwritten by pass 2. |
| S10 | `IdentityFinalConflictResolved` empty | 12 113 NaN / 3 736 True / 0 False | Single writer `processing.py:1642` (merge-time *realtime*-label arbitration, despite the `IdentityFinal*` name) writes only `True`; nothing initialises `False`. |

### 1.1 Why the solver assigns labels to the wrong fragments (S6/S7)

`_iterative_assign` (`offline.py:481`) in the shipped configuration
(`FRAGMENT_CNN_WEIGHT=0.1`, `FRAGMENT_TAG_WEIGHT=0`, `ONLINE_PRIOR_WEIGHT=0`,
`ASSIGNMENT_MARGIN_THRESHOLD=0`, `IDENTITY_DISPLAY_THRESHOLD=0.95`, 25 labels):

1. **`FRAGMENT_CNN_WEIGHT` acts as a softmax temperature, not a weight.** Support =
   `normalize(exp(cnn_w · mean log p))`. A track with mean-posterior 0.99999 on one
   label (log p ≈ 0 vs ≈ −12 on the rest) becomes `exp(−1.2)=0.30` per wrong label,
   i.e. a median top support of **0.21** (max 0.40) with 25 labels. The solver
   never sees the evidence it was given. The default 0.4 gives 0.95 median top
   support but only 0.45 after the length factor. With defaults, an *uninformative*
   online prior (all-zeros dict from `_build_prior_log_scores` when the online label
   is not in the catalog) still counts in the denominator, so even a convex
   combination would leave a 0.615 temperature unless "present" means "informative".
2. **The base assignment is dead.** `_base_assignment_via_substrate` builds
   temporal-overlap connected components *transitively*; on a 25-animal clip every
   fragment overlaps something, so the component is **all fragments** (214/214 in the
   re-run), competing for 25 labels as if simultaneously visible (hard cap: ≤25 of
   214 could ever be assigned). It then gates each slot at `display_threshold=0.95`
   on `support × length_factor`. Measured: **0/214 assigned** at the shipped weight
   (21/214 at cnn_w=0.4, threshold 0.5). Everything falls through with `current=None`.
3. **The rescue is greedy in the wrong order and the swap move is effectively
   dead.** Passes walk fragments by descending doubt = `(1−stability)(1−length_scale)
   + 0.5·[unknown]` — **short, unstable fragments first** (a 701-frame stable track
   scores 0.5, a 3-frame one ≈1.4). Each takes the best non-colliding label among its
   top-3. A long track then finds its label held by several short fragments across its
   span: `_find_blocker` returns only the first, and the swap additionally requires
   `alt_score_j ≥ cur_score_j` for the displaced blocker (`offline.py:800-802`) — a
   displaced fragment is almost always worse off under its alternative, so the swap is
   essentially never taken. The final "Unknown-rescue" pass iterates in plain index
   order and cannot displace anything. Net effect: labels go to the fragments with the
   *least* evidence mass, and long tracks go `unknown`.

Independent of the solver, the classifier evidence itself is still confused on
this catalog (23 of 25 labels appear as a track argmax; `blue_orange` is the
argmax of 2 380 track-frames ≈ 3.4 animals' worth, `pink_green` of 15). That is the
residual-accuracy follow-up from the previous spec and is **out of scope** here;
this design makes the solver faithful to whatever evidence it is given and makes
the output honest about what it does not know.

Not defects: no two tracks share a known label at the same frame (0 collisions
in 10 166 labelled rows); positions are untouched by identity (`identity_weight=0`).

### 1.2 Adversarial-review corrections folded in
Leading NaN run (S1); S4 has two causes; S7 counts; S8 root cause is
`media_export`'s column preference, not the live overlay; reorder in §3.1 must
keep the mirror/consensus/sort chain *after* the solve or AprilTag/realtime-sourced
Final labels get overwritten and identity-adjacent ID numbering is lost;
`NONE="none"` breaks `postprocess_df.py:146-151`'s `fillna("") == NONE` gate;
`IdentityFinalConflictResolved` is written `True` at merge time so a blanket
`False` init would clobber it; all nine equivalence fixtures have
`enable_tracklet_relinking=false`; the final-pass interpolation must never fill
*less* than the user's knob; the multi-blocker move needs a real termination
argument.

## 2. Goals / non-goals

**Goals**
1. **One identity per trajectory in the written CSV, by construction**: a
   post-write-time invariant, checked and logged, not a hope.
2. **Every cell is a value or an explicit denial**: no NaN position inside a
   trajectory, no position-less leading/trailing rows; `IdentityFinalSource="none"`
   (never empty) for unresolved rows; `IdentityFinalSmoothedLabel` always carries
   the argmax with its real confidence where a smoothed posterior exists and an
   explicit `unknown`/0.0 where none does; `IdentityFinalConflictResolved` is
   `True`/`False`.
3. **The solver honours evidence mass**: a long, consistent track beats short
   fragments for its label; a label is only ever given to a fragment whose own
   evidence supports it above a stated floor; otherwise the fragment is `unknown`
   with source `none`.
4. **What the user watches matches what the CSV says**: rendered video and
   exported media label and colour by `IdentityFinalLabel`.

**Non-goals**: classifier retraining / accuracy; the realtime online decoder's
commit policy (it does not affect tracking with `identity_weight=0`); PELT
tuning; GUI knobs; the relink duplicate-FrameID drop (`processing.py:4083`, logged,
left as is).

## 3. Design

### 3.1 Pipeline order: relink first, resolve identity once (S5, S9)
`apply_identity_postprocessing_to_df` is split into two pure functions:
- `derive_identity_keys(df, params)` — the `IdentityEvidence*` summary and
  `UniqueIdentityKey` derivation only (today's tail of the function). Relink needs
  these and nothing else.
- `resolve_identity(df, params, identity_evidence_cache_path)` — the fragment
  solver, `_stamp_non_identifying_labels`, `_mirror_realtime_and_tag_into_final`,
  `fill_identity_nans_with_consensus`, `sort_trajectories_by_identity`, then the
  summary/key derivation again — **in today's order**, so rows with
  `IdentityFinalSource ∈ {tag, realtime, nonidentifying}` are filled exactly as
  today and never overwritten by the solver, and identity-adjacent ID numbering is
  produced after the final structure is known.

`build_rich_export_dataframe(..., resolve=True)` calls `derive_identity_keys` and,
when `resolve`, `resolve_identity`. `relink_and_export_rich_csv` builds with
`resolve=False`, relinks, densifies chains (§3.4), calls `resolve_identity`, runs
the §3.2 check, writes. `export_rich_csv` (pass 1) passes
`resolve = not postpass_will_run` (`session.py` computes it from
`should_run_interpolated_postpass`), so every run gets exactly one solve. PELT
may still cut a chain the relinker joined across an identity switch; with PELT
off the solver labels whole chains (as today for un-split trajectories).

### 3.2 Written-CSV invariant (S5)
`assert_one_identity_per_trajectory(df) -> list[int]` (in `identity_postprocess.py`)
returns trajectory ids whose `IdentityFinalLabel`/`IdentityFinalID`/`IdentityFinalSource`
are not constant. `relink_and_export_rich_csv` and `export_rich_csv` run it on the
frame they are about to write (after the whole `resolve_identity` chain); offenders
are logged at ERROR and collapsed to their majority label/id/source with confidence
set to the row minimum — never silently. A test feeds a hand-built two-label chain.

### 3.3 Solver: evidence-faithful assignment (S6, S7)
- **Weights are convex over *informative* sources.** `combined_log = Σ_s w_s·log_s /
  Σ_s w_s` over sources present *and informative* for the fragment: `cnn` when the
  fragment has cache evidence; `tag` when `TagLogEvidence` is non-empty; `prior` when
  the online label is in the catalog with finite confidence. With one informative
  source the support is that source's geometric-mean posterior itself. The
  `FRAGMENT_*_WEIGHT`/`ONLINE_PRIOR_WEIGHT` knobs keep their meaning as relative source
  weights; their absolute scale no longer sharpens or flattens evidence.
- **Support floor.** New `FRAGMENT_MIN_SUPPORT` (engine param from config
  `fragment_min_support`, default **0.5**): a label whose normalised support is below
  the floor is not a candidate for that fragment. A fragment with no candidate above
  the floor is `unknown`/`none`. Documented as an absolute posterior floor (on a
  2-label catalog it is satisfied by any argmax; on 25 labels it is a real gate —
  that is the intent: "more likely than all alternatives combined").
- **Mass-first seeding replaces the dead component-Hungarian.** Initial assignment
  visits fragments in descending *evidence mass* = `duration × top_support`; each
  gets its top candidate above the floor if no time collision with an already-seeded
  same-label fragment and the spatial veto passes. `_base_assignment_via_substrate`
  is deleted from this path (`substrate.solve_unique_assignment` remains the
  realtime substrate's solver); `IDENTITY_DISPLAY_THRESHOLD` is no longer read by
  the offline solver.
- **Multi-blocker displacement with an exact, monotone objective.** Global objective
  `J = Σ_i score_i(current_i)` (score 0 for unknown). When label `c` for fragment `i`
  collides with blockers `B` (all overlapping same-label fragments, capped at
  `FRAGMENT_MAX_BLOCKERS=4`, else the move is skipped), tentatively apply: `i←c`,
  each `b∈B` ← its best non-colliding alternative above the floor or `unknown`;
  recompute `J` on the actual post-move schedule (spatial terms re-evaluated for
  every fragment under `c` and under each displaced blocker's new label); accept iff
  `J_after − J_before ≥ ε` with `ε = max(ASSIGNMENT_MARGIN_THRESHOLD, 1e-3)`, else
  revert. `J` is bounded (each score ≤ 1) and every accepted move raises it by ≥ ε, so
  the passes terminate; `FRAGMENT_MAX_PASSES` stays as a belt. With `|B|=1` this is a
  strict generalisation of today's move (today additionally required the displaced
  fragment to be no worse off, which is exactly the guard that made the move dead).
- **Refinement order** stays doubt-descending (it is a refinement now, not the
  seeding); the final unknown-rescue pass visits unknowns by descending mass and may
  use the displacement move.
- **Regression fixtures**: (a) synthetic 3-animal case — one 701-frame fragment with
  0.999 evidence overlaps four 10-frame fragments with 0.6 evidence on the same
  label: the long one must win, the short ones go to their own labels or unknown;
  (b) a fragment whose top support is 0.3 stays unknown; (c) the DEMO/ID numbers as
  a gate-record acceptance (§4).

### 3.4 Explicit values (S1, S2, S3, S4, S10)
- **Positions.** Final-pass interpolation uses
  `max_gap = max(round(interpolation_max_gap_seconds × FPS), MAX_OCCLUSION_GAP + 1)`
  — never less than today's fill. Applied in `_interpolate_and_scale` /
  `merge_trajectories` (final frame) *and* after relink, where chains are first
  re-indexed to a dense frame range (missing frames become `State="occluded"` rows
  with empty `DetectionID`, `DetectionConfidence=0.0`), so S2's rows exist and are
  filled. **Leading/trailing occluded runs are dropped** (`trim_positionless_ends`):
  they carry no position and no detection and cannot be interpolated. Interior gaps
  longer than `max_gap` (possible via stitching/merge) are still filled — a gap the
  tracker asserted continuity across is filled or the row is dropped, never left NaN;
  the gap length is logged. Filled rows keep `State="occluded"` so a consumer can
  exclude them.
- **`IdentityFinalSource.NONE = "none"`.** In the same commit: `postprocess_df.py`
  and every `fillna("")` on the source column become `fillna(NONE)`; readers of
  existing CSVs (`identity_postprocess`, `trajectory_writer`) normalise `""`/NaN to
  `none`. `tests/identity/test_identity_columns.py` pins the vocabulary.
- **Smoothed columns are a record, not a display.** `_annotate_smoothed_labels`
  writes the argmax known label and its posterior for every row that has cache
  evidence; `IDENTITY_DISPLAY_THRESHOLD` no longer touches them. Rows with no cache
  evidence (no `DetectionID` — the crop-pass rows) get `IdentityFinalSmoothedLabel=
  "unknown"`, confidence `0.0`: the column is *defined* as the cache-evidence
  forward-backward posterior, and those rows have none; their own per-row
  `CNN_*_Class/_Conf` (`CNN_*_Source="interp"`) stay as the per-row record, and the
  Final family (per-fragment) covers them.
- **`IdentityFinalConflictResolved`**: `_ensure_final_columns` creates it as `False`
  only when absent; the writer (`write_final_trajectories`) fills NaN → `False`, so
  merge-time `True`s are preserved and every row is boolean.

### 3.5 Media (S8)
`build_video_track_label_array` / `build_video_track_color_key_array` prefer
`IdentityFinalLabel`, then `IdentityFinalSmoothedLabel`, then `UniqueIdentityKey`
(update `test_overlay_priority_is_unique_key_then_final_then_final_smoothed`). No
user-guide claim about the live overlay.

### 3.6 Gate coverage (review O1)
Add one relink-enabled equivalence fixture config
(`tools/equivalence/fixtures/configs/ant_cnn_identity_relink.json`, a copy of
`ant_cnn_identity.json` with `enable_tracklet_relinking=true`) so §3.1/3.2/3.4's
relink path has matrix coverage; its baseline row is recorded on first run.

## 4. Verification
- Unit tests per task (plan).
- Equivalence matrix (MPS here, CUDA on mehek), current vs `main` @ `f2d4ca36`:
  `fly_obb`/`worm_bgsub` byte-identical (no postpass, no gaps > their max_gap);
  ant/emi clips: detection-row positions byte-identical, **expected divergence** =
  additional interpolated rows for occluded runs 6–11 frames and dropped
  leading/trailing occluded rows (§3.4), plus identity columns on the identity
  clips; each recorded per clip in the gate note, not papered over.
- `DEMO/ID/ONLINE` acceptance, rerun of post-processing on the same caches:
  0 NaN `X/Y/Theta`; 0 leading/trailing occluded rows; 0 missing frames inside a
  trajectory; 0 rows with empty `IdentityFinalSource`; 0 NaN
  `IdentityFinalConflictResolved`; 0 trajectories with >1 `IdentityFinalLabel`;
  every labelled trajectory's label equals the argmax of its own mean smoothed
  posterior; t111/t110/t115 labelled; labelled/unknown track counts reported with
  the honest residual (classifier confusion) called out; rendered video labels ==
  CSV `IdentityFinalLabel`.
