# Combined core/post Slice — Orientation Stability + Postproc Vectorization

**Date:** 2026-08-18
**Status:** Shipped — merged to main (b0871a96).
**Branch:** `perf/postproc-vectorize-and-orient` (worktree `.worktrees/postproc-combined`, from main @8c2288be)

## Context

Post-processing is ~28% of tracker wall-clock (now the largest share after Slice A
sped the crop warp ~7×), dominated by row-wise pandas access
(`.xs`/`_getitem_axis`/`Series.__getitem__`/`fast_xs` — millions of calls). Separately,
Slice A surfaced that the post-processor's trajectory head/tail **orientation resolution
is bistable**: a ~2e-4 crop perturbation leaves forward θ unchanged (~1e-3 rad) but flips
final θ by π on ~12–15% of trajectory-rows.

This slice does BOTH, in `core/post` + `core/individual/pose`. The two parts have opposite
verification stories (one deliberately changes θ; the other must be byte-identical), so
they are sequenced: **Part 1 (orientation stability) first, Part 2 (vectorization) second,
byte-identical to the post-Part-1 output.**

Grounded in two read-only investigations (2026-08-18). See memory
`project-pipeline-perf-slices-bcd`.

## Part 1 — Orientation stability (correctness; done FIRST)

### Root cause (verified)
`_fix_heading_globally` (`src/hydra_suite/core/post/processing.py:4082`, invoked at
`:3428`/`:3463` when `directed_heading_posthoc=True`, i.e. head/tail + pose models) is a
Viterbi DP where each frame is kept or π-flipped (`curr_flip = (curr_orig + np.pi) %
two_pi`, `:4129`) to minimize **pairwise** circular heading variation
(`cost = dp[prev_s] + diff`, `:4140-4143`; `dp` init `[0.0, 0.0]` at `:4118`). The cost is
invariant under a global π-flip of the whole trajectory, so `dp[0] == dp[1]` in exact
arithmetic. The terminal tie-break `states[-1] = 0 if dp[0] <= dp[1] else 1` (`:4155`) is
therefore decided by ~1e-15 float rounding; any ~1e-3 θ perturbation flips the residual's
sign, backtracking inverts the ENTIRE trajectory by π — the observed 12–15%.

### Fix (approved anchor: raw directed-heading agreement)
Keep the DP unchanged (it correctly makes a track internally consistent). Replace ONLY the
tied terminal selection at `:4153-4162` with a deterministic, perturbation-robust absolute
anchor:

- Compute both globally-consistent assignments (backtrack from terminal state 0 and from
  state 1 — they are exact π-complements).
- Choose the assignment minimizing the **summed circular distance to the RAW per-frame
  directed headings** (the head/tail-classifier / pose head calls that are the DP's own
  input, carrying absolute head direction). This is a majority over many frames → robust to
  1e-3 noise, and selects the model's intended head.
- Exact-tie fallback (both sums exactly equal): pick state 0 (lowest index) — deterministic.

Secondary hardening (same PR, minimal):
- `:4144` per-transition strict `<`: make the keep-vs-flip choice deterministic when the two
  costs are within an epsilon (prefer "keep" = state 0) so internal segments don't flip on
  noise either.
- `_merge_angle_mean` (`:3223-3241`) forward/backward disagreement uses a fixed `1e-12`
  bias; widen to a small principled epsilon consistent with the anchor so overlap-row θ is
  stable. (Only if it measurably contributes; gate on a test.)

### Part 1 acceptance
- **Determinism under perturbation** (the whole point): running the full pipeline on a
  head/tail + a pose clip with vs without a sub-pixel crop perturbation (or parent-vs-Slice-A
  crops) yields NO π-flips in final θ — `theta_mean` at the equivalence floor, not ~0.4.
- **Correct head preserved:** on clips with a known-correct orientation (goldens /
  head/tail fixtures), the anchored orientation matches the pre-existing intended head (no
  systematic 180° inversion of the whole dataset).
- Unit tests on `_fix_heading_globally` directly: a synthetic trajectory whose raw headings
  favor one global orientation returns that orientation regardless of a tiny input jitter or
  which terminal state has the lower float residual.
- This part is NOT byte-identical to current main (it changes θ on the rows that previously
  flipped) — that is the intended correctness change; re-baseline goldens after.

## Part 2 — Postproc vectorization (perf; byte-identical to post-Part-1 output)

Replace row-wise pandas loops with column-wise/numpy/groupby equivalents. **Capture a
committed value-level characterization golden of the rich `_with_individual.csv` FIRST**
(Explorer-identified gap: `tests/test_user_mode_golden.py` asserts only column-set, not
values). All four hotspots (approved scope):

| # | Site | Computes | Vectorized approach | Byte-identical risk |
|---|---|---|---|---|
| 3 | `core/individual/postprocess_df.py:153-156` `.apply(axis=1)`×3 | per-row identity evidence sources / conflict flag / top CNN-or-tag | boolean-column combine for sources+conflict; `idxmax(axis=1)` over CNN conf cols for top-evidence | LOW-MED (sorted-set join order, NaN handling) |
| 4 | `core/individual/pose/quality.py:381,494` calibration `iterrows()` | body-length + per-edge median/MAD priors (high-conf subset) | vectorized `(M,K,3)` keypoint stack → same sample lists → `np.median`/MAD | LOW |
| 1 | `core/individual/pose/quality.py:729-774` `apply_quality_to_dataframe` row loop | per-row keypoint cleaning, valid-fraction, score/state, conf+5 meta writeback | stack `(N,K,3)` from triplet cols; masks/fraction/score/state column-wise; block-write | MED (float32 round-trip; exact flag-string token order + `|`-join) |
| 2 | `core/post/pose_merge.py:333` + `quality.py:784-916,1030-1051` temporal postproc | per-traj rolling-z outlier suppression, short-gap interpolation, summary recompute | `groupby.rolling` for z; boolean-mask conf-zeroing; vectorized linear gap fill; `df[conf_cols].mean/(>0).sum` for summary | MED-HIGH (rolling ddof=1/min_periods=3/center; `1e-6` eps; `t=step/(gap+1)`; literal `0.3` fill; suppression→interp→recompute + X-before-Y order) |

Order: #3, #4 (isolated) → #1 → #2 (#1→#2 sequential: #2 consumes #1's cleaned conf +
`PoseQualityState`). Do not touch dtypes/formatting at the write boundary: float32↔float
conf round-trips, `Int64` NaN-aware rounding, flag/source strings all print into the CSVs.

### Output contract (must stay byte-identical in Part 2)
- `_tracking_final.csv`: bare `relinked_base.to_csv` (`rich_export.py:420`, deliberately not
  via `write_base_final_csv` to avoid Int64 rounding); row order
  `sort_values(["TrajectoryID","FrameID"], kind="stable")`.
- `_with_individual.csv` (Debug rich) and `_tracks.csv` (User) via `trajectory_writer.py`
  (`heading_deg = mod(degrees(theta),360)`, `frame` Int64-rounded, gated identity block,
  pose triples).

### Part 2 acceptance
- Value-level characterization golden of the rich CSV: byte-identical before/after each
  hotspot.
- `tests/test_pose_quality.py` (~70 unit tests), `tests/test_postproc_equivalence.py`,
  `tests/test_identity_postprocess.py`, `tests/test_core_rich_export.py` all green.
- Equivalence harness MPS + CUDA byte-identical vs the post-Part-1 baseline (positions AND θ
  at floor — Part 1 already stabilized θ, so no π-flip excuse remains here).
- Re-profile after: confirm the ~28% postproc share dropped.

## Components / files
- `core/post/processing.py` — Part 1 (`_fix_heading_globally` terminal anchor; `:4144`,
  `_merge_angle_mean` hardening).
- `core/individual/pose/quality.py` — Part 2 hotspots #1, #2, #4.
- `core/post/pose_merge.py` — Part 2 hotspot #2 driver.
- `core/individual/postprocess_df.py` — Part 2 hotspot #3.
- `tests/` — new Part 1 unit tests; new committed value-level rich-CSV golden; extend
  `test_user_mode_golden.py` to assert values.

## Sequencing / isolation summary
1. Part 1 orientation anchor (+ unit tests) → re-baseline any θ-dependent goldens →
   equivalence: θ now stable under perturbation.
2. Capture value-level rich-CSV golden (post-Part-1).
3. Part 2 hotspots #3, #4, #1, #2 — each byte-identical to that golden, committed
   separately, each re-running the unit suites; equivalence MPS+CUDA at the end.

## Risks
- Part 1 could systematically invert orientation if the raw-heading anchor is computed with
  a sign error — guard with the "correct head preserved" golden/fixture check, not just
  determinism.
- Part 2 #1/#2 byte-identity hinges on reproducing float32 round-trips, rolling `std`
  ddof/window semantics, flag-string token order, and interpolation constants exactly — the
  value-level golden + unit tests are the guard; the post-collapse oracle is otherwise
  tautological (memory `project-shared-engine-param-builder`).
