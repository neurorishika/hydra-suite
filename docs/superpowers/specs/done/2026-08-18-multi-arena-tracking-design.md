# Multi-Arena Tracking — independent trackers, one shared inference pass

**Date:** 2026-08-18
**Status:** Shipped — merged to main (see the multi-arena merge commit)
**Branch:** `feat/multi-arena-tracking` (worktree from local HEAD)

## Context

Users record several physically separate arenas in one camera view — dishes,
boxes, or well-plate grids — and want each arena tracked **completely
independently**: its own track slots, its own assignment, its own identity
decoding, its own tracklet matching. What they explicitly do *not* want is to
crop each region into its own video and run the pipeline N times: the detector
should see the frame once, and one run should produce all arenas' trajectories.

Target scale, from the requesting users: **10–100 arenas per video**, animals
permanently confined to one arena (no crossing), often a single animal per
arena, and **identical parameters across all arenas**. Identity catalogs
**repeat per arena** — arena 1 and arena 2 may both contain an "ant A" — so
identity labels are not globally unique.

### What the codebase already provides

The pipeline is unusually well-shaped for this, because arena-shaped structure
already exists in three places:

| Existing mechanism | Where | Why it matters |
|---|---|---|
| ROI as a **list of shapes** with include/exclude modes, rasterized to one binary mask | `trackerkit/engine_params.py:193` `build_roi_mask` | An arena is a shape plus a label. No new geometry concept is needed. |
| Inference is **arena-agnostic** — the ROI mask is purely a gate/tile filter, detections come out as a flat per-frame list | `core/inference/pipeline.py:113`, `core/inference/stages/filtering.py` | A single full-frame inference pass already serves all arenas. Nothing to change to get detection sharing. |
| Tracking state is **slot-indexed**, not a global graph: `KalmanFilterManager(p["MAX_TARGETS"], p)` plus parallel per-slot arrays | `core/tracking/worker.py:873` | Kalman is already fully independent per slot. Arena membership can be a static per-slot label rather than a restructuring. |

### What actually couples arenas today

Three places, and only three, reason across all tracks at once:

1. **Assignment** — `core/assigners/hungarian.py:32` builds a dense N×M cost
   matrix over every track and every detection.
2. **Online identity** — `OnlineIdentityDecoder` enforces a global
   one-individual-one-track constraint in its visible-slot solve
   (`core/individual/identity/online.py:498`), plus commitment tracking and
   swap detection (`online.py:620`).
3. **Post-processing** — `resolve_trajectories`, merge, and relink in
   `core/post/processing.py` operate over all trajectories at once, as does the
   offline identity uniqueness solver (`core/individual/identity/offline.py`).

### The one genuine obstacle

`run_tracking` in `core/tracking/worker.py` is a single ~3300-line function
holding all per-frame state in local variables. This is what makes the
"instantiate N tracker objects" design expensive, and it is why the chosen
design threads arena through as a *label* rather than restructuring control
flow.

## Approaches considered

**A. Block-diagonal single pass — CHOSEN.** One tracker with
`MAX_TARGETS = n_arenas × animals_per_arena`, a static per-slot arena label, and
arena-blocked assignment. One inference pass, one decode pass, runtime
essentially unchanged from today, flat scaling to 100 arenas. Does not require
touching `run_tracking`'s structure.

**B. N tracker instances behind a shared inference pass.** Extract per-frame
tracking state into an `ArenaTracker` object and fan detections out to per-arena
instances. Independence is guaranteed *by construction* — no shared local can
leak across arenas. Rejected as a *first* step: it requires the Slice-4 monolith
decomposition first, which is large, risky, and makes the byte-identity
equivalence gate much harder to hold. This remains the right eventual
destination; the chosen design does not block it.

**C. Re-run tracking per arena from a shared detection cache.** Run inference
once, then loop the tracking pass per arena over cached detections. Zero tracker
changes. Rejected: pays N× video decode and N× tracking orchestration. Viable at
2–6 arenas, disqualifying at 10–100.

## Design

### Data model

An arena is an ROI shape plus an `arena_id`. `roi_shapes` entries gain
`arena_id: int` and an optional display name. Two derived artifacts replace
today's single `ROI_MASK`:

- **`ROI_MASK`** — unchanged: the union of all arena include-shapes minus
  exclusions. Detection gating semantics are exactly as they are today.
- **`ARENA_LABELS`** — a `uint16` label image of the same H×W, where pixel value
  is `arena_id + 1` and 0 means outside every arena. Arena lookup for a
  detection is one array index, O(1) regardless of arena count.

Exclusion shapes subtract from both, so a hole inside an arena is outside every
arena. A detection outside all arenas is dropped by `ROI_MASK` exactly as today.

### Slot layout

`MAX_TARGETS = n_arenas × animals_per_arena`, with slots laid out in contiguous
per-arena blocks. A static `slot_arena` array of length `MAX_TARGETS` is built
once at session start. Every existing per-slot array (`track_states`,
`missed_frames`, `orientation_last`, `trajectory_ids`, …) is untouched.

Animal count is a **single shared value across all arenas**, per the
"identical parameters" requirement. `MAX_TARGETS` is derived, not entered.

### Touch point 1 — detection labeling

`core/inference/stages/filtering.py` already performs the exact centroid lookup
required (`filter_detections:71`, `filter_with_indices:298`, and the CUDA tensor
path `filter_from_tensors:134`). Add a parallel `arena_ids` output alongside
`keep`, sampled from `ARENA_LABELS` at the same clipped centroid coordinates.
Cost is one extra gather on an array already being indexed.

`arena_ids` flows with the detection result through the pipeline and into the
detection cache, so backward passes and cached reruns carry arena membership
without recomputation.

### Touch point 2 — arena-blocked assignment

`_compute_cost_matrix_numba` (`core/assigners/hungarian.py:32`) already uses
`cost = 1e6` as a **hard-reject sentinel**: the post-solve check at line 749
rejects any chosen pair with `cost >= 1e6`. Adding

```
if meas_arena[j] != track_arena[i]:
    cost[i, j] = 1e6
    continue
```

as the *first* test in the inner loop makes Hungarian decompose **exactly** into
independent per-arena assignment problems. This is an identity, not an
approximation, precisely because the sentinel is a hard reject rather than a
large finite cost — a blocked pair can never be selected even when the solver is
forced.

`_apply_bayesian_identity_cost` (`hungarian.py:253`) adds to blocked cells but
only increases them, so they remain `>= 1e6` and remain rejected.

**Blocking is also the performance enabler, not only the correctness
mechanism.** `_apply_bayesian_identity_cost` is a *Python* double loop over N×M.
At 100 arenas × 4 animals that is 400×400 = 160,000 Python iterations per frame,
which is unusable. Skipping off-block pairs collapses it to
`Σ_arena (n × m)` = 100 × 16 = 1,600. The same argument applies to the
spatial-candidate pass (`_get_spatial_candidates`). Therefore block structure
must be **exploited during cost construction**, never applied as a mask over an
already-dense matrix.

### Touch point 3 — per-arena identity decoding

Instantiate **one `OnlineIdentityDecoder` per arena** over the shared catalog,
rather than one global decoder. The decoder is already a self-contained,
slot-keyed object, so its uniqueness solve (`online.py:498`), commitment
tracking, swap detection (`online.py:620`), and slot clearing all become
per-arena with no changes to the decoder itself.

This is exactly what a repeated-label catalog requires: `n_arenas` decoders
enforcing "one ant A per arena" rather than one decoder enforcing "one ant A in
the whole video".

Offline identity resolution (`core/individual/identity/offline.py`) runs its own
uniqueness solver over trajectories and takes the same per-arena grouping.

### Touch point 4 — post-processing grouped by arena

`core/post/processing.py` (`resolve_trajectories`, merge candidates, relink) and
the offline identity pass group by `arena_id`. The pandas path is already
vectorized (merge `b0871a96`), so this is a groupby over existing vectorized
operations rather than a rewrite.

### Touch point 5 — output

The tracking CSV gains an **`arena_id` column**. Track IDs and trajectory IDs
remain **globally unique across arenas**, so RefineKit, post-processing,
dataset export, and every existing CSV consumer work unchanged, and one file
describes the whole video. Per-arena split files are explicitly out of scope.

### GUI — two arena-definition modes, one data model

Both modes produce the same `roi_shapes` list; nothing downstream distinguishes
them.

- **Manual.** The existing draw tool (`trackerkit/gui/orchestrators/session.py:2112`)
  gains an arena selector in the ROI toolbar: "new arena" versus "add to arena
  N". The second is required because one arena is often several shapes — an
  include circle plus an exclude hole — so shape count is not arena count.
- **Grid generator.** A new `trackerkit/gui/dialogs/arena_grid_dialog.py`
  (`BaseDialog` subclass) taking rows × cols, origin, pitch, and shape/size,
  with a live overlay preview on the reference frame. It emits ordinary shapes
  with sequential `arena_id`s that remain individually editable afterwards. The
  generator is a bulk-entry convenience, not a separate mode — this is what
  makes 96-well layouts practical.

### Config and backward compatibility

`roi_shapes` already persists through `TrackerConfig.to_dict`/`from_dict`
(`trackerkit/config/schemas.py:25`), so `arena_id` rides along for free; old
configs simply lack the key.

**Rule: shapes with no `arena_id` all map to arena 0.** A legacy user who drew
three shapes as one region keeps exactly today's behavior. Multi-arena is opt-in
and is never inferred from shape count.

`animals_per_arena` is added alongside, with `MAX_TARGETS` derived from it.

### Backward compatibility as the primary regression gate

With exactly one arena, `slot_arena` is uniform, every block test passes, and
there is a single identity decoder — so a single-arena run must be **byte-identical
to today**. This makes the existing equivalence harness
(`tools/equivalence/run_matrix.sh`, all 7 clips, MPS **and** CUDA) a hard gate on
the entire feature rather than a smoke test.

## Testing

1. **Non-regression (primary gate).** Full equivalence matrix byte-identical on
   MPS and CUDA. Single-arena runs take the uniform-`slot_arena` path, so any
   drift is a real defect.

2. **Synthetic tiling oracle (independence gate).** Tile an existing fixture
   clip 2×2 into one video with four arenas and run it. Each arena's
   trajectories must match the single-clip run **exactly**, modulo a constant
   coordinate offset. Any cross-arena leak — a shared per-frame local, a global
   identity constraint, an ungrouped post-processing step — surfaces as a
   mismatch. This tests the property users actually asked for: "the same as
   running each arena separately." Buildable from existing fixtures.

3. **Unit tests.**
   - Blocked assignment ≡ independent per-arena Hungarian, over random cost
     matrices.
   - `ARENA_LABELS` lookup against exclusion holes and arena boundaries.
   - Legacy `roi_shapes` without `arena_id` → single arena 0.
   - Grid generator produces the expected shape count, ids, and geometry.

4. **Scale profiling, early.** Profile at 100 arenas near the start of
   implementation rather than at the end. Per-frame Python-level overheads that
   are invisible at 10 slots may dominate at 400, and the fix — exploiting block
   structure — is the same lever as correctness.

## Out of scope (YAGNI)

Each is addable later on this data model without rework:

- Per-arena parameter overrides (animal count, thresholds).
- Arena crossing — arena membership as a per-frame track property.
- Per-arena output files.
- Automatic arena detection from the frame (Hough circles / contours).
- Globally-unique identity catalogs across arenas.

## Known limitation — detection-stage budgets remain global

Per-arena independence is achieved for assignment, identity decoding, and
post-processing (Touch points 2-4), but **not** for the detection stage in
`core/background/measure.py`. Two budgets there are still spent across the
whole frame rather than per arena:

- The frame-skip guard (`measure.py:210`) compares total contour count against
  `MAX_TARGETS * MAX_CONTOUR_MULTIPLIER`. Since `MAX_TARGETS` is the derived
  `n_arenas * animals_per_arena`, this threshold scales with arena count, so a
  noisy frame that would have been skipped in a single-arena run may be
  processed under multi-arena, and vice versa.
- The top-N-by-area truncation (`measure.py:261`) keeps the largest
  `N = MAX_TARGETS` contours across the whole frame, so a crowded arena can
  consume detection slots that "belong" to a quiet arena.

This is the one place the "same as running each arena separately" promise does
not hold. Fixing it means giving each arena its own detection budget — a
detection-stage restructure that is deliberately out of scope for this
feature (see Out of scope, above, and the YOLO/OBB path has the analogous
per-frame top-N behavior for the same reason).

In practice this only bites when total detections exceed total slots, i.e.
under noise or crowding; the nominal case (each arena's true detection count
within its own share of `MAX_TARGETS`) is unaffected.

## Risks

- **`run_tracking` local state.** Auditing ~3300 lines to confirm no per-frame
  local silently aggregates across arenas is real work. The tiling oracle is the
  mechanism that actually proves it; the audit alone is not sufficient evidence.
- **Scale-dependent overheads.** Python-level per-frame loops that are cheap at
  N=10 slots may dominate at N=400. Mitigated by exploiting block structure in
  cost construction and by profiling at target scale early.
