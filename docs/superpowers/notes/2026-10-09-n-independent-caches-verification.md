# N-independent inference caches: verification (MPS + CUDA)

Branch `feat/n-independent-caches` @ `138da28f`, checked against its merge-base with
main, `2f8f9726`. Both trees ran in a detached worktree, with the same conda env, models
and config on each side. `run_matrix.sh` ran base x1 and branch x2 per clip.

- **MPS**: this Mac, `hydra-mps`, `RUNTIME=mps`. Output in `/tmp/equiv_nfree_mps`.
- **CUDA**: diptera, `hydra-cuda`, driver 570, `CUDA_MAJOR=12`. The run was pinned to the idle GPU 4
  (`CUDA_VISIBLE_DEVICES=GPU-5b331492-...`, RTX 6000 Ada). Output in
  `/tmp/equiv_nfree_cuda` on diptera. mehek was unreachable (ssh timeout).
- **Import pin checked**: every run's `meta.json` and log name the expected tree. Base runs
  imported `nfree-equiv-base/src` (2f8f9726). Branch runs imported the branch worktree
  `src` (138da28f). The CUDA log has 9 base and 18 branch imports.
- **No empty CSVs**: every CSV has more than one line.
- `NUMBA_CACHE_DIR` pointed at a fresh directory for each box, so no stale JIT cache was used.

## Verdict

- **DETERMINISM (branch vs branch) is exact on every clip and every CSV, on both platforms.**
  Every pair of branch CSVs is byte-identical.
- **Tracked positions match base on every clip.** In the positional view, base vs branch shows
  position max |Δ| = 0 everywhere. Rows the matcher could not pair appear only on the
  bgsub clips (b) and on MPS relink (e).
- Every base-vs-branch difference is attributed below to (b), (c) or a new category (e).
  (a) and (d) produce no output difference on any fixture.
- **(e) is not a branch defect.** It is a pre-existing base bug: base attached CNN
  predictions to the wrong animal. The branch's raw-index keying fixes it, so the
  branch output is the correct one. It is flagged for the reviewer because it lies outside (a)-(d).
- **Performance**: worst new/base ratio is 1.22x (MPS `ant_obb_sleap`). Re-measured twice,
  interleaved base/new, it came out at 1.01x and 0.98x, so 1.22x was noise. Every clip on
  both platforms is within the 1.25x tolerance.

## Per-clip results (base 2f8f9726 vs branch 138da28f)

"Bytes" means base `==` branch for each CSV. The three groups are forward/backward, final,
and final_with_individual. Rows are given as base/branch. The determinism floor is exact (0)
everywhere.

### MPS

| clip | bytes fwd/bwd | bytes final | rows final | unmatched | pos p99 / max | θ max (final) | differing columns | cause |
|---|---|---|---|---|---|---|---|---|
| ant_pose_headtail | differ | differ | 9096/9096 | 0 | 0 / 0 | 0 | DetectionID (2 rows, frame 54) | (b) |
| ant_obb_sleap | identical | identical | 11885/11885 | 0 | 0 / 0 | 0 | none | none |
| ant_obb_sequential | identical | identical | 941/941 | 0 | 0 / 0 | 0 | none | none |
| worm_bgsub | differ | differ | 2706/2706 | 0 | 0 / 0 | π | TrajectoryID/Theta/State/DetectionID... | (b) bgsub order |
| worm_bgsub_scaled | differ | differ | 1002/1002 | 0 (fwd 1+1) | 0 / 0 | π | Theta/PositionUncertainty/DetectionID | (b) bgsub order |
| ant_cnn_identity | identical | differ | 10246/10246 | 0 | 0 / 0 | 0 | CNN class/conf (756 rows), IdentityEvidence*, UniqueIdentityKey; pose (3 rows) | (e), and (c) for pose |
| ant_cnn_identity_relink | identical | differ | 10262/10254 | 8+0 | 0 / 0 | 0 | as above, plus relink row count | (e), with relink downstream of it |
| fly_obb | identical | identical | 1500/1500 | 0 | 0 / 0 | 0 | none | none |
| fly_obb_roi | identical | identical | 1500/1500 | 0 | 0 / 0 | 0 | none | none |

### CUDA

| clip | bytes fwd/bwd | bytes final | rows final | unmatched | pos p99 / max | θ max (final) | differing columns | cause |
|---|---|---|---|---|---|---|---|---|
| ant_pose_headtail | identical | identical | 9065/9065 | 0 | 0 / 0 | 0 | none | none |
| ant_obb_sleap | identical | identical | 11843/11843 | 0 | 0 / 0 | 0 | none | none |
| ant_obb_sequential | identical | identical | 1076/1076 | 0 | 0 / 0 | 0 | none | none |
| worm_bgsub | differ | differ | 2724/2724 | 0 | 0 / 0 | π | Theta, DetectionID | (b) bgsub order |
| worm_bgsub_scaled | differ | differ | 1060/1061 | 1+2 (fwd 3+0) | 0 / 0 | π | Theta, assignment | (b) bgsub order |
| ant_cnn_identity | differ (θ ≤ 2.05e-4) | differ | 10503/10503 | 0 | 0 / 0 | 2.05e-4 | CNN class/conf (706 rows), IdentityEvidence* | (e), and (c) for θ/pose |
| ant_cnn_identity_relink | differ (θ ≤ 2.05e-4) | differ | 10512/10512 | 0 | 0 / 0 | 2.05e-4 | as above | (e), and (c) |
| fly_obb | identical | identical | 1500/1500 | 0 | 0 / 0 | 0 | none | none |
| fly_obb_roi | identical | identical | 1500/1500 | 0 | 0 / 0 | 0 | none | none |

CUDA shows no nondeterminism: its DETERMINISM floor is exactly 0 on every clip. The known
CUDA YOLO-OBB divergence between main and legacy does not apply here, because the baseline
is 2f8f9726, not `legacy/main`.

## Attribution

### (a) Confidence floor 0.001 -> 0.01: no output effect on any fixture

Step 4 temporarily set `EXTRACTION_CONFIDENCE_FLOOR = 0.001` in
`src/hydra_suite/core/inference/limits.py`. It then ran the branch alone through `runner.py` on
`fly_obb`, `worm_bgsub` and `ant_obb_sleap`, writing to a separate OUT
(`/tmp/equiv_nfree_mps_floor001`). The floor is part of the cache key, so no 0.01 cache was reused.

- The edit took effect. Stored detections per frame rose from mean 36.8 / max 50 to
  81.0 / max 100 on ant_obb_sleap, and from max 5 to max 8 on fly_obb.
- `fly_obb` and `ant_obb_sleap` at floor 0.001 are byte-identical to base on all three
  CSVs. They were already byte-identical at 0.01.
- `worm_bgsub` at floor 0.001 is byte-identical to the branch at 0.01. bgsub has no
  confidence floor, so its difference from base is not (a).
- The edit was reverted. `git diff --quiet` was clean before the commit.

The fixture thresholds (0.05 to 0.3) sit well above both floors, so detections between 0.001
and 0.01 can never survive the confidence gate.

### (b) Ranking order

- **YOLO-OBB tie order** (MPS `ant_pose_headtail` only). Frame 54 has two detections with
  identical confidence 0.949407, and their DetectionIDs 540001 and 540002 swap.
  The tie is bit-exact in both detection caches (float32 `0x3f730c58` = 0.949407101 on both
  rows), and base and branch break it in opposite order.
  Positions, θ and every other column are identical. CUDA shows no swap; that clip is
  byte-identical to base there.
- **Background-subtraction ranking** (worm_bgsub, worm_bgsub_scaled, both platforms). Spec §2
  says bgsub extraction stores every contour sorted by area, descending and stable, and that
  replay keeps the top N by area. Base kept findContours order when there were ≤ N contours.
  Per-frame comparison of the detection caches:
  - The set of detections is identical on **every** frame. The set differs on 0 frames on
    both clips and both platforms.
  - Only the order within a frame changes: 499/500 frames on worm_bgsub and 212/467 on
    worm_bgsub_scaled (MPS); 497/500 and 218/463 on CUDA.
  - The tracker is sensitive to the order detections arrive in (track birth order, ties in
    assignment). Different order gives different TrajectoryID assignment and π flips in
    heading, which is the documented head/tail bistability. It also gives the 1-3 rows
    the positional matcher leaves unpaired on worm_bgsub_scaled. Those unpaired rows are
    `active` rows (MPS forward frames 63 and 178; CUDA final frames 397-399). Each one's
    detection is present in BOTH detection caches on that frame; one tracker run left it
    unassigned because its assignment history differed. No detection was lost.
  - No detection is added or lost. The contour-budget skip was never triggered: it now
    counts stored contours rather than raw findContours output.

### (c) Per-animal superset: positions unchanged, per-animal values change only on superset frames

Per-animal caches were compared by raw detection index between base and branch.
Base pose/headtail caches are raw-keyed as well.

- **Head-tail**: identical on every row on both platforms (MPS 10615 rows, CUDA 10676 rows,
  ant_cnn_identity).
- **Pose**: every differing row lies on a frame where the branch superset had an extra filter
  survivor beyond the final N, meaning 26 per-animal rows against N=25. Frames without
  extras show **0** differing rows on both platforms.
  - **MPS**: 4 frames have extras, and 3 rows differ. Two of them, frame 399 dets 8 and 16,
    lie 109 px and 65 px from the extra detection. Foreign-suppression masks that detection
    inside their crops. Pose quality flips for one row (partial -> rejected).
  - **CUDA**: 23 frames have extras, and 575 rows differ by ≤ 4e-4 px, spread over every
    detection in those frames. This is batch-size-dependent kernel numerics (batch 26 vs 25).
    It reaches θ (≤ 2.05e-4 rad) through pose-direction heading on ant_cnn_identity.
- X and Y are unchanged on every clip.

### (d) Removed 128 clamp: explains nothing here

No fixture comes near 128 detections per frame. `cache_stats.py` reports a branch maximum of 50
stored detections per frame on any clip and platform (ant_obb_sleap MPS; CUDA reaches 49), and
`at_limit=0` everywhere. Per-animal rows reach at most 27. The 128 clamp never bound on these
fixtures, so it cannot account for any difference.

### (e) Not in (a)-(d): base attached CNN predictions to the wrong animal (base bug, fixed by the branch)

Seen on ant_cnn_identity and ant_cnn_identity_relink, on both platforms.

**The base bug.** Base's CNN cache stores `det_index` as a position 0..K-1 over the filtered
set. The base reader (`runner.py:_load_cnn_for_indices`, ~line 880 at 2f8f9726) selects
predictions with `p.det_index in det_set`, where `det_set` holds raw detection indices. So
whenever filtering drops an earlier raw row, positions and raw indices fall out of step.
The prediction for "raw r" is then the prediction of whichever detection sits at position r.
The same reader line (`aligned = [p for p in preds if p.det_index in det_set]`, runner.py:880)
is still on main @4617d692.

**Measured on MPS ant_cnn_identity.**
- 860 final rows have a position different from their raw index.
- 514 of them carry the conf of the detection at position r. That conf differs from the
  row's own detection's conf.
- 336 are NaN in base: no position r exists. This matches IdentityEvidenceSources
  differing on 336 rows.
- **CUDA**: 813 rows shifted, 502 wrong-animal, 304 NaN.

**The branch is correct.** The branch keys CNN by raw index. All 9680 branch rows (CUDA 9831)
carry the cached prediction of their own detection. Per detection, the CNN cache contents are
identical between base and branch. For example, frame 10 has the same 15 predictions in the same
detection order. Only the read-side join changed.

**Effect on output.**
- CNN class/conf changes on 706-756 rows.
- IdentityEvidence*, IdentityEvidenceTopLabel and UniqueIdentityKey change as a consequence.
- On MPS relink, identity-driven relinking produces 8 fewer final rows (10262 -> 10254). All 8 are base-only `occluded` interpolated
  rows of a single trajectory, covering frames 139-146.
  Forward and backward CSVs are byte-identical, so the change is purely post-processing.
- Tracked positions are unchanged.

## Performance (new/base wall-clock, tolerance 1.25x)

| clip | MPS base s | MPS new s | MPS ratio | CUDA base s | CUDA new s | CUDA ratio |
|---|---|---|---|---|---|---|
| ant_pose_headtail | 82.6 | 80.4 | 0.97 | 181.4 | 179.0 | 0.99 |
| ant_obb_sleap | 68.0 | 82.9 | 1.22 (re-run: 1.01, 0.98) | 123.4 | 128.8 | 1.04 |
| ant_obb_sequential | 69.0 | 62.1 | 0.90 | 77.9 | 77.1 | 0.99 |
| worm_bgsub | 29.7 | 29.7 | 1.00 | 80.1 | 76.9 | 0.96 |
| worm_bgsub_scaled | 13.4 | 13.4 | 1.00 | 33.8 | 33.4 | 0.99 |
| ant_cnn_identity | 196.3 | 197.6 | 1.01 | 398.1 | 398.4 | 1.00 |
| ant_cnn_identity_relink | 164.1 | 161.3 | 0.98 | 386.6 | 399.8 | 1.03 |
| fly_obb | 26.7 | 26.6 | 1.00 | 16.6 | 17.2 | 1.04 |
| fly_obb_roi | 26.5 | 26.7 | 1.01 | 16.5 | 17.0 | 1.03 |

**The ant_obb_sleap re-run.** It used `/tmp/equiv_nfree_mps_perf2`, interleaved
base/new/base/new on a quiet box (load average about 2.5):

| run | seconds | ratio |
|---|---|---|
| base | 65.6 | |
| new | 66.3 | 1.01 |
| base | 74.9 | |
| new | 73.7 | 0.98 |

The per-animal superset adds no material work. Per-animal rows per frame have the same mean
(22.6) in base and branch; the per-frame maximum rises by one (25 -> 26). Stored detections per frame
fall: the 2N=50 base store becomes mean 30-37 at the 0.01 floor.

**Diptera conditions.** Diptera was shared during the CUDA run (load average about 44, from
another session's SAM3/DetectKit jobs on GPUs 0, 2 and 9), so the CUDA wall times are noisy. All
ratios are still ≤ 1.04.

## Cost: stored detections and per-animal rows per frame (branch vs base, MPS)

| clip | base det/frame (mean/max) | branch det/frame (mean/max) | per-animal rows base -> branch (mean/max) |
|---|---|---|---|
| ant_pose_headtail | 50.0/50 | 30.1/43 | 18.1/21 -> 18.1/21 |
| ant_obb_sleap | 50.0/50 | 36.8/50 | 22.6/25 -> 22.6/26 |
| ant_obb_sequential | 50.0/50 | 21.5/33 | 1.4/6 -> 1.4/6 |
| ant_cnn_identity | 50.0/50 | 34.1/49 | 21.7/25 -> 21.7/26 |
| fly_obb | 4.0/6 | 3.2/5 | none |
| worm_bgsub | 5.3/8 | 5.3/8 | none |

CUDA matches within ±0.3 on every mean. The CUDA maxima are ant_cnn_identity per-animal 27
and ant_obb_sleap stored 49.

## Stress: 1024 detections in one frame (crop materialisation, `DOWNSTREAM_CHUNK_SIZE=256`)

The setup is one 4512x4512 frame with a 32x32 grid of OBBs, using foreign suppression. Crops
come out in chunks of [256, 256, 256, 256] on every device.

| device | measure | memory |
|---|---|---|
| CPU | tracemalloc peak | 48.1 MiB |
| CPU | max-RSS growth | 214.8 MiB (one float32 CHW frame copy) |
| MPS | driver_allocated delta, before -> after | +40.0 MiB (allocator-retained, not a high-water mark) |
| CUDA (RTX 6000 Ada) | `max_memory_allocated` above the on-device frame | **+99.4 MiB** |

- The CPU and MPS figures come from the Task 12 benchmark.
- The CUDA run used `/tmp/nfree_tools/stress_cuda.py` (run on diptera as `/tmp/nfree_stress_cuda.py`), a CUDA twin of the MPS case in
  `tests/test_downstream_1024_stress.py`. `hydra-cuda` on diptera has no pytest, so pytest was
  stubbed. CUDA memory before was 58.2 MiB and the peak was 157.6 MiB, a true high-water mark.
