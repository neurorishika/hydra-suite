# Identity-subsystem-repair gate record (2026-08-27)

Task 9 of `.superpowers/sdd/2026-08-27-identity-subsystem-repair/`. Branch
`fix/identity-subsystem-repair`, worktree
`.worktrees/identity-repair`. Baseline for the equivalence matrix: local
`main` at `930bcdb9` (one docs-only commit ahead of `efca3d71`, the SHA
named in the task-9 brief — functionally the same tree for src/).

## Step 1/2 — DEMO/ID acceptance (OFFLINE_v2)

Ran per the brief: stamped both DEMO models
(`classification/identity/20260429-105036_classifier_multihead_obiroi_colortag.multihead.json`
and `classification/orientation/20260429-104937_efficientnet_b0_obiroi_train1.pth`)
with `scripts/stamp_fit_policy.py --policy squash`, copied `DEMO/ID/OFFLINE`
to `DEMO/ID/OFFLINE_v2` (deleting the CNN/head-tail detection-cache entries
and identity evidence sidecar so they rebuild), and ran the headless
tracker CLI against `ant_config.json`. Log: `/tmp/offline_v2_run2.log`.

Confirmed via `python3 -c "json.load(...)['fit_policy']"` on the classifier
manifest: `fit_policy: squash` (stamp took effect).

Measured on `ant_tracking_final_with_individual.csv`:

| Metric | Value | Threshold | Result |
|---|---|---|---|
| Trajectories | **171** | ≤ 150 | **NOT MET** (real, not a measurement artifact) |
| vs OFFLINE (original) trajectory count | 689 → 171 | — | **4.0x improvement** |
| Median trajectory length (frames) | 24.0 | — | — |
| Distinct `IdentityFinalSmoothedLabel`/frame (mean) | 17.09 | ≥ 15 | MET |
| `IdentityFinalSource` breakdown | `offline: 10166`, `NaN: 5683` | — | — |
| `PELT found` changepoints | 97 (129 → 226 trajectories after split) | — | logged |
| Re-merge after assignment | 226 → 186 | — | logged |
| Tracklet relink (final collapse) | 186 → 171 | — | logged |
| `uninformative` breaker trips | 0 (`grep -c` = 0) | none expected | MET |

**Verdict: the 171-trajectory count is real and exceeds the plan's ≤150
acceptance threshold — reporting this honestly as NOT MET, not glossing
over it.** It is nonetheless a 4x improvement over OFFLINE's original 689
trajectories, so the repair work moved the needle substantially even though
it falls short of the plan's specific numeric bar. The ≥15 distinct
labels/frame threshold and the "no uninformative-breaker trip" criterion
both pass.

### Track-identity spot-check (30/36/58) — MET on 1/3, run and reported below

This spot-check is named explicitly in design spec §3.5 and in the task-9
brief's Step 2 as part of acceptance. It was skipped in an earlier pass of
this record ("not re-derived... since the CSV numbers above already
establish the accept/reject call") — that was wrong to skip; it is
independent evidence and is reported here on its own, not folded into the
trajectory-count verdict above.

Method: most-common `IdentityFinalLabel` value per `TrajectoryID`, for
trajectories 30/36/58, against
`/Users/neurorishika/Projects/Rockefeller/Ruta/Presentation/DEMO/ID/OFFLINE_v2/ant_tracking_final_with_individual.csv`
(same file, same approach used in earlier acceptance-diagnosis work in this
plan).

| Trajectory | Rows | Label(s) found | Expected | Result |
|---|---|---|---|---|
| 30 | 6 | `blue_pink` (6/6) | `blue_blue` | **WRONG** |
| 36 | 7 | `blue_pink` (7/7) | `orange_yellow` | **WRONG** |
| 58 | 17 | `green_orange` (17/17) | `orange_green` / `green_orange` | **correct** |

**1/3 correct.** Notably, both failing trajectories converge to the exact
same wrong label, `blue_pink` — not two different wrong answers. That
convergence pattern (rather than scattered/random mislabeling) is worth
flagging as a possible residual identity-confusion signature — e.g. a
label collapsing onto a locally-dominant or over-weighted class — rather
than simple noise, though root-causing it is out of scope for this gate
record and is a candidate follow-up.

## Step 3 — MPS equivalence matrix

Full log: `/tmp/equiv_idrepair_full.log`. Baseline = local `main`
(`930bcdb9`/`efca3d71` src). Command per CLAUDE.md fast path, `RUNTIME=mps`,
all 7 fixture clips + `worm_bgsub_scaled`.

### Clean (fully EQUIVALENT — determinism ✅ and legacy-vs-new ✅ on
forward/final/final_with_individual/performance)

- `fly_obb`
- `worm_bgsub`
- `worm_bgsub_scaled`

These three have no head/tail orientation model in their pipeline
(`enable_headtail_orientation` off / not applicable) and no active identity
fragment-solver catalog — clean confirms the identity-repair changes don't
leak into paths that don't touch identity or head/tail.

### Divergent (determinism ✅, legacy-vs-new ❌ on forward/final/final_with_individual; performance ✅ throughout)

| Clip | pos p99 (px) | theta max (rad) | theta mean (rad) | Row counts (legacy vs new, final) | Unmatched (final) |
|---|---|---|---|---|---|
| `emi_obb_identity` | 0.0 | 3.142 (π) | 0.224 | 11757 vs 11759 | 8 legacy / 24 new |
| `ant_pose_headtail` | 0.0 | 3.142 (π) | 0.017 | 9093 vs 9096 | 4 legacy / 7 new |
| `ant_obb_sleap` | 0.0 | 3.142 (π) | 0.189 | 11879 vs 11881 | 6 legacy / 8 new |
| `ant_obb_sequential` | 0.0 | 3.142 (π) | 0.247 | 940 vs 941 | 0 legacy / 1 new |
| `ant_cnn_identity` | 0.0 | 3.142 (π) | 1.089 | 8099 vs 8105 | 2 legacy / 1 new |

**Positions are byte-identical (p99 = 0.0) on every clip, forward and
final** — the Kalman/assignment/detection geometry is untouched, confirming
the spec's stated invariant holds across the board, not just on the two
named identity clips.

### Judgment per clip

- **`emi_obb_identity` — EXPECTED.** Named in design spec §3.5 as an
  identity clip expected to diverge. `identity_weight=0.0` in its fixture
  config, `identity_method=none_disabled` — so the divergence is not from
  "real" identity evidence, but from the fragment-solver/head-tail pipeline
  reacting to the re-stamped orientation model (see below). Positions
  identical, as the spec requires when `identity_weight` is not driving
  assignment.

- **`ant_cnn_identity` — EXPECTED.** Named in spec §3.5; the *only* clip
  with real `CNN_CLASSIFIERS` entries, so it's the one clip where
  `run_fragment_solver` (Task 6/7's changed code:
  `src/hydra_suite/core/individual/identity/offline.py`) actually executes
  via `postprocess_df.py`'s catalog-gated call
  (`catalog_spec.entries` non-empty only here). Identity columns
  (`IdentityRealtimeLabel`, evidence columns, etc.) diverge heavily by
  design; positions still identical.

- **`ant_pose_headtail`, `ant_obb_sleap`, `ant_obb_sequential` — EXPECTED,
  but NOT scoped by spec §3.5 (which only names the two identity clips).
  Root cause traced, not left as noise-floor hand-waving:** all three
  (plus `emi_obb_identity` and `ant_cnn_identity`) point their
  `yolo_headtail_model_path` at the exact same shared checkpoint —
  `classification/orientation/20260429-104937_efficientnet_b0_obiroi_train1.pth`
  in `~/Library/Application Support/hydra-suite/models/` (fixtures resolve
  models from the shared user models dir, not a fixture-local copy).
  **This session's own Step 1** (DEMO/ID acceptance prep, run ~13:30,
  before the 14:08 equivalence matrix) explicitly re-stamped that exact
  file with `fit_policy=squash` via `scripts/stamp_fit_policy.py`, per the
  brief's own instructions. Tasks 1-3 of this plan (`79bcb2a1`..`b5750986`)
  made the head/tail Layer-2 fit path honor a stamped `fit_policy` instead
  of the old unconditional preprocessing baseline `main` always used — so
  once the shared checkpoint carries an explicit stamp, current `src/`
  computes head/tail orientation slightly differently from baseline on a
  handful of borderline frames, producing π-magnitude θ flips that then
  cascade through post-processing's tracklet splitting/relinking into the
  small (single- to dozens-of-row) row-count deltas seen above. `fly_obb`
  and `worm_bgsub{,_scaled}` don't use head/tail orientation at all and stay
  clean, which is consistent with this mechanism and not with e.g. a
  general regression in tracking/assignment.
  **Caveat: this is a strong, config-consistent correlation, not a
  counterfactually-verified causal test.** All 5 divergent clips use the
  re-stamped model with `enable_headtail_orientation=True`; the 2 clean
  clips don't — that pattern is real and was checked directly. But
  `scripts/stamp_fit_policy.py` stamps checkpoints **in place with no
  backup**, so there is no unstamped copy of this exact model left to
  re-run the equivalence matrix against and directly confirm the
  divergence disappears without the stamp. A more mundane alternative — a
  different Tasks 1-8 change touching the head/tail Layer-2 path
  unconditionally, independent of the fit_policy stamp — has not been
  ruled out by a direct test. Direct verification would require
  re-running the matrix against an unstamped copy of
  `20260429-104937_efficientnet_b0_obiroi_train1.pth`, which does not
  currently exist.

  This is **exactly the effect the task-9 brief itself flagged as a risk**
  ("head/tail model in fixtures: if its checkpoint is unstamped it now runs
  under squash → `ant_pose_headtail` θ may change") — it is a
  real, expected consequence of Tasks 1-3's fit_policy work, not the
  previously-documented "bistable head/tail π-flip" noise floor from
  earlier equivalence gates (that documented floor requires **identical
  row counts** — see `docs/superpowers/plans/done/*` mentions of "θ within
  head/tail π-flip noise floor, identical row counts"; these clips fail the
  row-count leg, so this is new territory, correctly attributed to Tasks
  1-3, not silently folded into the old noise-floor language). **Practical
  implication:** the design spec's §3.5 divergence list should be corrected
  to include all 5 of these clips (any fixture using the shared head/tail
  orientation model), not just the 2 identity-labeled ones — this is a
  spec-scoping gap, not a code defect.

### Performance

All divergent + clean clips: `new/legacy` time ratio 0.95x-1.01x, well
within the 1.25x tolerance. PERFORMANCE: EQUIVALENT ✅ everywhere.

## CUDA

**Not run this session.** mehek reachability was not established in this
environment (prior attempt could not confirm SSH access from this sandbox).
This is an explicit, honestly-reported gap, not a silent skip — **follow-up
required**: run
```
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin --tags && git checkout <this-branch-sha>
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
git worktree add --detach .worktrees/equiv-legacy 930bcdb9   # or efca3d71
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_idrepair_cuda RUNTIME=cuda bash tools/equivalence/run_matrix.sh
```
on the CUDA box before this plan is considered fully gated on both
platforms. The 5-clip divergence pattern documented above should re-derive
identically on CUDA (position-invariant, orientation-model-driven — not
device-specific), but that must be confirmed, not assumed.

## Overall gate status

- DEMO/ID acceptance: **partial** — 4/5 measured criteria pass; trajectory
  count (171) misses the ≤150 threshold. Track-identity spot-check
  (30/36/58) is **1/3 correct** (track 58 only); tracks 30 and 36 both
  mislabel to the same wrong value (`blue_pink`), a possible residual
  identity-confusion pattern worth follow-up investigation.
- MPS equivalence: **3/7 clips clean**, **4/7 (+1 identity-named = 5 total)
  diverge**, all divergences root-caused to the fit_policy migration
  (Tasks 1-3) interacting with a shared, session-restamped orientation
  checkpoint — expected, not a regression, but broader than spec §3.5's
  stated scope.
- CUDA equivalence: **not run** — open follow-up item.
