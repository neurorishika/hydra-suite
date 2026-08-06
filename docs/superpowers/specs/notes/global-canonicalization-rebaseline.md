# Global canonicalization — re-baseline

Date: 2026-08-06. Branch `feat/global-canonicalization` vs baseline `e6882c0e`
(the branch's merge-base). Runtime MPS, `hydra-mps`, `sleap` env present.

**This supersedes the 2026-08-05 version of this file, which was wrong.** That
run compared against `main` at `a31353f1`, not the merge-base: `main` moved
mid-session and gained ~50 commits from `feat/headless-qt-free`, two of which
(`d46bd7af`, `a1bde2ba`) changed the CLI parameter builder's pose and identity
blocks. The harness drives tracking through the CLI, so those alone shift the
crop clips. The differences were therefore not attributable to this branch.
Re-running against the merge-base showed the confound was real for
`ant_pose_headtail` (11739 -> 9100 matched rows) and absent for
`ant_cnn_identity` (identical numbers) — there was no way to know which without
re-running.

It also predates the six deviations the post-implementation audits found and
fixed (see below), so it measured an intermediate state.

## Correctness results (MPS, vs `e6882c0e`)

| clip | crop consumer | determinism | equivalence | mean \|Δθ\| |
|---|---|---|---|---|
| `fly_obb` | none (control) | EQUIVALENT | **EQUIVALENT** | 0.000 |
| `worm_bgsub` | none (control) | EQUIVALENT | **EQUIVALENT** | 0.000 |
| `ant_obb_sequential` | head-tail | EQUIVALENT | DIFFERENCES | 0.160 (final) |
| `ant_obb_sleap` | SLEAP pose | EQUIVALENT | DIFFERENCES | 0.905 (final) |
| `ant_pose_headtail` | head-tail + pose | EQUIVALENT | DIFFERENCES | 0.534 |
| `ant_cnn_identity` | head-tail + CNN identity | EQUIVALENT | DIFFERENCES | 0.862 (fwd) |
| `emi_obb_identity` | head-tail + identity | EQUIVALENT | — | 0.000 (fwd matched 11067) |

- **Both controls byte-identical.** They run no crop-consuming stage. The blast
  radius is exactly the designed one.
- **Determinism exact on every target**: `new_a` vs `new_b` matched every row
  with `θ max = 0.000e+00`, zero unmatched. The differences below are the
  change, not noise.
- **CSV row counts verified > 1 on every clip**, so no comparison is the
  empty-CSV false pass an inactive conda env produces.

Positions and track structure barely move; heading and identity move. Most clips
match every row with only θ differing. Head-tail and CNN identity are
unretrained models reading correctly-canonicalized crops for the first time.

## Performance — read this before quoting a number

**Wall-clock on the development machine is not usable for ±30 % decisions.**
The same unchanged baseline binary on the same clip measured:

| clip | observed baseline wall-clock across runs |
|---|---|
| `ant_obb_sleap` | 53.4 s – 90.6 s |
| `emi_obb_identity` | 52.1 s – 158.1 s |

A 1.31x "regression" reported on 2026-08-05 for `ant_obb_sleap` was inside that
spread. Measured by **process CPU time** (user+sys, insensitive to competing
load), the branch cost **+2.4 s on a ~59 s run (+4 %)** before the perf fixes
below, of which ~1.1 s was the crop path.

Attribution of that +2.4 s, measured:

| bucket | cost |
|---|---|
| Layer 2 `apply_fit` — the second resample (`cv2.resize` calls 11778 -> 23056, exactly +1/detection) | +1.00 s |
| float32 round trip around the canvas crops | +0.10 s net |
| bigger warp destination + batch bookkeeping | +0.25 s |
| unattributed (allocator, GC) | +1.05 s |

A hypothesis that the fixed canvas warps *more* pixels than the old per-animal
extent was **refuted and is backwards**: baseline warped straight to the
classifier's 224x224 input (50 176 px/detection) while the branch warps to a
154x72 canvas (11 088 px) — baseline resampled 4.5x more pixels per detection.

### Perf fixes landed (all byte-identity preserving)

| commit | change | measured saving |
|---|---|---|
| `dc91b091` | drop the `uint8 -> float32 -> uint8` round trip in the CPU classifier batch | 0.75 s |
| `a081680e` | skip `_preprocess`'s now same-size `cv2.resize`; `apply_fit` no-pad fast path | ~1.3 s |
| `8168bbe6` | run the classifier warp + Layer 2 fit across the existing warp pool | ~1.2 s wall |

Together these remove more work than the branch added. Post-fix wall-clock
ratios were 0.98x (`ant_obb_sleap`) and 0.67x (`emi_obb_identity`) — both inside
the noise band and not quotable as speedups.

**Rejected**: composing Layer 1 ∘ Layer 2 into a single `warpAffine` (~1.0 s).
Geometrically identical, but one bilinear sample instead of two changes pixel
values, and training applies Layer 2 only — it would break the train/inference
byte-identity guarantee the branch exists to establish.

## Deviations found by post-implementation audit and fixed

The plan scoped work by file list, so it converted the call sites it enumerated
and missed consumers that were not on it. The correct scoping question was
"who reaches a model", not "which files does the spec name".

| # | deviation | fix |
|---|---|---|
| A | tiny-classifier training (head-tail's backbone, default non-square 128x64) stretched anisotropically via `cv2.resize`; ClassKit inference workers likewise | shared Layer 2 fit; new non-square guard **demonstrated failing** pre-fix at 99.4 % mismatched pixels |
| B | interpolated-crops worker bypassed Layer 2 for both CNN identity (anisotropic) and pose (wrong scale) | pre-fits like `stages/*` |
| C | `model_input_wh` collapsed a backend's `(W,H)` to a square of its long side | true non-square fit; content scale 4.267 -> 6.4 on the 60x25 -> 384x256 case |
| D | ViTPose training keyed `box2cs` off the tight COCO bbox, inference off the full crop extent | training uses full extent; PoseKit images verified to be one-animal crops |
| E | three fill policies coexisted (`BORDER_REPLICATE`, zeros, `bg_color`) | zeros everywhere for "no data"; background colour retained for foreign-animal masking only |
| F | clipping counted but discarded at all 13 tracking call sites; `warn_on_geometry_mismatch` had no caller; exporter silently fell back to legacy AABB | all three now fire |

The byte-identity guard passed the entire time deviation A existed, because it
only exercised square inputs and never touched the tiny path.

## Operational conclusion

**Head-tail and CNN identity must be retrained before this branch produces
trustworthy tracking output.** Suggested order — validating a downstream model
on top of an unretrained upstream one measures only the compound error:

1. Retrain **head-tail** on new-convention crops.
2. Measure direction agreement against the current model on a held-out clip and
   report the number. `stages/crops.py` records that a mere extra resample once
   flipped 1-2 % of its decisions.
3. Retrain **CNN identity**, then **ViTPose** and **SLEAP**.
4. Re-run this matrix. θ and identity should converge toward baseline; they will
   not return to byte-identity, and should not.

**Regenerate crop datasets rather than reusing them.** The exporter's canonical
path was never live on the tracking path before this branch (`worker.py` assigns
`raw_canonical_affines = None` in all six dispatch branches), so every existing
ClassKit crop dataset came from the legacy AABB path.

## Outstanding

- CUDA matrix on mehek: running at time of writing, branch `b3bec1f0` vs
  `e6882c0e`. Results to be appended.
- Clipping reporting covers the core tracking loop but not the GUI
  interpolated-crops path.
- Re-running old exports will diverge at crop edges (`BORDER_REPLICATE` removed).
