# Global canonicalization — MPS re-baseline

Date: 2026-08-05. Branch `feat/global-canonicalization` (29 commits) vs `main` @ `e6882c0e`.
Runtime: MPS, `hydra-mps`, `sleap` env present (SLEAP clips produced real output).

**Baseline choice:** this matrix compares **`main` vs the branch**, NOT `legacy/main` vs the
branch as the CLAUDE.md fast path does. The question here is what *this* change did; using
the legacy tag would conflate it with the whole preceding migration.

## Results

| clip | crop consumer | determinism | equivalence | mean \|Δθ\| | implied flip rate | perf |
|---|---|---|---|---|---|---|
| `fly_obb` | none (control) | EQUIVALENT | **EQUIVALENT** | 0.000 | 0% | 1.00x |
| `worm_bgsub` | none (control) | EQUIVALENT | **EQUIVALENT** | 0.000 | 0% | 1.13x |
| `ant_obb_sequential` | head-tail | EQUIVALENT | DIFFERENCES | 0.160 (final) | ~5% | 1.00x |
| `ant_obb_sleap` | SLEAP pose | EQUIVALENT | DIFFERENCES | 0.905 (final) | ~29% | 1.00x |
| `emi_obb_identity` | head-tail + identity | EQUIVALENT | DIFFERENCES | 0.720 (fwd) | ~23% | 1.19x |
| `ant_cnn_identity` | head-tail + CNN identity | EQUIVALENT | DIFFERENCES | 0.862 (fwd) | ~27% | 1.10x |
| `ant_pose_headtail` | head-tail + pose | EQUIVALENT | DIFFERENCES | 0.993 (final) | ~32% | 1.06x |

Flip rate is inferred from the mean: `|Δθ|` is bimodal at 0 or π (head/tail inversion), so
`mean / π` estimates the fraction of rows whose heading inverted.

## What passes

- **Both controls are byte-identical.** `fly_obb` and `worm_bgsub` run no crop-consuming
  stage; they are unchanged. The blast radius is exactly the designed one.
- **Determinism is exact everywhere.** `new_a` vs `new_b` matched every row with
  `θ max = 0.000e+00` and zero unmatched on all seven targets. The new pipeline is
  fully reproducible; the differences below are the change, not noise.
- **Performance is within tolerance on every clip** (1.00x-1.19x, tolerance 1.25x).
- **CSV row counts verified > 1 on every clip**, so no comparison is the empty-CSV
  false pass that an inactive conda env produces.

## What changed, and why

Positions and track structure barely move; **heading and identity move a lot**. Most clips
match every row (`unmatched = 0/0`) with only θ differing. That is the signature of crop
geometry changing underneath classifiers that have not been retrained: the detector, Kalman
filter and assigner are untouched, so tracks stay put, while every head/tail and identity
decision is now made on inputs those models have never seen.

`ant_cnn_identity` shows the compounding: 8181 `IdentityAssignedLabel` mismatches, 4298
`IdentitySlotLockLabel`, 1727 `State`. Head-tail feeds identity, so an unretrained head-tail
and an unretrained identity classifier do not merely coexist — their errors multiply.

`ant_obb_sleap` fares best of the crop clips (forward pass: 11161 matched, 0/0 unmatched,
76 `State` mismatches) because SLEAP's pose path was always fed an isotropic zero-padded
canvas — its internal fit already matched what Layer 2 does. Head-tail and CNN identity are
the consumers whose input framing genuinely changed.

`ant_obb_sequential` moves least (~5%) — it exercises head-tail on far fewer rows (940).

## Operational conclusion

**This branch must not be used for production tracking until head-tail and CNN identity are
retrained.** The heading field is materially wrong without it — roughly a quarter to a third
of rows inverted — and identity assignment is reshuffled. This is not a defect in the change
(determinism is exact, controls are clean, differences are confined to unretrained
consumers), but it is a hard dependency, not a recommendation.

Suggested order, because validating a downstream model on top of an unretrained upstream one
only measures the compound error:

1. Retrain **head-tail** on new-convention crops.
2. Measure direction agreement against the current model on a held-out clip and report it —
   `stages/crops.py:238-247` records that a mere extra resample once flipped 1-2% of its
   decisions, so this needs a number, not an inference from tracking output.
3. Retrain **CNN identity**.
4. Re-run this matrix. Expect θ and identity to converge back toward `main`; they will not
   return to byte-identity, and should not.
5. Retrain **ViTPose** and **SLEAP** (lower urgency — SLEAP already moves least).

Regenerate crop datasets rather than reusing existing ones: the crop-dataset exporter's
canonical path was never live on the tracking path before this branch
(`worker.py` assigns `raw_canonical_affines = None` in all six dispatch branches), so every
existing ClassKit crop dataset was produced by the legacy AABB path.

## Still outstanding

- **CUDA re-baseline on mehek has not been run.** Required before merge. The `sleap` conda
  env must be active there or the pose clips emit empty CSVs that compare as *falsely*
  equivalent.
- The interpolated-crop ROI canvas in `crops_worker.py` now uses the shared geometry
  instead of its old `INDIVIDUAL_CROP_PADDING` native-scale canvas — an intended change,
  recorded here so the movement is attributed rather than rediscovered.
