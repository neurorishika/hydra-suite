# PoseKit Smart Select aligned with ClassKit selection

**Status:** Shipped — merged to main (6941adc8d).

## Problem

PoseKit's Smart Select and ClassKit's candidate selection diverged:

| | PoseKit (before) | ClassKit |
|---|---|---|
| Clustering | Ward linkage on cosine distances (n ≤ 2500), else farthest-point centres with no k-means iterations | K-means / MiniBatch, n_init=5, seeded |
| Labeled frames | Only excluded from the pool | Coverage-aware: under-labeled and rare clusters weighted up |
| Within cluster | Centroid-nearest then farthest-point | Diversity, error neighbours, representative picks |
| Model signal | None (separate Active Learning dialog) | One batch mixing errors, uncertainty, rare clusters, diversity |
| Result | Unordered set added to the labeling list | Ordered candidate list, Start Labeling |

PoseKit also has outright bugs: the near-duplicate filter indexes the wrong
frames, the Embedding Explorer raises on close, settings only persist on a
title-bar close, and the embedding cache key includes device and batch size.

## Decisions (agreed with the user)

1. **Full alignment:** ClassKit's clustering and coverage-aware selection plus
   one batch builder that mixes pose-model signals.
2. **Shared core:** the generic pieces move to `hydra_suite.core.selection`
   (pure numpy, light import: `hydra_suite.core` is lazy, unlike
   `hydra_suite.data`/`utils`, which pull in torch). ClassKit keeps its import
   paths through thin re-exports.
3. **UI:** keep Smart Select as a dialog, rebuilt around ClassKit's controls,
   with an ordered candidate table and *Start Labeling*.

## Design

### `hydra_suite.core.selection` (shared, Qt/torch-free)

- `clustering.py` — `SKLearnClusterer` (moved from ClassKit), `label_coverage`.
- `density.py` — kNN density (exact blockwise numpy up to 20k points, hnswlib
  above; never an N×N×D broadcast), cluster densities, k-center
  `select_diverse_samples` (optional *reference* set, seeded generator) and
  greedy-coverage `select_coverage_samples` (facility location: each pick most
  reduces the distance from uncovered points to the covered set).
- `selectors.py` — `UncertaintySelector`, `RepresentativeSelector`,
  `AuditSelector`, `ErrorSet`, `ErrorNeighborSelector`, `RareClusterSelector`,
  all taking an explicit `np.random.Generator`.
- `batch.py` — `SlotBatch`: slot budgets from fractions, claim-once bookkeeping,
  per-reason breakdown, optional near-duplicate suppression on claim.

ClassKit's `BatchAcquisition` is rebuilt on these (same slot order and
semantics) and becomes deterministic for a given seed.

### PoseKit (`posekit/core/selection.py`, pure numpy)

- **Units:** individual crops, or source frames in Frame Mode (unit embedding =
  normalised mean of its instances; labeled if any instance is labeled; signals
  take the worst instance). One code path for both modes.
- **Pose error** (error-neighbour severity): the Evaluation dialog's
  `mean_error_norm` — mean pixel error over visible ground-truth keypoints
  divided by object scale — except that a keypoint the model missed counts as
  error 1.0 instead of being skipped.
- **Uncertainty:** `1 - mean keypoint confidence` (missed keypoint = 0).
- **Batch recipe** (defaults): error neighbours 20%, uncertainty 25%,
  rare clusters 10% (alpha 0.75), fill gaps 30% (greedy coverage from the
  labeled set; k-center chased outlier crops on real data), representative 10%,
  remainder random audit. Slot sizes use largest-remainder rounding. Slots without a signal
  (no model predictions, no errors yet) hand their budget to diversity.
- Predictions come from the main window's prediction cache for the current
  model ("Predict Dataset…"); the dialog reports coverage and never runs
  inference itself.

### Dialog and labeling flow

- Embeddings computed for all frames (needed for coverage), cached with a key
  that excludes device and batch size. Unreadable images are skipped and
  remembered (not fatal), and take no part in clustering or selection.
- Clustering, prediction scoring and selection run in a background worker.
- The candidate table shows order, file, cluster, reason and detail.
- *Add to Labeling Set* / *Start Labeling*: the batch order is stored on the
  project; a "Smart Select batch" sort mode shows the labeling list in that
  order and frame navigation follows it.
- `clusters.csv` covers every frame (used by the cluster sort and the
  cluster-stratified export split).
